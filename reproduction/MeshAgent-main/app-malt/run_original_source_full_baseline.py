"""
Compatible runner for the original app-malt Full source-code flow.

This is not the paper-style Full runner. It intentionally mirrors the public
app-malt/full_cot_with_tools.py structure as closely as possible while using the
local OpenAI-compatible client, local JSON RAG data, unified evaluator, timeouts,
and JSON outputs used by this reproduction workspace.

Key source-style choices:
- pure vector top-k constraint retrieval, not hybrid/RRF;
- pure vector top-1 tool retrieval;
- fixed three-step CoT;
- per-step execution error debugging;
- per-step MyChecker validation debugging;
- DEBUG_LOOP_TOTAL defaults to 3;
- no confidence score and no abstention.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import benchmark
import networkx as nx
from networkx.readwrite import json_graph

from error_check import MyChecker


DEFAULT_OUTPUT_PREFIX = "results_original_source_full50_r3"
DEFAULT_CONSTRAINT_TOP_K = 13
DEFAULT_TOOL_TOP_K = 1
DEFAULT_MAX_DEBUG = 3


SYS_COT_TOOL_SOURCE = """
For the given breakdown step, generate the Python code needed to process the network graph to answer the user question or request.
If there is code available from the last step, you should expand the new code based on it. If there is no code available, just generate from scratch.
If a new step is not needed, just use the same code from last step.
Before generating, check if the extracted tool is useful for the current query, if it is, then you should try to leverage it.

Strictly follow the data input and out format:
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph object) and returns a single object return_object.

The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should contain a networkx graph.
"""


USR_COT_TOOL_SOURCE = """Begin! Your code should only contain the process_graph(). Strictly generate Python code with the following format, without comments:

Answer:
```python
${{Code that will answer the user question or request}}
```

Question: {input}
Constraints: {constraints}
Step: {step}
Code_from_last_step: {code}
Extracted tool: {tool}
"""


SYS_STEP_SOURCE = """
You should behave with chain of thoughts, the first answer is three summarized steps you need to take to answer the user query.

Each node has a 鈥榯ype鈥?attribute and other attributes depending on its type. The 鈥榯ype鈥?attribute is a list, and each element is in the format of 鈥楨K_{{TYPE}}鈥? For example, EK_PACKET_SWITCH indicates this node is a packet switch node. Because it is a list, each node can have multiple types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, EK_AGG_BLOCK, EK_JUPITER, EK_PORT, EK_SPINEBLOCK, EK_PACKET_SWITCH, EK_CONTROL_POINT, EK_CONTROL_DOMAIN.
Each directed edge also has a 鈥榯ype鈥?attribute, where the value RK_CONTAINS indicates the source node contains the destination node, and the value RK_CONTROLS indicates the source node controls the destination node. 
"""


USR_STEP_SOURCE = """Begin! Strictly generate steps with the following string format:

'
Step 1: the first step in your chain of thoughts.
Step 2: the second step in your chain of thoughts.
Step 3: the third step in your chain of thoughts.
'

Question: {input}
"""


SYS_DEBUG_SOURCE = """
Generate the Python code needed to process the network graph to answer the user query. 
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object. 
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query. 
If the output type is 'text' then the 'data' key should be convert to a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.
"""


USR_DEBUG_SOURCE = """Please debug the following code you generated before:
Question: {input}
Constraints: {constraints}
Code: {code}
Error: {error}
"""


@dataclass
class SourceAttemptResult:
    ret: dict[str, Any] | None
    generated_code: str | None = None
    error: str | None = None
    debug_count: int = 0
    checker_passed: bool | None = None
    validation_error: str | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)


class PureVectorJsonRetriever:
    """Small pure vector retriever matching the original Azure vector-only calls."""

    def __init__(self, path: Path, text_key: str, label_key: str | None = None):
        self.path = path
        self.text_key = text_key
        self.label_key = label_key
        self.entries = json.loads(path.read_text(encoding="utf-8"))
        self._embeddings: list[list[float]] | None = None

    def _entry_text(self, entry: dict[str, Any]) -> str:
        parts = []
        if self.label_key and entry.get(self.label_key):
            parts.append(str(entry[self.label_key]))
        parts.append(str(entry.get(self.text_key, "")))
        return " ".join(parts)

    def _embed(self, text: str) -> list[float]:
        resp = benchmark.client.embeddings.create(model="text-embedding-ada-002", input=text)
        return resp.data[0].embedding

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        print(f"[source_vector_rag] Embedding {len(self.entries)} items from {self.path} ...")
        self._embeddings = [self._embed(self._entry_text(entry)) for entry in self.entries]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self._ensure_embeddings()
        q_emb = self._embed(query)
        assert self._embeddings is not None
        scored = [(self._cosine(q_emb, emb), idx) for idx, emb in enumerate(self._embeddings)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        for score, idx in scored[: min(top_k, len(scored))]:
            entry = dict(self.entries[idx])
            entry["vector_score"] = round(score, 6)
            selected.append(entry)
        return selected


_constraint_retriever: PureVectorJsonRetriever | None = None
_tool_retriever: PureVectorJsonRetriever | None = None


def get_constraint_retriever() -> PureVectorJsonRetriever:
    global _constraint_retriever
    if _constraint_retriever is None:
        _constraint_retriever = PureVectorJsonRetriever(Path("data/rag_constraints.json"), "constraint", "label")
    return _constraint_retriever


def get_tool_retriever() -> PureVectorJsonRetriever:
    global _tool_retriever
    if _tool_retriever is None:
        _tool_retriever = PureVectorJsonRetriever(Path("data/rag_tools.json"), "description")
    return _tool_retriever


def entries_to_constraint_text(entries: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("constraint", "")) for item in entries)


def entries_to_tool_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "no tools available"
    return "\n\n".join(str(item.get("tool", "")) for item in entries)


def split_three_steps(raw: str, query: str) -> list[str]:
    parts = re.split(r"Step\s*\d+\s*:\s*", raw)
    steps = [part.strip(" '\n\t") for part in parts if part.strip(" '\n\t")]
    while len(steps) < 3:
        steps.append(f"Complete the original query: {query}")
    return steps[:3]


def _chat_code(system_prompt: str, user_prompt: str) -> str:
    raw = benchmark._chat(system_prompt, user_prompt)
    return benchmark.clean_up_llm_output_func(raw)


def _ret_preview(ret: dict[str, Any] | None, limit: int = 1200) -> dict[str, Any] | None:
    if not isinstance(ret, dict):
        return None
    data = ret.get("data")
    if isinstance(data, nx.Graph):
        preview = {
            "graph_type": data.__class__.__name__,
            "node_count": data.number_of_nodes(),
            "edge_count": data.number_of_edges(),
            "nodes_sample": list(data.nodes())[:20],
            "edges_sample": list(data.edges())[:20],
        }
    else:
        try:
            preview = json.dumps(data, ensure_ascii=False, default=str)
        except TypeError:
            preview = repr(data)
        if isinstance(preview, str) and len(preview) > limit:
            preview = preview[:limit] + f"... <truncated {len(preview) - limit} chars>"
    return {"type": ret.get("type"), "data_preview": preview}


def source_validate_return(ret: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Replicate original source validation scope: MyChecker on graph/non-graph outputs."""
    if not isinstance(ret, dict):
        return False, "return_object is not a dict"
    if "type" not in ret or "data" not in ret:
        return False, "return_object must contain 'type' and 'data'"
    try:
        if ret.get("type") == "graph":
            graph = ret["data"] if isinstance(ret["data"], nx.Graph) else json_graph.node_link_graph(ret["data"])
            ok, err = MyChecker(ret_graph=graph).evaluate_all()
        else:
            ok, err = MyChecker(ret_list=ret).evaluate_all()
        return bool(ok), None if ok else str(err)
    except Exception as exc:
        return False, str(exc)


def execution_debug_loop(
    *,
    query: str,
    graph: nx.Graph,
    code: str,
    constraints_text: str,
    max_debug: int,
) -> tuple[str, dict[str, Any] | None, int, str | None, list[dict[str, Any]]]:
    records = []
    current_code = code
    debug_count = 0
    last_error = None
    for loop in range(max_debug + 1):
        try:
            ret = benchmark._exec(current_code, graph)
            records.append({
                "loop": loop,
                "phase": "execution_check",
                "ok": True,
                "error": None,
                "code": current_code,
                "return_preview": _ret_preview(ret),
            })
            return current_code, ret, debug_count, None, records
        except Exception as exc:
            last_error = str(exc)
            records.append({
                "loop": loop,
                "phase": "execution_check",
                "ok": False,
                "error": last_error,
                "code": current_code,
            })
            if loop >= max_debug:
                break
            debugged = _chat_code(
                SYS_DEBUG_SOURCE,
                USR_DEBUG_SOURCE.format(
                    input=query,
                    constraints=constraints_text,
                    code=current_code,
                    error=last_error,
                ),
            )
            debug_count += 1
            records.append({
                "loop": loop,
                "phase": "execution_debug_regenerate",
                "ok": bool(debugged and "process_graph" in debugged),
                "error": None,
                "code": debugged,
            })
            if debugged and "process_graph" in debugged:
                current_code = debugged
    return current_code, None, debug_count, last_error, records


def validation_debug_loop(
    *,
    query: str,
    graph: nx.Graph,
    code: str,
    ret: dict[str, Any] | None,
    constraints_text: str,
    max_debug: int,
) -> tuple[str, dict[str, Any] | None, int, bool, str | None, list[dict[str, Any]]]:
    records = []
    current_code = code
    current_ret = ret
    debug_count = 0
    ok, validation_error = source_validate_return(current_ret)
    records.append({
        "loop": 0,
        "phase": "source_validation",
        "ok": ok,
        "error": validation_error,
        "code": current_code,
        "return_preview": _ret_preview(current_ret),
    })
    if ok:
        return current_code, current_ret, debug_count, True, None, records

    for loop in range(max_debug):
        extra_entries = get_constraint_retriever().retrieve(str(validation_error), top_k=2)
        debug_constraints = constraints_text + " " + entries_to_constraint_text(extra_entries)
        debugged = _chat_code(
            SYS_DEBUG_SOURCE,
            USR_DEBUG_SOURCE.format(
                input=query,
                constraints=debug_constraints,
                code=current_code,
                error=validation_error,
            ),
        )
        debug_count += 1
        records.append({
            "loop": loop,
            "phase": "validation_debug_regenerate",
            "ok": bool(debugged and "process_graph" in debugged),
            "error": None,
            "code": debugged,
            "extra_constraints": extra_entries,
        })
        if not debugged or "process_graph" not in debugged:
            validation_error = "debugger returned no process_graph code"
            continue
        current_code = debugged
        try:
            current_ret = benchmark._exec(current_code, copy.deepcopy(graph))
            ok, validation_error = source_validate_return(current_ret)
            records.append({
                "loop": loop + 1,
                "phase": "source_validation",
                "ok": ok,
                "error": validation_error,
                "code": current_code,
                "return_preview": _ret_preview(current_ret),
            })
            if ok:
                return current_code, current_ret, debug_count, True, None, records
        except Exception as exc:
            validation_error = str(exc)
            records.append({
                "loop": loop + 1,
                "phase": "validation_debug_execution",
                "ok": False,
                "error": validation_error,
                "code": current_code,
            })
    return current_code, current_ret, debug_count, False, validation_error, records


def run_original_source_full(
    query: str,
    graph: nx.Graph,
    *,
    constraint_top_k: int,
    tool_top_k: int,
    max_debug: int,
) -> SourceAttemptResult:
    constraints = get_constraint_retriever().retrieve(query, top_k=constraint_top_k)
    tools = get_tool_retriever().retrieve(query, top_k=tool_top_k)
    constraints_text = entries_to_constraint_text(constraints)
    tool_text = entries_to_tool_text(tools)
    step_records = []
    debug_count = 0
    steps: list[str] = []
    final_code = None
    final_ret = None
    final_checker_passed = None
    final_validation_error = None

    try:
        steps_raw = benchmark._chat(SYS_STEP_SOURCE, USR_STEP_SOURCE.format(input=query))
        steps = split_three_steps(steps_raw, query)
        previous_code = "None"
        step_codes: list[str] = []

        for step_no, step in enumerate(steps, 1):
            initial_code = _chat_code(
                SYS_COT_TOOL_SOURCE,
                USR_COT_TOOL_SOURCE.format(
                    input=query,
                    constraints=constraints_text,
                    step=step,
                    code=previous_code,
                    tool=tool_text,
                ),
            )
            record = {
                "step": step_no,
                "summary": step,
                "initial_code": initial_code,
                "execution_records": [],
                "validation_records": [],
                "final_code": None,
                "return_preview": None,
                "debug_count": 0,
                "checker_passed": None,
                "validation_error": None,
            }

            if not initial_code or "process_graph" not in initial_code:
                record["checker_passed"] = False
                record["validation_error"] = "no process_graph code generated"
                step_records.append(record)
                continue

            code, ret, exec_debugs, exec_error, exec_records = execution_debug_loop(
                query=query,
                graph=graph,
                code=initial_code,
                constraints_text=constraints_text,
                max_debug=max_debug,
            )
            debug_count += exec_debugs
            record["execution_records"] = exec_records
            if exec_error is not None:
                record["debug_count"] = exec_debugs
                record["checker_passed"] = False
                record["validation_error"] = exec_error
                record["final_code"] = code
                step_records.append(record)
                previous_code = code
                step_codes.append(code)
                continue

            code, ret, val_debugs, ok, validation_error, val_records = validation_debug_loop(
                query=query,
                graph=graph,
                code=code,
                ret=ret,
                constraints_text=constraints_text,
                max_debug=max_debug,
            )
            debug_count += val_debugs
            record["validation_records"] = val_records
            record["debug_count"] = exec_debugs + val_debugs
            record["checker_passed"] = ok
            record["validation_error"] = validation_error
            record["final_code"] = code
            record["return_preview"] = _ret_preview(ret)
            step_records.append(record)
            previous_code = code
            step_codes.append(code)

        for code in reversed(step_codes):
            if code and "process_graph" in code:
                final_code = code
                break

        if not final_code:
            return SourceAttemptResult(
                ret=None,
                error="no process_graph code generated",
                debug_count=debug_count,
                checker_passed=False,
                validation_error="no process_graph code generated",
                constraints=constraints,
                tools=tools,
                steps=steps,
                step_records=step_records,
            )

        try:
            final_ret = benchmark._exec(final_code, graph)
            final_checker_passed, final_validation_error = source_validate_return(final_ret)
        except Exception as exc:
            return SourceAttemptResult(
                ret=None,
                generated_code=final_code,
                error=str(exc),
                debug_count=debug_count,
                checker_passed=False,
                validation_error=str(exc),
                constraints=constraints,
                tools=tools,
                steps=steps,
                step_records=step_records,
            )

        return SourceAttemptResult(
            ret=final_ret,
            generated_code=final_code,
            debug_count=debug_count,
            checker_passed=final_checker_passed,
            validation_error=final_validation_error,
            constraints=constraints,
            tools=tools,
            steps=steps,
            step_records=step_records,
        )
    except Exception as exc:
        return SourceAttemptResult(
            ret=final_ret,
            generated_code=final_code,
            error=str(exc),
            debug_count=debug_count,
            checker_passed=False,
            validation_error=str(exc),
            constraints=constraints,
            tools=tools,
            steps=steps,
            step_records=step_records,
        )


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    raw_correct = sum(1 for item in attempts if item.get("correct"))
    answered = sum(1 for item in attempts if not item.get("abstained"))
    abstained = total - answered
    correct_answered = sum(1 for item in attempts if item.get("correct") and not item.get("abstained"))
    wrong_answered = sum(1 for item in attempts if (not item.get("correct")) and not item.get("abstained"))
    abstained_wrong = sum(1 for item in attempts if (not item.get("correct")) and item.get("abstained"))
    abstained_correct = sum(1 for item in attempts if item.get("correct") and item.get("abstained"))
    raw_wrong = total - raw_correct
    return {
        "total_attempts": total,
        "raw_correct": raw_correct,
        "answered": answered,
        "abstained": abstained,
        "correct_answered": correct_answered,
        "wrong_answered": wrong_answered,
        "abstained_wrong": abstained_wrong,
        "abstained_correct": abstained_correct,
        "raw_accuracy_before_abstention": _ratio(raw_correct, total),
        "total_accuracy": _ratio(correct_answered, total),
        "reliable_accuracy": _ratio(correct_answered, answered),
        "abstain_rate": _ratio(abstained, total),
        "abstain_accuracy": _ratio(correct_answered + abstained_wrong, total),
        "abstain_precision": _ratio(abstained_wrong, abstained),
        "abstain_recall": _ratio(abstained_wrong, raw_wrong),
    }


def strip_private_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_private_fields(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_private_fields(child) for key, child in value.items() if not key.startswith("_")}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(strip_private_fields(value), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def attempt_to_json(
    *,
    query: str,
    attempt_index: int,
    result: SourceAttemptResult,
    gt: dict[str, Any] | None,
    gt_code: str,
    gt_error: str | None,
    correct: bool,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "query": query,
        "attempt": attempt_index,
        "correct": correct,
        "elapsed": round(elapsed, 1),
        "error": result.error,
        "ground_truth_error": gt_error,
        "debug_count": result.debug_count,
        "checker_passed": result.checker_passed,
        "validation_error": result.validation_error,
        "return_type": result.ret.get("type") if isinstance(result.ret, dict) else None,
        "ground_truth_type": gt.get("type") if isinstance(gt, dict) else None,
        "generated_code": result.generated_code,
        "ground_truth_code": gt_code,
        "return_preview": _ret_preview(result.ret),
        "ground_truth_preview": _ret_preview(gt),
        "constraints": result.constraints,
        "tools": result.tools,
        "steps": result.steps,
        "step_records": result.step_records,
        "confidence": None,
        "semantic_consistency": None,
        "abstained": False,
        "_ret": result.ret,
    }


def parse_query_indices(value: str | None, total: int) -> list[int]:
    if not value:
        return list(range(1, total + 1))
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            indices.update(range(int(left), int(right) + 1))
        else:
            indices.add(int(part))
    return sorted(index for index in indices if 1 <= index <= total)


def run(
    *,
    query_indices: list[int],
    runs: int,
    output_prefix: str,
    timeout: float,
    max_debug: int,
    constraint_top_k: int,
    tool_top_k: int,
) -> dict[str, Any]:
    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = timeout

    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    all_queries = list(golden.keys())
    selected_queries = [(idx, all_queries[idx - 1]) for idx in query_indices]

    output_path = Path(f"{output_prefix}_OriginalSourceFull.json")
    summary_path = Path(f"{output_prefix}_summary.json")
    group_output = {
        "experiment": "Original Source Full",
        "source_style_features": {
            "compatible_runner": True,
            "direct_original_script": False,
            "prompt_origin": "MeshAgent-main/app-malt/ai_models_cot.py summary_gen_chain, cot_plus_tool_chain, pySelfDebugger",
            "prompt_modified": False,
            "pure_vector_constraint_retrieval": True,
            "constraint_top_k": constraint_top_k,
            "pure_vector_tool_retrieval": True,
            "tool_top_k": tool_top_k,
            "fixed_three_step_cot": True,
            "execution_error_debug": True,
            "mychecker_validation_debug": True,
            "max_debug": max_debug,
            "confidence_abstention": False,
        },
        "query_indices": query_indices,
        "total_queries": len(selected_queries),
        "runs_per_query": runs,
        "queries": [],
        "metrics": {},
    }

    print(
        f"Original Source Full compatible runner: {len(selected_queries)} queries x {runs} runs "
        f"(timeout={timeout:g}s, max_debug={max_debug})"
    )
    print("No confidence/abstention; all attempts are counted as answered.")
    print()

    for position, (query_index, query) in enumerate(selected_queries, 1):
        query_attempts = []
        print("=" * 78)
        print(f"[{position:02d}/{len(selected_queries):02d}] q={query_index}: {query[:100]}")

        for attempt_index in range(1, runs + 1):
            t0 = time.time()
            _, graph_run = benchmark.getGraphData()
            _, graph_gt = benchmark.getGraphData()

            gt = None
            gt_error = None
            try:
                gt = benchmark._exec_gt(golden[query], graph_gt)
            except Exception as exc:
                gt_error = str(exc)

            result = run_original_source_full(
                query=query,
                graph=graph_run,
                constraint_top_k=constraint_top_k,
                tool_top_k=tool_top_k,
                max_debug=max_debug,
            )

            correct = False
            if result.ret is not None and result.error is None and gt is not None and gt_error is None:
                try:
                    correct = benchmark._cmp(result.ret, gt)
                except Exception as exc:
                    gt_error = str(exc)

            elapsed = time.time() - t0
            attempt_json = attempt_to_json(
                query=query,
                attempt_index=attempt_index,
                result=result,
                gt=gt,
                gt_code=golden[query],
                gt_error=gt_error,
                correct=correct,
                elapsed=elapsed,
            )
            query_attempts.append(attempt_json)

            status = "PASS" if correct else "FAIL"
            err = result.error or result.validation_error or gt_error
            print(
                f"    run {attempt_index}/{runs}: {status} "
                f"({elapsed:.1f}s, dc={result.debug_count}, ret={attempt_json['return_type']}, "
                f"check={result.checker_passed})"
            )
            if err:
                print(f"        {str(err)[:180]}")

        query_metrics = compute_metrics(query_attempts)
        group_output["queries"].append(
            {
                "query_index": query_index,
                "query": query,
                "attempts": query_attempts,
                "metrics": query_metrics,
            }
        )
        flat_attempts = [attempt for item in group_output["queries"] for attempt in item["attempts"]]
        group_output["metrics"] = compute_metrics(flat_attempts)
        write_json(output_path, group_output)

        print(
            f"    query metrics: raw={query_metrics['raw_accuracy_before_abstention']} "
            f"wrong_answered={query_metrics['wrong_answered']}/{query_metrics['answered']}"
        )

    flat_attempts = [attempt for item in group_output["queries"] for attempt in item["attempts"]]
    group_output["metrics"] = compute_metrics(flat_attempts)
    write_json(output_path, group_output)

    summary = {
        "experiment": "Original Source Full",
        "output": str(output_path),
        "total_queries": len(selected_queries),
        "runs_per_query": runs,
        **group_output["metrics"],
    }
    write_json(summary_path, [summary])

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved output to {output_path}")
    print(f"Saved summary to {summary_path}")
    return group_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the original Source Full style MALT baseline.")
    parser.add_argument("--query-indices", default=None, help="Comma/range query indices, e.g. 1-50 or 1,2,10.")
    parser.add_argument("--limit", type=int, default=None, help="Shortcut for first N queries when --query-indices is omitted.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-debug", type=int, default=DEFAULT_MAX_DEBUG)
    parser.add_argument("--constraint-top-k", type=int, default=DEFAULT_CONSTRAINT_TOP_K)
    parser.add_argument("--tool-top-k", type=int, default=DEFAULT_TOOL_TOP_K)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    golden = json.loads(Path("golden_answer_generator/prompt_golden_ans.json").read_text(encoding="utf-8"))
    if args.query_indices:
        query_indices = parse_query_indices(args.query_indices, total=len(golden))
    else:
        limit = args.limit or 50
        query_indices = list(range(1, min(limit, len(golden)) + 1))
    if not query_indices:
        raise ValueError("No query indices selected.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    run(
        query_indices=query_indices,
        runs=args.runs,
        output_prefix=args.output_prefix,
        timeout=args.timeout,
        max_debug=args.max_debug,
        constraint_top_k=args.constraint_top_k,
        tool_top_k=args.tool_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
