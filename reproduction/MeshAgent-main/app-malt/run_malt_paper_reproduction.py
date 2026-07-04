import argparse
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import benchmark
import networkx as nx
from networkx.readwrite import json_graph


DEBUG_MAX_DEFAULT = 5
CONFIDENCE_THRESHOLD_DEFAULT = 0.7
MAX_PREVIEW_CHARS = 2000


SYS_COT_TOOL = """
For the given breakdown step, generate the Python code needed to process the network graph to answer the user question or request.
If there is code available from the last step, expand the new code based on it. If there is no code available, generate from scratch.
If a new step is not needed, keep the same process_graph logic from the last step.
Before generating, check if the extracted tool is useful for the current query. If it is useful, adapt it carefully.

The Python code must define process_graph(graph_data) and return a JSON-compatible object with keys 'type' and 'data'.
The 'type' must be one of 'text', 'list', 'table', or 'graph'.
Use a networkx graph as input. If returning a graph, return a networkx graph object in 'data'.
"""

USR_COT_TOOL = """Begin! Your output must only contain one Python code block.

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

SYS_DEBUG_CONTEXT = """
Generate corrected Python code for the network graph query.
The code must define process_graph(graph_data) and return a dict with keys 'type' and 'data'.
The returned type must be one of 'text', 'list', 'table', or 'graph'.
Only output the corrected function in a Python code block.
"""

USR_DEBUG_CONTEXT = """The previous code failed during the MeshAgent error-reduction loop.

Question: {input}
Step: {step}
Constraints: {constraints}
Extracted tool: {tool}
Previous code:
{code}

Error or violated constraint:
{error}
"""


@dataclass
class AttemptResult:
    ret: dict[str, Any] | None
    generated_code: str | None = None
    error: str | None = None
    debug_count: int = 0
    checker_passed: bool | None = None
    return_type_match: bool | None = None
    validation_error: str | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)


def _safe_group_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.replace("+", "")).strip("_")


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


def _truncate(text: str, limit: int = MAX_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _graph_preview(graph: nx.Graph) -> dict[str, Any]:
    nodes = list(graph.nodes())[:20]
    edges = list(graph.edges())[:20]
    return {
        "graph_type": graph.__class__.__name__,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes_sample": nodes,
        "edges_sample": edges,
    }


def preview_value(value: Any) -> Any:
    if isinstance(value, nx.Graph):
        return _graph_preview(value)
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    return _truncate(text)


def preview_return(ret: dict[str, Any] | None) -> dict[str, Any] | None:
    if ret is None:
        return None
    return {
        "type": ret.get("type"),
        "data_preview": preview_value(ret.get("data")),
    }


def canonical_signature(ret: dict[str, Any] | None) -> str:
    if not isinstance(ret, dict):
        return "null"
    rtype = ret.get("type")
    data = ret.get("data")
    if isinstance(data, nx.Graph):
        payload = _graph_preview(data)
    else:
        payload = data
    try:
        return json.dumps({"type": rtype, "data": payload}, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr((rtype, payload))


def infer_expected_type_from_query(query: str) -> str | None:
    q = query.lower()
    if "return the new graph" in q or "return the balanced graph" in q or "return graph" in q:
        return "graph"
    if "return a table" in q or "return table" in q:
        return "table"
    if "return a list" in q or "return list" in q:
        return "list"
    if "return one number" in q or "return a number" in q or "return a string" in q:
        return "text"
    if "output bandwidth" in q and "return one number" in q:
        return "text"
    return None


def _simple_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(tok) > 1}


class HybridJsonRetriever:
    """Small local hybrid retriever: keyword rank + embedding rank with RRF fusion."""

    def __init__(self, path: Path, text_key: str, label_key: str | None = None):
        self.path = path
        self.text_key = text_key
        self.label_key = label_key
        self.entries = json.loads(path.read_text(encoding="utf-8"))
        self._embeddings: list[list[float]] | None = None
        self.embedding_error: str | None = None

    def _entry_text(self, entry: dict[str, Any]) -> str:
        parts = []
        if self.label_key and entry.get(self.label_key):
            parts.append(str(entry[self.label_key]))
        parts.append(str(entry.get(self.text_key, "")))
        return " ".join(parts)

    def _keyword_rank(self, query: str) -> list[int]:
        q_tokens = _simple_tokens(query)
        scored = []
        for idx, entry in enumerate(self.entries):
            text = self._entry_text(entry).lower()
            tokens = _simple_tokens(text)
            overlap = len(q_tokens & tokens)
            phrase_bonus = sum(1 for token in q_tokens if token in text)
            score = overlap * 2 + phrase_bonus
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for score, idx in scored if score > 0] or [idx for _, idx in scored]

    def _embed(self, text: str) -> list[float]:
        resp = benchmark.client.embeddings.create(model="text-embedding-ada-002", input=text)
        return resp.data[0].embedding

    def _ensure_embeddings(self):
        if self._embeddings is not None or self.embedding_error is not None:
            return
        try:
            self._embeddings = [self._embed(self._entry_text(entry)) for entry in self.entries]
        except Exception as exc:
            self.embedding_error = str(exc)
            self._embeddings = None

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _vector_rank(self, query: str) -> list[int]:
        self._ensure_embeddings()
        if self._embeddings is None:
            return []
        try:
            q_emb = self._embed(query)
        except Exception as exc:
            self.embedding_error = str(exc)
            return []
        scored = [(self._cosine(q_emb, emb), idx) for idx, emb in enumerate(self._embeddings)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored]

    def retrieve(self, query: str, top_k: int, rrf_k: int = 60) -> list[dict[str, Any]]:
        keyword_rank = self._keyword_rank(query)
        vector_rank = self._vector_rank(query)
        scores: Counter[int] = Counter()
        rank_sources = [keyword_rank]
        if vector_rank:
            rank_sources.append(vector_rank)

        for ranked in rank_sources:
            for rank, idx in enumerate(ranked, 1):
                scores[idx] += 1 / (rrf_k + rank)

        selected = []
        for idx, score in scores.most_common(top_k):
            entry = dict(self.entries[idx])
            entry["rrf_score"] = round(score, 6)
            entry["retrieval_warning"] = self.embedding_error
            selected.append(entry)
        return selected


_constraint_retriever: HybridJsonRetriever | None = None
_tool_retriever: HybridJsonRetriever | None = None


def get_constraint_retriever() -> HybridJsonRetriever:
    global _constraint_retriever
    if _constraint_retriever is None:
        _constraint_retriever = HybridJsonRetriever(Path("data/rag_constraints.json"), "constraint", "label")
    return _constraint_retriever


def get_tool_retriever() -> HybridJsonRetriever:
    global _tool_retriever
    if _tool_retriever is None:
        _tool_retriever = HybridJsonRetriever(Path("data/rag_tools.json"), "description")
    return _tool_retriever


def entries_to_constraint_text(entries: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{item.get('id')}] {item.get('constraint', '')}" for item in entries)


def entries_to_tool_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "no tools available"
    return "\n\n".join(f"[{item.get('id')}] {item.get('description', '')}\n{item.get('tool', '')}" for item in entries)


def _get_all_constraints_text() -> str:
    path = Path("data/rag_constraints.json")
    entries = json.loads(path.read_text(encoding="utf-8"))
    return entries_to_constraint_text(entries)


def validate_return(ret: dict[str, Any] | None, expected_type: str | None) -> tuple[bool, str | None]:
    if not isinstance(ret, dict):
        return False, "return_object is not a dict"
    if "type" not in ret or "data" not in ret:
        return False, "return_object must contain 'type' and 'data'"

    actual_type = ret.get("type")
    if actual_type not in {"text", "list", "table", "graph"}:
        return False, f"invalid return type: {actual_type}"
    if expected_type and actual_type != expected_type:
        return False, f"expected return type {expected_type}, got {actual_type}"

    data = ret.get("data")
    if actual_type == "list" and not isinstance(data, list):
        return False, "list output must use a list in data"
    if actual_type == "table" and not (isinstance(data, list) and all(isinstance(row, list) for row in data)):
        return False, "table output must be a list of rows"
    if actual_type == "text" and not isinstance(data, (str, int, float)):
        return False, "text output must be string-like"
    if actual_type == "graph":
        try:
            graph = data if isinstance(data, nx.Graph) else json_graph.node_link_graph(data)
            checker = benchmark.MyChecker(ret_graph=graph)
            ok, err = checker.evaluate_all()
            if not ok:
                return False, str(err)
        except Exception as exc:
            return False, str(exc)

    return True, None


def _chat_code(system_prompt: str, user_prompt: str) -> str:
    raw = benchmark._chat(system_prompt, user_prompt)
    return benchmark.clean_up_llm_output_func(raw)


def run_single_pass(query: str, graph: nx.Graph, expected_type: str | None, constraints: str) -> AttemptResult:
    try:
        code = _chat_code(benchmark.SYS_SINGLE, benchmark.USR_SINGLE.format(input=query, constraints=constraints))
        ret = benchmark._exec(code, graph)
        checker_passed, validation_error = validate_return(ret, expected_type)
        return AttemptResult(
            ret=ret,
            generated_code=code,
            checker_passed=checker_passed,
            return_type_match=(ret.get("type") == expected_type) if isinstance(ret, dict) and expected_type else None,
            validation_error=validation_error,
        )
    except Exception as exc:
        return AttemptResult(ret=None, error=str(exc), checker_passed=False, validation_error=str(exc))


def run_baseline(query: str, graph: nx.Graph, expected_type: str | None) -> AttemptResult:
    return run_single_pass(query, graph, expected_type, constraints="")


def run_all_constraints(query: str, graph: nx.Graph, expected_type: str | None) -> AttemptResult:
    return run_single_pass(query, graph, expected_type, constraints=_get_all_constraints_text())


def split_steps(raw: str, query: str) -> list[str]:
    parts = re.split(r"Step\s*\d+\s*:\s*", raw)
    steps = [part.strip(" '\n\t") for part in parts if part.strip(" '\n\t")]
    if len(steps) < 3:
        steps.extend([f"Complete the original query: {query}"] * (3 - len(steps)))
    return steps[:3]


def _execute_and_reduce(
    query: str,
    step: str,
    graph: nx.Graph,
    code: str,
    constraints_text: str,
    tool_text: str,
    expected_type: str | None,
    max_debug: int,
) -> tuple[str, dict[str, Any] | None, int, bool, str | None, list[dict[str, Any]]]:
    debug_count = 0
    records = []
    current_code = code

    for loop_index in range(max_debug + 1):
        try:
            ret = benchmark._exec(current_code, graph)
            ok, validation_error = validate_return(ret, expected_type)
            records.append({
                "loop": loop_index,
                "phase": "execute_validate",
                "ok": ok,
                "error": validation_error,
                "return_type": ret.get("type") if isinstance(ret, dict) else None,
            })
            if ok:
                return current_code, ret, debug_count, True, None, records
            error_text = validation_error or "constraint validation failed"
        except Exception as exc:
            ret = None
            error_text = str(exc)
            records.append({
                "loop": loop_index,
                "phase": "execute",
                "ok": False,
                "error": error_text,
                "return_type": None,
            })

        if loop_index >= max_debug:
            return current_code, ret, debug_count, False, error_text, records

        extra_constraints = get_constraint_retriever().retrieve(error_text, top_k=3)
        full_constraints = constraints_text + "\n" + entries_to_constraint_text(extra_constraints)
        debug_prompt = USR_DEBUG_CONTEXT.format(
            input=query,
            step=step,
            constraints=full_constraints,
            tool=tool_text,
            code=current_code,
            error=error_text,
        )
        current_code = _chat_code(SYS_DEBUG_CONTEXT, debug_prompt)
        debug_count += 1
        records.append({
            "loop": loop_index,
            "phase": "debug_regenerate",
            "ok": bool(current_code and "process_graph" in current_code),
            "error": None if current_code else "debugger returned empty code",
            "return_type": None,
        })

    return current_code, None, debug_count, False, "debug loop exhausted", records


def run_full_meshagent(
    query: str,
    graph: nx.Graph,
    expected_type: str | None,
    constraint_top_k: int = 9,
    tool_top_k: int = 1,
    max_debug: int = DEBUG_MAX_DEFAULT,
) -> AttemptResult:
    constraints = get_constraint_retriever().retrieve(query, top_k=constraint_top_k)
    tools = get_tool_retriever().retrieve(query, top_k=tool_top_k)
    constraints_text = entries_to_constraint_text(constraints)
    tool_text = entries_to_tool_text(tools)
    debug_count = 0
    step_records: list[dict[str, Any]] = []
    generated_code = None
    ret = None

    try:
        steps_raw = benchmark._chat(benchmark.SYS_STEP, benchmark.USR_STEP.format(input=query))
        steps = split_steps(steps_raw, query)
        previous_code = "None"

        for step_no, step in enumerate(steps, 1):
            code = _chat_code(
                SYS_COT_TOOL,
                USR_COT_TOOL.format(
                    input=query,
                    constraints=constraints_text,
                    step=step,
                    code=previous_code,
                    tool=tool_text,
                ),
            )
            if not code or "process_graph" not in code:
                step_records.append({
                    "step": step_no,
                    "summary": step,
                    "error": "no process_graph code generated",
                    "debug_count": 0,
                })
                continue

            code, ret, dc, ok, validation_error, records = _execute_and_reduce(
                query=query,
                step=step,
                graph=graph,
                code=code,
                constraints_text=constraints_text,
                tool_text=tool_text,
                expected_type=None,
                max_debug=max_debug,
            )
            debug_count += dc
            generated_code = code
            previous_code = code
            step_records.append({
                "step": step_no,
                "summary": step,
                "debug_count": dc,
                "checker_passed": ok,
                "validation_error": validation_error,
                "records": records,
            })

        if generated_code is None:
            return AttemptResult(
                ret=None,
                error="Full MeshAgent did not produce process_graph code",
                debug_count=debug_count,
                checker_passed=False,
                validation_error="no process_graph code generated",
                constraints=constraints,
                tools=tools,
                steps=steps,
                step_records=step_records,
            )

        ret = benchmark._exec(generated_code, graph)
        checker_passed, validation_error = validate_return(ret, expected_type)
        if not checker_passed and debug_count < max_debug:
            generated_code, ret, dc, checker_passed, validation_error, records = _execute_and_reduce(
                query=query,
                step="final output validation",
                graph=graph,
                code=generated_code,
                constraints_text=constraints_text,
                tool_text=tool_text,
                expected_type=expected_type,
                max_debug=max_debug - debug_count,
            )
            debug_count += dc
            step_records.append({
                "step": "final",
                "summary": "final output validation",
                "debug_count": dc,
                "checker_passed": checker_passed,
                "validation_error": validation_error,
                "records": records,
            })

        return AttemptResult(
            ret=ret,
            generated_code=generated_code,
            debug_count=debug_count,
            checker_passed=checker_passed,
            return_type_match=(ret.get("type") == expected_type) if isinstance(ret, dict) and expected_type else None,
            validation_error=validation_error,
            constraints=constraints,
            tools=tools,
            steps=steps,
            step_records=step_records,
        )
    except Exception as exc:
        return AttemptResult(
            ret=ret,
            generated_code=generated_code,
            error=str(exc),
            debug_count=debug_count,
            checker_passed=False,
            validation_error=str(exc),
            constraints=constraints,
            tools=tools,
            steps=steps if "steps" in locals() else [],
            step_records=step_records,
        )


def assign_confidence(attempts: list[dict[str, Any]], threshold: float, max_debug: int):
    signatures = [
        canonical_signature(item.get("_ret"))
        for item in attempts
        if item.get("_ret") is not None and not item.get("error") and item.get("checker_passed") is not False
    ]
    counts = Counter(signatures)

    for item in attempts:
        if item.get("_ret") is None or item.get("error") or item.get("checker_passed") is False:
            confidence = 0.0
            semantic_consistency = 0.0
        else:
            signature = canonical_signature(item.get("_ret"))
            semantic_consistency = counts[signature] / len(signatures) if signatures else 0.0
            debug_component = 1 - min(item.get("debug_count", 0), max_debug) / max_debug if max_debug else 1.0
            confidence = 0.5 * semantic_consistency + 0.5 * debug_component

        item["semantic_consistency"] = round(semantic_consistency, 4)
        item["confidence"] = round(confidence, 4)
        item["abstained"] = confidence < threshold


def attempt_to_json(
    *,
    query: str,
    attempt_index: int,
    result: AttemptResult,
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
        "return_type_match": result.return_type_match,
        "validation_error": result.validation_error,
        "return_type": result.ret.get("type") if isinstance(result.ret, dict) else None,
        "ground_truth_type": gt.get("type") if isinstance(gt, dict) else None,
        "generated_code": result.generated_code,
        "ground_truth_code": gt_code,
        "return_preview": preview_return(result.ret),
        "ground_truth_preview": preview_return(gt),
        "constraints": result.constraints,
        "tools": result.tools,
        "steps": result.steps,
        "step_records": result.step_records,
        "_ret": result.ret,
    }


def strip_private_fields(obj: Any) -> Any:
    if isinstance(obj, list):
        return [strip_private_fields(item) for item in obj]
    if isinstance(obj, dict):
        return {key: strip_private_fields(value) for key, value in obj.items() if not key.startswith("_")}
    return obj


def write_json(path: Path, data: Any):
    path.write_text(json.dumps(strip_private_fields(data), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_methods(max_debug: int) -> dict[str, Callable[[str, nx.Graph, str | None], AttemptResult]]:
    return {
        "Baseline": run_baseline,
        "+Constraints": run_all_constraints,
        "Full MeshAgent": lambda query, graph, expected_type: run_full_meshagent(
            query=query,
            graph=graph,
            expected_type=expected_type,
            max_debug=max_debug,
        ),
    }


def run(
    limit: int,
    runs: int,
    output_prefix: str,
    groups: list[str],
    methods: dict[str, Callable[[str, nx.Graph, str | None], AttemptResult]] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    max_debug: int = DEBUG_MAX_DEFAULT,
) -> list[dict[str, Any]]:
    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    queries = list(golden.keys())[:limit]
    methods = methods or build_methods(max_debug=max_debug)

    print(f"MALT paper-style reproduction: {len(queries)} queries x {runs} runs x {len(groups)} groups")
    print(f"confidence_threshold={confidence_threshold}, max_debug={max_debug}")
    print()

    summary = []
    for group in groups:
        if group not in methods:
            raise ValueError(f"Unknown group {group!r}. Available: {', '.join(methods)}")

        print("=" * 78)
        print(f"  {group}")
        print("=" * 78)

        group_output = {
            "experiment": group,
            "paper_style_features": {
                "query_specific_constraints": group == "Full MeshAgent",
                "hybrid_rrf_retrieval": group == "Full MeshAgent",
                "tool_retrieval": group == "Full MeshAgent",
                "cot_decomposition": group == "Full MeshAgent",
                "execution_error_reduction": group == "Full MeshAgent",
                "constraint_error_reduction": group == "Full MeshAgent",
                "confidence_abstention": True,
                "runs_per_query": runs,
            },
            "total_queries": len(queries),
            "runs_per_query": runs,
            "confidence_threshold": confidence_threshold,
            "queries": [],
            "metrics": {},
        }
        out_path = Path(f"{output_prefix}_{_safe_group_name(group)}.json")

        for query_index, query in enumerate(queries, 1):
            query_attempts = []
            expected_type = infer_expected_type_from_query(query)
            print(f"[{query_index:02d}/{len(queries)}] {query[:95]}")

            for attempt_index in range(1, runs + 1):
                t0 = time.time()
                _, graph_run = benchmark.getGraphData()
                _, graph_gt = benchmark.getGraphData()

                gt = None
                gt_error = None
                try:
                    gt = benchmark._exec_gt(golden[query], graph_gt)
                    expected_type = expected_type or gt.get("type")
                except Exception as exc:
                    gt_error = str(exc)

                result = methods[group](query, graph_run, expected_type)
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

            assign_confidence(query_attempts, threshold=confidence_threshold, max_debug=max_debug)
            query_metrics = compute_metrics(query_attempts)
            group_output["queries"].append({
                "query_index": query_index,
                "query": query,
                "expected_type": expected_type,
                "attempts": query_attempts,
                "metrics": query_metrics,
            })

            flat_attempts = [attempt for item in group_output["queries"] for attempt in item["attempts"]]
            group_output["metrics"] = compute_metrics(flat_attempts)
            write_json(out_path, group_output)

            print(
                f"    query metrics: raw={query_metrics['raw_accuracy_before_abstention']} "
                f"reliable={query_metrics['reliable_accuracy']} abstain={query_metrics['abstain_rate']}"
            )

        flat_attempts = [attempt for item in group_output["queries"] for attempt in item["attempts"]]
        group_output["metrics"] = compute_metrics(flat_attempts)
        write_json(out_path, group_output)

        metrics = group_output["metrics"]
        row = {
            "experiment": group,
            "total_queries": len(queries),
            "runs_per_query": runs,
            **metrics,
        }
        summary.append(row)
        print(f"\n>>> {group}")
        print(
            f"    raw={metrics['raw_accuracy_before_abstention']} "
            f"total={metrics['total_accuracy']} reliable={metrics['reliable_accuracy']} "
            f"abstain={metrics['abstain_rate']}"
        )
        print(f"    Saved to {out_path}\n")

    summary_path = Path(f"{output_prefix}_summary.json")
    write_json(summary_path, summary)
    print(f"Summary saved to {summary_path}")
    return summary


def parse_groups(group_arg: str) -> list[str]:
    default = ["Baseline", "+Constraints", "Full MeshAgent"]
    if not group_arg:
        return default
    return [item.strip() for item in group_arg.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-style MALT MeshAgent reproduction runner.")
    parser.add_argument("--limit", type=int, default=50, help="Number of MALT queries to run from the start.")
    parser.add_argument("--runs", type=int, default=3, help="Repeated runs per query, matching the paper-style variance control.")
    parser.add_argument("--timeout", type=float, default=120, help="Generated-code execution timeout in seconds.")
    parser.add_argument("--output-prefix", default="results_malt_paper50", help="Output JSON prefix.")
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD_DEFAULT)
    parser.add_argument("--max-debug", type=int, default=DEBUG_MAX_DEFAULT, help="Max error-reduction loops before abstention.")
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated groups. Default: Baseline,+Constraints,Full MeshAgent",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout
    groups = parse_groups(args.groups)
    run(
        limit=args.limit,
        runs=args.runs,
        output_prefix=args.output_prefix,
        groups=groups,
        confidence_threshold=args.confidence_threshold,
        max_debug=args.max_debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
