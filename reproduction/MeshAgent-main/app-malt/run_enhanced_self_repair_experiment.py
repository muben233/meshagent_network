import argparse
import copy
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import benchmark
import networkx as nx

import reanalyze_enhanced_validation as enhanced
import run_malt_paper_reproduction as paper


DEFAULT_LIMIT = 50
DEFAULT_RUNS = 3
DEFAULT_OUTPUT_PREFIX = "results_enhanced_selfrepair50_r3"
EXPERIMENT_NAME = "Full+EnhancedValidationSelfRepair"


SYS_ENHANCED_REPAIR = """
Generate corrected Python code for a MeshAgent graph query.
The code must define process_graph(graph_data) and return a dict with keys 'type' and 'data'.
The returned type must be one of 'text', 'list', 'table', or 'graph'.
If returning a graph, return a networkx graph object in 'data'.
Use the original question, current step, retrieved constraints, available tool, previous code,
execution error, and validation failures to repair the code.
Only output the corrected function in a Python code block.
"""


USR_ENHANCED_REPAIR = """The previous code failed enhanced validation in the MeshAgent error-reduction loop.

Original question:
{input}

Current step:
{step}

Expected final return type:
{expected_type}

Retrieved constraints:
{constraints}

Extracted tool:
{tool}

Previous code:
{code}

Execution error:
{execution_error}

Enhanced validation failures:
{validation_failures}

Please revise the code so that it:
- still answers the original question;
- keeps the process_graph(graph_data) function signature;
- returns the expected output type;
- satisfies all retrieved constraints;
- fixes every listed validation failure;
- avoids changing unrelated graph structure.
"""


def parse_query_indices(value: str | None, total: int, limit: int | None) -> list[int]:
    if value:
        indices: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start = int(left)
                end = int(right)
                if end < start:
                    raise ValueError(f"invalid query range: {part}")
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
    else:
        count = limit or DEFAULT_LIMIT
        indices = list(range(1, min(count, total) + 1))

    deduped = []
    seen = set()
    for idx in indices:
        if idx < 1 or idx > total:
            raise ValueError(f"query index out of range: {idx}; valid range is 1-{total}")
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _failure_text(validation: dict[str, Any] | None, execution_error: str | None = None) -> str:
    lines: list[str] = []
    if execution_error:
        lines.append(f"- [error_check/execution] {execution_error}")
    if validation:
        for failure in validation.get("critical_failures", []):
            source = failure.get("source") or "validation_test"
            name = failure.get("name") or "unknown"
            message = failure.get("message") or "failed"
            lines.append(f"- [{source}/{name}] {message}")
        warnings = validation.get("warnings", [])
        if warnings:
            lines.append("Warnings:")
            for warning in warnings:
                source = warning.get("source") or "validation_test"
                name = warning.get("name") or "unknown"
                message = warning.get("message") or "warning"
                lines.append(f"- [{source}/{name}] {message}")
    return "\n".join(lines) if lines else "- enhanced validation failed"


def _critical_error_text(validation: dict[str, Any] | None, execution_error: str | None = None) -> str | None:
    text = _failure_text(validation, execution_error=execution_error)
    return None if text.strip() == "- enhanced validation failed" and not validation and not execution_error else text


def _execution_failure_validation(error: str) -> dict[str, Any]:
    return {
        "passed": False,
        "critical_failures": [
            {
                "name": "execution",
                "severity": "critical",
                "passed": False,
                "message": error,
                "source": "error_check",
            }
        ],
        "warnings": [],
        "checks": [
            {
                "name": "execution",
                "severity": "critical",
                "passed": False,
                "message": error,
                "source": "error_check",
            }
        ],
    }


def _constraint_text_for_repair(base_constraints: list[dict[str, Any]], error_text: str, extra_top_k: int) -> str:
    entries = list(base_constraints)
    if extra_top_k > 0 and error_text:
        try:
            extra = paper.get_constraint_retriever().retrieve(error_text, top_k=extra_top_k)
        except Exception:
            extra = []
        seen_ids = {item.get("id") for item in entries}
        for item in extra:
            if item.get("id") not in seen_ids:
                entries.append(item)
                seen_ids.add(item.get("id"))
    return paper.entries_to_constraint_text(entries)


def _run_code_on_fresh_graph(code: str, base_graph: nx.Graph) -> dict[str, Any]:
    return benchmark._exec(code, copy.deepcopy(base_graph))


def _validate_generated_return(
    query: str,
    ret: Any,
    expected_type: str | None,
    base_graph: nx.Graph,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return enhanced.run_enhanced_validation(
        query=query,
        ret=ret,
        expected_type=expected_type,
        base_graph=base_graph,
        constraints=constraints,
    )


def apply_enhanced_self_repair(
    query: str,
    base_graph: nx.Graph,
    result: paper.AttemptResult,
    expected_type: str | None,
    repair_loops: int,
    extra_constraint_top_k: int,
) -> paper.AttemptResult:
    checked = copy.deepcopy(result)
    current_code = checked.generated_code or ""
    current_ret = checked.ret
    current_execution_error = checked.error

    if current_ret is None and current_code:
        try:
            current_ret = _run_code_on_fresh_graph(current_code, base_graph)
            current_execution_error = None
        except Exception as exc:
            current_execution_error = str(exc)

    if current_execution_error:
        validation = _execution_failure_validation(current_execution_error)
    else:
        validation = _validate_generated_return(query, current_ret, expected_type, base_graph, checked.constraints)

    initial_validation = copy.deepcopy(validation)
    initial_error_text = _failure_text(validation, execution_error=current_execution_error)
    loop_records: list[dict[str, Any]] = [
        {
            "loop": 0,
            "phase": "enhanced_validate",
            "ok": validation.get("passed", False),
            "execution_error": current_execution_error,
            "validation_failures": validation.get("critical_failures", []),
            "warnings": validation.get("warnings", []),
            "return_type": current_ret.get("type") if isinstance(current_ret, dict) else None,
        }
    ]

    repaired = False
    debug_increments = 0
    last_error_text = initial_error_text

    for loop_index in range(1, repair_loops + 1):
        if validation.get("passed", False):
            break
        if not current_code or "process_graph" not in current_code:
            break

        constraints_text = _constraint_text_for_repair(
            checked.constraints,
            error_text=last_error_text,
            extra_top_k=extra_constraint_top_k,
        )
        tool_text = paper.entries_to_tool_text(checked.tools)
        prompt = USR_ENHANCED_REPAIR.format(
            input=query,
            step="final enhanced validation",
            expected_type=expected_type or "not explicitly specified",
            constraints=constraints_text,
            tool=tool_text,
            code=current_code,
            execution_error=current_execution_error or "None",
            validation_failures=last_error_text,
        )

        try:
            repaired_code = paper._chat_code(SYS_ENHANCED_REPAIR, prompt)
        except Exception as exc:
            current_execution_error = str(exc)
            loop_records.append({
                "loop": loop_index,
                "phase": "enhanced_repair_regenerate",
                "ok": False,
                "execution_error": current_execution_error,
                "validation_failures": [],
                "warnings": [],
                "return_type": None,
                "code": None,
            })
            break

        debug_increments += 1
        if not repaired_code or "process_graph" not in repaired_code:
            current_execution_error = "debugger returned no process_graph code"
            validation = _execution_failure_validation(current_execution_error)
            last_error_text = _failure_text(validation, execution_error=current_execution_error)
            loop_records.append({
                "loop": loop_index,
                "phase": "enhanced_repair_regenerate",
                "ok": False,
                "execution_error": current_execution_error,
                "validation_failures": validation.get("critical_failures", []),
                "warnings": [],
                "return_type": None,
                "code": repaired_code,
            })
            continue

        current_code = repaired_code
        try:
            current_ret = _run_code_on_fresh_graph(current_code, base_graph)
            current_execution_error = None
            validation = _validate_generated_return(query, current_ret, expected_type, base_graph, checked.constraints)
        except Exception as exc:
            current_ret = None
            current_execution_error = str(exc)
            validation = _execution_failure_validation(current_execution_error)

        last_error_text = _failure_text(validation, execution_error=current_execution_error)
        ok = validation.get("passed", False)
        repaired = repaired or ok
        loop_records.append({
            "loop": loop_index,
            "phase": "enhanced_repair_execute_validate",
            "ok": ok,
            "execution_error": current_execution_error,
            "validation_failures": validation.get("critical_failures", []),
            "warnings": validation.get("warnings", []),
            "return_type": current_ret.get("type") if isinstance(current_ret, dict) else None,
            "code": current_code,
        })

    checked.ret = current_ret
    checked.generated_code = current_code
    checked.error = None if current_execution_error is None else current_execution_error
    checked.debug_count += debug_increments
    checked.checker_passed = validation.get("passed", False)
    checked.return_type_match = (
        current_ret.get("type") == expected_type
        if isinstance(current_ret, dict) and expected_type
        else None
    )
    checked.validation_error = None if validation.get("passed", False) else _critical_error_text(
        validation,
        execution_error=current_execution_error,
    )

    enhanced_record = {
        "step": "enhanced_validation_self_repair",
        "summary": "final output enhanced validation with repair feedback",
        "debug_count": debug_increments,
        "checker_passed": checked.checker_passed,
        "validation_error": checked.validation_error,
        "initial_passed": initial_validation.get("passed", False),
        "final_passed": validation.get("passed", False),
        "repaired": bool(not initial_validation.get("passed", False) and validation.get("passed", False)),
        "records": loop_records,
    }
    checked.step_records.append(enhanced_record)
    checked.enhanced_initial_validation = initial_validation
    checked.enhanced_validation = validation
    checked.enhanced_repair = {
        "initial_passed": initial_validation.get("passed", False),
        "final_passed": validation.get("passed", False),
        "attempted": bool(repair_loops > 0 and not initial_validation.get("passed", False)),
        "loops_used": debug_increments,
        "repaired": bool(not initial_validation.get("passed", False) and validation.get("passed", False)),
    }
    return checked


def evaluate_attempt(
    query: str,
    gt_code: str,
    result: paper.AttemptResult,
    elapsed: float,
    attempt_index: int,
) -> dict[str, Any]:
    _, graph_gt = benchmark.getGraphData()
    gt = None
    gt_error = None
    correct = False
    try:
        gt = benchmark._exec_gt(gt_code, graph_gt)
        if result.ret is not None and result.error is None:
            correct = benchmark._cmp(result.ret, gt)
    except Exception as exc:
        gt_error = str(exc)
        correct = False

    item = paper.attempt_to_json(
        query=query,
        attempt_index=attempt_index,
        result=result,
        gt=gt,
        gt_code=gt_code,
        gt_error=gt_error,
        correct=correct,
        elapsed=elapsed,
    )
    item["enhanced_initial_validation"] = getattr(result, "enhanced_initial_validation", None)
    item["enhanced_validation"] = getattr(result, "enhanced_validation", None)
    item["enhanced_repair"] = getattr(result, "enhanced_repair", None)
    return item


def compute_enhanced_diagnostics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    initial_failed = []
    final_failed = []
    repaired = []
    failure_names: Counter[str] = Counter()
    initial_failure_names: Counter[str] = Counter()
    loops_total = 0

    for item in attempts:
        repair = item.get("enhanced_repair") or {}
        initial = item.get("enhanced_initial_validation") or {}
        final = item.get("enhanced_validation") or {}
        loops_total += int(repair.get("loops_used") or 0)

        if initial and not initial.get("passed", False):
            initial_failed.append(item)
            for failure in initial.get("critical_failures", []):
                initial_failure_names[failure.get("name", "unknown")] += 1
        if final and not final.get("passed", False):
            final_failed.append(item)
            for failure in final.get("critical_failures", []):
                failure_names[failure.get("name", "unknown")] += 1
        if repair.get("repaired"):
            repaired.append(item)

    return {
        "enhanced_initial_failed_total": len(initial_failed),
        "enhanced_final_failed_total": len(final_failed),
        "enhanced_repaired_total": len(repaired),
        "enhanced_repair_loops_total": loops_total,
        "enhanced_initial_failed_correct": sum(1 for item in initial_failed if item.get("correct")),
        "enhanced_final_failed_correct": sum(1 for item in final_failed if item.get("correct")),
        "enhanced_repaired_correct": sum(1 for item in repaired if item.get("correct")),
        "initial_critical_failure_names": dict(initial_failure_names),
        "final_critical_failure_names": dict(failure_names),
    }


def run(
    query_indices: list[int],
    runs: int,
    output_prefix: str,
    confidence_threshold: float,
    max_debug: int,
    enhanced_repair_loops: int,
    extra_constraint_top_k: int,
) -> dict[str, Any]:
    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    all_queries = list(golden.keys())
    selected = [(idx, all_queries[idx - 1]) for idx in query_indices]

    print(
        f"{EXPERIMENT_NAME}: {len(selected)} queries x {runs} runs "
        f"(max_debug={max_debug}, enhanced_repair_loops={enhanced_repair_loops})"
    )
    print(f"confidence_threshold={confidence_threshold}")
    print()

    output = {
        "experiment": EXPERIMENT_NAME,
        "query_indices": query_indices,
        "runs_per_query": runs,
        "confidence_threshold": confidence_threshold,
        "max_debug": max_debug,
        "enhanced_repair_loops": enhanced_repair_loops,
        "extra_constraint_top_k": extra_constraint_top_k,
        "paper_style_features": {
            "query_specific_constraints": True,
            "hybrid_rrf_retrieval": True,
            "tool_retrieval": True,
            "cot_decomposition": True,
            "execution_error_reduction": True,
            "constraint_error_reduction": True,
            "enhanced_validation_self_repair": True,
            "repair_feedback_contains_query_step_code_errors_constraints_tools": True,
            "confidence_abstention": True,
            "runs_per_query": runs,
        },
        "queries": [],
        "metrics": {},
        "enhanced_diagnostics": {},
    }
    out_path = Path(f"{output_prefix}_{paper._safe_group_name(EXPERIMENT_NAME)}.json")

    for position, (query_index, query) in enumerate(selected, 1):
        query_attempts = []
        expected_type = paper.infer_expected_type_from_query(query)
        print(f"[{position:02d}/{len(selected)}] q{query_index}: {query[:100]}")

        for attempt_index in range(1, runs + 1):
            t0 = time.time()
            _, base_graph = benchmark.getGraphData()
            graph_for_full = copy.deepcopy(base_graph)

            gt_type = None
            try:
                _, graph_gt_probe = benchmark.getGraphData()
                gt_probe = benchmark._exec_gt(golden[query], graph_gt_probe)
                gt_type = gt_probe.get("type") if isinstance(gt_probe, dict) else None
                expected_type = expected_type or gt_type
            except Exception:
                pass

            result = paper.run_full_meshagent(
                query=query,
                graph=graph_for_full,
                expected_type=expected_type,
                max_debug=max_debug,
            )
            result = apply_enhanced_self_repair(
                query=query,
                base_graph=base_graph,
                result=result,
                expected_type=expected_type,
                repair_loops=enhanced_repair_loops,
                extra_constraint_top_k=extra_constraint_top_k,
            )

            elapsed = time.time() - t0
            attempt_json = evaluate_attempt(
                query=query,
                gt_code=golden[query],
                result=result,
                elapsed=elapsed,
                attempt_index=attempt_index,
            )
            query_attempts.append(attempt_json)

            repair = attempt_json.get("enhanced_repair") or {}
            status = "PASS" if attempt_json["correct"] else "FAIL"
            print(
                f"    run {attempt_index}/{runs}: {status} "
                f"({elapsed:.1f}s, dc={attempt_json['debug_count']}, "
                f"ret={attempt_json['return_type']}, check={attempt_json['checker_passed']}, "
                f"ev0={repair.get('initial_passed')}, ev={repair.get('final_passed')}, "
                f"repair={repair.get('loops_used')})"
            )
            err = attempt_json.get("validation_error") or attempt_json.get("error") or attempt_json.get("ground_truth_error")
            if err:
                print(f"        {str(err)[:180]}")

        paper.assign_confidence(query_attempts, threshold=confidence_threshold, max_debug=max_debug)
        query_metrics = paper.compute_metrics(query_attempts)
        output["queries"].append({
            "query_index": query_index,
            "query": query,
            "expected_type": expected_type,
            "attempts": query_attempts,
            "metrics": query_metrics,
        })

        flat_attempts = [attempt for item in output["queries"] for attempt in item["attempts"]]
        output["metrics"] = paper.compute_metrics(flat_attempts)
        output["enhanced_diagnostics"] = compute_enhanced_diagnostics(flat_attempts)
        paper.write_json(out_path, output)

        print(
            f"    query metrics: raw={query_metrics['raw_accuracy_before_abstention']} "
            f"reliable={query_metrics['reliable_accuracy']} abstain={query_metrics['abstain_rate']}"
        )

    flat_attempts = [attempt for item in output["queries"] for attempt in item["attempts"]]
    output["metrics"] = paper.compute_metrics(flat_attempts)
    output["enhanced_diagnostics"] = compute_enhanced_diagnostics(flat_attempts)
    paper.write_json(out_path, output)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "query_indices": query_indices,
        "runs_per_query": runs,
        **output["metrics"],
        **output["enhanced_diagnostics"],
        "output": str(out_path),
    }
    summary_path = Path(f"{output_prefix}_summary.json")
    paper.write_json(summary_path, summary)

    print()
    print(f">>> {EXPERIMENT_NAME}")
    print(
        f"    raw={output['metrics']['raw_accuracy_before_abstention']} "
        f"total={output['metrics']['total_accuracy']} "
        f"reliable={output['metrics']['reliable_accuracy']} "
        f"abstain={output['metrics']['abstain_rate']}"
    )
    print(
        f"    repaired={output['enhanced_diagnostics']['enhanced_repaired_total']} "
        f"initial_failed={output['enhanced_diagnostics']['enhanced_initial_failed_total']} "
        f"final_failed={output['enhanced_diagnostics']['enhanced_final_failed_total']}"
    )
    print(f"Saved output to {out_path}")
    print(f"Saved summary to {summary_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Full MeshAgent with enhanced validation self-repair.")
    parser.add_argument("--query-indices", default="", help="Comma/range query indices, e.g. 1-50 or 2,15,16.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Use first N queries when --query-indices is omitted.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--confidence-threshold", type=float, default=paper.CONFIDENCE_THRESHOLD_DEFAULT)
    parser.add_argument("--max-debug", type=int, default=paper.DEBUG_MAX_DEFAULT)
    parser.add_argument("--enhanced-repair-loops", type=int, default=2)
    parser.add_argument("--extra-constraint-top-k", type=int, default=3)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.max_debug < 0:
        raise ValueError("--max-debug must be non-negative")
    if args.enhanced_repair_loops < 0:
        raise ValueError("--enhanced-repair-loops must be non-negative")

    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    total = len(json.loads(golden_path.read_text(encoding="utf-8")))
    query_indices = parse_query_indices(args.query_indices, total=total, limit=args.limit)

    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout

    run(
        query_indices=query_indices,
        runs=args.runs,
        output_prefix=args.output_prefix,
        confidence_threshold=args.confidence_threshold,
        max_debug=args.max_debug,
        enhanced_repair_loops=args.enhanced_repair_loops,
        extra_constraint_top_k=args.extra_constraint_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
