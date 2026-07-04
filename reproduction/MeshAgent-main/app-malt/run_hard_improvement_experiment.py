import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import benchmark
import networkx as nx

import query_intent_validators as intent
import run_malt_paper_reproduction as paper


DEFAULT_QUERY_INDICES = [2, 15, 16, 17, 19, 21]
VARIANTS = ["Full", "Full+IntentCheck", "Full+IntentDebug"]


def parse_query_indices(value: str) -> list[int]:
    if not value:
        return DEFAULT_QUERY_INDICES
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def _copy_result(result: paper.AttemptResult) -> paper.AttemptResult:
    return copy.deepcopy(result)


def _intent_error_text(validation: intent.IntentValidationResult) -> str:
    if not validation.applied_validators:
        return ""
    if validation.ok:
        return ""
    return (
        "Intent-level postcondition validation failed. "
        f"Applied validators: {', '.join(validation.applied_validators)}. "
        "Errors: " + " | ".join(validation.errors)
    )


def apply_intent_check(query: str, before_graph: nx.Graph, result: paper.AttemptResult) -> paper.AttemptResult:
    checked = _copy_result(result)
    validation = intent.validate_query_intent(query, before_graph, checked.ret)
    checked.step_records.append({
        "step": "intent_check",
        "summary": "query-derived postcondition validation",
        "checker_passed": validation.ok,
        "validation_error": _intent_error_text(validation) or None,
        "applied_validators": validation.applied_validators,
    })
    if validation.applied_validators and not validation.ok:
        checked.checker_passed = False
        checked.validation_error = _intent_error_text(validation)
    return checked


def apply_intent_debug(
    query: str,
    before_graph: nx.Graph,
    result: paper.AttemptResult,
    expected_type: str | None,
    max_debug: int,
    intent_debug_loops: int,
) -> paper.AttemptResult:
    debugged = apply_intent_check(query, before_graph, result)
    if debugged.checker_passed is not False:
        return debugged
    if not debugged.generated_code:
        return debugged

    constraints_text = paper.entries_to_constraint_text(debugged.constraints)
    tool_text = paper.entries_to_tool_text(debugged.tools)
    current_code = debugged.generated_code
    last_error = debugged.validation_error or "intent validation failed"
    last_ret = debugged.ret
    loop_records = []

    for loop_index in range(1, intent_debug_loops + 1):
        prompt = paper.USR_DEBUG_CONTEXT.format(
            input=query,
            step="final intent-level postcondition validation",
            constraints=constraints_text,
            tool=tool_text,
            code=current_code,
            error=last_error,
        )
        try:
            current_code = paper._chat_code(paper.SYS_DEBUG_CONTEXT, prompt)
            graph_for_exec = copy.deepcopy(before_graph)
            last_ret = benchmark._exec(current_code, graph_for_exec)
            return_ok, return_error = paper.validate_return(last_ret, expected_type)
            validation = intent.validate_query_intent(query, before_graph, last_ret)
            ok = return_ok and validation.ok
            last_error = return_error or _intent_error_text(validation)
            loop_records.append({
                "loop": loop_index,
                "ok": ok,
                "return_error": return_error,
                "intent_error": _intent_error_text(validation) or None,
                "applied_validators": validation.applied_validators,
                "return_type": last_ret.get("type") if isinstance(last_ret, dict) else None,
            })
            debugged.debug_count += 1
            debugged.generated_code = current_code
            debugged.ret = last_ret
            debugged.return_type_match = (last_ret.get("type") == expected_type) if isinstance(last_ret, dict) and expected_type else None
            if ok:
                debugged.checker_passed = True
                debugged.validation_error = None
                break
            debugged.checker_passed = False
            debugged.validation_error = last_error
        except Exception as exc:
            last_error = str(exc)
            debugged.debug_count += 1
            loop_records.append({
                "loop": loop_index,
                "ok": False,
                "return_error": str(exc),
                "intent_error": None,
                "applied_validators": [],
                "return_type": None,
            })
            debugged.checker_passed = False
            debugged.validation_error = last_error

    debugged.step_records.append({
        "step": "intent_debug",
        "summary": "debug with query-derived postcondition error",
        "debug_count": len(loop_records),
        "checker_passed": debugged.checker_passed,
        "validation_error": debugged.validation_error,
        "records": loop_records,
    })
    return debugged


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
    return paper.attempt_to_json(
        query=query,
        attempt_index=attempt_index,
        result=result,
        gt=gt,
        gt_code=gt_code,
        gt_error=gt_error,
        correct=correct,
        elapsed=elapsed,
    )


def run_experiment(
    query_indices: list[int],
    runs: int,
    output_prefix: str,
    confidence_threshold: float,
    max_debug: int,
    intent_debug_loops: int,
) -> list[dict[str, Any]]:
    golden = json.loads(Path("golden_answer_generator/prompt_golden_ans.json").read_text(encoding="utf-8"))
    queries = list(golden.keys())
    outputs = {
        variant: {
            "experiment": variant,
            "query_indices": query_indices,
            "runs_per_query": runs,
            "confidence_threshold": confidence_threshold,
            "max_debug": max_debug,
            "intent_debug_loops": intent_debug_loops,
            "queries": [],
            "metrics": {},
        }
        for variant in VARIANTS
    }

    print(f"Hard improvement experiment: query_indices={query_indices}, runs={runs}")
    print("Variants: " + ", ".join(VARIANTS))
    print()

    for query_index in query_indices:
        query = queries[query_index - 1]
        gt_code = golden[query]
        expected_type = paper.infer_expected_type_from_query(query)
        print("=" * 78)
        print(f"[#{query_index:02d}] {query}")
        print("=" * 78)
        per_variant_attempts = {variant: [] for variant in VARIANTS}

        for attempt_index in range(1, runs + 1):
            _, graph_run = benchmark.getGraphData()
            before_graph = copy.deepcopy(graph_run)
            expected_type_attempt = expected_type
            try:
                _, graph_gt = benchmark.getGraphData()
                gt = benchmark._exec_gt(gt_code, graph_gt)
                expected_type_attempt = expected_type_attempt or gt.get("type")
            except Exception:
                pass

            start = time.time()
            full_result = paper.run_full_meshagent(
                query=query,
                graph=graph_run,
                expected_type=expected_type_attempt,
                max_debug=max_debug,
            )
            base_elapsed = time.time() - start

            variants = {
                "Full": _copy_result(full_result),
                "Full+IntentCheck": apply_intent_check(query, before_graph, full_result),
                "Full+IntentDebug": apply_intent_debug(
                    query=query,
                    before_graph=before_graph,
                    result=full_result,
                    expected_type=expected_type_attempt,
                    max_debug=max_debug,
                    intent_debug_loops=intent_debug_loops,
                ),
            }

            for variant, result in variants.items():
                attempt = evaluate_attempt(
                    query=query,
                    gt_code=gt_code,
                    result=result,
                    elapsed=base_elapsed,
                    attempt_index=attempt_index,
                )
                per_variant_attempts[variant].append(attempt)
                status = "PASS" if attempt["correct"] else "FAIL"
                print(
                    f"  run {attempt_index}/{runs} {variant:17s}: {status} "
                    f"dc={attempt['debug_count']} check={attempt['checker_passed']} "
                    f"err={(attempt['validation_error'] or attempt['error'] or '')[:120]}"
                )

        for variant in VARIANTS:
            paper.assign_confidence(per_variant_attempts[variant], confidence_threshold, max_debug)
            query_metrics = paper.compute_metrics(per_variant_attempts[variant])
            outputs[variant]["queries"].append({
                "query_index": query_index,
                "query": query,
                "expected_type": expected_type,
                "attempts": per_variant_attempts[variant],
                "metrics": query_metrics,
            })
            print(
                f"  {variant:17s} query metrics: raw={query_metrics['raw_accuracy_before_abstention']} "
                f"reliable={query_metrics['reliable_accuracy']} abstain={query_metrics['abstain_rate']}"
            )

    summary = []
    for variant in VARIANTS:
        attempts = [attempt for query in outputs[variant]["queries"] for attempt in query["attempts"]]
        outputs[variant]["metrics"] = paper.compute_metrics(attempts)
        out_path = Path(f"{output_prefix}_{paper._safe_group_name(variant)}.json")
        paper.write_json(out_path, outputs[variant])
        row = {
            "experiment": variant,
            "output": str(out_path),
            **outputs[variant]["metrics"],
        }
        summary.append(row)

    summary_path = Path(f"{output_prefix}_summary.json")
    paper.write_json(summary_path, summary)
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hard-query improvement variants for MALT.")
    parser.add_argument("--query-indices", default=",".join(str(i) for i in DEFAULT_QUERY_INDICES))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-prefix", default="results_hard_improve")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--max-debug", type=int, default=5)
    parser.add_argument("--intent-debug-loops", type=int, default=2)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout
    run_experiment(
        query_indices=parse_query_indices(args.query_indices),
        runs=args.runs,
        output_prefix=args.output_prefix,
        confidence_threshold=args.confidence_threshold,
        max_debug=args.max_debug,
        intent_debug_loops=args.intent_debug_loops,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
