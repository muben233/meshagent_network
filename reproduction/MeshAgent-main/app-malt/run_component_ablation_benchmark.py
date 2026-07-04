import argparse
import json
import os
import re
import time
from pathlib import Path

import benchmark
import run_malt_paper_reproduction as paper


def _safe_group_name(name: str) -> str:
    safe = name.replace("+", "plus").replace("/", "_")
    return "_".join(part for part in safe.replace("-", " ").split())


def retrieve_hybrid_constraints_text(query: str, top_k: int = 9) -> str:
    """Retrieve Paper-style query-specific constraints with keyword+vector RRF."""
    entries = paper.get_constraint_retriever().retrieve(query, top_k=top_k)
    return paper.entries_to_constraint_text(entries)


def run_old_qs_single_pass(query: str, graph):
    """Single-pass code generation with the original local query-specific retriever."""
    try:
        constraints = benchmark.retrieve_constraints(query, top_k=9)
        raw = benchmark._chat(
            benchmark.SYS_SINGLE,
            benchmark.USR_SINGLE.format(input=query, constraints=constraints),
        )
        code = benchmark.clean_up_llm_output_func(raw)
        ret = benchmark._exec(code, graph)
        return ret, 0, None
    except Exception as exc:
        return None, 0, str(exc)


def run_hybrid_qs_single_pass(query: str, graph):
    """Single-pass code generation with Paper-style Hybrid/RRF query-specific constraints."""
    try:
        constraints = retrieve_hybrid_constraints_text(query, top_k=9)
        raw = benchmark._chat(
            benchmark.SYS_SINGLE,
            benchmark.USR_SINGLE.format(input=query, constraints=constraints),
        )
        code = benchmark.clean_up_llm_output_func(raw)
        ret = benchmark._exec(code, graph)
        return ret, 0, None
    except Exception as exc:
        return None, 0, str(exc)


def run_qs_cot_debug_hybrid(query: str, graph):
    """CoT/debug ablation with Paper-style Hybrid/RRF query-specific constraints."""
    constraints = retrieve_hybrid_constraints_text(query, top_k=9)
    debug_count = 0

    try:
        steps_raw = benchmark._chat(benchmark.SYS_STEP, benchmark.USR_STEP.format(input=query))
        steps = re.split(r"Step \d+:\s*", steps_raw)
        steps = [step.strip() for step in steps if step.strip()][:3]
        while len(steps) < 3:
            steps.append(f"Complete: {query}")

        previous_code = "None"
        for step in steps:
            raw = benchmark._chat(
                benchmark.SYS_COT,
                benchmark.USR_COT.format(
                    input=query,
                    constraints=constraints,
                    step=step,
                    code=previous_code,
                ),
            )
            code = benchmark.clean_up_llm_output_func(raw)
            if not code or "def process_graph" not in code:
                continue

            for _ in range(benchmark.DEBUG_MAX):
                try:
                    benchmark._exec(code, graph)
                    break
                except Exception as exc:
                    debug_count += 1
                    debug_raw = benchmark._chat(
                        benchmark.SYS_DBG,
                        benchmark.USR_DBG.format(
                            input=query,
                            constraints=constraints,
                            code=code,
                            error=str(exc),
                        ),
                    )
                    debug_code = benchmark.clean_up_llm_output_func(debug_raw)
                    if debug_code and "def process_graph" in debug_code:
                        code = debug_code

            try:
                ret_temp = benchmark._exec(code, graph)
                if ret_temp.get("type") == "graph":
                    ret_graph = ret_temp["data"]
                    if not isinstance(ret_graph, benchmark.nx.Graph):
                        ret_graph = benchmark.json_graph.node_link_graph(ret_graph)
                    checker = benchmark.MyChecker(ret_graph=ret_graph)
                    ok, validation_error = checker.evaluate_all()
                    if not ok:
                        debug_count += 1
                        extra_constraints = retrieve_hybrid_constraints_text(str(validation_error), top_k=3)
                        full_constraints = constraints + "\n" + extra_constraints
                        debug_raw = benchmark._chat(
                            benchmark.SYS_DBG,
                            benchmark.USR_DBG.format(
                                input=query,
                                constraints=full_constraints,
                                code=code,
                                error=validation_error,
                            ),
                        )
                        debug_code = benchmark.clean_up_llm_output_func(debug_raw)
                        if debug_code and "def process_graph" in debug_code:
                            code = debug_code
            except Exception:
                pass

            previous_code = code

        ret = benchmark._exec(previous_code, graph)
        return ret, debug_count, None
    except Exception as exc:
        return None, debug_count, str(exc)


COMPONENT_GROUPS = {
    "No-Constraint Single-Pass": {
        "kind": "baseline",
        "description": "No constraints; one-shot process_graph generation.",
        "run_fn": benchmark.run_none_single,
    },
    "All-Constraints Single-Pass": {
        "kind": "constraint_ablation",
        "description": "All constraints injected; one-shot process_graph generation.",
        "run_fn": benchmark.run_all_single,
    },
    "Old-QS-Constraints Single-Pass": {
        "kind": "constraint_ablation",
        "description": "Original local query-specific constraints; one-shot process_graph generation.",
        "run_fn": run_old_qs_single_pass,
    },
    "Hybrid-QS-Constraints Single-Pass": {
        "kind": "constraint_ablation",
        "description": "Query-specific constraints retrieved with Paper-style Hybrid/RRF; one-shot process_graph generation.",
        "run_fn": run_hybrid_qs_single_pass,
    },
    "QS-CoT-Debug": {
        "kind": "cot_debug_ablation",
        "description": "Paper-style Hybrid/RRF query-specific constraints + 3-step CoT + execution debug + MyChecker graph verification.",
        "run_fn": run_qs_cot_debug_hybrid,
    },
}


def parse_query_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(part))
    return sorted(selected)


def parse_groups(raw: str | None) -> list[str]:
    if not raw:
        return list(COMPONENT_GROUPS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [name for name in requested if name not in COMPONENT_GROUPS]
    if missing:
        raise ValueError(
            f"Unknown group(s): {', '.join(missing)}. "
            f"Available: {', '.join(COMPONENT_GROUPS)}"
        )
    return requested


def select_queries(golden: dict[str, str], limit: int | None, query_indices: list[int] | None):
    all_queries = list(golden.keys())
    if query_indices is not None:
        pairs = [(idx, all_queries[idx - 1]) for idx in query_indices if 1 <= idx <= len(all_queries)]
    else:
        upper = limit if limit is not None else len(all_queries)
        pairs = list(enumerate(all_queries[:upper], 1))
    return pairs


def run(limit: int | None, query_indices: list[int] | None, output_prefix: str, groups: list[str]):
    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    query_pairs = select_queries(golden, limit=limit, query_indices=query_indices)

    print(f"Component ablation benchmark: {len(query_pairs)} queries x {len(groups)} groups")
    print(f"Groups: {', '.join(groups)}")
    print()

    summary = []
    for group_name in groups:
        group = COMPONENT_GROUPS[group_name]
        run_fn = group["run_fn"]

        print("=" * 78)
        print(f"  {group_name}")
        print(f"  {group['description']}")
        print("=" * 78)

        correct = 0
        query_results = []

        for position, (query_index, query) in enumerate(query_pairs, 1):
            started = time.time()
            _, graph_run = benchmark.getGraphData()
            _, graph_gt = benchmark.getGraphData()

            ret, debug_count, err = run_fn(query, graph_run)
            elapsed = time.time() - started

            is_correct = False
            gt_error = None
            if ret is not None and err is None:
                try:
                    gt = benchmark._exec_gt(golden[query], graph_gt)
                    is_correct = benchmark._cmp(ret, gt)
                except Exception as exc:
                    gt_error = str(exc)

            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"
            print(
                f"[{position:02d}/{len(query_pairs):02d}] q{query_index:02d} "
                f"{status} ({elapsed:.1f}s, dc={debug_count}) {query[:80]}"
            )
            if err:
                print(f"    Error: {err[:180]}")
            if gt_error:
                print(f"    GT Error: {gt_error[:180]}")

            query_results.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "correct": is_correct,
                    "debug_count": debug_count,
                    "elapsed": round(elapsed, 1),
                    "error": err,
                    "ground_truth_error": gt_error,
                    "return_type": ret.get("type") if isinstance(ret, dict) else None,
                }
            )

        accuracy = correct / len(query_pairs) if query_pairs else 0.0
        output = {
            "experiment": group_name,
            "kind": group["kind"],
            "description": group["description"],
            "total": len(query_pairs),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "queries": query_results,
        }

        output_path = Path(f"{output_prefix}_{_safe_group_name(group_name)}.json")
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n>>> {group_name}: {correct}/{len(query_pairs)} = {accuracy:.2%}")
        print(f"Saved to {output_path}\n")

        summary.append(
            {
                "experiment": group_name,
                "kind": group["kind"],
                "correct": correct,
                "total": len(query_pairs),
                "accuracy": round(accuracy, 4),
                "output": str(output_path),
            }
        )

    summary_path = Path(f"{output_prefix}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")
    print("Done.")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run MALT component ablation benchmark.")
    parser.add_argument("--limit", type=int, default=50, help="Number of benchmark queries from the start.")
    parser.add_argument("--query-indices", default=None, help="Optional query indices, e.g. 1,2,10-15.")
    parser.add_argument("--timeout", type=float, default=120, help="Generated-code execution timeout in seconds.")
    parser.add_argument("--output-prefix", default="results_component_ablation50", help="Output JSON prefix.")
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names. Default: all component groups.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout

    groups = parse_groups(args.groups)
    query_indices = parse_query_indices(args.query_indices)
    run(limit=args.limit, query_indices=query_indices, output_prefix=args.output_prefix, groups=groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
