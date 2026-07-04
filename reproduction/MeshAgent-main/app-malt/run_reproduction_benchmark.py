import argparse
import json
import os
import time
from pathlib import Path

import benchmark


def _safe_group_name(name: str) -> str:
    return name.replace(" ", "_").replace("+", "")


def _parse_groups(group_arg: str):
    if not group_arg:
        return list(benchmark.EXPERIMENTS.keys())

    requested = [item.strip() for item in group_arg.split(",") if item.strip()]
    available = set(benchmark.EXPERIMENTS)
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(f"Unknown group(s): {', '.join(missing)}. Available: {', '.join(benchmark.EXPERIMENTS)}")
    return requested


def run(limit: int, output_prefix: str, groups: list[str]):
    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    queries = list(golden.keys())[:limit]

    print(f"Benchmark: {len(queries)} queries x {len(groups)} groups")
    print()

    summary = []
    for exp_name in groups:
        cmode, run_fn = benchmark.EXPERIMENTS[exp_name]
        print("=" * 70)
        print(f"  {exp_name} ({cmode})")
        print("=" * 70)

        correct = 0
        query_results = []

        for i, query in enumerate(queries, 1):
            t0 = time.time()
            _, g_run = benchmark.getGraphData()
            _, g_gt = benchmark.getGraphData()

            ret, debug_count, err = run_fn(query, g_run)
            elapsed = time.time() - t0

            is_correct = False
            gt_error = None
            if ret is not None and err is None:
                try:
                    gt = benchmark._exec_gt(golden[query], g_gt)
                    is_correct = benchmark._cmp(ret, gt)
                except Exception as exc:
                    gt_error = str(exc)

            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"
            print(f"[{i:02d}/{len(queries)}] {status} ({elapsed:.1f}s, dc={debug_count}) {query[:80]}")
            if err:
                print(f"    Error: {err[:180]}")
            if gt_error:
                print(f"    GT Error: {gt_error[:180]}")

            query_results.append({
                "query": query,
                "correct": is_correct,
                "debug_count": debug_count,
                "elapsed": round(elapsed, 1),
                "error": err,
                "ground_truth_error": gt_error,
                "return_type": ret.get("type") if isinstance(ret, dict) else None,
            })

        accuracy = correct / len(queries) if queries else 0
        out = {
            "experiment": exp_name,
            "total": len(queries),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "queries": query_results,
        }

        out_path = Path(f"{output_prefix}_{_safe_group_name(exp_name)}.json")
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n>>> {exp_name}: {correct}/{len(queries)} = {accuracy:.1%}")
        print(f"Saved to {out_path}\n")
        summary.append({"experiment": exp_name, "correct": correct, "total": len(queries), "accuracy": round(accuracy, 4)})

    summary_path = Path(f"{output_prefix}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to {summary_path}")
    print("Done.")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run the MALT MeshAgent reproduction benchmark.")
    parser.add_argument("--limit", type=int, default=50, help="Number of benchmark queries to run from the start.")
    parser.add_argument("--timeout", type=float, default=120, help="Generated-code execution timeout in seconds.")
    parser.add_argument("--output-prefix", default="results_50", help="Prefix for output JSON files.")
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names. Default: all groups in benchmark.EXPERIMENTS.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout

    groups = _parse_groups(args.groups)
    run(limit=args.limit, output_prefix=args.output_prefix, groups=groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
