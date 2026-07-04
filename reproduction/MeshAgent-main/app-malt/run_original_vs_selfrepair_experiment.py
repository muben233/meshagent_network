import argparse
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import benchmark

import run_enhanced_self_repair_experiment as selfrepair
import run_original_source_full_baseline as original


DEFAULT_OUTPUT_PREFIX = "results_original_vs_selfrepair50_r3"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_query_indices(value: str | None, limit: int | None, total: int) -> list[int]:
    if value:
        indices = original.parse_query_indices(value, total=total)
    else:
        count = limit or 50
        indices = list(range(1, min(count, total) + 1))
    if not indices:
        raise ValueError("No query indices selected.")
    return indices


def flatten_attempts(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not output:
        return []
    return [attempt for query in output.get("queries", []) for attempt in query.get("attempts", [])]


def compare_metrics(original_output: dict[str, Any] | None, selfrepair_output: dict[str, Any] | None) -> dict[str, Any]:
    original_metrics = (original_output or {}).get("metrics", {})
    selfrepair_metrics = (selfrepair_output or {}).get("metrics", {})
    delta = {}
    for key in sorted(set(original_metrics) & set(selfrepair_metrics)):
        left = original_metrics.get(key)
        right = selfrepair_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta[key] = round(right - left, 4)
    return {
        "original_metrics": original_metrics,
        "selfrepair_metrics": selfrepair_metrics,
        "delta_selfrepair_minus_original": delta,
    }


def per_query_compare(original_output: dict[str, Any] | None, selfrepair_output: dict[str, Any] | None) -> list[dict[str, Any]]:
    original_by_index = {
        item.get("query_index"): item
        for item in (original_output or {}).get("queries", [])
    }
    selfrepair_by_index = {
        item.get("query_index"): item
        for item in (selfrepair_output or {}).get("queries", [])
    }
    rows = []
    for query_index in sorted(set(original_by_index) | set(selfrepair_by_index)):
        original_query = original_by_index.get(query_index, {})
        selfrepair_query = selfrepair_by_index.get(query_index, {})
        original_metrics = original_query.get("metrics", {})
        selfrepair_metrics = selfrepair_query.get("metrics", {})
        rows.append({
            "query_index": query_index,
            "query": original_query.get("query") or selfrepair_query.get("query"),
            "original_raw_correct": original_metrics.get("raw_correct"),
            "selfrepair_raw_correct": selfrepair_metrics.get("raw_correct"),
            "original_wrong_answered": original_metrics.get("wrong_answered"),
            "selfrepair_wrong_answered": selfrepair_metrics.get("wrong_answered"),
            "original_reliable_accuracy": original_metrics.get("reliable_accuracy"),
            "selfrepair_reliable_accuracy": selfrepair_metrics.get("reliable_accuracy"),
            "selfrepair_enhanced_initial_failed": sum(
                1
                for attempt in selfrepair_query.get("attempts", [])
                if (attempt.get("enhanced_repair") or {}).get("initial_passed") is False
            ),
            "selfrepair_enhanced_repaired": sum(
                1
                for attempt in selfrepair_query.get("attempts", [])
                if (attempt.get("enhanced_repair") or {}).get("repaired")
            ),
        })
    return rows


def run_variant(
    *,
    name: str,
    fn,
    manifest: dict[str, Any],
    stop_on_error: bool,
) -> dict[str, Any] | None:
    print("=" * 88)
    print(f"START {name} at {now_iso()}")
    print("=" * 88)
    t0 = time.time()
    try:
        output = fn()
        manifest["variants"][name] = {
            "status": "completed",
            "started_at": manifest["variants"][name]["started_at"],
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.time() - t0, 1),
            "metrics": output.get("metrics", {}),
        }
        print("=" * 88)
        print(f"END {name} at {manifest['variants'][name]['finished_at']} elapsed={manifest['variants'][name]['elapsed_seconds']}s")
        print("=" * 88)
        return output
    except Exception as exc:
        manifest["variants"][name] = {
            "status": "failed",
            "started_at": manifest["variants"][name]["started_at"],
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.time() - t0, 1),
            "error": repr(exc),
        }
        print("=" * 88)
        print(f"FAILED {name} at {manifest['variants'][name]['finished_at']}: {repr(exc)}")
        print("=" * 88)
        if stop_on_error:
            raise
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    golden = json.loads(Path("golden_answer_generator/prompt_golden_ans.json").read_text(encoding="utf-8"))
    query_indices = parse_query_indices(args.query_indices, limit=args.limit, total=len(golden))

    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout

    output_prefix = args.output_prefix
    original_prefix = f"{output_prefix}_original"
    selfrepair_prefix = f"{output_prefix}_selfrepair"
    combined_summary_path = Path(f"{output_prefix}_comparison_summary.json")
    manifest_path = Path(f"{output_prefix}_manifest.json")

    manifest: dict[str, Any] = {
        "experiment": "OriginalSourceFull_vs_FullEnhancedValidationSelfRepair",
        "started_at": now_iso(),
        "cwd": str(Path.cwd()),
        "query_indices": query_indices,
        "runs_per_query": args.runs,
        "timeout": args.timeout,
        "original": {
            "output_prefix": original_prefix,
            "max_debug": args.original_max_debug,
            "constraint_top_k": args.original_constraint_top_k,
            "tool_top_k": args.original_tool_top_k,
            "expected_output": f"{original_prefix}_OriginalSourceFull.json",
            "expected_summary": f"{original_prefix}_summary.json",
        },
        "selfrepair": {
            "output_prefix": selfrepair_prefix,
            "max_debug": args.selfrepair_max_debug,
            "enhanced_repair_loops": args.enhanced_repair_loops,
            "extra_constraint_top_k": args.extra_constraint_top_k,
            "confidence_threshold": args.confidence_threshold,
            "expected_output": f"{selfrepair_prefix}_FullEnhancedValidationSelfRepair.json",
            "expected_summary": f"{selfrepair_prefix}_summary.json",
        },
        "outputs": {
            "comparison_summary": str(combined_summary_path),
            "manifest": str(manifest_path),
            "run_log": f"{output_prefix}_run.log",
        },
        "variants": {
            "Original Source Full": {"status": "pending", "started_at": None},
            "Full+EnhancedValidationSelfRepair": {"status": "pending", "started_at": None},
        },
    }
    write_json(manifest_path, manifest)

    print("Original Source Full vs Full+EnhancedValidationSelfRepair")
    print(f"queries={query_indices[0]}..{query_indices[-1]} count={len(query_indices)} runs={args.runs}")
    print(f"original_prefix={original_prefix}")
    print(f"selfrepair_prefix={selfrepair_prefix}")
    print(f"comparison_summary={combined_summary_path}")
    print()

    original_output = None
    selfrepair_output = None

    if "original" in args.only:
        manifest["variants"]["Original Source Full"]["started_at"] = now_iso()
        write_json(manifest_path, manifest)
        original_output = run_variant(
            name="Original Source Full",
            manifest=manifest,
            stop_on_error=args.stop_on_error,
            fn=lambda: original.run(
                query_indices=query_indices,
                runs=args.runs,
                output_prefix=original_prefix,
                timeout=args.timeout,
                max_debug=args.original_max_debug,
                constraint_top_k=args.original_constraint_top_k,
                tool_top_k=args.original_tool_top_k,
            ),
        )
        write_json(manifest_path, manifest)

    if "selfrepair" in args.only:
        manifest["variants"]["Full+EnhancedValidationSelfRepair"]["started_at"] = now_iso()
        write_json(manifest_path, manifest)
        selfrepair_output = run_variant(
            name="Full+EnhancedValidationSelfRepair",
            manifest=manifest,
            stop_on_error=args.stop_on_error,
            fn=lambda: selfrepair.run(
                query_indices=query_indices,
                runs=args.runs,
                output_prefix=selfrepair_prefix,
                confidence_threshold=args.confidence_threshold,
                max_debug=args.selfrepair_max_debug,
                enhanced_repair_loops=args.enhanced_repair_loops,
                extra_constraint_top_k=args.extra_constraint_top_k,
            ),
        )
        write_json(manifest_path, manifest)

    comparison = {
        "experiment": manifest["experiment"],
        "started_at": manifest["started_at"],
        "finished_at": now_iso(),
        "query_indices": query_indices,
        "runs_per_query": args.runs,
        "outputs": manifest["outputs"],
        "variant_outputs": {
            "original": manifest["original"],
            "selfrepair": manifest["selfrepair"],
        },
        "variant_status": manifest["variants"],
        **compare_metrics(original_output, selfrepair_output),
        "selfrepair_enhanced_diagnostics": (selfrepair_output or {}).get("enhanced_diagnostics", {}),
        "per_query": per_query_compare(original_output, selfrepair_output),
    }
    write_json(combined_summary_path, comparison)
    manifest["finished_at"] = comparison["finished_at"]
    manifest["comparison_summary_written"] = str(combined_summary_path)
    write_json(manifest_path, manifest)

    print()
    print("Combined comparison summary:")
    print(json.dumps({
        "original_metrics": comparison.get("original_metrics"),
        "selfrepair_metrics": comparison.get("selfrepair_metrics"),
        "delta_selfrepair_minus_original": comparison.get("delta_selfrepair_minus_original"),
        "selfrepair_enhanced_diagnostics": comparison.get("selfrepair_enhanced_diagnostics"),
    }, ensure_ascii=False, indent=2))
    print(f"Saved comparison summary to {combined_summary_path}")
    print(f"Saved manifest to {manifest_path}")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Original Source Full and Full+EnhancedValidationSelfRepair on the same MALT queries.")
    parser.add_argument("--query-indices", default="1-50", help="Comma/range query indices, e.g. 1-50 or 2,16,21.")
    parser.add_argument("--limit", type=int, default=None, help="Use first N queries when --query-indices is omitted.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--original-max-debug", type=int, default=3)
    parser.add_argument("--original-constraint-top-k", type=int, default=13)
    parser.add_argument("--original-tool-top-k", type=int, default=1)
    parser.add_argument("--selfrepair-max-debug", type=int, default=5)
    parser.add_argument("--enhanced-repair-loops", type=int, default=2)
    parser.add_argument("--extra-constraint-top-k", type=int, default=3)
    parser.add_argument(
        "--only",
        default="original,selfrepair",
        help="Comma-separated variants: original,selfrepair. Useful for resuming one side.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if a variant crashes.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    args.only = {item.strip().lower() for item in args.only.split(",") if item.strip()}
    valid = {"original", "selfrepair"}
    unknown = args.only - valid
    if unknown:
        raise ValueError(f"unknown --only value(s): {sorted(unknown)}; valid values are {sorted(valid)}")
    if not args.only:
        raise ValueError("--only selected no variants")

    log_path = Path(f"{args.output_prefix}_run.log")
    with log_path.open("a", encoding="utf-8") as log_file:
        tee_out = Tee(sys.__stdout__, log_file)
        tee_err = Tee(sys.__stderr__, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"\n\n===== run started {now_iso()} =====")
            print("command:", " ".join(sys.argv))
            run(args)
            print(f"===== run finished {now_iso()} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
