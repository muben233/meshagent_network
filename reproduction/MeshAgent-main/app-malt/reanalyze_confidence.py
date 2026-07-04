import argparse
import copy
import json
import math
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import benchmark
import networkx as nx
from networkx.readwrite import json_graph


CONFIDENCE_THRESHOLD_DEFAULT = 0.7
MAX_DEBUG_DEFAULT = 5


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


def _decimal_string(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        value = repr(value)
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    normalized = dec.normalize()
    if normalized == normalized.to_integral():
        normalized = normalized.quantize(Decimal(1))
    return format(normalized, "f")


def normalize_value(value: Any) -> Any:
    numeric = _decimal_string(value)
    if numeric is not None:
        return {"number": numeric}
    if isinstance(value, str):
        return {"string": value.strip()}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(value[key]) for key in sorted(value)}
    return value


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def normalize_graph(graph_or_data: Any) -> dict[str, Any]:
    if isinstance(graph_or_data, dict) and "node_count" in graph_or_data and "edge_count" in graph_or_data:
        return {"preview": normalize_value(graph_or_data)}
    graph = graph_or_data if isinstance(graph_or_data, nx.Graph) else json_graph.node_link_graph(graph_or_data)
    nodes = []
    for node, attrs in graph.nodes(data=True):
        nodes.append([str(node), normalize_value(dict(attrs))])
    edges = []
    for u, v, attrs in graph.edges(data=True):
        edges.append([str(u), str(v), normalize_value(dict(attrs))])
    nodes.sort(key=_sort_key)
    edges.sort(key=_sort_key)
    return {
        "graph_type": graph.__class__.__name__,
        "nodes": nodes,
        "edges": edges,
    }


def normalize_return(ret: dict[str, Any] | None) -> Any:
    if not isinstance(ret, dict):
        return {"type": None, "data": None}
    rtype = ret.get("type")
    data = ret.get("data")
    if rtype == "text":
        return {"type": "text", "data": normalize_value(data)}
    if rtype == "list":
        items = [normalize_value(item) for item in data] if isinstance(data, list) else normalize_value(data)
        if isinstance(items, list):
            items = sorted(items, key=_sort_key)
        return {"type": "list", "data": items}
    if rtype == "table":
        return {"type": "table", "data": normalize_value(data)}
    if rtype == "graph":
        return {"type": "graph", "data": normalize_graph(data)}
    return {"type": rtype, "data": normalize_value(data)}


def normalized_signature(ret: dict[str, Any] | None) -> str:
    return json.dumps(normalize_return(ret), sort_keys=True, ensure_ascii=False, default=str)


def _parse_preview_data(rtype: str | None, data_preview: Any) -> Any:
    if not isinstance(data_preview, str):
        return data_preview
    if "<truncated" in data_preview:
        return data_preview
    if rtype in {"list", "table"}:
        try:
            return json.loads(data_preview)
        except json.JSONDecodeError:
            return data_preview
    if rtype == "text":
        try:
            parsed = json.loads(data_preview)
            if isinstance(parsed, (str, int, float)):
                return parsed
        except json.JSONDecodeError:
            pass
        return data_preview
    return data_preview


def ret_from_preview(attempt: dict[str, Any]) -> dict[str, Any] | None:
    preview = attempt.get("return_preview")
    if not isinstance(preview, dict):
        return None
    rtype = preview.get("type")
    if rtype is None:
        return None
    return {
        "type": rtype,
        "data": _parse_preview_data(rtype, preview.get("data_preview")),
    }


def _attempt_invalid(attempt: dict[str, Any]) -> bool:
    return bool(
        attempt.get("reanalyze_error")
        or attempt.get("error")
        or attempt.get("ground_truth_error")
        or attempt.get("checker_passed") is False
        or attempt.get("_ret") is None
    )


def assign_normalized_confidence(attempts: list[dict[str, Any]], threshold: float, max_debug: int):
    valid_signatures = [
        normalized_signature(item.get("_ret"))
        for item in attempts
        if not _attempt_invalid(item)
    ]
    counts = Counter(valid_signatures)

    for item in attempts:
        item.setdefault("original_confidence", item.get("confidence"))
        item.setdefault("original_abstained", item.get("abstained"))
        if _attempt_invalid(item):
            semantic_consistency = 0.0
            confidence = 0.0
            signature = None
        else:
            signature = normalized_signature(item.get("_ret"))
            semantic_consistency = counts[signature] / len(valid_signatures) if valid_signatures else 0.0
            debug_component = 1 - min(item.get("debug_count", 0) or 0, max_debug) / max_debug if max_debug else 1.0
            confidence = 0.5 * semantic_consistency + 0.5 * debug_component

        item["normalized_signature"] = signature
        item["semantic_consistency"] = round(semantic_consistency, 4)
        item["confidence"] = round(confidence, 4)
        item["abstained"] = confidence < threshold


def _execute_code(code: str | None, function: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    if not code:
        return None, "missing generated code"
    _, graph = benchmark.getGraphData()
    try:
        if function == "process_graph":
            return benchmark._exec(code, graph, timeout=timeout), None
        return benchmark._exec_gt(code, graph, timeout=timeout), None
    except Exception as exc:
        return None, str(exc)


def _reexecute_attempt(attempt: dict[str, Any], timeout: float):
    ret, ret_error = _execute_code(attempt.get("generated_code"), "process_graph", timeout)
    gt, gt_error = _execute_code(attempt.get("ground_truth_code"), "ground_truth_process_graph", timeout)
    attempt["_ret"] = ret
    attempt["_ground_truth_ret"] = gt
    attempt["reanalyze_error"] = ret_error
    attempt["ground_truth_error"] = gt_error
    if ret is not None and gt is not None and not ret_error and not gt_error:
        try:
            attempt["correct"] = benchmark._cmp(ret, gt)
        except Exception as exc:
            attempt["correct"] = False
            attempt["ground_truth_error"] = str(exc)
    else:
        attempt["correct"] = False


def strip_private_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_private_fields(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_private_fields(item) for key, item in value.items() if not key.startswith("_")}
    return value


def reanalyze_group_data(
    data: dict[str, Any],
    threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    max_debug: int = MAX_DEBUG_DEFAULT,
    timeout: float = 120,
    reexecute: bool = True,
) -> dict[str, Any]:
    improved = copy.deepcopy(data)
    improved["confidence_reanalysis"] = {
        "method": "normalized output consistency",
        "normalizations": [
            "unordered list outputs",
            "numeric text equivalence",
            "numeric values in nested structures",
        ],
        "threshold": threshold,
        "max_debug": max_debug,
        "reexecuted_generated_code": reexecute,
    }

    for query in improved.get("queries", []):
        attempts = query.get("attempts", [])
        if reexecute:
            for attempt in attempts:
                _reexecute_attempt(attempt, timeout=timeout)
        else:
            for attempt in attempts:
                if attempt.get("_ret") is None:
                    attempt["_ret"] = ret_from_preview(attempt)
        assign_normalized_confidence(attempts, threshold=threshold, max_debug=max_debug)
        query["metrics"] = compute_metrics(attempts)

    all_attempts = [attempt for query in improved.get("queries", []) for attempt in query.get("attempts", [])]
    improved["metrics"] = compute_metrics(all_attempts)
    return improved


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_confidence_improved.json")


def _default_summary_output(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_summary.json")


def write_json(path: Path, data: Any):
    path.write_text(json.dumps(strip_private_fields(data), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline confidence reanalysis for MALT paper-style results.")
    parser.add_argument("--input", required=True, help="Input result JSON from run_malt_paper_reproduction.py")
    parser.add_argument("--output", default="", help="Improved result JSON path")
    parser.add_argument("--summary-output", default="", help="Improved summary JSON path")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD_DEFAULT)
    parser.add_argument("--max-debug", type=int, default=MAX_DEBUG_DEFAULT)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--no-reexecute",
        action="store_true",
        help="Do not re-run generated code. Intended only for tests that already provide _ret.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["MESHAGENT_EXEC_TIMEOUT_SECONDS"] = str(args.timeout)
    benchmark.EXEC_TIMEOUT_SECONDS = args.timeout

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output(input_path)
    summary_path = Path(args.summary_output) if args.summary_output else _default_summary_output(output_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    improved = reanalyze_group_data(
        data,
        threshold=args.threshold,
        max_debug=args.max_debug,
        timeout=args.timeout,
        reexecute=not args.no_reexecute,
    )
    write_json(output_path, improved)
    summary = [{
        "experiment": improved.get("experiment"),
        "source": str(input_path),
        "output": str(output_path),
        **improved["metrics"],
    }]
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved improved result to {output_path}")
    print(f"Saved improved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
