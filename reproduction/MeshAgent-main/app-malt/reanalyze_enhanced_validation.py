"""
Offline enhanced validation + paper confidence reanalysis for saved MALT runs.

This script does not call any LLM or embedding API. It reads saved attempts that
already contain generated_code, re-executes the code locally, applies conservative
validation checks, and recomputes the paper-style confidence/abstention score.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from error_check import MyChecker
from helper import check_list_equal, getGraphData, node_attributes_are_equal
import constraint_validators


DEFAULT_INPUT = "results_intentcheck50_v2_r3_Full.json"
DEFAULT_OUTPUT = "results_enhanced_validation_offline.json"
DEFAULT_SUMMARY = "results_enhanced_validation_offline_summary.json"
VALID_NODE_TYPES = {
    "EK_SUPERBLOCK",
    "EK_CHASSIS",
    "EK_RACK",
    "EK_AGG_BLOCK",
    "EK_JUPITER",
    "EK_PORT",
    "EK_SPINEBLOCK",
    "EK_PACKET_SWITCH",
    "EK_CONTROL_POINT",
    "EK_CONTROL_DOMAIN",
}
VALID_EDGE_TYPES = {"RK_CONTAINS", "RK_CONTROLS", "RK_CONTROL"}
CONTAINS_HIERARCHY = {
    "EK_JUPITER": {"EK_SPINEBLOCK", "EK_SUPERBLOCK"},
    "EK_SPINEBLOCK": {"EK_PACKET_SWITCH"},
    "EK_SUPERBLOCK": {"EK_AGG_BLOCK"},
    "EK_AGG_BLOCK": {"EK_PACKET_SWITCH"},
    "EK_CHASSIS": {"EK_CONTROL_POINT", "EK_PACKET_SWITCH"},
    "EK_CONTROL_POINT": {"EK_PACKET_SWITCH"},
    "EK_RACK": {"EK_CHASSIS"},
    "EK_PACKET_SWITCH": {"EK_PORT"},
    "EK_CONTROL_DOMAIN": {"EK_CONTROL_POINT"},
}


_EXEC_SUBPROCESS_SCRIPT = r"""
import json
import pickle
import sys
import traceback

import networkx as nx

in_path, out_path, function_name = sys.argv[1:4]
try:
    with open(in_path, "rb") as f:
        code, graph = pickle.load(f)
    ns = {"json": json, "nx": nx, "networkx": nx}
    exec(code, ns)
    if function_name not in ns:
        raise NameError(f"{function_name} is not defined")
    ret = ns[function_name](graph)
    if isinstance(ret, str):
        ret = json.loads(ret)
    payload = ("ok", ret, None)
except Exception:
    payload = ("error", None, traceback.format_exc())

with open(out_path, "wb") as f:
    pickle.dump(payload, f)
"""


def _exec_with_timeout(code: str, graph: nx.Graph, function_name: str, timeout: float) -> Any:
    if timeout <= 0:
        ns = {"json": json, "nx": nx, "networkx": nx}
        exec(code, ns)
        ret = ns[function_name](graph)
        return json.loads(ret) if isinstance(ret, str) else ret

    in_file = tempfile.NamedTemporaryFile(prefix="meshagent_offline_in_", suffix=".pkl", delete=False)
    out_file = tempfile.NamedTemporaryFile(prefix="meshagent_offline_out_", suffix=".pkl", delete=False)
    in_path = in_file.name
    out_path = out_file.name
    in_file.close()
    out_file.close()

    try:
        with open(in_path, "wb") as f:
            pickle.dump((code, graph), f)

        try:
            completed = subprocess.run(
                [sys.executable, "-c", _EXEC_SUBPROCESS_SCRIPT, in_path, out_path, function_name],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"{function_name} exceeded {timeout:g}s") from exc

        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(err.splitlines()[-1] if err else f"{function_name} subprocess failed")

        with open(out_path, "rb") as f:
            status, ret, err = pickle.load(f)
    finally:
        for path in (in_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    if status == "error":
        raise RuntimeError((err or "").strip().splitlines()[-1])
    return ret


def _as_graph(data: Any) -> nx.Graph:
    if isinstance(data, nx.Graph):
        return data
    return json_graph.node_link_graph(data)


def _graph_signature(graph: nx.Graph) -> dict[str, Any]:
    nodes = []
    for node, attrs in graph.nodes(data=True):
        nodes.append([str(node), _jsonable(attrs)])
    edges = []
    for u, v, attrs in graph.edges(data=True):
        edges.append([str(u), str(v), _jsonable(attrs)])
    return {
        "directed": graph.is_directed(),
        "nodes": sorted(nodes, key=lambda item: item[0]),
        "edges": sorted(edges, key=lambda item: (item[0], item[1], json.dumps(item[2], sort_keys=True, default=str))),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, nx.Graph):
        return _graph_signature(value)
    return value


def canonical_signature(ret: dict[str, Any] | None) -> str:
    if not isinstance(ret, dict):
        return "null"
    rtype = ret.get("type")
    data = ret.get("data")
    if rtype == "graph":
        try:
            payload = _graph_signature(_as_graph(data))
        except Exception:
            payload = _jsonable(data)
    elif rtype == "list" and isinstance(data, list):
        payload = sorted((_jsonable(item) for item in data), key=lambda item: json.dumps(item, sort_keys=True, default=str))
    else:
        payload = _jsonable(data)
    return json.dumps({"type": rtype, "data": payload}, sort_keys=True, ensure_ascii=False, default=str)


def preview_value(value: Any, limit: int = 500) -> str:
    if isinstance(value, nx.Graph):
        value = {"nodes": value.number_of_nodes(), "edges": value.number_of_edges()}
    try:
        text = json.dumps(_jsonable(value), ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def preview_return(ret: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ret, dict):
        return None
    return {
        "type": ret.get("type"),
        "data_preview": preview_value(ret.get("data")),
    }


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
    if "output bandwidth" in q and "return only the number" in q:
        return "text"
    return None


def add_check(results: list[dict[str, Any]], name: str, severity: str, passed: bool, message: str, source: str) -> None:
    results.append(
        {
            "name": name,
            "severity": severity,
            "passed": bool(passed),
            "message": message,
            "source": source,
        }
    )


def extract_requested_headers(query: str) -> list[str]:
    match = re.search(r"headers?\s+(.+)", query, flags=re.IGNORECASE)
    if not match:
        return []
    tail = match.group(1)
    quoted = [item.strip() for item in re.findall(r"'([^']+)'", tail)]
    # Only enforce when the query gives at least two unambiguous quoted headers.
    return quoted if len(quoted) >= 2 else []


def iter_numeric_values(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        yield float(value)
    elif isinstance(value, str):
        stripped = value.strip().replace(",", "")
        try:
            yield float(stripped)
        except ValueError:
            return
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_numeric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_numeric_values(child)


def validate_schema(ret: Any, expected_type: str | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not isinstance(ret, dict):
        add_check(checks, "return_object_schema", "critical", False, "return_object is not a dict", "error_check")
        return checks
    missing = [key for key in ("type", "data") if key not in ret]
    add_check(
        checks,
        "return_object_schema",
        "critical",
        not missing,
        "ok" if not missing else f"missing keys: {missing}",
        "error_check",
    )
    if missing:
        return checks

    rtype = ret.get("type")
    add_check(
        checks,
        "return_type_enum",
        "critical",
        rtype in {"text", "list", "table", "graph"},
        "ok" if rtype in {"text", "list", "table", "graph"} else f"invalid return type: {rtype}",
        "error_check",
    )
    if expected_type:
        add_check(
            checks,
            "expected_return_type",
            "critical",
            rtype == expected_type,
            "ok" if rtype == expected_type else f"expected {expected_type}, got {rtype}",
            "error_check",
        )

    data = ret.get("data")
    if rtype == "text":
        passed = isinstance(data, (str, int, float))
        add_check(checks, "text_data_shape", "critical", passed, "ok" if passed else "text data must be string-like", "error_check")
    elif rtype == "list":
        passed = isinstance(data, list)
        add_check(checks, "list_data_shape", "critical", passed, "ok" if passed else "list data must be a list", "error_check")
    elif rtype == "table":
        passed = isinstance(data, list) and all(isinstance(row, list) for row in data)
        add_check(checks, "table_data_shape", "critical", passed, "ok" if passed else "table data must be a list of rows", "error_check")
    elif rtype == "graph":
        try:
            _as_graph(data)
            add_check(checks, "graph_data_shape", "critical", True, "ok", "error_check")
        except Exception as exc:
            add_check(checks, "graph_data_shape", "critical", False, str(exc), "error_check")
    return checks


def validate_table_contract(query: str, ret: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if ret.get("type") != "table" or not isinstance(ret.get("data"), list) or not ret["data"]:
        return checks
    expected = extract_requested_headers(query)
    if expected:
        actual = ret["data"][0]
        passed = [str(x).strip() for x in actual] == expected
        add_check(
            checks,
            "explicit_table_header",
            "critical",
            passed,
            "ok" if passed else f"expected header {expected}, got {actual}",
            "validation_test",
        )
    return checks


def _edge_type_values(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    return {str(raw)}


def validate_graph_contract(ret: dict[str, Any], base_graph: nx.Graph) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if ret.get("type") != "graph":
        return checks
    try:
        graph = _as_graph(ret.get("data"))
    except Exception as exc:
        add_check(checks, "graph_parse", "critical", False, str(exc), "validation_test")
        return checks

    try:
        ok, err = MyChecker(ret_graph=graph).evaluate_all()
        add_check(checks, "original_mychecker", "critical", ok, "ok" if ok else str(err), "validation_test")
    except Exception as exc:
        add_check(checks, "original_mychecker", "critical", False, str(exc), "validation_test")

    missing_name = [node for node, attrs in graph.nodes(data=True) if not attrs.get("name")]
    add_check(
        checks,
        "node_name_attribute",
        "critical",
        not missing_name,
        "ok" if not missing_name else f"nodes missing name attr: {missing_name[:5]}",
        "validation_test",
    )

    bad_node_types = []
    for node, attrs in graph.nodes(data=True):
        node_types = attrs.get("type")
        if not isinstance(node_types, list) or not node_types:
            bad_node_types.append((node, node_types))
            continue
        invalid = [node_type for node_type in node_types if node_type not in VALID_NODE_TYPES]
        if invalid:
            bad_node_types.append((node, invalid))
    add_check(
        checks,
        "node_type_contract",
        "critical",
        not bad_node_types,
        "ok" if not bad_node_types else f"invalid node types: {bad_node_types[:5]}",
        "validation_test",
    )

    bad_edge_types = []
    for u, v, attrs in graph.edges(data=True):
        edge_types = _edge_type_values(attrs.get("type"))
        if not edge_types or not edge_types <= VALID_EDGE_TYPES:
            bad_edge_types.append((u, v, sorted(edge_types)))
    add_check(
        checks,
        "edge_type_contract",
        "critical",
        not bad_edge_types,
        "ok" if not bad_edge_types else f"invalid edge types: {bad_edge_types[:5]}",
        "validation_test",
    )

    bad_hierarchy = []
    for u, v, attrs in graph.edges(data=True):
        if "RK_CONTAINS" not in _edge_type_values(attrs.get("type")):
            continue
        source_types = graph.nodes[u].get("type", [])
        target_types = graph.nodes[v].get("type", [])
        if not any(src in CONTAINS_HIERARCHY and tgt in CONTAINS_HIERARCHY[src] for src in source_types for tgt in target_types):
            bad_hierarchy.append((u, source_types, v, target_types))
    add_check(
        checks,
        "contains_hierarchy_contract",
        "critical",
        not bad_hierarchy,
        "ok" if not bad_hierarchy else f"invalid contains hierarchy: {bad_hierarchy[:5]}",
        "validation_test",
    )

    new_nodes = set(graph.nodes()) - set(base_graph.nodes())
    isolated_new_nodes = [node for node in new_nodes if graph.degree(node) == 0]
    add_check(
        checks,
        "new_node_connectivity",
        "critical",
        not isolated_new_nodes,
        "ok" if not isolated_new_nodes else f"new isolated nodes: {isolated_new_nodes[:5]}",
        "validation_test",
    )
    return checks


def validate_unit_sanity(query: str, ret: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    q = query.lower()
    if "mbps" not in q or not isinstance(ret, dict):
        return checks
    numbers = list(iter_numeric_values(ret.get("data")))
    if not numbers:
        add_check(checks, "mbps_numeric_sanity", "warning", False, "query asks Mbps but no numeric output was found", "validation_test")
        return checks
    suspicious = [num for num in numbers if abs(num) >= 1e8]
    add_check(
        checks,
        "mbps_numeric_sanity",
        "warning",
        not suspicious,
        "ok" if not suspicious else f"large Mbps-looking values, possible bps unit: {suspicious[:5]}",
        "validation_test",
    )
    return checks


def run_enhanced_validation(
    query: str,
    ret: Any,
    expected_type: str | None,
    base_graph: nx.Graph,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(validate_schema(ret, expected_type))
    if isinstance(ret, dict):
        checks.extend(validate_table_contract(query, ret))
        checks.extend(validate_graph_contract(ret, base_graph))
        checks.extend(validate_unit_sanity(query, ret))
        checks.extend(
            constraint_validators.run_constraint_validators(
                query=query,
                ret=ret,
                expected_type=expected_type,
                base_graph=base_graph,
                constraints=constraints,
            )
        )

    critical_failures = [check for check in checks if check["severity"] == "critical" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    return {
        "passed": not critical_failures,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "checks": checks,
    }


def compare_ret(ret: dict[str, Any], gt: dict[str, Any]) -> bool:
    gt_type = gt.get("type")
    if gt_type == "text":
        return str(ret.get("data", "")) == str(gt.get("data", ""))
    if gt_type == "list":
        return check_list_equal(ret.get("data", []), gt.get("data", []))
    if gt_type == "table":
        return ret.get("data") == gt.get("data")
    if gt_type == "graph":
        gt_graph = nx.DiGraph(gt["data"])
        ret_graph = _as_graph(ret["data"])
        return nx.is_isomorphic(gt_graph, ret_graph, node_match=node_attributes_are_equal)
    return False


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    raw_correct = sum(1 for item in attempts if item.get("correct"))
    answered = sum(1 for item in attempts if not item.get("abstained"))
    abstained = total - answered
    correct_answered = sum(1 for item in attempts if item.get("correct") and not item.get("abstained"))
    wrong_answered = sum(1 for item in attempts if (not item.get("correct")) and not item.get("abstained"))
    abstained_wrong = sum(1 for item in attempts if (not item.get("correct")) and item.get("abstained"))
    abstained_correct = sum(1 for item in attempts if item.get("correct") and item.get("abstained"))
    raw_wrong = total - raw_correct

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "total_attempts": total,
        "raw_correct": raw_correct,
        "answered": answered,
        "abstained": abstained,
        "correct_answered": correct_answered,
        "wrong_answered": wrong_answered,
        "abstained_wrong": abstained_wrong,
        "abstained_correct": abstained_correct,
        "raw_accuracy_before_abstention": ratio(raw_correct, total),
        "total_accuracy": ratio(correct_answered, total),
        "reliable_accuracy": ratio(correct_answered, answered),
        "abstain_rate": ratio(abstained, total),
        "abstain_accuracy": ratio(correct_answered + abstained_wrong, total),
        "abstain_precision": ratio(abstained_wrong, abstained),
        "abstain_recall": ratio(abstained_wrong, raw_wrong),
    }


def assign_paper_confidence(attempts: list[dict[str, Any]], rets: list[dict[str, Any] | None], threshold: float, max_debug: int) -> None:
    signatures = []
    for item, ret in zip(attempts, rets):
        if ret is not None and not item.get("reexecute_error") and item.get("enhanced_validation", {}).get("passed"):
            signatures.append(canonical_signature(ret))
    counts = Counter(signatures)

    for item, ret in zip(attempts, rets):
        if ret is None or item.get("reexecute_error") or not item.get("enhanced_validation", {}).get("passed"):
            semantic_consistency = 0.0
            confidence = 0.0
        else:
            semantic_consistency = counts[canonical_signature(ret)] / len(signatures) if signatures else 0.0
            debug_component = 1 - min(item.get("debug_count", 0), max_debug) / max_debug if max_debug else 1.0
            confidence = 0.5 * semantic_consistency + 0.5 * debug_component
        item["semantic_consistency"] = round(semantic_consistency, 4)
        item["confidence"] = round(confidence, 4)
        item["abstained"] = confidence < threshold


def process_query(query_obj: dict[str, Any], base_graph: nx.Graph, timeout: float, threshold: float, max_debug: int) -> dict[str, Any]:
    query = query_obj["query"]
    expected_type = query_obj.get("expected_type") or infer_expected_type_from_query(query)
    new_query = copy.deepcopy(query_obj)
    new_attempts = []
    rets: list[dict[str, Any] | None] = []

    gt = None
    gt_error = None
    gt_code = None
    for attempt in query_obj.get("attempts", []):
        if attempt.get("ground_truth_code"):
            gt_code = attempt.get("ground_truth_code")
            break
    if gt_code:
        try:
            gt = _exec_with_timeout(gt_code, copy.deepcopy(base_graph), "ground_truth_process_graph", timeout)
        except Exception as exc:
            gt_error = str(exc)

    for attempt in query_obj.get("attempts", []):
        item = copy.deepcopy(attempt)
        item["original_correct"] = attempt.get("correct")
        item["original_checker_passed"] = attempt.get("checker_passed")
        item["original_confidence"] = attempt.get("confidence")
        item["original_abstained"] = attempt.get("abstained")

        code = attempt.get("generated_code") or ""
        ret = None
        reexecute_error = None
        if not code.strip():
            reexecute_error = "missing generated_code"
        else:
            try:
                ret = _exec_with_timeout(code, copy.deepcopy(base_graph), "process_graph", timeout)
            except Exception as exc:
                reexecute_error = str(exc)

        item["reexecute_error"] = reexecute_error
        item["ground_truth_reexecute_error"] = gt_error
        item["enhanced_return_preview"] = preview_return(ret)
        validation = run_enhanced_validation(
            query,
            ret,
            expected_type,
            base_graph,
            constraints=attempt.get("constraints"),
        ) if reexecute_error is None else {
            "passed": False,
            "critical_failures": [
                {
                    "name": "execution",
                    "severity": "critical",
                    "passed": False,
                    "message": reexecute_error,
                    "source": "error_check",
                }
            ],
            "warnings": [],
            "checks": [],
        }
        item["enhanced_validation"] = validation
        item["checker_passed"] = validation["passed"]
        item["validation_error"] = "; ".join(check["message"] for check in validation["critical_failures"]) or None

        if ret is not None and gt is not None and gt_error is None:
            try:
                item["correct"] = compare_ret(ret, gt)
            except Exception as exc:
                item["correct"] = False
                item["ground_truth_reexecute_error"] = str(exc)
        else:
            item["correct"] = False
        item["reexecute_correct_mismatch"] = item.get("original_correct") != item.get("correct")

        rets.append(ret if isinstance(ret, dict) else None)
        new_attempts.append(item)

    assign_paper_confidence(new_attempts, rets, threshold=threshold, max_debug=max_debug)
    new_query["attempts"] = new_attempts
    new_query["metrics"] = summarize_attempts(new_attempts)
    return new_query


def validation_diagnostics(queries: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [attempt for query in queries for attempt in query.get("attempts", [])]
    critical_failed = [item for item in attempts if not item.get("enhanced_validation", {}).get("passed")]
    correct_failed = [item for item in critical_failed if item.get("correct")]
    wrong_failed = [item for item in critical_failed if not item.get("correct")]
    mismatches = [item for item in attempts if item.get("reexecute_correct_mismatch")]
    warning_count = sum(len(item.get("enhanced_validation", {}).get("warnings", [])) for item in attempts)
    failure_names = Counter()
    for item in critical_failed:
        for failure in item.get("enhanced_validation", {}).get("critical_failures", []):
            failure_names[failure.get("name", "unknown")] += 1
    return {
        "critical_failed_total": len(critical_failed),
        "critical_failed_correct": len(correct_failed),
        "critical_failed_wrong": len(wrong_failed),
        "warning_count": warning_count,
        "reexecute_correct_mismatch": len(mismatches),
        "critical_failure_names": dict(failure_names.most_common()),
    }


def parse_query_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline enhanced validation and paper confidence reanalysis.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Saved detailed result JSON with generated_code attempts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-debug", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0, help="Local execution timeout per generated code.")
    parser.add_argument("--query-indices", default=None, help="Optional comma/range filter, e.g. 1,2,10-15.")
    parser.add_argument("--limit-queries", type=int, default=None, help="Optional smoke limit after filtering.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    selected_indices = parse_query_indices(args.query_indices)

    _, base_graph = getGraphData()
    queries = data.get("queries", [])
    if selected_indices is not None:
        queries = [query for query in queries if int(query.get("query_index", 0)) in selected_indices]
    if args.limit_queries is not None:
        queries = queries[: args.limit_queries]

    print(
        f"Offline enhanced validation: {len(queries)} queries, "
        f"threshold={args.threshold}, max_debug={args.max_debug}, timeout={args.timeout}s"
    )
    new_queries = []
    for index, query_obj in enumerate(queries, 1):
        processed = process_query(
            query_obj,
            base_graph=base_graph,
            timeout=args.timeout,
            threshold=args.threshold,
            max_debug=args.max_debug,
        )
        new_queries.append(processed)
        metrics = processed["metrics"]
        print(
            f"[{index:02d}/{len(queries):02d}] q={processed.get('query_index')} "
            f"raw={metrics['raw_accuracy_before_abstention']} "
            f"reliable={metrics['reliable_accuracy']} "
            f"abstain={metrics['abstain_rate']}"
        )

    all_attempts = [attempt for query in new_queries for attempt in query.get("attempts", [])]
    metrics = summarize_attempts(all_attempts)
    diagnostics = validation_diagnostics(new_queries)

    output = copy.deepcopy(data)
    output["experiment"] = f"{data.get('experiment', 'Full')}+EnhancedValidationOffline"
    output["offline_enhanced_validation"] = {
        "source_input": str(input_path),
        "threshold": args.threshold,
        "max_debug": args.max_debug,
        "timeout": args.timeout,
        "notes": "No LLM/API calls. Re-executes saved generated_code and recomputes paper confidence using enhanced validation.",
    }
    output["queries"] = new_queries
    output["metrics"] = metrics
    output["validation_diagnostics"] = diagnostics

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {
        "experiment": output["experiment"],
        "output": str(output_path),
        **metrics,
        **diagnostics,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved output to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
