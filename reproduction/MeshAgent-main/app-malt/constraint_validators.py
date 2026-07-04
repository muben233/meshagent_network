from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


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

PORT_NAME_RE = re.compile(r"^.+\.p\d+$")
NODE_NAME_RE = re.compile(r"\bju\d+(?:\.[A-Za-z]+\d*|\.[ms]\d+|\.[ps]\d+c\d+|\.[a-z]+\d*)*(?:\.p\d+)?\b")


CONSTRAINT_VALIDATOR_MAP = {
    "1": ["graph_directed_contract", "node_name_attribute_contract"],
    "2": ["node_type_contract"],
    "3": ["type_specific_attribute_contract"],
    "4": ["edge_type_contract"],
    "5": ["relationship_edge_contract"],
    "6": ["new_node_hierarchy_contract", "new_node_count_hint_contract"],
    "7": ["new_node_edge_contract"],
    "8": ["port_capacity_attribute_contract"],
    "9": ["port_name_contract"],
    "10": ["capacity_output_contract", "capacity_table_shape_contract"],
    "11": ["contains_hierarchy_contract"],
    "12": ["graph_projection_attribute_contract", "remove_mutation_diff_contract"],
    "13": ["graph_update_output_contract"],
    "14": ["packet_switch_attr_contract"],
    "15": ["node_lookup_type_contract"],
}


def make_check(name: str, severity: str, passed: bool, message: str, constraint_ids: Iterable[str]) -> dict[str, Any]:
    return {
        "name": name,
        "severity": severity,
        "passed": bool(passed),
        "message": message,
        "source": "constraint_validator",
        "constraint_ids": sorted({str(item) for item in constraint_ids}),
    }


def _as_graph(data: Any) -> nx.Graph:
    if isinstance(data, nx.Graph):
        return data
    return json_graph.node_link_graph(data)


def _ret_graph(ret: Any) -> nx.Graph | None:
    if not isinstance(ret, dict) or ret.get("type") != "graph":
        return None
    return _as_graph(ret.get("data"))


def _node_types(attrs: dict[str, Any]) -> list[str]:
    value = attrs.get("type")
    return value if isinstance(value, list) else []


def _edge_type_values(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if raw is None:
        return set()
    return {str(raw)}


def _edge_attr_dicts(graph: nx.Graph, source: Any, target: Any) -> list[dict[str, Any]]:
    data = graph.get_edge_data(source, target)
    if not data:
        return []
    if "type" in data:
        return [data]
    if isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
        return [value for value in data.values() if isinstance(value, dict)]
    return [data] if isinstance(data, dict) else []


def _has_edge_type(graph: nx.Graph, source: Any, target: Any, edge_type: str) -> bool:
    return any(edge_type in _edge_type_values(attrs.get("type")) for attrs in _edge_attr_dicts(graph, source, target))


def _iter_numbers(value: Any):
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
            yield from _iter_numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_numbers(child)


def _query_node_names(query: str) -> set[str]:
    return set(NODE_NAME_RE.findall(query))


def _new_nodes(graph: nx.Graph, base_graph: nx.Graph) -> set[Any]:
    return set(graph.nodes()) - set(base_graph.nodes())


def _removed_nodes(graph: nx.Graph, base_graph: nx.Graph) -> set[Any]:
    return set(base_graph.nodes()) - set(graph.nodes())


def _graph_diff(graph: nx.Graph, base_graph: nx.Graph) -> dict[str, Any]:
    graph_edges = {(u, v) for u, v in graph.edges()}
    base_edges = {(u, v) for u, v in base_graph.edges()}
    return {
        "new_nodes": _new_nodes(graph, base_graph),
        "removed_nodes": _removed_nodes(graph, base_graph),
        "new_edges": graph_edges - base_edges,
        "removed_edges": base_edges - graph_edges,
    }


def _is_add_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ("add ", "adding", "new ", "create", "insert"))


def _is_remove_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ("remove", "delete"))


def _is_graph_mutation_query(query: str) -> bool:
    q = query.lower()
    return _is_add_query(q) or _is_remove_query(q) or "update" in q or "balanced graph" in q


def graph_directed_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    return [
        make_check(
            "graph_directed_contract",
            "critical",
            graph.is_directed(),
            "ok" if graph.is_directed() else "graph output must be directed",
            constraint_ids,
        )
    ]


def node_name_attribute_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    existing_missing = [node for node in graph.nodes() if node in base_graph.nodes and not graph.nodes[node].get("name")]
    new_missing = [node for node in graph.nodes() if node not in base_graph.nodes and not graph.nodes[node].get("name")]
    checks = [
        make_check(
            "node_name_existing_attribute_contract",
            "critical",
            not existing_missing,
            "ok" if not existing_missing else f"existing nodes lost name attr: {existing_missing[:5]}",
            constraint_ids,
        )
    ]
    if new_missing:
        checks.append(
            make_check(
                "node_name_new_attribute_warning",
                "warning",
                False,
                f"new nodes missing name attr: {new_missing[:5]}",
                constraint_ids,
            )
        )
    return checks


def node_type_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    bad = []
    for node, attrs in graph.nodes(data=True):
        node_types = attrs.get("type")
        if not isinstance(node_types, list) or not node_types:
            bad.append((node, node_types))
            continue
        invalid = [item for item in node_types if item not in VALID_NODE_TYPES]
        if invalid:
            bad.append((node, invalid))
    return [
        make_check(
            "node_type_contract",
            "critical",
            not bad,
            "ok" if not bad else f"invalid or missing node types: {bad[:5]}",
            constraint_ids,
        )
    ]


def node_lookup_type_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    checks = node_type_contract(query, ret, expected_type, base_graph, constraint_ids)
    if isinstance(ret, dict) and ret.get("type") in {"list", "table"}:
        names = _query_node_names(query)
        if names and any("type" in query.lower() or token in query for token in ("EK_", "PACKET_SWITCH", "CONTROL_POINT")):
            data_text = str(ret.get("data"))
            missing_mentioned = [name for name in names if name not in data_text and name in base_graph.nodes]
            if missing_mentioned and "identify" not in query.lower():
                checks.append(
                    make_check(
                        "node_lookup_reference_warning",
                        "warning",
                        False,
                        f"mentioned existing nodes not reflected in output: {missing_mentioned[:5]}",
                        constraint_ids,
                    )
                )
    return checks


def type_specific_attribute_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    missing_port_capacity = [
        node
        for node, attrs in graph.nodes(data=True)
        if "EK_PORT" in _node_types(attrs) and "physical_capacity_bps" not in attrs
    ]
    return [
        make_check(
            "type_specific_port_capacity_attribute",
            "critical",
            not missing_port_capacity,
            "ok" if not missing_port_capacity else f"PORT nodes missing physical_capacity_bps: {missing_port_capacity[:5]}",
            constraint_ids,
        )
    ]


def edge_type_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    bad = []
    for u, v, attrs in graph.edges(data=True):
        edge_types = _edge_type_values(attrs.get("type"))
        if not edge_types or not edge_types <= VALID_EDGE_TYPES:
            bad.append((u, v, sorted(edge_types)))
    return [
        make_check(
            "edge_type_contract",
            "critical",
            not bad,
            "ok" if not bad else f"invalid edge types: {bad[:5]}",
            constraint_ids,
        )
    ]


def contains_hierarchy_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    bad_hierarchy = []
    self_loops = []
    for u, v, attrs in graph.edges(data=True):
        if "RK_CONTAINS" not in _edge_type_values(attrs.get("type")):
            continue
        if u == v:
            self_loops.append((u, v))
            continue
        source_types = _node_types(graph.nodes[u])
        target_types = _node_types(graph.nodes[v])
        if not any(src in CONTAINS_HIERARCHY and tgt in CONTAINS_HIERARCHY[src] for src in source_types for tgt in target_types):
            bad_hierarchy.append((u, source_types, v, target_types))
    return [
        make_check(
            "contains_hierarchy_contract",
            "critical",
            not bad_hierarchy,
            "ok" if not bad_hierarchy else f"invalid contains hierarchy: {bad_hierarchy[:5]}",
            constraint_ids,
        ),
        make_check(
            "contains_no_self_loop_contract",
            "critical",
            not self_loops,
            "ok" if not self_loops else f"RK_CONTAINS self-loops are invalid: {self_loops[:5]}",
            constraint_ids,
        ),
    ]


def relationship_edge_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    return contains_hierarchy_contract(query, ret, expected_type, base_graph, constraint_ids)


def port_capacity_attribute_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    bad = []
    for node, attrs in graph.nodes(data=True):
        if "EK_PORT" not in _node_types(attrs):
            continue
        value = attrs.get("physical_capacity_bps")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            bad.append((node, value))
    return [
        make_check(
            "port_capacity_attribute_contract",
            "critical",
            not bad,
            "ok" if not bad else f"PORT nodes have missing/non-numeric physical_capacity_bps: {bad[:5]}",
            constraint_ids,
        )
    ]


def port_name_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    bad = [node for node, attrs in graph.nodes(data=True) if "EK_PORT" in _node_types(attrs) and not PORT_NAME_RE.match(str(node))]
    return [
        make_check(
            "port_name_contract",
            "critical",
            not bad,
            "ok" if not bad else f"PORT node names do not match dotted .pN format: {bad[:5]}",
            constraint_ids,
        )
    ]


def new_node_hierarchy_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None or not _is_add_query(query):
        return []
    new_nodes = _new_nodes(graph, base_graph)
    dotted_parent_errors = []
    for node in sorted(new_nodes, key=str):
        node_text = str(node)
        if "." not in node_text:
            continue
        parent = node_text.rsplit(".", 1)[0]
        if parent in graph and not _has_edge_type(graph, parent, node, "RK_CONTAINS"):
            dotted_parent_errors.append((parent, node))
    return [
        make_check(
            "new_node_hierarchy_contract",
            "critical",
            not dotted_parent_errors,
            "ok" if not dotted_parent_errors else f"new nodes are not contained by dotted-path parent: {dotted_parent_errors[:5]}",
            constraint_ids,
        )
    ]


def new_node_count_hint_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None or not _is_add_query(query):
        return []
    q = query.lower()
    match = re.search(r"with\s+(\d+)\s+ports?", q)
    if not match:
        return []
    expected_ports = int(match.group(1))
    new_port_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if node not in base_graph.nodes and "EK_PORT" in _node_types(attrs)
    ]
    return [
        make_check(
            "new_port_count_hint_contract",
            "critical",
            len(new_port_nodes) == expected_ports,
            "ok" if len(new_port_nodes) == expected_ports else f"expected {expected_ports} new PORT nodes, got {len(new_port_nodes)}",
            constraint_ids,
        )
    ]


def new_node_edge_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None or not _is_add_query(query):
        return []
    new_nodes = _new_nodes(graph, base_graph)
    isolated = [node for node in new_nodes if graph.degree(node) == 0]
    no_contains = []
    for node in new_nodes:
        incident = list(graph.in_edges(node, data=True)) + list(graph.out_edges(node, data=True))
        if not any("RK_CONTAINS" in _edge_type_values(attrs.get("type")) for _, _, attrs in incident):
            no_contains.append(node)
    return [
        make_check(
            "new_node_connectivity_contract",
            "critical",
            not isolated,
            "ok" if not isolated else f"new isolated nodes: {isolated[:5]}",
            constraint_ids,
        ),
        make_check(
            "new_node_contains_edge_contract",
            "critical",
            not no_contains,
            "ok" if not no_contains else f"new nodes without RK_CONTAINS relationship: {no_contains[:5]}",
            constraint_ids,
        ),
    ]


def capacity_output_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    q = query.lower()
    if not isinstance(ret, dict) or not any(token in q for token in ("capacity", "bandwidth", "mbps", "bps")):
        return []
    if ret.get("type") == "graph" or "return the balanced graph" in q or "return graph" in q:
        return []
    numbers = list(_iter_numbers(ret.get("data")))
    checks = [
        make_check(
            "capacity_numeric_output_contract",
            "critical",
            bool(numbers),
            "ok" if numbers else "capacity/bandwidth query produced no numeric output",
            constraint_ids,
        )
    ]
    if "mbps" in q and numbers:
        suspicious = [num for num in numbers if abs(num) >= 1e8]
        checks.append(
            make_check(
                "capacity_mbps_unit_contract",
                "warning",
                not suspicious,
                "ok" if not suspicious else f"large Mbps-looking values, possible bps unit: {suspicious[:5]}",
                constraint_ids,
            )
        )
    return checks


def capacity_table_shape_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    q = query.lower()
    if not isinstance(ret, dict) or ret.get("type") != "table" or not isinstance(ret.get("data"), list):
        return []
    data = ret.get("data")
    checks = []
    if "first and the second largest chassis" in q:
        row_count_ok = len(data) == 3
        checks.append(
            make_check(
                "top_two_chassis_table_row_count",
                "critical",
                row_count_ok,
                "ok" if row_count_ok else f"expected header plus 2 rows for top-two chassis, got {len(data)} rows",
                constraint_ids,
            )
        )
        if len(data) >= 2:
            bad_rows = [row for row in data[1:] if not (isinstance(row, list) and len(row) >= 2 and isinstance(row[1], (int, float)))]
            checks.append(
                make_check(
                    "top_two_chassis_numeric_bandwidth",
                    "critical",
                    not bad_rows,
                    "ok" if not bad_rows else f"top chassis rows need numeric bandwidth: {bad_rows[:3]}",
                    constraint_ids,
                )
            )
            values = [row[1] for row in data[1:] if isinstance(row, list) and len(row) >= 2 and isinstance(row[1], (int, float))]
            sorted_ok = all(values[idx] >= values[idx + 1] for idx in range(len(values) - 1))
            checks.append(
                make_check(
                    "top_two_chassis_sorted_bandwidth",
                    "critical",
                    sorted_ok,
                    "ok" if sorted_ok else f"bandwidth values are not sorted descending: {values}",
                    constraint_ids,
                )
            )
    return checks


def graph_projection_attribute_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    q = query.lower()
    if "new graph" not in q and "contains all" not in q:
        return []
    if _is_add_query(q) or _is_remove_query(q) or "update" in q:
        return []
    diff = _graph_diff(graph, base_graph)
    unknown_nodes = sorted(diff["new_nodes"], key=str)
    missing_attrs = []
    for node in set(graph.nodes()) & set(base_graph.nodes()):
        for key, value in base_graph.nodes[node].items():
            if key not in graph.nodes[node]:
                missing_attrs.append((node, key))
                break
    return [
        make_check(
            "graph_projection_no_unknown_nodes",
            "critical",
            not unknown_nodes,
            "ok" if not unknown_nodes else f"projection graph introduced nodes not in source graph: {unknown_nodes[:5]}",
            constraint_ids,
        ),
        make_check(
            "graph_projection_preserve_node_attrs",
            "critical",
            not missing_attrs,
            "ok" if not missing_attrs else f"projected nodes lost source attributes: {missing_attrs[:5]}",
            constraint_ids,
        ),
    ]


def remove_mutation_diff_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None or not _is_remove_query(query):
        return []
    q = query.lower()
    diff = _graph_diff(graph, base_graph)
    checks = []
    if not _is_add_query(q):
        unexpected_new = sorted(diff["new_nodes"], key=str)
        checks.append(
            make_check(
                "remove_no_unexpected_new_nodes",
                "critical",
                not unexpected_new,
                "ok" if not unexpected_new else f"remove/delete query introduced new nodes: {unexpected_new[:10]}",
                constraint_ids,
            )
        )
    target_match = re.search(r"packet switch\s+'([^']+)'", query, flags=re.IGNORECASE)
    if target_match:
        target = target_match.group(1)
        checks.append(
            make_check(
                "remove_packet_switch_target_absent",
                "critical",
                target not in graph.nodes,
                "ok" if target not in graph.nodes else f"removed packet switch target is still present: {target}",
                constraint_ids,
            )
        )
        unexpected_removed_ports = [
            node for node in diff["removed_nodes"]
            if str(node).startswith(target + ".p")
        ]
        if unexpected_removed_ports:
            checks.append(
                make_check(
                    "remove_packet_switch_ports_preserved_warning",
                    "warning",
                    False,
                    f"removing packet switch also removed child ports: {sorted(unexpected_removed_ports, key=str)[:10]}",
                    constraint_ids,
                )
            )
    excessive_new_edges = len(diff["new_edges"]) > max(50, len(diff["removed_edges"]) * 3 + 10)
    checks.append(
        make_check(
            "remove_mutation_edge_delta_sanity",
            "warning",
            not excessive_new_edges,
            "ok" if not excessive_new_edges else f"remove/delete query added unusually many edges: added={len(diff['new_edges'])}, removed={len(diff['removed_edges'])}",
            constraint_ids,
        )
    )
    return checks


def graph_update_output_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    q = query.lower()
    if "update" not in q:
        return []
    graph = _ret_graph(ret)
    if graph is None:
        return []
    diff = _graph_diff(graph, base_graph)
    changed_any = bool(diff["new_nodes"] or diff["removed_nodes"] or diff["new_edges"] or diff["removed_edges"])
    return [
        make_check(
            "graph_update_returns_graph_contract",
            "critical",
            ret.get("type") == "graph",
            "ok" if ret.get("type") == "graph" else "update graph queries should return graph",
            constraint_ids,
        ),
        make_check(
            "graph_update_diff_sanity_warning",
            "warning",
            changed_any or "all" in q,
            "ok" if changed_any or "all" in q else "update query returned graph with no structural diff; attribute-only updates may still be valid",
            constraint_ids,
        ),
    ]


def packet_switch_attr_contract(query: str, ret: Any, expected_type: str | None, base_graph: nx.Graph, constraint_ids: set[str]):
    graph = _ret_graph(ret)
    if graph is None:
        return []
    q = query.lower()
    if "switch_loc" not in q and "packet_switch_attr" not in q and "stage" not in q:
        return []
    stage_values = [int(value) for value in re.findall(r"stage\s*:\s*(\d+)", query)]
    if len(stage_values) >= 2 and "update" in q:
        old_stage, new_stage = stage_values[0], stage_values[-1]
        not_updated = []
        for node, attrs in base_graph.nodes(data=True):
            if "EK_PACKET_SWITCH" not in _node_types(attrs):
                continue
            packet_attr = attrs.get("packet_switch_attr")
            switch_loc = packet_attr.get("switch_loc") if isinstance(packet_attr, dict) else None
            if not isinstance(switch_loc, dict) or switch_loc.get("stage") != old_stage:
                continue
            if node not in graph.nodes:
                not_updated.append((node, "missing in output"))
                continue
            out_packet_attr = graph.nodes[node].get("packet_switch_attr")
            out_switch_loc = out_packet_attr.get("switch_loc") if isinstance(out_packet_attr, dict) else None
            out_stage = out_switch_loc.get("stage") if isinstance(out_switch_loc, dict) else None
            if out_stage != new_stage:
                not_updated.append((node, out_stage))
        return [
            make_check(
                "packet_switch_stage_update_contract",
                "critical",
                not not_updated,
                "ok" if not not_updated else f"PACKET_SWITCH stage update did not apply to expected nodes: {not_updated[:5]}",
                constraint_ids,
            )
        ]

    missing = []
    for node, attrs in graph.nodes(data=True):
        if "EK_PACKET_SWITCH" not in _node_types(attrs):
            continue
        packet_attr = attrs.get("packet_switch_attr")
        if not isinstance(packet_attr, dict) or "switch_loc" not in packet_attr:
            missing.append(node)
    return [
        make_check(
            "packet_switch_attr_contract",
            "warning",
            not missing,
            "ok" if not missing else f"PACKET_SWITCH nodes missing packet_switch_attr.switch_loc: {missing[:5]}",
            constraint_ids,
        )
    ]


VALIDATOR_FUNCTIONS = {
    name: obj
    for name, obj in globals().items()
    if callable(obj) and name.endswith("_contract")
}


def validators_for_constraints(constraints: list[dict[str, Any]] | None) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    if not constraints:
        return selected
    for constraint in constraints:
        cid = str(constraint.get("id", ""))
        names = constraint.get("validators") or CONSTRAINT_VALIDATOR_MAP.get(cid, [])
        for name in names:
            selected.setdefault(name, set()).add(cid)
    return selected


def run_constraint_validators(
    *,
    query: str,
    ret: Any,
    expected_type: str | None,
    base_graph: nx.Graph,
    constraints: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    selected = validators_for_constraints(constraints)
    for name in sorted(selected):
        fn = VALIDATOR_FUNCTIONS.get(name)
        if fn is None:
            checks.append(
                make_check(
                    "unknown_constraint_validator",
                    "warning",
                    False,
                    f"validator is not implemented: {name}",
                    selected[name],
                )
            )
            continue
        try:
            checks.extend(fn(query, ret, expected_type, base_graph, selected[name]))
        except Exception as exc:
            checks.append(
                make_check(
                    f"{name}_runtime_error",
                    "warning",
                    False,
                    f"validator raised {type(exc).__name__}: {exc}",
                    selected[name],
                )
            )
    return checks
