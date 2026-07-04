import re
import heapq
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


TYPE_ALIASES = {
    "SUPERBLOCK": "EK_SUPERBLOCK",
    "AGG_BLOCK": "EK_AGG_BLOCK",
    "PACKET_SWITCH": "EK_PACKET_SWITCH",
    "PORT": "EK_PORT",
    "CHASSIS": "EK_CHASSIS",
    "RACK": "EK_RACK",
    "CONTROL_POINT": "EK_CONTROL_POINT",
    "CONTROL_DOMAIN": "EK_CONTROL_DOMAIN",
    "JUPITER": "EK_JUPITER",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass
class IntentValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    applied_validators: list[str] = field(default_factory=list)

    def merge(self, name: str, errors: list[str]):
        self.applied_validators.append(name)
        self.errors.extend(errors)
        self.ok = self.ok and not errors


def _node_types(graph: nx.Graph, node: str) -> list[str]:
    if node not in graph.nodes:
        return []
    types = graph.nodes[node].get("type", [])
    if isinstance(types, str):
        return [types]
    return list(types)


def _has_type(graph: nx.Graph, node: str, node_type: str) -> bool:
    return node_type in _node_types(graph, node)


def _edge_type_matches(data: dict[str, Any], expected: str) -> bool:
    etype = data.get("type")
    if isinstance(etype, str):
        return etype == expected
    if isinstance(etype, list):
        return expected in etype
    return False


def _contains_successors(graph: nx.Graph, node: str, child_type: str | None = None) -> list[str]:
    if node not in graph:
        return []
    children = []
    for _, child, data in graph.out_edges(node, data=True):
        if not _edge_type_matches(data, "RK_CONTAINS"):
            continue
        if child_type is None or _has_type(graph, child, child_type):
            children.append(child)
    return children


def _expected_contains_parent(node_name: str) -> str | None:
    if "." not in node_name:
        return None
    return node_name.rsplit(".", 1)[0]


def _ret_graph(ret: dict[str, Any] | None) -> nx.Graph | None:
    if not isinstance(ret, dict) or ret.get("type") != "graph":
        return None
    data = ret.get("data")
    if isinstance(data, nx.Graph):
        return data
    try:
        return json_graph.node_link_graph(data)
    except Exception:
        try:
            return nx.DiGraph(data)
        except Exception:
            return None


def _word_or_digit_to_int(value: str) -> int | None:
    value = value.lower().strip()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _extract_switch_names(text: str) -> list[str]:
    return re.findall(r"ju\d+(?:\.[a-z]\d+)*(?:\.s\d+c\d+)", text)


def _extract_new_packet_switch_request(query: str) -> tuple[str, int] | None:
    match = re.search(
        r"(?:Add|Determine).*new\s+packet_switch(?:\s+node)?\s+(?:'([^']+)'|([A-Za-z0-9_.]+s\d+c\d+)).*?with\s+(\d+)\s+(?:PORT\s+nodes|ports)",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1) or match.group(2), int(match.group(3))


def _capacity_by_chassis(graph: nx.Graph) -> dict[str, float]:
    capacities = {}
    for node in graph.nodes:
        if not _has_type(graph, node, "EK_CHASSIS"):
            continue
        total = 0.0
        descendants = nx.descendants(graph, node) if node in graph else []
        for child in descendants:
            if _has_type(graph, child, "EK_PORT"):
                total += float(graph.nodes[child].get("physical_capacity_bps", 0) or 0)
        capacities[node] = total
    return capacities


def _spread(values: dict[str, float]) -> float | None:
    if not values:
        return None
    vals = list(values.values())
    return max(vals) - min(vals)


def _port_capacity(graph: nx.Graph, port: str) -> float:
    return float(graph.nodes[port].get("physical_capacity_bps", 0) or 0)


def _capacity_after_port_removal(graph: nx.Graph, switch: str, removed_ports: set[str]) -> float | None:
    if switch not in graph:
        return None
    ports = _contains_successors(graph, switch, "EK_PORT")
    if not ports:
        return None
    return sum(_port_capacity(graph, port) for port in ports if port not in removed_ports)


def _possible_remaining_capacities(graph: nx.Graph, switch: str, remove_count: int) -> set[float] | None:
    ports = _contains_successors(graph, switch, "EK_PORT")
    if len(ports) < remove_count:
        return None
    total = sum(_port_capacity(graph, port) for port in ports)
    removal_sums_by_count: list[set[float]] = [set() for _ in range(remove_count + 1)]
    removal_sums_by_count[0].add(0.0)
    for port in ports:
        cap = _port_capacity(graph, port)
        for count in range(remove_count - 1, -1, -1):
            for current_sum in tuple(removal_sums_by_count[count]):
                removal_sums_by_count[count + 1].add(current_sum + cap)
    return {total - removed for removed in removal_sums_by_count[remove_count]}


def _minimum_spread(capacity_sets: list[set[float]]) -> float | None:
    if not capacity_sets or any(not values for values in capacity_sets):
        return None
    sorted_sets = [sorted(values) for values in capacity_sets]
    heap = []
    current_max = None
    for list_index, values in enumerate(sorted_sets):
        value = values[0]
        heapq.heappush(heap, (value, list_index, 0))
        current_max = value if current_max is None else max(current_max, value)

    best = float("inf")
    while heap:
        current_min, list_index, value_index = heapq.heappop(heap)
        best = min(best, current_max - current_min)
        next_index = value_index + 1
        if next_index >= len(sorted_sets[list_index]):
            break
        next_value = sorted_sets[list_index][next_index]
        current_max = max(current_max, next_value)
        heapq.heappush(heap, (next_value, list_index, next_index))
    return best if best != float("inf") else None


def validate_add_packet_switch(query: str, _before: nx.Graph, ret: dict[str, Any] | None) -> list[str] | None:
    request = _extract_new_packet_switch_request(query)
    if not request:
        return None

    switch_name, port_count = request
    capacity_match = re.search(r"physical_capacity_bps\s+(?:as|=)?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
    capacity = float(capacity_match.group(1)) if capacity_match else None
    graph = _ret_graph(ret)
    errors = []
    if graph is None:
        return ["add_packet_switch expected graph output"]
    if switch_name not in graph:
        errors.append(f"new packet switch {switch_name} is missing")
        return errors
    if not _has_type(graph, switch_name, "EK_PACKET_SWITCH"):
        errors.append(f"{switch_name} must have type EK_PACKET_SWITCH")

    ports = _contains_successors(graph, switch_name, "EK_PORT")
    if len(ports) != port_count:
        errors.append(f"{switch_name} expected {port_count} contained ports, found {len(ports)}")
    for i in range(1, port_count + 1):
        port_name = f"{switch_name}.p{i}"
        if port_name not in graph:
            errors.append(f"expected port {port_name} is missing")
            continue
        if not _has_type(graph, port_name, "EK_PORT"):
            errors.append(f"{port_name} must have type EK_PORT")
        if capacity is not None:
            actual = graph.nodes[port_name].get("physical_capacity_bps")
            try:
                if float(actual) != capacity:
                    errors.append(f"{port_name} expected physical_capacity_bps {capacity:g}, got {actual}")
            except (TypeError, ValueError):
                errors.append(f"{port_name} missing numeric physical_capacity_bps")

    parents = [u for u, _, data in graph.in_edges(switch_name, data=True) if _edge_type_matches(data, "RK_CONTAINS")]
    expected_parent = _expected_contains_parent(switch_name)
    has_expected_parent = (
        expected_parent is not None
        and graph.has_edge(expected_parent, switch_name)
        and _edge_type_matches(graph.get_edge_data(expected_parent, switch_name, default={}), "RK_CONTAINS")
    )
    if expected_parent and not has_expected_parent:
        errors.append(f"{switch_name} expected RK_CONTAINS parent {expected_parent}, found {parents}")
    elif not parents:
        errors.append(f"{switch_name} must be contained by a hierarchy parent")
    return errors


def validate_remove_packet_switch(query: str, before: nx.Graph, ret: dict[str, Any] | None) -> list[str] | None:
    match = re.search(r"Remove\s+packet\s+switch\s+'([^']+)'", query, re.IGNORECASE)
    if not match or "Return the balanced graph" not in query:
        return None
    switch_name = match.group(1)
    graph = _ret_graph(ret)
    errors = []
    if graph is None:
        return ["remove_packet_switch expected graph output"]
    if switch_name in graph:
        errors.append(f"removed packet switch {switch_name} is still present")
    isolated = list(nx.isolates(graph))
    if isolated:
        errors.append(f"graph contains isolated nodes after removal: {isolated[:5]}")

    before_spread = _spread(_capacity_by_chassis(before))
    after_spread = _spread(_capacity_by_chassis(graph))
    if before_spread is not None and after_spread is not None and after_spread > before_spread:
        errors.append(f"chassis capacity spread worsened from {before_spread:g} to {after_spread:g}")
    return errors


def validate_remove_ports_from_switches(query: str, _before: nx.Graph, ret: dict[str, Any] | None) -> list[str] | None:
    count_match = re.search(r"Remove\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+PORT\s+nodes", query, re.IGNORECASE)
    if not count_match:
        return None
    count = _word_or_digit_to_int(count_match.group(1))
    switches = _extract_switch_names(query)
    if not count or not switches:
        return None

    errors = []
    if not isinstance(ret, dict) or ret.get("type") != "list":
        return ["remove_ports_from_switches expected list output"]
    data = ret.get("data")
    if not isinstance(data, list):
        return ["remove_ports_from_switches output data must be a list"]

    flat_ports = []
    for item in data:
        if isinstance(item, str):
            flat_ports.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, list):
                    flat_ports.extend(str(v) for v in value)
        elif isinstance(item, list):
            flat_ports.extend(str(v) for v in item)

    expected_total = count * len(switches)
    if len(flat_ports) != expected_total:
        errors.append(f"expected {expected_total} moved ports, found {len(flat_ports)}")
    for switch in switches:
        prefix = f"{switch}.p"
        switch_ports = [port for port in flat_ports if port.startswith(prefix)]
        if len(switch_ports) != count:
            errors.append(f"{switch} expected {count} moved ports, found {len(switch_ports)}")

    if "balanced" in query.lower() and all(switch in _before for switch in switches):
        removed_ports = set(flat_ports)
        after_caps = {
            switch: _capacity_after_port_removal(_before, switch, removed_ports)
            for switch in switches
        }
        possible_sets = [
            _possible_remaining_capacities(_before, switch, count)
            for switch in switches
        ]
        if all(value is not None for value in after_caps.values()) and all(values is not None for values in possible_sets):
            actual_spread = _spread(after_caps)  # type: ignore[arg-type]
            best_spread = _minimum_spread(possible_sets)  # type: ignore[arg-type]
            if actual_spread is not None and best_spread is not None and actual_spread > best_spread + 1e-9:
                errors.append(f"capacity spread {actual_spread:g} exceeds best achievable spread {best_spread:g}")
    return errors


def validate_subgraph_types(query: str, before: nx.Graph, ret: dict[str, Any] | None) -> list[str] | None:
    if "graph that contains all" not in query.lower():
        return None
    mentioned = [name for name in TYPE_ALIASES if name in query.upper()]
    if not mentioned:
        return None
    required_types = {TYPE_ALIASES[name] for name in mentioned}
    graph = _ret_graph(ret)
    errors = []
    if graph is None:
        return ["subgraph_type expected graph output"]

    required_nodes = {
        node for node, attrs in before.nodes(data=True)
        if required_types.intersection(set(attrs.get("type", [])))
    }
    missing_nodes = sorted(node for node in required_nodes if node not in graph)
    if missing_nodes:
        errors.append(f"missing required {mentioned} nodes: {missing_nodes[:8]}")

    for u, v, data in before.edges(data=True):
        if not _edge_type_matches(data, "RK_CONTAINS"):
            continue
        if u in required_nodes or v in required_nodes:
            if not graph.has_edge(u, v):
                errors.append(f"missing required edge touching selected node: {u}->{v}")
                if len(errors) > 8:
                    break
    return errors


def validate_query_intent(query: str, before_graph: nx.Graph, ret: dict[str, Any] | None) -> IntentValidationResult:
    result = IntentValidationResult(ok=True)
    validators = [
        ("add_packet_switch", validate_add_packet_switch),
        ("remove_packet_switch", validate_remove_packet_switch),
        ("remove_ports_from_switches", validate_remove_ports_from_switches),
        ("subgraph_type", validate_subgraph_types),
    ]
    for name, validator in validators:
        errors = validator(query, before_graph, ret)
        if errors is not None:
            result.merge(name, errors)
    return result
