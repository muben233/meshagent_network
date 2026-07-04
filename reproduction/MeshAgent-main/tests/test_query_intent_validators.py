import importlib.util
import sys
import unittest
from pathlib import Path

import networkx as nx


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"
VALIDATORS_PATH = APP_MALT / "query_intent_validators.py"


def import_validators():
    spec = importlib.util.spec_from_file_location("query_intent_validators", VALIDATORS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["query_intent_validators"] = module
    spec.loader.exec_module(module)
    return module


class QueryIntentValidatorTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("query_intent_validators", None)

    def test_add_packet_switch_validator_is_query_derived(self):
        validators = import_validators()
        before = nx.DiGraph()
        after = nx.DiGraph()
        after.add_node("ju1.a1.m1", type=["EK_AGG_BLOCK"])
        after.add_node("ju1.a1.m1.s4c7", type=["EK_PACKET_SWITCH"])
        after.add_edge("ju1.a1.m1", "ju1.a1.m1.s4c7", type="RK_CONTAINS")
        for i in range(1, 6):
            port = f"ju1.a1.m1.s4c7.p{i}"
            after.add_node(port, type=["EK_PORT"], physical_capacity_bps=1000)
            after.add_edge("ju1.a1.m1.s4c7", port, type="RK_CONTAINS")

        query = "Add a new packet_switch 'ju1.a1.m1.s4c7' with 5 ports, each port has physical_capacity_bps as 1000. Return the new graph."
        result = validators.validate_query_intent(query, before, {"type": "graph", "data": after})

        self.assertTrue(result.ok, result.errors)
        self.assertIn("add_packet_switch", result.applied_validators)

    def test_add_packet_switch_validator_rejects_missing_port(self):
        validators = import_validators()
        before = nx.DiGraph()
        after = nx.DiGraph()
        after.add_node("ju1.a1.m1.s4c7", type=["EK_PACKET_SWITCH"])
        for i in range(1, 5):
            port = f"ju1.a1.m1.s4c7.p{i}"
            after.add_node(port, type=["EK_PORT"], physical_capacity_bps=1000)
            after.add_edge("ju1.a1.m1.s4c7", port, type="RK_CONTAINS")

        query = "Add a new packet_switch 'ju1.a1.m1.s4c7' with 5 ports, each port has physical_capacity_bps as 1000. Return the new graph."
        result = validators.validate_query_intent(query, before, {"type": "graph", "data": after})

        self.assertFalse(result.ok)
        self.assertTrue(any("expected 5 contained ports" in err for err in result.errors))

    def test_add_packet_switch_validator_rejects_wrong_hierarchy_parent(self):
        validators = import_validators()
        before = nx.DiGraph()
        after = nx.DiGraph()
        after.add_node("ju1.a1.m1", type=["EK_CONTROL_DOMAIN"])
        after.add_node("ju1.a1.m2", type=["EK_CONTROL_DOMAIN"])
        after.add_node("ju1.a1.m1.s2c9", type=["EK_PACKET_SWITCH"])
        after.add_edge("ju1.a1.m2", "ju1.a1.m1.s2c9", type="RK_CONTAINS")
        for i in range(1, 6):
            port = f"ju1.a1.m1.s2c9.p{i}"
            after.add_node(port, type=["EK_PORT"], physical_capacity_bps=1000000000)
            after.add_edge("ju1.a1.m1.s2c9", port, type="RK_CONTAINS")

        query = (
            "Determine the optimal placement of a new PACKET_SWITCH node "
            "ju1.a1.m1.s2c9 with 5 PORT nodes in the format ju1.a1.m1.s2c9.p{i} "
            "(each has physical_capacity_bps 1000000000). Return the networkx graph."
        )
        result = validators.validate_query_intent(query, before, {"type": "graph", "data": after})

        self.assertFalse(result.ok)
        self.assertTrue(any("expected RK_CONTAINS parent ju1.a1.m1" in err for err in result.errors))

    def test_remove_ports_validator_checks_count_per_switch(self):
        validators = import_validators()
        query = (
            "Remove five PORT nodes (start from p1) from each PACKET_SWITCH node "
            "ju1.a1.m1.s2c1, ju1.a1.m1.s2c2. Return the list of ports that will be moved."
        )
        ret = {
            "type": "list",
            "data": [
                "ju1.a1.m1.s2c1.p1",
                "ju1.a1.m1.s2c1.p2",
                "ju1.a1.m1.s2c1.p3",
                "ju1.a1.m1.s2c1.p4",
                "ju1.a1.m1.s2c1.p5",
                "ju1.a1.m1.s2c2.p1",
                "ju1.a1.m1.s2c2.p2",
                "ju1.a1.m1.s2c2.p3",
                "ju1.a1.m1.s2c2.p4",
                "ju1.a1.m1.s2c2.p5",
            ],
        }

        result = validators.validate_query_intent(query, nx.DiGraph(), ret)

        self.assertTrue(result.ok, result.errors)
        self.assertIn("remove_ports_from_switches", result.applied_validators)

    def test_remove_ports_validator_rejects_non_optimal_balancing_choice(self):
        validators = import_validators()
        before = nx.DiGraph()
        switch_caps = {
            "ju1.a1.m1.s2c1": [9, 3, 2, 1, 1, 1],
            "ju1.a1.m1.s2c2": [9, 3, 2, 1, 1, 1],
        }
        for switch, caps in switch_caps.items():
            before.add_node(switch, type=["EK_PACKET_SWITCH"])
            for idx, cap in enumerate(caps, start=1):
                port = f"{switch}.p{idx}"
                before.add_node(port, type=["EK_PORT"], physical_capacity_bps=cap)
                before.add_edge(switch, port, type="RK_CONTAINS")

        query = (
            "Remove two PORT nodes from each PACKET_SWITCH node "
            "ju1.a1.m1.s2c1, ju1.a1.m1.s2c2. Make sure after the removal "
            "the capacity between switches is still balanced. Return the list of ports that will be moved."
        )
        ret = {
            "type": "list",
            "data": [
                "ju1.a1.m1.s2c1.p1",
                "ju1.a1.m1.s2c1.p2",
                "ju1.a1.m1.s2c2.p5",
                "ju1.a1.m1.s2c2.p6",
            ],
        }

        result = validators.validate_query_intent(query, before, ret)

        self.assertFalse(result.ok)
        self.assertTrue(any("capacity spread" in err for err in result.errors))

    def test_subgraph_validator_requires_touching_edges_from_original(self):
        validators = import_validators()
        before = nx.DiGraph()
        before.add_node("ju1", type=["EK_JUPITER"])
        before.add_node("ju1.a1", type=["EK_SUPERBLOCK"])
        before.add_node("ju1.a1.m1", type=["EK_AGG_BLOCK"])
        before.add_edge("ju1", "ju1.a1", type="RK_CONTAINS")
        before.add_edge("ju1.a1", "ju1.a1.m1", type="RK_CONTAINS")

        after = nx.DiGraph()
        after.add_node("ju1.a1", type=["EK_SUPERBLOCK"])
        after.add_node("ju1.a1.m1", type=["EK_AGG_BLOCK"])
        after.add_edge("ju1.a1", "ju1.a1.m1", type="RK_CONTAINS")

        query = "Provide a graph that contains all SUPERBLOCK and AGG_BLOCK. Create the new graph."
        result = validators.validate_query_intent(query, before, {"type": "graph", "data": after})

        self.assertFalse(result.ok)
        self.assertTrue(any("missing required edge" in err for err in result.errors))


if __name__ == "__main__":
    unittest.main()
