import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"


def import_benchmark_with_stubs():
    sys.path.insert(0, str(APP_MALT))

    helper = types.ModuleType("helper")
    helper.getGraphData = lambda: (None, {})
    helper.clean_up_llm_output_func = lambda text: text
    helper.check_list_equal = lambda left, right: left == right
    helper.node_attributes_are_equal = lambda left, right: left == right

    local_rag = types.ModuleType("local_rag")
    local_rag.retrieve_constraints = lambda query, top_k=9: ""

    class ConstraintRetriever:
        def __init__(self, path):
            self.texts = []

    local_rag.ConstraintRetriever = ConstraintRetriever

    error_check = types.ModuleType("error_check")

    class MyChecker:
        def __init__(self, ret_graph=None, ret_list=None):
            self.ret_graph = ret_graph
            self.ret_list = ret_list

        def evaluate_all(self):
            return True, ""

    error_check.MyChecker = MyChecker

    with patch.dict(sys.modules, {
        "helper": helper,
        "local_rag": local_rag,
        "error_check": error_check,
    }):
        sys.modules.pop("benchmark", None)
        return importlib.import_module("benchmark")


class BenchmarkHarnessTests(unittest.TestCase):
    def test_main_uses_fresh_graphs_for_each_query_and_ground_truth(self):
        benchmark = import_benchmark_with_stubs()

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "golden_answer_generator").mkdir()
            (root / "golden_answer_generator" / "prompt_golden_ans.json").write_text(
                json.dumps({"q1": "gt1", "q2": "gt2"}),
                encoding="utf-8",
            )

            graph_counter = {"value": 0}
            run_graphs = []
            gt_graphs = []

            def fake_get_graph_data():
                graph_counter["value"] += 1
                return None, {"id": graph_counter["value"], "mutated": False}

            def run_fn(query, graph):
                graph["mutated"] = True
                run_graphs.append(graph)
                return {"type": "text", "data": "ok"}, 0, None

            def fake_exec_gt(code, graph):
                gt_graphs.append(graph)
                return {"type": "text", "data": "ok"}

            cwd = Path.cwd()
            with patch.object(benchmark, "getGraphData", fake_get_graph_data), \
                 patch.object(benchmark, "EXPERIMENTS", {"Fake": ("none", run_fn)}), \
                 patch.object(benchmark, "_exec_gt", fake_exec_gt), \
                 patch.object(benchmark, "_cmp", lambda ret, gt: True), \
                 patch.object(benchmark, "client", object()), \
                 patch.object(benchmark, "MODEL", "fake-model"):
                try:
                    import os
                    os.chdir(root)
                    benchmark.main()
                finally:
                    import os
                    os.chdir(cwd)

            self.assertEqual(2, len(run_graphs))
            self.assertEqual(2, len(gt_graphs))
            self.assertIsNot(run_graphs[0], run_graphs[1])
            for run_graph, gt_graph in zip(run_graphs, gt_graphs):
                self.assertIsNot(run_graph, gt_graph)
                self.assertFalse(gt_graph["mutated"])

    def test_exec_times_out_generated_code(self):
        benchmark = import_benchmark_with_stubs()
        code = "def process_graph(graph_data):\n    while True:\n        pass\n"
        with self.assertRaises(TimeoutError):
            benchmark._exec(code, {}, timeout=0.5)


if __name__ == "__main__":
    unittest.main()
