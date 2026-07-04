import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"
RUNNER_PATH = APP_MALT / "run_malt_paper_reproduction.py"


def import_runner_with_benchmark(fake_benchmark):
    sys.modules["benchmark"] = fake_benchmark
    spec = importlib.util.spec_from_file_location("run_malt_paper_reproduction", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_malt_paper_reproduction"] = module
    spec.loader.exec_module(module)
    return module


class MaltPaperRunnerTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("benchmark", None)
        sys.modules.pop("run_malt_paper_reproduction", None)

    def test_metrics_include_reliable_accuracy_and_abstention(self):
        fake = types.ModuleType("benchmark")
        runner = import_runner_with_benchmark(fake)

        attempts = [
            {"correct": True, "abstained": False},
            {"correct": False, "abstained": True},
            {"correct": False, "abstained": False},
            {"correct": True, "abstained": True},
        ]

        metrics = runner.compute_metrics(attempts)

        self.assertEqual(4, metrics["total_attempts"])
        self.assertEqual(2, metrics["answered"])
        self.assertEqual(2, metrics["abstained"])
        self.assertEqual(1, metrics["correct_answered"])
        self.assertEqual(0.25, metrics["total_accuracy"])
        self.assertEqual(0.5, metrics["reliable_accuracy"])
        self.assertEqual(0.5, metrics["abstain_accuracy"])
        self.assertEqual(0.5, metrics["abstain_precision"])
        self.assertEqual(0.5, metrics["abstain_recall"])

    def test_runner_repeats_queries_and_saves_generated_code(self):
        fake = types.ModuleType("benchmark")
        fake.getGraphData = lambda: (None, {})
        fake._exec_gt = lambda code, graph: {"type": "text", "data": code}
        fake._cmp = lambda ret, gt: ret["data"] == gt["data"]

        runner = import_runner_with_benchmark(fake)

        calls = []

        def fake_method(query, graph, expected_type):
            calls.append((query, expected_type))
            return runner.AttemptResult(
                ret={"type": "text", "data": query},
                generated_code=f"def process_graph(graph_data): return {query!r}",
                debug_count=0,
                checker_passed=True,
                return_type_match=True,
                validation_error=None,
            )

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "golden_answer_generator").mkdir()
            (root / "golden_answer_generator" / "prompt_golden_ans.json").write_text(
                json.dumps({"q1": "q1", "q2": "q2", "q3": "q3"}),
                encoding="utf-8",
            )

            cwd = Path.cwd()
            try:
                import os
                os.chdir(root)
                summary = runner.run(
                    limit=2,
                    runs=3,
                    output_prefix="paper_test",
                    groups=["Fake Full"],
                    methods={"Fake Full": fake_method},
                    confidence_threshold=0.7,
                    max_debug=5,
                )
            finally:
                import os
                os.chdir(cwd)

            self.assertEqual(6, len(calls))
            self.assertEqual(1, len(summary))
            self.assertEqual(1.0, summary[0]["raw_accuracy_before_abstention"])

            out = json.loads((root / "paper_test_Fake_Full.json").read_text(encoding="utf-8"))
            self.assertEqual(2, out["total_queries"])
            self.assertEqual(6, out["metrics"]["total_attempts"])
            self.assertEqual(3, len(out["queries"][0]["attempts"]))
            self.assertIn("generated_code", out["queries"][0]["attempts"][0])
            self.assertIn("ground_truth_code", out["queries"][0]["attempts"][0])


if __name__ == "__main__":
    unittest.main()
