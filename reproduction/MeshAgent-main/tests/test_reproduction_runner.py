import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"
RUNNER_PATH = APP_MALT / "run_reproduction_benchmark.py"


def import_runner_with_benchmark(fake_benchmark):
    sys.modules["benchmark"] = fake_benchmark
    spec = importlib.util.spec_from_file_location("run_reproduction_benchmark", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionRunnerTests(unittest.TestCase):
    def test_runner_writes_limited_group_results(self):
        fake = types.ModuleType("benchmark")
        fake.EXPERIMENTS = {
            "Fake Group": ("fake", lambda query, graph: ({"type": "text", "data": query}, 0, None))
        }
        fake.getGraphData = lambda: (None, {})
        fake._exec_gt = lambda code, graph: {"type": "text", "data": code}
        fake._cmp = lambda ret, gt: ret["data"] == gt["data"]

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "golden_answer_generator").mkdir()
            (root / "golden_answer_generator" / "prompt_golden_ans.json").write_text(
                json.dumps({"q1": "q1", "q2": "q2", "q3": "q3"}),
                encoding="utf-8",
            )

            runner = import_runner_with_benchmark(fake)
            cwd = Path.cwd()
            try:
                import os
                os.chdir(root)
                exit_code = runner.main(["--limit", "2", "--output-prefix", "results_test"])
            finally:
                import os
                os.chdir(cwd)
                sys.modules.pop("benchmark", None)

            self.assertEqual(0, exit_code)
            out = json.loads((root / "results_test_Fake_Group.json").read_text(encoding="utf-8"))
            self.assertEqual("Fake Group", out["experiment"])
            self.assertEqual(2, out["total"])
            self.assertEqual(2, out["correct"])
            self.assertEqual(1.0, out["accuracy"])
            self.assertEqual(["q1", "q2"], [item["query"] for item in out["queries"]])


if __name__ == "__main__":
    unittest.main()
