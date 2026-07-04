import importlib.util
import sys
import types
import unittest
from pathlib import Path


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"
SCRIPT_PATH = APP_MALT / "reanalyze_confidence.py"


def import_reanalyzer_with_fake_benchmark():
    fake = types.ModuleType("benchmark")
    sys.modules["benchmark"] = fake
    spec = importlib.util.spec_from_file_location("reanalyze_confidence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reanalyze_confidence"] = module
    spec.loader.exec_module(module)
    return module


class ReanalyzeConfidenceTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("benchmark", None)
        sys.modules.pop("reanalyze_confidence", None)

    def test_normalized_signature_matches_unordered_lists_and_numeric_text(self):
        reanalyzer = import_reanalyzer_with_fake_benchmark()

        self.assertEqual(
            reanalyzer.normalized_signature({"type": "list", "data": ["b", "a", "a"]}),
            reanalyzer.normalized_signature({"type": "list", "data": ["a", "b", "a"]}),
        )
        self.assertEqual(
            reanalyzer.normalized_signature({"type": "text", "data": "16000"}),
            reanalyzer.normalized_signature({"type": "text", "data": "16000.0"}),
        )

    def test_reanalysis_does_not_abstain_on_equivalent_correct_outputs(self):
        reanalyzer = import_reanalyzer_with_fake_benchmark()
        attempts = [
            {"correct": True, "debug_count": 0, "_ret": {"type": "list", "data": ["b", "a"]}},
            {"correct": True, "debug_count": 0, "_ret": {"type": "list", "data": ["a", "b"]}},
            {"correct": True, "debug_count": 0, "_ret": {"type": "list", "data": ["b", "a"]}},
        ]

        reanalyzer.assign_normalized_confidence(attempts, threshold=0.7, max_debug=5)

        self.assertEqual([1.0, 1.0, 1.0], [item["confidence"] for item in attempts])
        self.assertEqual([False, False, False], [item["abstained"] for item in attempts])

    def test_reanalysis_can_use_saved_return_preview_without_reexecution(self):
        reanalyzer = import_reanalyzer_with_fake_benchmark()
        data = {
            "experiment": "Full MeshAgent",
            "queries": [{
                "query": "q",
                "attempts": [
                    {
                        "correct": True,
                        "debug_count": 0,
                        "return_preview": {"type": "list", "data_preview": '["b", "a"]'},
                    },
                    {
                        "correct": True,
                        "debug_count": 0,
                        "return_preview": {"type": "list", "data_preview": '["a", "b"]'},
                    },
                ],
            }],
        }

        improved = reanalyzer.reanalyze_group_data(data, reexecute=False)

        attempts = improved["queries"][0]["attempts"]
        self.assertEqual([False, False], [item["abstained"] for item in attempts])
        self.assertEqual(1.0, improved["metrics"]["reliable_accuracy"])


if __name__ == "__main__":
    unittest.main()
