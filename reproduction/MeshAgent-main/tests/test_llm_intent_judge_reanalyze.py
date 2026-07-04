import importlib.util
import sys
import unittest
from pathlib import Path


APP_MALT = Path(__file__).resolve().parents[1] / "app-malt"
JUDGE_PATH = APP_MALT / "llm_intent_judge_reanalyze.py"


def import_judge_module():
    spec = importlib.util.spec_from_file_location("llm_intent_judge_reanalyze", JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_intent_judge_reanalyze"] = module
    spec.loader.exec_module(module)
    return module


class LlmIntentJudgeReanalyzeTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("llm_intent_judge_reanalyze", None)

    def test_extract_judge_json_from_fenced_response(self):
        judge = import_judge_module()
        raw = """Here is the judgment:
```json
{"pass": false, "confidence": 0.82, "reason": "missing required edge", "checks": ["edge check"]}
```
"""

        parsed = judge.normalize_judge_response(raw)

        self.assertFalse(parsed["pass"])
        self.assertEqual(parsed["confidence"], 0.82)
        self.assertEqual(parsed["reason"], "missing required edge")
        self.assertEqual(parsed["checks"], ["edge check"])

    def test_reanalyze_marks_failed_judgement_as_abstained(self):
        judge = import_judge_module()
        data = {
            "experiment": "Full",
            "queries": [
                {
                    "query_index": 21,
                    "query": "Determine the optimal placement of a new PACKET_SWITCH node.",
                    "expected_type": "graph",
                    "attempts": [
                        {
                            "attempt": 1,
                            "correct": False,
                            "abstained": False,
                            "confidence": 1.0,
                            "checker_passed": True,
                            "return_preview": {"type": "graph", "data_preview": {"node_count": 10}},
                            "generated_code": "def process_graph(graph_data): pass",
                        }
                    ],
                }
            ],
        }

        def fake_judge(_query, _attempt, _expected_type):
            return {
                "pass": False,
                "confidence": 0.91,
                "reason": "The new switch is attached to the wrong parent.",
                "checks": ["hierarchy"],
            }

        improved = judge.reanalyze_data(data, query_indices=[21], judge_func=fake_judge, judge_threshold=0.7)
        attempt = improved["queries"][0]["attempts"][0]

        self.assertTrue(attempt["abstained"])
        self.assertEqual(attempt["llm_judge"]["reason"], "The new switch is attached to the wrong parent.")
        self.assertEqual(improved["metrics"]["wrong_answered"], 0)
        self.assertEqual(improved["metrics"]["abstained_wrong"], 1)

    def test_reanalyze_filters_query_indices_and_keeps_passed_answered(self):
        judge = import_judge_module()
        data = {
            "experiment": "Full",
            "queries": [
                {
                    "query_index": 1,
                    "query": "q1",
                    "expected_type": "list",
                    "attempts": [
                        {
                            "attempt": 1,
                            "correct": True,
                            "abstained": True,
                            "confidence": 0.4,
                            "checker_passed": True,
                            "return_preview": {"type": "list", "data_preview": "[1]"},
                            "generated_code": "def process_graph(graph_data): pass",
                        }
                    ],
                },
                {
                    "query_index": 2,
                    "query": "q2",
                    "expected_type": "graph",
                    "attempts": [
                        {
                            "attempt": 1,
                            "correct": True,
                            "abstained": False,
                            "confidence": 1.0,
                            "checker_passed": True,
                            "return_preview": {"type": "graph", "data_preview": {"node_count": 2}},
                            "generated_code": "def process_graph(graph_data): pass",
                        }
                    ],
                },
            ],
        }

        def fake_judge(_query, _attempt, _expected_type):
            return {"pass": True, "confidence": 0.95, "reason": "Looks consistent.", "checks": []}

        improved = judge.reanalyze_data(data, query_indices=[2], judge_func=fake_judge, judge_threshold=0.7)

        self.assertEqual([q["query_index"] for q in improved["queries"]], [2])
        attempt = improved["queries"][0]["attempts"][0]
        self.assertFalse(attempt["abstained"])
        self.assertEqual(improved["metrics"]["correct_answered"], 1)
        self.assertEqual(improved["metrics"]["reliable_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
