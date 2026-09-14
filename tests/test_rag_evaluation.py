import unittest
from types import SimpleNamespace

import rag_evaluation


class FakeIndex:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, question, k):
        return self.results[question][:k]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_include_hits_misses_and_reciprocal_rank(self):
        cases = (
            {"id": "hit", "question": "hit question", "expected_terms": ("target section",)},
            {"id": "miss", "question": "miss question", "expected_terms": ("absent",)},
        )
        results = {
            "hit question": [
                (SimpleNamespace(page_content="unrelated"), 1.0),
                (SimpleNamespace(page_content="The target\n\n  section applies."), 0.4),
            ],
            "miss question": [(SimpleNamespace(page_content="unrelated"), 1.2)],
        }

        summary, rows = rag_evaluation.evaluate_retrieval(FakeIndex(results), cases=cases, k=4)

        self.assertEqual(summary["hits"], 1)
        self.assertEqual(summary["hit_rate"], 0.5)
        self.assertEqual(summary["mrr"], 0.25)
        self.assertEqual(rows[0]["first_rank"], 2)
        self.assertEqual(rows[0]["result"], "Pass")
        self.assertEqual(rows[1]["result"], "Miss")

    def test_empty_case_set_is_well_defined(self):
        summary, rows = rag_evaluation.evaluate_retrieval(FakeIndex({}), cases=(), k=4)

        self.assertEqual(summary, {"cases": 0, "hits": 0, "hit_rate": 0.0, "mrr": 0.0})
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
