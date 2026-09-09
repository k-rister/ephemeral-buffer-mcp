"""Tests for deterministic search relevance evaluation."""

import unittest

from benchmark_relevance import relevance_cases, run_relevance_benchmark


class TestRelevanceBenchmark(unittest.TestCase):
    def test_corpus_is_reproducible_and_content_free(self):
        first = relevance_cases()
        second = relevance_cases()
        self.assertEqual(first, second)
        self.assertTrue(all("synthetic" not in "\n".join(case["lines"]) for case in first))
        self.assertEqual(len(first), 4)

    def test_all_modes_report_machine_readable_relevance_scores(self):
        result = run_relevance_benchmark()

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(set(result["summaries"]), {"bm25", "semantic", "hybrid"})
        self.assertEqual(len(result["records"]), 12)
        self.assertTrue(all("rank" in record for record in result["records"]))
        self.assertTrue(all(0.0 <= summary["mrr"] <= 1.0 for summary in result["summaries"].values()))
        self.assertGreaterEqual(result["summaries"]["bm25"]["hit_at_1"], 0.75)
        self.assertEqual(result["summaries"]["semantic"]["hit_at_1"], 1.0)
        self.assertEqual(result["summaries"]["hybrid"]["hit_at_1"], 1.0)

    def test_relevance_benchmark_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            run_relevance_benchmark(top_k=0)


if __name__ == "__main__":
    unittest.main()
