"""Tests for deterministic search relevance evaluation."""

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmark_relevance import (
    compare_relevance,
    load_baseline,
    relevance_cases,
    run_relevance_benchmark,
)


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
        self.assertEqual(result["fixture_version"], 1)
        self.assertEqual(result["embedding_mode"], "deterministic-test")
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

    def test_baseline_comparison_passes_current_scores(self):
        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "1"}):
            result = run_relevance_benchmark()
        baseline = load_baseline(Path("benchmark_relevance_baseline.json"))
        comparison = compare_relevance(result, baseline)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["regressions"], [])

    def test_baseline_comparison_reports_material_regression(self):
        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "1"}):
            result = run_relevance_benchmark()
        baseline = load_baseline(Path("benchmark_relevance_baseline.json"))
        result["summaries"]["hybrid"]["hit_at_1"] = 0.0
        comparison = compare_relevance(result, baseline)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("hybrid.hit_at_1" in item for item in comparison["regressions"]))

    def test_baseline_loader_rejects_schema_and_missing_metrics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"schema_version": 99}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_baseline(path)

            path.write_text(
                '{"schema_version": 1, "fixture_version": 1, "benchmark": "search-relevance", '
                '"tolerances": {}, "summaries": {}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_baseline(path)


if __name__ == "__main__":
    unittest.main()
