"""Tests for deterministic search relevance evaluation."""

import io
import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import benchmark_relevance
import workload_results as wr
from benchmark_relevance import (
    compare_relevance,
    load_baseline,
    relevance_cases,
    run_relevance_benchmark,
    workload_result,
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


    def test_workload_result_reports_modes_and_baseline_regressions(self):
        record = run_relevance_benchmark()
        result = workload_result(record)
        self.assertEqual(result["workload"]["name"], "search-relevance")
        self.assertEqual(result["workload"]["fixture_version"], 1)
        self.assertFalse(result["workload"]["parameters"]["baseline_compared"])
        self.assertEqual([item["id"] for item in result["runs"]], ["bm25", "semantic", "hybrid"])
        self.assertEqual(result["runs"][0]["labels"], {"mode": "bm25", "top_k": 3})
        self.assertEqual(set(result["runs"][0]["measurements"]), {"queries", "hit_at_1", "hit_at_k", "mrr"})

        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "1"}):
            record["baseline_comparison"] = compare_relevance(
                record, load_baseline(Path(__file__).with_name("benchmark_relevance_baseline.json"))
            )
        compared = workload_result(record)
        self.assertTrue(compared["workload"]["parameters"]["baseline_compared"])
        self.assertEqual(compared["status"], "success")
        record["baseline_comparison"]["regressions"] = ["hybrid.mrr dropped from 1.0000 to 0.5000 (allowed drop 0.0500)"]
        regressed = workload_result(record)
        self.assertEqual(regressed["status"], "partial")
        self.assertEqual(regressed["runs"][2]["status"], "failure")
        self.assertEqual(regressed["runs"][2]["errors"], record["baseline_comparison"]["regressions"])
        self.assertEqual(regressed["runs"][0]["status"], "success")

    def test_result_flag_emits_json_on_stdout(self):
        argv = ["benchmark_relevance.py", "--top-k", "2", "--result", "-"]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_relevance.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["workload"]["parameters"]["top_k"], 2)
        self.assertEqual(json.loads(stderr.getvalue())["top_k"], 2)


if __name__ == "__main__":
    unittest.main()
