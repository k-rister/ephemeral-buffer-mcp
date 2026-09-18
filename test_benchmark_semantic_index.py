"""Tests for the semantic indexing latency benchmark."""

import io
import unittest
from unittest.mock import patch

import benchmark_semantic_index
from benchmark_semantic_index import (
    NEEDLE_LINE,
    build_fixture,
    format_measurement,
    needle_line_number,
    run_benchmark,
    summarize,
)


class TestSemanticIndexBenchmark(unittest.TestCase):
    def test_fixture_is_deterministic_and_places_needle(self):
        first = build_fixture(64, seed=3)
        second = build_fixture(64, seed=3)
        self.assertEqual(first, second)
        lines = first.splitlines()
        self.assertEqual(len(lines), 64)
        self.assertEqual(lines[needle_line_number(64) - 1], NEEDLE_LINE)
        self.assertEqual(lines.count(NEEDLE_LINE), 1)
        self.assertNotEqual(first, build_fixture(64, seed=4))
        self.assertEqual(build_fixture(1), NEEDLE_LINE)
        with self.assertRaises(ValueError):
            build_fixture(0)

    def test_benchmark_reports_phases_and_needle_rank(self):
        result = run_benchmark((8, 32), samples=2)
        self.assertEqual(result["schema_version"], 1)
        self.assertTrue(result["test_embeddings"])
        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(result["engine_options"]["semantic_prefetch"], False)
        self.assertEqual([m["line_count"] for m in result["measurements"]], [8, 32])
        for measurement in result["measurements"]:
            self.assertEqual(measurement["samples"], 2)
            self.assertGreater(measurement["chunk_count"], 0)
            self.assertGreater(measurement["chunks_per_second_median"], 0)
            for phase in benchmark_semantic_index.PHASES:
                self.assertGreaterEqual(measurement[f"{phase}_median"], 0)
                self.assertGreaterEqual(measurement[f"{phase}_p95"], measurement[f"{phase}_median"])
            self.assertEqual(measurement["needle_ranks"], [1, 1])
            self.assertIn("needle_ranks", format_measurement(measurement))

    def test_semantic_mode_times_lazy_index_inside_first_search(self):
        result = run_benchmark((32,), samples=1, mode="semantic")
        measurement = result["measurements"][0]
        self.assertGreater(measurement["semantic_index_seconds_median"], 0)
        self.assertLessEqual(
            measurement["semantic_index_seconds_median"],
            measurement["first_search_seconds_median"],
        )
        self.assertLess(
            measurement["subsequent_search_seconds_median"],
            measurement["first_search_seconds_median"],
        )

    def test_summarize_handles_zero_index_time_and_rejects_empty(self):
        sample = {
            "line_count": 4,
            "output_bytes": 10,
            "chunk_count": 1,
            "needle_rank": None,
            "ingest_seconds": 0.0,
            "semantic_index_seconds": 0.0,
            "first_search_seconds": 0.0,
            "subsequent_search_seconds": 0.0,
        }
        summary = summarize([sample])
        self.assertIsNone(summary["chunks_per_second_median"])
        self.assertIn("chunks_per_second=n/a", format_measurement(summary))
        with self.assertRaises(ValueError):
            summarize([])

    def test_engine_options_are_forwarded_and_reported(self):
        result = run_benchmark((4,), samples=1, engine_options={"embedding_threads": 2})
        self.assertEqual(result["embedding_threads"], 2)
        self.assertEqual(result["engine_options"]["embedding_threads"], 2)
        self.assertNotIn("max_captures", result["engine_options"])

    def test_main_forwards_cli_flags(self):
        captured = {}

        def fake_run_benchmark(line_counts, samples, mode="hybrid", engine_options=None):
            captured.update(line_counts=line_counts, samples=samples, mode=mode, engine_options=engine_options)
            return {
                "embedding_model": "m",
                "embedding_threads": 3,
                "test_embeddings": True,
                "mode": mode,
                "model_load_seconds": 0.0,
                "measurements": [],
            }

        argv = [
            "benchmark_semantic_index.py", "--line-counts", "4", "--samples", "1", "--mode", "semantic",
            "--embedding-model", "m", "--embedding-threads", "3",
        ]
        with patch.object(benchmark_semantic_index, "run_benchmark", fake_run_benchmark), \
                patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            benchmark_semantic_index.main()
        self.assertEqual(captured["line_counts"], (4,))
        self.assertEqual(captured["mode"], "semantic")
        self.assertEqual(captured["engine_options"], {"embedding_model_name": "m", "embedding_threads": 3})
        self.assertIn("threads=3", stdout.getvalue())

    def test_benchmark_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            run_benchmark((), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((0,), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((4,), samples=0)
        with self.assertRaises(ValueError):
            run_benchmark((4,), samples=1, mode="bm25")
        with self.assertRaises(ValueError):
            benchmark_semantic_index.measure_once(None, 4, mode="bm25")
        with self.assertRaises(ValueError):
            benchmark_semantic_index._nearest_rank([])
        with self.assertRaises(ValueError):
            benchmark_semantic_index._nearest_rank([1.0], percentile=0)


if __name__ == "__main__":
    unittest.main()
