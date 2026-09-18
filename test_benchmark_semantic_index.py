"""Tests for the semantic indexing latency benchmark."""

import io
import json
import unittest
from unittest.mock import patch

import benchmark_semantic_index
import workload_results as wr
from benchmark_semantic_index import (
    NEEDLES,
    build_fixture,
    format_measurement,
    needle_positions,
    run_benchmark,
    summarize,
)


class TestSemanticIndexBenchmark(unittest.TestCase):
    def test_fixture_is_deterministic_and_places_needles(self):
        first = build_fixture(64, seed=3)
        second = build_fixture(64, seed=3)
        self.assertEqual(first, second)
        lines = first.splitlines()
        self.assertEqual(len(lines), 64)
        positions = needle_positions(64)
        self.assertEqual([line for _, line in positions], [17, 33, 49])
        for index, line_number in positions:
            self.assertEqual(lines[line_number - 1], NEEDLES[index]["line"])
        self.assertNotEqual(first, build_fixture(64, seed=4))

        # Small captures keep only needles with distinct positions.
        self.assertEqual(needle_positions(1), [(0, 1)])
        self.assertEqual(needle_positions(3), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(build_fixture(1), NEEDLES[0]["line"])
        with self.assertRaises(ValueError):
            build_fixture(0)

    def test_benchmark_reports_phases_chunking_and_needle_quality(self):
        result = run_benchmark((8, 32), samples=2)
        self.assertEqual(result["schema_version"], 3)
        self.assertTrue(result["test_embeddings"])
        self.assertGreater(result["semantic_wait_seconds"], 0)
        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(result["engine_options"]["semantic_prefetch"], False)
        self.assertEqual(set(result["semantic_chunking"]), {"lines", "bytes", "overlap"})
        self.assertEqual([m["line_count"] for m in result["measurements"]], [8, 32])
        for measurement in result["measurements"]:
            self.assertEqual(measurement["samples"], 2)
            self.assertGreater(measurement["chunk_count"], 0)
            self.assertLessEqual(measurement["semantic_chunk_count"], measurement["chunk_count"])
            self.assertGreater(measurement["semantic_chunks_per_second_median"], 0)
            for phase in benchmark_semantic_index.PHASES:
                self.assertGreaterEqual(measurement[f"{phase}_median"], 0)
                self.assertGreaterEqual(measurement[f"{phase}_p95"], measurement[f"{phase}_median"])
            self.assertEqual(len(measurement["needle_ranks"]), 2)
            self.assertEqual(set(measurement["needle_ranks"][0]), {needle["id"] for needle in NEEDLES})
            self.assertTrue(0 <= measurement["needle_hit_at_1"] <= 1)
            self.assertTrue(0 <= measurement["needle_mrr"] <= 1)
            # Deterministic embeddings finish inside the budget, so the first search is complete.
            self.assertEqual(measurement["first_search_semantic_coverage"], ["complete", "complete"])
            self.assertEqual(measurement["first_search_pending_rate"], 0.0)
            self.assertEqual(len(measurement["first_search_needle_ranks"]), 2)
            self.assertTrue(0 <= measurement["first_search_needle_mrr"] <= 1)
            self.assertIn("first_search_pending_rate=0.00", format_measurement(measurement))
            self.assertIn("needle_mrr", format_measurement(measurement))

    def test_first_search_reports_pending_coverage_under_a_zero_budget(self):
        result = run_benchmark((32,), samples=2, engine_options={"semantic_wait_seconds": 0})
        self.assertEqual(result["semantic_wait_seconds"], 0)
        self.assertEqual(result["engine_options"]["semantic_wait_seconds"], 0)
        measurement = result["measurements"][0]
        self.assertEqual(measurement["first_search_semantic_coverage"], ["pending", "pending"])
        self.assertEqual(measurement["first_search_pending_rate"], 1.0)
        # The index is still timed to completion and subsequent searches use it.
        self.assertGreater(measurement["semantic_index_seconds_median"], 0)
        self.assertEqual(len(measurement["needle_ranks"][0]), len(NEEDLES))
        self.assertIn("first_search_pending_rate=1.00", format_measurement(measurement))
        self.assertIsNone(benchmark_semantic_index._needle_rank({"matches": []}, 1))

    def test_measure_once_rejects_an_index_that_never_becomes_ready(self):
        class BrokenEngine:
            def ingest(self, text, label):
                return type("Capture", (), {"capture_id": "cap", "chunks": [], "semantic_chunks": []})()

            def search(self, *args, **kwargs):
                return {"matches": [], "semantic_coverage": "unavailable"}

            def wait_for_semantic_index(self, capture):
                return "failed"

            _ensure_embeddings = None

        with self.assertRaisesRegex(RuntimeError, "did not become ready: failed"):
            benchmark_semantic_index.measure_once(BrokenEngine(), 4)

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

    def test_single_needle_capture_measures_a_subsequent_search(self):
        measurement = run_benchmark((1,), samples=1)["measurements"][0]
        self.assertEqual(list(measurement["needle_ranks"][0]), [NEEDLES[0]["id"]])
        self.assertGreater(measurement["subsequent_search_seconds_median"], 0)

    def test_summarize_handles_zero_index_time_and_rejects_empty(self):
        sample = {
            "line_count": 4,
            "output_bytes": 10,
            "chunk_count": 1,
            "semantic_chunk_count": 1,
            "needle_ranks": {"a": None, "b": 2},
            "first_search_semantic_coverage": "pending",
            "first_search_needle_rank": None,
            "ingest_seconds": 0.0,
            "semantic_index_seconds": 0.0,
            "first_search_seconds": 0.0,
            "subsequent_search_seconds": 0.0,
        }
        summary = summarize([sample])
        self.assertIsNone(summary["semantic_chunks_per_second_median"])
        self.assertEqual(summary["needle_hit_at_1"], 0.0)
        self.assertEqual(summary["needle_mrr"], 0.25)
        self.assertEqual(summary["first_search_pending_rate"], 1.0)
        self.assertEqual(summary["first_search_needle_hit_at_1"], 0.0)
        self.assertEqual(summary["first_search_needle_mrr"], 0.0)
        self.assertIn("semantic_chunks_per_second=n/a", format_measurement(summary))
        with self.assertRaises(ValueError):
            summarize([])

    def test_engine_options_are_forwarded_and_reported(self):
        result = run_benchmark(
            (4,), samples=1,
            engine_options={"embedding_threads": 2, "semantic_chunk_lines": 2, "semantic_chunk_overlap": 1},
        )
        self.assertEqual(result["embedding_threads"], 2)
        self.assertEqual(result["semantic_chunking"], {"lines": 2, "bytes": 1024, "overlap": 1})
        self.assertEqual(result["engine_options"]["embedding_threads"], 2)
        self.assertNotIn("max_captures", result["engine_options"])

    def test_main_forwards_cli_flags(self):
        captured = {}

        def fake_run_benchmark(line_counts, samples, mode="hybrid", engine_options=None):
            captured.update(line_counts=line_counts, samples=samples, mode=mode, engine_options=engine_options)
            return {
                "embedding_model": "m",
                "embedding_threads": 3,
                "semantic_chunking": {"lines": 6, "bytes": 512, "overlap": 1},
                "test_embeddings": True,
                "mode": mode,
                "semantic_wait_seconds": 1.5,
                "model_load_seconds": 0.0,
                "measurements": [],
            }

        argv = [
            "benchmark_semantic_index.py", "--line-counts", "4", "--samples", "1", "--mode", "semantic",
            "--embedding-model", "m", "--embedding-threads", "3",
            "--semantic-chunk-lines", "6", "--semantic-chunk-bytes", "512", "--semantic-chunk-overlap", "1",
            "--semantic-wait-seconds", "1.5",
        ]
        with patch.object(benchmark_semantic_index, "run_benchmark", fake_run_benchmark), \
                patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            benchmark_semantic_index.main()
        self.assertEqual(captured["line_counts"], (4,))
        self.assertEqual(captured["mode"], "semantic")
        self.assertEqual(captured["engine_options"], {
            "embedding_model_name": "m", "embedding_threads": 3,
            "semantic_chunk_lines": 6, "semantic_chunk_bytes": 512, "semantic_chunk_overlap": 1,
            "semantic_wait_seconds": 1.5,
        })
        self.assertIn("threads=3", stdout.getvalue())
        self.assertIn("semantic_wait=1.5s", stdout.getvalue())
        self.assertIn("lines:6/bytes:512/overlap:1", stdout.getvalue())

    def test_benchmark_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            run_benchmark((), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((0,), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((4,), samples=0)
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            run_benchmark((4, 4), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((4,), samples=1, mode="bm25")
        with self.assertRaises(ValueError):
            benchmark_semantic_index.measure_once(None, 4, mode="bm25")
        with self.assertRaises(ValueError):
            benchmark_semantic_index._nearest_rank([])
        with self.assertRaises(ValueError):
            benchmark_semantic_index._nearest_rank([1.0], percentile=0)


    def test_workload_result_reports_model_load_and_sized_runs(self):
        record = run_benchmark((8,), samples=1, engine_options={"semantic_wait_seconds": float("inf")})
        result = benchmark_semantic_index.workload_result(record)
        self.assertEqual(result["workload"]["name"], "semantic-index")
        self.assertEqual(result["workload"]["fixture_version"], benchmark_semantic_index.FIXTURE_VERSION)
        parameters = result["workload"]["parameters"]
        self.assertEqual(parameters["semantic_wait_seconds"], "unbounded")
        self.assertEqual(parameters["engine_options"]["semantic_wait_seconds"], "unbounded")
        self.assertEqual(parameters["line_counts"], [8])
        self.assertEqual(result["environment"]["embedding_model"], record["embedding_model"])
        self.assertEqual([item["id"] for item in result["runs"]], ["model-load", "lines-8"])
        sized = result["runs"][1]
        self.assertEqual(sized["labels"], {"line_count": 8, "mode": "hybrid", "cache_state": "warm"})
        self.assertEqual(
            [entry["name"] for entry in sized["phases"]],
            ["ingest", "semantic_index", "first_search", "subsequent_search"],
        )
        self.assertEqual(sized["measurements"]["needle_mrr"]["unit"], "score")
        self.assertEqual(sized["measurements"]["throughput_per_second"]["unit"], "per_second")
        json.dumps(result, allow_nan=False)

    def test_result_flag_emits_json_on_stdout(self):
        argv = ["benchmark_semantic_index.py", "--line-counts", "4", "--samples", "1", "--result", "-"]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_semantic_index.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["runs"][1]["id"], "lines-4")
        self.assertIn("first_search_pending_rate", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
