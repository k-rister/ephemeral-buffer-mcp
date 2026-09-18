"""Tests for direct-versus-captured routing measurements."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_routing
import workload_results as wr
from benchmark_latency import _nearest_rank, _summary as latency_summary
from benchmark_routing import _summary as routing_summary, run_benchmark, workload_result


class TestRoutingBenchmark(unittest.TestCase):
    def test_nearest_rank_p95_preserves_small_sample_tail(self):
        self.assertEqual(_nearest_rank([1, 2, 100]), 100)
        self.assertEqual(_nearest_rank([100, 1, 2]), 100)
        self.assertEqual(_nearest_rank([1]), 1)
        self.assertEqual(_nearest_rank([1, 1, 2, 2]), 2)

        latency_samples = [
            {
                "line_count": 16,
                "output_bytes": 100,
                "command_seconds": value,
                "ingest_seconds": value,
                "semantic_index_seconds": value,
                "summary_seconds": value,
                "total_seconds": value,
            }
            for value in (1, 2, 100)
        ]
        self.assertEqual(latency_summary(latency_samples)["total_seconds_p95"], 100)

        routing_samples = [
            {
                "direct_seconds": value,
                "captured_seconds": value,
                "output_bytes": 100,
            }
            for value in (1, 2, 100)
        ]
        routing_result = routing_summary(routing_samples, 16)
        self.assertEqual(routing_result["direct_seconds_p95"], 100)
        self.assertEqual(routing_result["captured_seconds_p95"], 100)

    def test_summaries_reject_empty_samples(self):
        with self.assertRaises(ValueError):
            latency_summary([])
        with self.assertRaises(ValueError):
            routing_summary([], 16)

    def test_benchmark_reports_profiles_and_comparable_timings(self):
        result = run_benchmark((2, 4), samples=2)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual([profile["profile"] for profile in result["profiles"]], ["custom", "custom"])
        self.assertEqual([profile["samples"] for profile in result["profiles"]], [2, 2])
        self.assertTrue(all(profile["direct_seconds_median"] > 0 for profile in result["profiles"]))
        self.assertTrue(all(profile["captured_seconds_median"] > 0 for profile in result["profiles"]))
        self.assertTrue(all(profile["output_bytes"] > 0 for profile in result["profiles"]))

    def test_benchmark_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            run_benchmark((), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((0,), samples=1)
        with self.assertRaises(ValueError):
            run_benchmark((1,), samples=0)
        # Sizes become run IDs, so a repeated size is rejected before any work runs.
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            run_benchmark((2, 2), samples=1)


    def test_workload_result_reports_one_run_per_profile(self):
        results = run_benchmark((2,), samples=1)
        result = workload_result(results)
        self.assertEqual(result["workload"]["name"], "capture-routing")
        self.assertEqual(result["workload"]["parameters"], {"line_counts": [2], "samples": 1})
        run = result["runs"][0]
        self.assertEqual(run["id"], "custom-2")
        self.assertEqual(run["labels"], {"profile": "custom", "line_count": 2})
        self.assertEqual(
            set(run["measurements"]),
            {"output_bytes", "direct_seconds", "captured_seconds", "capture_overhead_seconds", "capture_overhead_ratio"},
        )
        self.assertEqual(run["measurements"]["capture_overhead_ratio"]["unit"], "ratio")
        self.assertEqual(result["details"], results)

    def test_result_flag_writes_a_file_next_to_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "routing-result.json"
            argv = ["benchmark_routing.py", "--line-counts", "2", "--samples", "1", "--result", str(path)]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                benchmark_routing.main()
            result = wr.load_result(path)
        self.assertEqual(result["runs"][0]["id"], "custom-2")
        self.assertIn("profile=custom", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
