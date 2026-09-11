"""Tests for direct-versus-captured routing measurements."""

import unittest

from benchmark_latency import _nearest_rank, _summary as latency_summary
from benchmark_routing import _summary as routing_summary, run_benchmark


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


if __name__ == "__main__":
    unittest.main()
