"""Tests for the embedding startup warm-up comparison benchmark."""

import unittest

from benchmark_warmup import run_benchmark


class TestWarmupBenchmark(unittest.TestCase):
    def test_benchmark_reports_lazy_and_warmup_modes(self):
        result = run_benchmark(samples=1)

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(set(result["summaries"]), {"false", "true"})
        self.assertEqual(
            {record["warmup_state"] for record in result["records"]},
            {"disabled", "ready"},
        )
        for summary in result["summaries"].values():
            self.assertGreaterEqual(summary["engine_init_seconds"], 0)
            self.assertGreaterEqual(summary["startup_ready_seconds"], 0)
            self.assertGreaterEqual(summary["embedding_ready_seconds"], 0)
            self.assertGreaterEqual(summary["first_search_seconds"], 0)
        lazy = next(record for record in result["records"] if not record["warmup"])
        self.assertEqual(lazy["embedding_ready_seconds"], lazy["first_search_seconds"])

    def test_benchmark_rejects_invalid_samples(self):
        with self.assertRaises(ValueError):
            run_benchmark(samples=0)


if __name__ == "__main__":
    unittest.main()
