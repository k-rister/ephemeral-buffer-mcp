"""Tests for the semantic prefetch comparison benchmark."""

import unittest

from benchmark_prefetch import run_benchmark


class TestPrefetchBenchmark(unittest.TestCase):
    def test_benchmark_reports_both_modes(self):
        result = run_benchmark(line_count=4, samples=1)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(set(result["summaries"]), {"false", "true"})
        self.assertTrue(all(value >= 0 for summary in result["summaries"].values() for value in summary.values()))

    def test_benchmark_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            run_benchmark(line_count=0, samples=1)
        with self.assertRaises(ValueError):
            run_benchmark(line_count=1, samples=0)


if __name__ == "__main__":
    unittest.main()
