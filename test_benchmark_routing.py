"""Tests for direct-versus-captured routing measurements."""

import unittest

from benchmark_routing import run_benchmark


class TestRoutingBenchmark(unittest.TestCase):
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
