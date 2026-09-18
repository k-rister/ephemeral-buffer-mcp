"""Tests for the semantic prefetch comparison benchmark."""

import io
import json
import unittest
from unittest.mock import patch

import benchmark_prefetch
import workload_results as wr
from benchmark_prefetch import run_benchmark, workload_result


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


    def test_workload_result_reports_a_phase_timeline_per_mode(self):
        result = workload_result(run_benchmark(line_count=4, samples=1))
        self.assertEqual(result["workload"]["name"], "semantic-prefetch")
        self.assertEqual([item["id"] for item in result["runs"]], ["prefetch-off", "prefetch-on"])
        self.assertEqual(result["runs"][1]["labels"], {"mode": "prefetch", "prefetch": True, "line_count": 4})
        for item in result["runs"]:
            self.assertEqual([entry["name"] for entry in item["phases"]], ["ingest", "first_search", "subsequent_search"])
            self.assertTrue(all(entry["samples"] == 1 for entry in item["phases"]))

    def test_result_flag_emits_json_on_stdout(self):
        argv = [
            "benchmark_prefetch.py", "--line-count", "2", "--samples", "1", "--result", "-",
            "--experiment", "prefetch-sweep", "--metadata", "variant=default", "--metadata", "api_key=hidden",
        ]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_prefetch.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["workload"]["name"], "semantic-prefetch")
        self.assertEqual(result["experiment"], {"group": "prefetch-sweep", "metadata": {"variant": "default", "api_key": wr.REDACTED}})
        self.assertIn("prefetch=true", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
