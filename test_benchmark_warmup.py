"""Tests for the embedding startup warm-up comparison benchmark."""

import io
import json
import unittest
from unittest.mock import patch

import benchmark_warmup
import workload_results as wr
from benchmark_warmup import run_benchmark, workload_result


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

        converted = workload_result(result)
        self.assertEqual(converted["workload"]["name"], "embedding-warmup")
        self.assertEqual([item["id"] for item in converted["runs"]], ["warmup-off", "warmup-on"])
        self.assertEqual(converted["runs"][1]["labels"], {"mode": "warmup", "warmup": True, "cache_state": "cold"})
        for item in converted["runs"]:
            self.assertEqual(
                set(item["measurements"]),
                set(benchmark_warmup.TIMING_FIELDS) | {"rss_delta_bytes"},
            )
            self.assertEqual(item["measurements"]["first_search_seconds"]["samples"], 1)

    def test_benchmark_rejects_invalid_samples(self):
        with self.assertRaises(ValueError):
            run_benchmark(samples=0)


    def test_result_flag_emits_json_on_stdout(self):
        canned = {
            "schema_version": 1,
            "samples": 1,
            "records": [],
            "summaries": {
                "false": {field: 0.5 for field in benchmark_warmup.TIMING_FIELDS} | {"rss_delta_bytes": None},
                "true": {field: 0.25 for field in benchmark_warmup.TIMING_FIELDS} | {"rss_delta_bytes": 2048},
            },
        }
        argv = ["benchmark_warmup.py", "--samples", "1", "--result", "-"]
        with patch.object(benchmark_warmup, "run_benchmark", return_value=canned), patch("sys.argv", argv), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            benchmark_warmup.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["runs"][0]["measurements"]["rss_delta_bytes"]["median"], None)
        self.assertEqual(result["runs"][1]["measurements"]["rss_delta_bytes"]["median"], 2048)
        self.assertIn("warmup=false", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
