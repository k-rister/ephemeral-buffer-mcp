import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_concurrency
import workload_results as wr
from benchmark_concurrency import build_record, check_regression, load_baseline, workload_result, write_results


class TestBenchmarkResults(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "ingest_per_second": 100.0,
            "reads_per_second": 1000.0,
            "minimum_ratio": 0.8,
        }

    def test_measurements_within_tolerance_have_no_regression(self):
        results = {"ingest_per_second": 80.0, "reads_per_second": 800.0}

        self.assertEqual(check_regression(results, self.baseline), [])

    def test_measurements_below_tolerance_are_reported(self):
        results = {"ingest_per_second": 79.9, "reads_per_second": 799.9}

        failures = check_regression(results, self.baseline)

        self.assertEqual(len(failures), 2)
        self.assertIn("ingest_per_second", failures[0])
        self.assertIn("reads_per_second", failures[1])

    def test_baseline_is_loaded_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(self.baseline), encoding="utf-8")

            self.assertEqual(load_baseline(path), self.baseline)

    def test_invalid_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"minimum_ratio": 2}', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_baseline(path)

    def test_results_are_written_with_regression_status(self):
        results = {"captures": 32, "ingest_per_second": 100.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.json"
            write_results(path, results, self.baseline, [])

            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(record["passed"])
        self.assertEqual(record["baseline"], self.baseline)
        self.assertEqual(record["regressions"], [])


    def test_workload_result_reports_throughput_and_regression_status(self):
        results = {
            "schema_version": 1,
            "captures": 4,
            "workers": 2,
            "ingest_seconds": 0.5,
            "ingest_per_second": 8.0,
            "read_seconds": 0.25,
            "reads_per_second": 16.0,
            "buffer_stats": {},
        }
        passed = workload_result(build_record(results, None, []))
        self.assertEqual(passed["workload"]["name"], "concurrency")
        self.assertEqual(passed["status"], "success")
        run = passed["runs"][0]
        self.assertEqual(run["id"], "captures-4-workers-2")
        self.assertEqual(run["measurements"]["ingest_per_second"], {"unit": "per_second", "value": 8.0, "samples": 1})
        self.assertEqual([entry["name"] for entry in run["phases"]], ["ingest", "read"])
        failed = workload_result(build_record(results, self.baseline, ["ingest too slow"]))
        self.assertEqual(failed["status"], "failure")
        self.assertEqual(failed["runs"][0]["errors"], ["ingest too slow"])
        self.assertEqual(failed["workload"]["parameters"]["baseline"], self.baseline)

    def test_result_flag_reports_regressions_and_still_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(
                json.dumps({"ingest_per_second": 1e9, "reads_per_second": 1e9, "minimum_ratio": 1.0}),
                encoding="utf-8",
            )
            argv = [
                "benchmark_concurrency.py", "--captures", "2", "--workers", "2",
                "--baseline", str(baseline), "--result", "-",
            ]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr, self.assertRaises(SystemExit) as raised:
                benchmark_concurrency.main()
        self.assertEqual(raised.exception.code, 1)
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["status"], "failure")
        self.assertEqual(len(result["runs"][0]["errors"]), 2)
        self.assertIn("REGRESSION:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
