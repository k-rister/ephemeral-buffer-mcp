import json
import tempfile
import unittest
from pathlib import Path

from benchmark_concurrency import check_regression, load_baseline, write_results


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


if __name__ == "__main__":
    unittest.main()
