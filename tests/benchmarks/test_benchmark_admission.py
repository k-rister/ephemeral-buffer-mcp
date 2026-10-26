"""Tests for the admission benchmark's JSON and human-readable output streams."""

import io
import json
import unittest
from unittest.mock import patch

import benchmarks.benchmark_admission as benchmark_admission
import benchmarks.workload_results as wr


class TestAdmissionBenchmarkOutput(unittest.TestCase):
    def test_result_stdout_contains_only_one_json_document(self):
        summary = {
            "repetition_count": 1,
            "success_rate": 1.0,
            "busy_rejection_rate": 0.25,
            "accepted_throughput_per_second_median": 10.0,
            "max_active_at_saturation": 2,
            "max_queued_at_saturation": 1,
            "passed": True,
        }
        record = {"passed": True, "scenarios": {"mcp": summary, "socket": summary}}
        result_document = wr.build_result(
            workload="foreground-admission",
            kind="benchmark",
            producer="benchmark_admission.py",
            producer_schema_version=1,
            runs=[wr.run("sample")],
        )

        async def fake_run_benchmark(_args):
            return record

        with patch.object(benchmark_admission, "run_benchmark", fake_run_benchmark), patch.object(
            benchmark_admission, "_workload_result", return_value=result_document
        ), patch("sys.argv", ["benchmark_admission.py", "--result", "-"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            benchmark_admission.main()

        parsed = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(parsed["workload"]["name"], "foreground-admission")
        summary_lines = stderr.getvalue().splitlines()
        self.assertEqual(len(summary_lines), 2)
        self.assertIn("transport=mcp", summary_lines[0])
        self.assertIn("transport=socket", summary_lines[1])


if __name__ == "__main__":
    unittest.main()
