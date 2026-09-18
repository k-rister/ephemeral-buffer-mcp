"""Tests for the command-capture latency benchmark."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_latency
import workload_results as wr


class FakeEngine:
    instances = []

    def __init__(self, max_captures, **_kwargs):
        self.max_buffer_bytes = 1024
        self.embedding_model_name = "test-model"
        self.events = []
        self._next_capture = 1
        self.__class__.instances.append(self)

    def ingest(self, content, *, label, **_kwargs):
        capture = SimpleNamespace(capture_id=f"cap-{self._next_capture}")
        self._next_capture += 1
        self.events.append(("ingest", label, content))
        return capture

    def _ensure_embeddings(self, capture):
        self.events.append(("ensure_embeddings", capture.capture_id))

    def shutdown(self):
        self.events.append(("shutdown",))

    def get_summary(self, capture_id):
        self.events.append(("summary", capture_id))
        return {}


class TestBenchmarkLatency(unittest.TestCase):
    def test_warmup_materializes_embeddings_before_warm_samples(self):
        FakeEngine.instances = []
        with patch.object(benchmark_latency, "EphemeralEngine", FakeEngine), patch.object(
            benchmark_latency,
            "run_command_bounded",
            return_value=("benchmark line\n", 0, False, 15, False),
        ):
            benchmark_latency.run_benchmark((1,), samples=1)

        self.assertEqual(len(FakeEngine.instances), 2)
        warm_events = FakeEngine.instances[1].events
        self.assertEqual(warm_events[:2], [
            ("ingest", "latency-warmup", "warmup"),
            ("ensure_embeddings", "cap-1"),
        ])
        self.assertLess(
            warm_events.index(("ensure_embeddings", "cap-1")),
            warm_events.index(("ingest", "latency-1", "benchmark line\n")),
        )


    def test_workload_result_reports_cold_and_warm_runs_with_phases(self):
        FakeEngine.instances = []
        with patch.object(benchmark_latency, "EphemeralEngine", FakeEngine), patch.object(
            benchmark_latency,
            "run_command_bounded",
            return_value=("benchmark line\n", 0, False, 15, False),
        ):
            results = benchmark_latency.run_benchmark((1, 2), samples=2)
        result = benchmark_latency.workload_result(results)
        self.assertEqual(result["workload"]["name"], "capture-latency")
        self.assertEqual(result["workload"]["parameters"], {"line_counts": [1, 2], "samples": 2})
        self.assertEqual(result["environment"]["embedding_model"], "test-model")
        self.assertEqual([item["id"] for item in result["runs"]], ["cold-start", "lines-1", "lines-2"])
        self.assertEqual(result["runs"][0]["labels"], {"cache_state": "cold", "line_count": 1})
        warm = result["runs"][1]
        self.assertEqual(warm["labels"], {"cache_state": "warm", "line_count": 1})
        self.assertEqual(warm["measurements"]["output_bytes"], {"unit": "bytes", "value": 15})
        self.assertEqual(warm["measurements"]["wall_time_seconds"]["samples"], 2)
        self.assertEqual([entry["name"] for entry in warm["phases"]], list(benchmark_latency.PHASE_NAMES))
        self.assertEqual(result["details"], results)

    def test_benchmark_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            benchmark_latency.run_benchmark((), samples=1)
        with self.assertRaises(ValueError):
            benchmark_latency.run_benchmark((0,), samples=1)
        with self.assertRaises(ValueError):
            benchmark_latency.run_benchmark((1,), samples=0)
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            benchmark_latency.run_benchmark((1, 1), samples=1)

    def test_duplicate_line_counts_are_rejected_before_the_benchmark_runs(self):
        FakeEngine.instances = []
        argv = ["benchmark_latency.py", "--line-counts", "16", "16", "--samples", "1", "--result", "-"]
        with patch.object(benchmark_latency, "EphemeralEngine", FakeEngine), patch("sys.argv", argv), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as exit_info:
            benchmark_latency.main()
        self.assertEqual(exit_info.exception.code, 2)
        self.assertIn("line_counts must not contain duplicates", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(FakeEngine.instances, [])

    def test_result_flag_moves_the_report_to_stderr(self):
        FakeEngine.instances = []
        argv = ["benchmark_latency.py", "--line-counts", "1", "--samples", "1", "--result", "-"]
        with patch.object(benchmark_latency, "EphemeralEngine", FakeEngine), patch.object(
            benchmark_latency,
            "run_command_bounded",
            return_value=("benchmark line\n", 0, False, 15, False),
        ), patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_latency.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["workload"]["name"], "capture-latency")
        self.assertIn("cold_start=", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
