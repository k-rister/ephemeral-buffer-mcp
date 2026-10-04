"""Tests for the semantic-memory benchmark's shared workload result export."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_semantic_memory
import workload_results as wr


def sample_record(validation_status="not_needed"):
    return {
        "schema_version": 1,
        "environment": {"python": "3.12", "platform": "test", "cpu_count": 4},
        "configuration": {
            "embedding_model": "test-model",
            "embedding_threads": 1,
            "embedding_batch_size": 16,
            "embedding_max_batch_tokens": 4096,
            "embedding_cpu_mem_arena_enabled": False,
            "semantic_max_index_input_bytes": 1024,
            "semantic_chunk_lines": 8,
            "semantic_chunk_bytes": 1024,
            "rss_sample_interval_seconds": 0.005,
        },
        "input": {
            "capture_bytes": 255,
            "line_count": 4,
            "semantic_chunk_count": 1,
            "semantic_input_bytes": 255,
        },
        "timing_seconds": {"model_load": 0.25, "semantic_index": 0.5},
        "semantic_index_status": "ready",
        "budget_fallback_validation": {"status": validation_status},
        "rss_bytes": {
            "before_model_load": 1000,
            "model_load_sampled_peak": 1200,
            "after_model_load": 1100,
            "before_semantic_index": 1100,
            "semantic_index_sampled_peak": 1400,
            "after_semantic_index": 1300,
            "after_capture_clear_and_gc": 1000,
        },
        "retained_embedding_bytes": 128,
    }


class FakeEngine:
    def __init__(self, **options):
        self.options = options
        self.capture = SimpleNamespace(capture_id="capture-1")

    def ingest(self, text, label):
        return self.capture

    def load_embedding_model(self):
        return None

    def index_capture(self, capture_id):
        return "ready"

    def get_capture_diagnostics(self, capture_id):
        return SimpleNamespace(
            semantic_chunk_count=1,
            semantic_input_bytes=len("synthetic semantic chunk".encode("utf-8")),
            retained_embedding_bytes=128,
        )

    def get_buffer_stats(self):
        return {
            "embedding_model": self.options.get("embedding_model_name", "test-model"),
            "embedding_threads": self.options["embedding_threads"],
            "embedding_batch_size": self.options["embedding_batch_size"],
            "embedding_max_batch_tokens": self.options["embedding_max_batch_tokens"],
            "embedding_cpu_mem_arena_enabled": self.options["embedding_cpu_mem_arena_enabled"],
            "semantic_max_index_input_bytes": self.options["semantic_max_index_input_bytes"],
            "semantic_chunk_lines": self.options["semantic_chunk_lines"],
            "semantic_chunk_bytes": self.options["semantic_chunk_bytes"],
        }

    def clear(self, capture_id):
        return True

    def shutdown(self):
        return None


class TestSemanticMemoryWorkloadResult(unittest.TestCase):
    def test_workload_result_reports_memory_stages_and_budget_validation(self):
        converted = benchmark_semantic_memory.workload_result(sample_record("passed"))
        self.assertEqual(converted["workload"]["name"], "semantic-memory")
        self.assertEqual([run["id"] for run in converted["runs"]], ["model-load", "semantic-index"])
        self.assertEqual(converted["status"], "success")
        index_run = converted["runs"][1]
        self.assertEqual(index_run["measurements"]["retained_embedding_bytes"]["value"], 128)
        self.assertEqual(index_run["measurements"]["rss_delta_bytes"]["value"], 200)
        self.assertEqual(index_run["measurements"]["rss_after_clear_bytes"]["value"], 1000)
        self.assertEqual([phase["name"] for phase in index_run["phases"]], ["semantic_index"])
        wr.validate_result(converted)

    def test_failed_fallback_validation_marks_the_result_failed(self):
        record = sample_record("failed")
        record["semantic_index_status"] = "budget_exceeded"
        converted = benchmark_semantic_memory.workload_result(record)
        self.assertEqual(converted["status"], "failure")
        self.assertEqual(converted["runs"][1]["status"], "failure")
        self.assertIn("fallback validation failed", converted["errors"][0])

    def test_main_rejects_failed_semantic_index_status(self):
        argv = [
            "benchmark_semantic_memory.py",
            "--line-count", "4",
            "--line-bytes", "32",
        ]
        with patch.object(benchmark_semantic_memory, "EphemeralEngine", FakeEngine), patch.object(
            FakeEngine, "index_capture", return_value="failed"
        ), patch.object(
            benchmark_semantic_memory,
            "measure_rss_stage",
            side_effect=lambda action, interval: (action(), 0.25, 2048),
        ), patch.object(benchmark_semantic_memory, "process_rss_bytes", return_value=1024), patch.object(
            benchmark_semantic_memory.time, "sleep"
        ), patch("sys.argv", argv):
            with self.assertRaisesRegex(RuntimeError, "semantic indexing failed with status: failed"):
                benchmark_semantic_memory.main()

    def test_result_stdout_contains_only_shared_json_and_report_goes_to_stderr(self):
        argv = [
            "benchmark_semantic_memory.py",
            "--line-count", "4",
            "--line-bytes", "32",
            "--result", "-",
            "--experiment", "memory-sweep",
            "--metadata", "model=test-model",
            "--metadata", "owner=team",
            "--metadata", "api_key=hidden",
            "--redact", "owner",
        ]
        with patch.object(benchmark_semantic_memory, "EphemeralEngine", FakeEngine), patch.object(
            benchmark_semantic_memory,
            "measure_rss_stage",
            side_effect=lambda action, interval: (action(), 0.25, 2048),
        ), patch.object(benchmark_semantic_memory, "process_rss_bytes", return_value=1024), patch.object(
            benchmark_semantic_memory.time, "sleep"
        ), patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_semantic_memory.main()

        exported = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(exported["workload"]["name"], "semantic-memory")
        self.assertEqual(
            exported["experiment"],
            {
                "group": "memory-sweep",
                "metadata": {"model": "test-model", "owner": wr.REDACTED, "api_key": wr.REDACTED},
            },
        )
        self.assertEqual(exported["details"]["schema_version"], 1)
        self.assertIn("semantic_index=ready", stderr.getvalue())
        self.assertIn("fallback_validation=not_needed", stderr.getvalue())

    def test_result_file_preserves_the_native_stdout_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "nested" / "memory.result.json"
            argv = [
                "benchmark_semantic_memory.py",
                "--line-count", "4",
                "--line-bytes", "32",
                "--result", str(result_path),
            ]
            with patch.object(benchmark_semantic_memory, "EphemeralEngine", FakeEngine), patch.object(
                benchmark_semantic_memory,
                "measure_rss_stage",
                side_effect=lambda action, interval: (action(), 0.25, 2048),
            ), patch.object(benchmark_semantic_memory, "process_rss_bytes", return_value=1024), patch.object(
                benchmark_semantic_memory.time, "sleep"
            ), patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                benchmark_semantic_memory.main()

            native = json.loads(stdout.getvalue())
            exported = wr.load_result(result_path)
            self.assertEqual(native["schema_version"], 1)
            self.assertEqual(exported["workload"]["producer"], "benchmark_semantic_memory.py")
            self.assertEqual(exported["details"], native)


if __name__ == "__main__":
    unittest.main()
