"""Tests for opt-in, content-free local metrics."""

import os
import unittest
from unittest.mock import patch

from engine import EphemeralEngine
from metrics import LocalMetrics, metrics_enabled


TEST_TOOL_NAMES = ("capture_file", "start_execution", "get_runtime_diagnostics")


class TestMetrics(unittest.TestCase):
    def test_environment_switch_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(metrics_enabled())
        with patch.dict(os.environ, {"EPHEMERAL_METRICS": "true"}, clear=True):
            self.assertTrue(metrics_enabled())

    def test_disabled_metrics_keep_empty_snapshot(self):
        metrics = LocalMetrics(enabled=False)
        metrics.record_event("secret-event")
        self.assertEqual(metrics.snapshot(), {"enabled": False})

    def test_tool_measurement_tracks_success_failure_and_duration(self):
        metrics = LocalMetrics(enabled=True)
        with metrics.measure("sample"):
            pass
        with metrics.measure("sample") as state:
            state["result_count"] = 2
        metrics.record_result_count("sample", 3)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with metrics.measure("sample"):
                raise RuntimeError("expected")

        stats = metrics.snapshot()["tools"]["sample"]
        self.assertEqual(stats["calls"], 3)
        self.assertEqual(stats["successes"], 2)
        self.assertEqual(stats["failures"], 1)
        self.assertEqual(stats["result_count"], 5)
        self.assertGreaterEqual(stats["total_duration_ms"], 0)

    def test_in_flight_tool_is_included_in_coverage_snapshot(self):
        metrics = LocalMetrics(enabled=True)

        with metrics.measure("get_runtime_diagnostics"):
            snapshot = metrics.snapshot(available_tools=TEST_TOOL_NAMES)
            self.assertEqual(snapshot["interface_coverage"]["used"], 1)
            self.assertNotIn("get_runtime_diagnostics", snapshot["interface_coverage"]["unused_tools"])
            self.assertEqual(snapshot["tools"]["get_runtime_diagnostics"]["calls"], 1)
            self.assertEqual(snapshot["tools"]["get_runtime_diagnostics"]["successes"], 0)

        self.assertEqual(
            metrics.snapshot()["tools"]["get_runtime_diagnostics"]["successes"],
            1,
        )

    def test_interface_coverage_reports_unused_tools(self):
        metrics = LocalMetrics(enabled=True)
        with metrics.measure("capture_file"):
            pass
        with metrics.measure("start_execution"):
            pass

        coverage = metrics.snapshot(available_tools=TEST_TOOL_NAMES)["interface_coverage"]
        self.assertEqual(coverage["used"], 2)
        self.assertEqual(coverage["available"], len(TEST_TOOL_NAMES))
        self.assertEqual(coverage["percentage"], 66.7)
        self.assertEqual(
            coverage["unused_tools"],
            ["get_runtime_diagnostics"],
        )

    def test_interface_coverage_reaches_full_coverage(self):
        metrics = LocalMetrics(enabled=True)
        for tool_name in TEST_TOOL_NAMES:
            with metrics.measure(tool_name):
                pass

        self.assertEqual(
            metrics.snapshot(available_tools=TEST_TOOL_NAMES)["interface_coverage"],
            {
                "used": len(TEST_TOOL_NAMES),
                "available": len(TEST_TOOL_NAMES),
                "percentage": 100.0,
                "unused_tools": [],
            },
        )

    def test_engine_records_content_free_usage_events(self):
        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(max_captures=1, embedding_model_name="test", metrics=metrics)
        engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[0.0] * 384 for _ in texts]},
        )()

        capture = engine.ingest("one useful line", label="private label")
        engine.search("useful", mode="bm25", capture_id=capture.capture_id)
        engine.search("not-present", mode="bm25", capture_id=capture.capture_id)
        engine.get_slice(1, 1, capture_id=capture.capture_id)
        engine.ingest("second line", label="another private label")

        snapshot = metrics.snapshot()
        self.assertTrue(snapshot["enabled"])
        self.assertEqual(snapshot["events"]["captures"], 2)
        self.assertEqual(snapshot["events"]["searches"], 2)
        self.assertEqual(snapshot["events"]["empty_searches"], 1)
        self.assertEqual(snapshot["events"]["capture_to_search"], 2)
        self.assertEqual(snapshot["events"]["search_to_retrieval"], 1)
        self.assertEqual(snapshot["events"]["evictions"], 1)
        self.assertNotIn("private", repr(snapshot))
        self.assertEqual(snapshot["bytes"]["capture_input_bytes"], len("one useful line".encode()) + len("second line".encode()))
        self.assertEqual(snapshot["bytes"]["capture_original_bytes"], len("one useful line".encode()) + len("second line".encode()))

    def test_byte_counters_are_content_free_and_zero_filled(self):
        metrics = LocalMetrics(enabled=True)
        with self.assertRaisesRegex(ValueError, "unknown byte counter"):
            metrics.record_bytes("unknown", 1)
        metrics.record_bytes("capture_input_bytes", 12)
        metrics.record_bytes("socket_response_bytes", 34)

        self.assertEqual(metrics.snapshot()["bytes"], {
            "capture_input_bytes": 12,
            "capture_retained_bytes": 0,
            "capture_original_bytes": 0,
            "tool_response_bytes": 0,
            "search_response_bytes": 0,
            "retrieval_response_bytes": 0,
            "socket_request_bytes": 0,
            "socket_response_bytes": 34,
        })
        self.assertNotIn("private", repr(metrics.snapshot()))

    def test_invalid_original_size_does_not_partially_record_capture(self):
        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(max_captures=1, metrics=metrics)
        try:
            with self.assertRaisesRegex(ValueError, "original_byte_size"):
                engine.ingest(
                    "payload",
                    truncated=True,
                    original_byte_size="not-an-integer",
                )

            self.assertEqual(engine.captures, {})
            self.assertEqual(metrics.snapshot()["events"]["captures"], 0)
            self.assertEqual(metrics.snapshot()["bytes"]["capture_original_bytes"], 0)
        finally:
            engine.shutdown()

    def test_snapshot_has_stable_zero_filled_event_schema(self):
        metrics = LocalMetrics(enabled=True)

        snapshot = metrics.snapshot(available_tools=TEST_TOOL_NAMES)
        self.assertEqual(snapshot["scope"], "process")
        self.assertRegex(snapshot["started_at"], r"Z$")
        self.assertRegex(snapshot["snapshot_at"], r"Z$")
        self.assertEqual(
            snapshot["interface_coverage"],
            {
                "used": 0,
                "available": len(TEST_TOOL_NAMES),
                "percentage": 0.0,
                "unused_tools": list(TEST_TOOL_NAMES),
            },
        )
        self.assertEqual(
            metrics.snapshot()["events"],
            {
                "captures": 0,
                "searches": 0,
                "empty_searches": 0,
                "capture_to_search": 0,
                "retrievals": 0,
                "search_to_retrieval": 0,
                "evictions": 0,
                "cleanups": 0,
            },
        )
        self.assertEqual(
            metrics.snapshot()["bytes"],
            {
                "capture_input_bytes": 0,
                "capture_retained_bytes": 0,
                "capture_original_bytes": 0,
                "tool_response_bytes": 0,
                "search_response_bytes": 0,
                "retrieval_response_bytes": 0,
                "socket_request_bytes": 0,
                "socket_response_bytes": 0,
            },
        )

    def test_search_correlation_state_only_tracks_active_captures(self):
        metrics = LocalMetrics(enabled=True)
        metrics.record_search("unknown-capture", 0)

        self.assertEqual(metrics.snapshot()["events"]["searches"], 1)
        self.assertEqual(metrics.snapshot()["events"]["capture_to_search"], 0)
        self.assertEqual(metrics._searched, set())


if __name__ == "__main__":
    unittest.main()
