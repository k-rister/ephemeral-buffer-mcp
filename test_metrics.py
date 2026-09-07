"""Tests for opt-in, content-free local metrics."""

import os
import unittest
from unittest.mock import patch

from engine import EphemeralEngine
from metrics import LocalMetrics, metrics_enabled


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

    def test_snapshot_has_stable_zero_filled_event_schema(self):
        metrics = LocalMetrics(enabled=True)

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

    def test_search_correlation_state_only_tracks_active_captures(self):
        metrics = LocalMetrics(enabled=True)
        metrics.record_search("unknown-capture", 0)

        self.assertEqual(metrics.snapshot()["events"]["searches"], 1)
        self.assertEqual(metrics.snapshot()["events"]["capture_to_search"], 0)
        self.assertEqual(metrics._searched, set())


if __name__ == "__main__":
    unittest.main()
