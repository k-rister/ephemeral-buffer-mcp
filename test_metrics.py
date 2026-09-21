"""Tests for opt-in, content-free local metrics."""

import os
import threading
import unittest
from unittest.mock import patch

from engine import EphemeralEngine
from metrics import MAX_SNAPSHOT_TOKENS, LocalMetrics, metrics_enabled


TEST_TOOL_NAMES = ("capture_file", "start_execution", "get_runtime_diagnostics")
TEST_TOOL_CATEGORIES = {
    "capture_file": "capture",
    "start_execution": "execution",
    "get_runtime_diagnostics": "diagnostics",
}


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
        self.assertEqual(stats["latency_ms"]["count"], 3)
        self.assertEqual(sum(stats["failure_categories"].values()), 1)
        self.assertEqual(stats["failure_categories"]["other"], 1)

    def test_tool_measurement_reports_bounded_latency_and_failure_categories(self):
        metrics = LocalMetrics(enabled=True)
        with patch(
            "metrics.time.perf_counter",
            side_effect=[0.0, 0.002, 1.0, 1.010, 2.0, 2.100],
        ):
            with metrics.measure("sample"):
                pass
            with metrics.measure("sample"):
                pass
            with self.assertRaisesRegex(RuntimeError, "private failure"):
                with metrics.measure("sample") as state:
                    state["failure_category"] = "timeout"
                    raise RuntimeError("private failure")

        stats = metrics.snapshot()["tools"]["sample"]
        self.assertEqual(stats["latency_ms"]["count"], 3)
        self.assertEqual(stats["latency_ms"]["p50"], 10.0)
        self.assertEqual(stats["latency_ms"]["p95"], 100.0)
        self.assertEqual(stats["latency_ms"]["p99"], 100.0)
        self.assertEqual(stats["latency_ms"]["overflow_count"], 0)
        self.assertEqual(stats["failure_categories"]["timeout"], 1)
        self.assertEqual(sum(stats["failure_categories"].values()), 1)
        self.assertNotIn("private failure", repr(stats))

        overflow = LocalMetrics._latency_distribution([0] * 14 + [1])
        self.assertIsNone(overflow["p50"])
        self.assertEqual(overflow["overflow_count"], 1)
        self.assertEqual(LocalMetrics._latency_distribution([1])["count"], 1)
        self.assertEqual(LocalMetrics._latency_bucket_index(60_000.1), 14)

    def test_latency_and_failure_categories_are_delta_additive(self):
        metrics = LocalMetrics(enabled=True)
        baseline = metrics.snapshot(include_snapshot_token=True)

        with metrics.measure("sample") as state:
            state["failure_category"] = "validation"
            state["success"] = False

        delta = metrics.snapshot(since_snapshot=baseline["snapshot_token"])
        stats = delta["tools"]["sample"]
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["failures"], 1)
        self.assertEqual(stats["latency_ms"]["count"], 1)
        self.assertEqual(stats["failure_categories"]["validation"], 1)
        self.assertEqual(sum(stats["failure_categories"].values()), 1)

    def test_discarded_measurement_does_not_count_as_a_tool_call(self):
        metrics = LocalMetrics(enabled=True)
        with metrics.measure("validation_probe") as state:
            state["record"] = False

        self.assertNotIn("validation_probe", metrics.snapshot()["tools"])

    def test_discarded_measurement_does_not_remove_concurrent_tool_record(self):
        metrics = LocalMetrics(enabled=True)
        validation_entered = threading.Event()
        release_validation = threading.Event()

        def discarded_validation():
            with metrics.measure("race_probe") as state:
                state["record"] = False
                validation_entered.set()
                self.assertTrue(release_validation.wait(timeout=2))

        validation_thread = threading.Thread(target=discarded_validation)
        validation_thread.start()
        self.assertTrue(validation_entered.wait(timeout=2))

        with metrics.measure("race_probe"):
            pass

        release_validation.set()
        validation_thread.join(timeout=2)
        self.assertFalse(validation_thread.is_alive())
        stats = metrics.snapshot()["tools"]["race_probe"]
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["successes"], 1)

    def test_in_flight_tool_is_included_in_coverage_snapshot(self):
        metrics = LocalMetrics(enabled=True)

        with metrics.measure("get_runtime_diagnostics"):
            snapshot = metrics.snapshot(
                available_tools=TEST_TOOL_NAMES,
                tool_categories=TEST_TOOL_CATEGORIES,
            )
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

        coverage = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
        )["interface_coverage"]
        self.assertEqual(coverage["used"], 2)
        self.assertEqual(coverage["available"], len(TEST_TOOL_NAMES))
        self.assertEqual(coverage["percentage"], 66.7)
        self.assertEqual(
            coverage["unused_tools"],
            ["get_runtime_diagnostics"],
        )
        self.assertEqual(
            coverage["by_category"],
            {
                "capture": {
                    "used": 1,
                    "available": 1,
                    "percentage": 100.0,
                    "unused_tools": [],
                },
                "diagnostics": {
                    "used": 0,
                    "available": 1,
                    "percentage": 0.0,
                    "unused_tools": ["get_runtime_diagnostics"],
                },
                "execution": {
                    "used": 1,
                    "available": 1,
                    "percentage": 100.0,
                    "unused_tools": [],
                },
            },
        )

    def test_interface_coverage_reaches_full_coverage(self):
        metrics = LocalMetrics(enabled=True)
        for tool_name in TEST_TOOL_NAMES:
            with metrics.measure(tool_name):
                pass

        self.assertEqual(
            metrics.snapshot(
                available_tools=TEST_TOOL_NAMES,
                tool_categories=TEST_TOOL_CATEGORIES,
            )["interface_coverage"],
            {
                "used": len(TEST_TOOL_NAMES),
                "available": len(TEST_TOOL_NAMES),
                "percentage": 100.0,
                "unused_tools": [],
                "by_category": {
                    "capture": {
                        "used": 1,
                        "available": 1,
                        "percentage": 100.0,
                        "unused_tools": [],
                    },
                    "diagnostics": {
                        "used": 1,
                        "available": 1,
                        "percentage": 100.0,
                        "unused_tools": [],
                    },
                    "execution": {
                        "used": 1,
                        "available": 1,
                        "percentage": 100.0,
                        "unused_tools": [],
                    },
                },
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

    def test_workflow_effectiveness_reports_rates_and_reductions(self):
        metrics = LocalMetrics(enabled=True)
        with metrics.measure("capture_text"):
            pass
        with metrics.measure("search_capture"):
            pass
        with metrics.measure("get_capture_slice"):
            pass
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with metrics.measure("failed_tool"):
                raise RuntimeError("failed")

        metrics.record_event("searches")
        metrics.record_event("capture_to_search")
        metrics.record_event("retrievals")
        metrics.record_event("search_to_retrieval")
        metrics.record_bytes("capture_input_bytes", 100)
        metrics.record_bytes("tool_response_bytes", 20)
        metrics.record_bytes("search_response_bytes", 10)
        metrics.record_bytes("retrieval_response_bytes", 5)

        effectiveness = metrics.snapshot()["workflow_effectiveness"]
        self.assertEqual(
            effectiveness["capture_to_search_rate"],
            {
                "status": "ok",
                "numerator": 1,
                "denominator": 1,
                "denominator_name": "searches",
                "percentage": 100.0,
            },
        )
        self.assertEqual(effectiveness["search_to_retrieval_rate"]["percentage"], 100.0)
        self.assertEqual(effectiveness["empty_search_rate"]["percentage"], 0.0)
        self.assertEqual(effectiveness["successful_call_rate"]["percentage"], 75.0)
        self.assertEqual(
            effectiveness["response_bytes_per_captured_byte"]["ratio"],
            0.2,
        )
        self.assertEqual(effectiveness["search_response_reduction"]["percentage"], 90.0)
        self.assertEqual(effectiveness["retrieval_response_reduction"]["percentage"], 95.0)

    def test_workflow_effectiveness_marks_zero_denominators_unavailable(self):
        metrics = LocalMetrics(enabled=True)
        effectiveness = metrics.snapshot()["workflow_effectiveness"]

        for metric in effectiveness.values():
            self.assertEqual(metric["status"], "unavailable")
            self.assertIsNone(metric.get("percentage", metric.get("ratio")))
            self.assertEqual(metric["reason"], "zero_denominator")

        metrics.record_bytes("capture_input_bytes", 100)
        effectiveness = metrics.snapshot()["workflow_effectiveness"]
        self.assertEqual(
            effectiveness["search_response_reduction"]["reason"],
            "zero_operation_count",
        )
        self.assertEqual(
            effectiveness["retrieval_response_reduction"]["reason"],
            "zero_operation_count",
        )

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

        snapshot = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
        )
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
                "by_category": {
                    "capture": {
                        "used": 0,
                        "available": 1,
                        "percentage": 0.0,
                        "unused_tools": ["capture_file"],
                    },
                    "diagnostics": {
                        "used": 0,
                        "available": 1,
                        "percentage": 0.0,
                        "unused_tools": ["get_runtime_diagnostics"],
                    },
                    "execution": {
                        "used": 0,
                        "available": 1,
                        "percentage": 0.0,
                        "unused_tools": ["start_execution"],
                    },
                },
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

    def test_snapshot_tokens_provide_independent_deltas_and_zero_windows(self):
        metrics = LocalMetrics(enabled=True)
        baseline = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
            include_snapshot_token=True,
        )

        with metrics.measure("capture_file") as state:
            state["result_count"] = 2
        metrics.record_event("captures")
        metrics.record_bytes("capture_input_bytes", 12)

        delta = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
            since_snapshot=baseline["snapshot_token"],
            include_snapshot_token=True,
        )
        self.assertEqual(delta["window"]["status"], "ok")
        self.assertEqual(delta["window"]["kind"], "delta")
        self.assertEqual(delta["window"]["started_at"], baseline["snapshot_at"])
        self.assertEqual(delta["tools"]["capture_file"]["calls"], 1)
        self.assertEqual(delta["tools"]["capture_file"]["result_count"], 2)
        self.assertEqual(delta["events"]["captures"], 1)
        self.assertEqual(delta["bytes"]["capture_input_bytes"], 12)
        self.assertEqual(delta["interface_coverage"]["used"], 1)
        self.assertEqual(
            delta["interface_coverage"]["unused_tools"],
            ["start_execution", "get_runtime_diagnostics"],
        )

        empty = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
            since_snapshot=delta["snapshot_token"],
        )
        self.assertEqual(empty["window"]["status"], "ok")
        self.assertEqual(empty["window"]["kind"], "delta")
        self.assertEqual(empty["tools"], {})
        self.assertEqual(empty["interface_coverage"]["used"], 0)
        self.assertEqual(empty["interface_coverage"]["percentage"], 0.0)
        self.assertEqual(empty["events"]["captures"], 0)
        self.assertEqual(empty["bytes"]["capture_input_bytes"], 0)
        self.assertNotEqual(empty["snapshot_token"], delta["snapshot_token"])
        self.assertEqual(metrics.snapshot()["events"]["captures"], 1)

    def test_unavailable_snapshot_tokens_are_distinct_from_zero_activity(self):
        metrics = LocalMetrics(enabled=True)
        unavailable = metrics.snapshot(
            available_tools=TEST_TOOL_NAMES,
            tool_categories=TEST_TOOL_CATEGORIES,
            since_snapshot="not-a-current-process-token",
        )
        self.assertEqual(unavailable["window"]["status"], "unavailable")
        self.assertEqual(unavailable["window"]["reason"], "snapshot_token_unavailable")
        self.assertNotIn("tools", unavailable)
        self.assertIn("snapshot_token", unavailable)

    def test_snapshot_token_history_is_bounded(self):
        metrics = LocalMetrics(enabled=True)
        tokens = [
            metrics.snapshot(include_snapshot_token=True)["snapshot_token"]
            for _ in range(MAX_SNAPSHOT_TOKENS + 1)
        ]

        unavailable = metrics.snapshot(since_snapshot=tokens[0])
        self.assertEqual(unavailable["window"]["status"], "unavailable")

    def test_snapshot_tokens_attribute_in_flight_calls_to_following_window(self):
        metrics = LocalMetrics(enabled=True)
        with metrics.measure("sample") as state:
            state["result_count"] = 4
            metrics.record_result_count("sample", 3)
            metrics.record_capture("capture-1")
            metrics.record_search("capture-1", 0)
            metrics.record_retrieval("capture-1")
            metrics.record_event("cleanups")
            metrics.record_bytes("capture_input_bytes", 9)
            baseline = metrics.snapshot(include_snapshot_token=True)
            delta = metrics.snapshot(since_snapshot=baseline["snapshot_token"])

            self.assertEqual(delta["tools"], {})
            self.assertEqual(delta["interface_coverage"]["used"], 0)
            self.assertEqual(delta["events"]["captures"], 0)
            self.assertEqual(delta["bytes"]["capture_input_bytes"], 0)

        following = metrics.snapshot(since_snapshot=delta["snapshot_token"])
        self.assertEqual(following["tools"]["sample"]["calls"], 1)
        self.assertEqual(following["tools"]["sample"]["successes"], 1)
        self.assertEqual(following["tools"]["sample"]["result_count"], 7)
        self.assertEqual(following["events"]["captures"], 1)
        self.assertEqual(following["events"]["searches"], 1)
        self.assertEqual(following["events"]["empty_searches"], 1)
        self.assertEqual(following["events"]["capture_to_search"], 1)
        self.assertEqual(following["events"]["retrievals"], 1)
        self.assertEqual(following["events"]["search_to_retrieval"], 1)
        self.assertEqual(following["events"]["cleanups"], 1)
        self.assertEqual(following["bytes"]["capture_input_bytes"], 9)

    def test_search_correlation_state_only_tracks_active_captures(self):
        metrics = LocalMetrics(enabled=True)
        metrics.record_search("unknown-capture", 0)

        self.assertEqual(metrics.snapshot()["events"]["searches"], 1)
        self.assertEqual(metrics.snapshot()["events"]["capture_to_search"], 0)
        self.assertEqual(metrics._searched, set())


if __name__ == "__main__":
    unittest.main()
