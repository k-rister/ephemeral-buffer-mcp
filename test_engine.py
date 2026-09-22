"""
Comprehensive unit & integration tests for EphemeralEngine and MCP Server tools.
"""

import io
import sqlite3
import sys
import threading
import time
import unittest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
from config import CATALOGUE_EMBEDDING_MODEL, FP32_EMBEDDING_MODEL
from engine import (
    Chunk,
    HYBRID_LEXICAL_WEIGHT,
    register_bundled_embedding_models,
    Capture,
    EphemeralEngine,
    PREVIEW_MAX_BYTES,
    SEARCH_SNIPPET_MAX_BYTES,
    _bounded_preview,
    _decode_git_path,
    _parse_git_diff_paths,
    detect_content_type,
    detect_signals,
    estimate_tokens,
    estimate_tokens_from_bytes,
    parse_unified_diff,
    process_rss_bytes,
    sqlite_fts5_available,
)


def _semantic_index_threads():
    """Return the live on-demand semantic index pool threads, by their public names."""
    return {thread for thread in threading.enumerate() if thread.name.startswith("semantic-index")}


class TestEngineClassification(unittest.TestCase):
    def test_capture_preserves_legacy_positional_field_order(self):
        capture = Capture(
            "cap-legacy",
            "legacy",
            0.0,
            [],
            0,
            [],
            None,
            None,
            "text",
            None,
            False,
            None,
            None,
            False,
            "not-requested",
            0,
            False,
        )
        self.assertEqual(capture.semantic_index_state, "not-requested")
        self.assertEqual(capture.active_readers, 0)
        self.assertFalse(capture.storage_close_pending)
        self.assertEqual(capture.source, "capture")
        self.assertIsNone(capture.duration_ms)
        self.assertEqual(capture.structured_metrics, {})

    def test_preview_budget_smaller_than_marker_remains_bounded(self):
        preview = _bounded_preview("content", max_bytes=1)
        self.assertLessEqual(len(preview.encode("utf-8")), 1)

    def test_sqlite_fts5_probe_reports_missing_capability(self):
        class BrokenConnection:
            def __init__(self):
                self.closed = False

            def execute(self, _statement):
                raise sqlite3.OperationalError("fts5 unavailable")

            def close(self):
                self.closed = True

        connection = BrokenConnection()
        with patch("engine.sqlite3.connect", return_value=connection):
            self.assertFalse(sqlite_fts5_available())
        self.assertTrue(connection.closed)

    def test_empty_and_non_diff_input_has_no_diff_metadata(self):
        self.assertIsNone(parse_unified_diff([]))
        self.assertIsNone(parse_unified_diff(["ordinary text", "with no patch markers"]))
        self.assertIsNone(parse_unified_diff(["@@ -1 +1 @@", "context without a file header"]))

    def test_parse_unified_diff_handles_fallback_and_file_statuses(self):
        fallback = parse_unified_diff([
            "--- a/fallback.py",
            "+++ b/fallback.py",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ])
        self.assertEqual(fallback["total_files"], 1)
        self.assertEqual(fallback["files"][0]["path"], "fallback.py")
        self.assertEqual(fallback["total_additions"], 1)
        self.assertEqual(fallback["total_deletions"], 1)

        statuses = parse_unified_diff([
            "diff --git a/added.py b/added.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/added.py",
            "@@ -0,0 +1 @@",
            "+added",
            "diff --git a/deleted.py b/deleted.py",
            "deleted file mode 100644",
            "--- a/deleted.py",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-deleted",
            "diff --git a/old.py b/new.py",
            "similarity index 90%",
            "rename from old.py",
            "rename to new.py",
        ])
        self.assertEqual(
            [item["status"] for item in statuses["files"]],
            ["added", "deleted", "renamed"],
        )
        self.assertEqual(statuses["files"][0]["path"], "added.py")
        self.assertEqual(statuses["files"][1]["path"], "deleted.py")

    def test_parse_unified_diff_counts_header_like_hunk_content(self):
        parsed = parse_unified_diff([
            "diff --git a/loop.c b/loop.c",
            "--- a/loop.c",
            "+++ b/loop.c",
            "@@ -1 +1 @@",
            "---i;",
            "+++i;",
            "diff --git a/other.c b/other.c",
            "--- a/other.c",
            "+++ b/other.c",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ])

        self.assertEqual(parsed["total_additions"], 2)
        self.assertEqual(parsed["total_deletions"], 2)
        self.assertEqual(parsed["files"][0]["additions"], 1)
        self.assertEqual(parsed["files"][0]["deletions"], 1)

    def test_parse_unified_diff_decodes_git_quoted_paths(self):
        parsed = parse_unified_diff([
            'diff --git "a/src/caf\\303\\251 notes.txt" "b/src/caf\\303\\251 notes.txt"',
            '--- "a/src/caf\\303\\251 notes.txt"',
            '+++ "b/src/caf\\303\\251 notes.txt"',
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ])

        self.assertEqual(parsed["files"][0]["path"], "src/café notes.txt")
        self.assertEqual(parsed["files"][0]["old_path"], "src/café notes.txt")
        self.assertEqual(parsed["files"][0]["new_path"], "src/café notes.txt")

    def test_git_path_decoder_handles_malformed_and_unknown_escapes(self):
        self.assertEqual(_decode_git_path("trailing\\"), "trailing\\")
        self.assertEqual(_decode_git_path(r"unknown\q"), "unknownq")
        self.assertIsNone(_parse_git_diff_paths("diff --git a/only-one-path"))

    def test_content_type_hints_and_label_detection(self):
        self.assertEqual(detect_content_type(["error"], content_type_hint="log"), ("log", None))
        self.assertEqual(detect_content_type(["source"], content_type_hint="text"), ("text", None))
        self.assertEqual(detect_content_type(["not a patch"], label="git diff command")[0], "diff")
        self.assertEqual(detect_content_type(["build complete"], label="nightly build")[0], "log")
        self.assertEqual(detect_content_type(["plain source"])[0], "text")

    def test_signal_detection_covers_success_timeout_and_non_log_paths(self):
        self.assertEqual(
            detect_signals(["2 tests passed", "OK"], "log"),
            ({}, "None (successful test run)"),
        )
        self.assertEqual(
            detect_signals(["2 tests passed", "WARNING: slow test", "OK"], "log"),
            ({"warning": 1}, "warning: 1"),
        )
        self.assertEqual(
            detect_signals(["2 tests passed", "2 warnings", "OK"], "log"),
            ({"warning": 1}, "warning: 1"),
        )
        self.assertEqual(
            detect_signals(["2 tests passed", "0 warnings", "OK"], "log"),
            ({}, "None (successful test run)"),
        )
        timeout_signals, timeout_summary = detect_signals(
            ["command stopped"], "log", command_exit_code=0, timed_out=True
        )
        self.assertEqual(timeout_signals, {})
        self.assertEqual(timeout_summary, "None detected")
        self.assertEqual(
            detect_signals(["error in source"], "text"),
            ({}, "None (non-log content)"),
        )

        signals, summary = detect_signals(
            ["0 errors", "ERROR: command crashed"], "log"
        )
        self.assertEqual(signals, {"error": 1})
        self.assertEqual(summary, "error: 1")

        mixed_signals, mixed_summary = detect_signals(
            ["FAILED: 1 failure, 0 errors"], "log"
        )
        self.assertEqual(mixed_signals, {"failure": 1})
        self.assertEqual(mixed_summary, "failure: 1")

    def test_token_estimates_are_deterministic_and_null_safe(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)
        self.assertEqual(estimate_tokens_from_bytes(None), None)
        self.assertEqual(estimate_tokens_from_bytes(0), 0)
        self.assertEqual(estimate_tokens_from_bytes(9), 3)

    def test_storage_cleanup_ignores_captures_without_storage(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("no storage payload", label="no-storage")
        capture.fts_conn = None

        engine._close_capture_storage(capture)
        self.assertIsNone(capture.fts_conn)

    def test_storage_cleanup_logs_and_swallows_close_failure(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("cleanup failure payload", label="cleanup-failure")

        class BrokenStorage:
            def close(self):
                raise RuntimeError("close failed")

        capture.fts_conn = BrokenStorage()
        with self.assertLogs("ephemeral_buffer.engine", level="ERROR") as events:
            engine._close_capture_storage(capture)

        self.assertTrue(any("capture_storage_cleanup_failed" in event for event in events.output))


class TestEphemeralEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Initializing EphemeralEngine for testing ---")
        cls.engine = EphemeralEngine(max_captures=3)

    def test_01_ingest_and_bm25_search(self):
        sample_log = """
2026-08-26 09:00:01 [INFO] Server starting up on port 8080...
2026-08-26 09:00:02 [INFO] Initializing authentication middleware
2026-08-26 09:00:03 [DEBUG] Connecting to redis cache at 127.0.0.1:6379
2026-08-26 09:00:04 [INFO] Ready to receive traffic
2026-08-26 09:05:12 [WARN] High latency detected in worker-thread-4
2026-08-26 09:05:13 [ERROR] Unhandled exception occurred:
java.lang.NullPointerException: Cannot invoke "UserSession.getRoles()" because "session" is null
    at com.example.auth.SecurityFilter.doFilter(SecurityFilter.java:142)
    at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:189)
2026-08-26 09:05:14 [INFO] Request completed with status HTTP 500
2026-08-26 09:05:15 [INFO] Worker healthcheck OK
"""
        cap = self.engine.ingest(sample_log.strip(), label="auth-service-log")
        self.assertEqual(cap.label, "auth-service-log")
        self.assertGreater(cap.line_count, 0)

        # 1. Exact BM25 keyword query
        res_bm25 = self.engine.search("NullPointerException UserSession", mode="bm25")
        self.assertEqual(res_bm25["status"], "ok")
        self.assertGreater(len(res_bm25["matches"]), 0)
        best_match = res_bm25["matches"][0]
        self.assertIn("NullPointerException", best_match["snippet"])
        self.assertIn("SecurityFilter.java:142", best_match["snippet"])
        print("\n[BM25 Test Passed] Matched exact exception successfully:")
        print(best_match["snippet"])

    def test_02_semantic_search(self):
        sample_log = """
[10:00:00] Ingesting telemetry data batch #440
[10:00:01] Worker pool health check: 8/8 nodes healthy
[10:00:02] Initiating replication sync across cluster nodes
[10:00:03] Node us-west-2b reported: IO stream unexpectedly terminated; remote host closed TCP connection
[10:00:04] Fallback to secondary read replica initiated
[10:00:05] Resuming telemetry pipeline batch #441
"""
        cap = self.engine.ingest(sample_log.strip(), label="cluster-telemetry")

        # Query uses semantic phrasing without using the literal words "TCP connection" or "unexpectedly terminated"
        res_sem = self.engine.search("where did the network disconnect?", mode="semantic")
        self.assertEqual(res_sem["status"], "ok")
        self.assertGreater(len(res_sem["matches"]), 0)
        best_match = res_sem["matches"][0]
        self.assertIn("remote host closed TCP connection", best_match["snippet"])
        print("\n[Semantic Test Passed] Semantic query matched conceptual meaning:")
        print(best_match["snippet"])

    def test_03_hybrid_search(self):
        sample_log = """
STEP 1: Compiling typescript sources...
[TS] src/index.ts: Compilation succeeded.
STEP 2: Running integration tests...
[TEST] TestSuite 'PaymentGateway' started.
[TEST] Testing Stripe webhook handler...
[FAIL] PaymentGateway: Expected status 200 but received 402 Payment Required.
       Details: The customer's credit card was declined by the cardholder bank.
STEP 3: Summary
2 tests passed, 1 test failed.
"""
        cap = self.engine.ingest(sample_log.strip(), label="ci-test-run")

        res_hybrid = self.engine.search("why did payment test fail card declined?", mode="hybrid")
        self.assertEqual(res_hybrid["status"], "ok")
        self.assertGreater(len(res_hybrid["matches"]), 0)
        best = res_hybrid["matches"][0]
        self.assertIn("Payment Required", best["snippet"])
        print("\n[Hybrid Test Passed] Hybrid RRF successfully retrieved failing test context:")
        print(best["snippet"])

    def test_search_query_semantics_and_mode_validation(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest(
            "alpha beta\nECONNREFUSED: port=5432\nfoo.bar literal\n",
            label="query-semantics",
        )

        self.assertGreater(len(engine.search_bm25(capture, "foo.*")), 0)
        self.assertGreater(len(engine.search_bm25(capture, "foo.bar")), 0)
        self.assertEqual(engine.search_bm25(capture, "!!!"), [])

        invalid = engine.search("alpha", mode="regex")
        self.assertEqual(invalid["status"], "error")
        self.assertIn("Unsupported search mode", invalid["message"])

    def test_hybrid_prioritizes_lexical_matches(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("noise line\n" * 8, label="hybrid-ranking")

        with patch.object(engine, "search_bm25", return_value=[(1, 0.1)]), \
                patch.object(engine, "search_semantic", return_value=[(0, 0.9)]):
            result = engine.search("exact error", mode="hybrid", capture_id=capture.capture_id, top_k=1)

        self.assertEqual(result["matches"][0]["chunk_id"], 1)

    def test_search_context_is_exact_and_bounded(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("first\nMATCH alpha\nthird\nfourth\nfifth", label="context-boundaries")

        result = engine.search("MATCH", mode="bm25", capture_id=capture.capture_id, top_k=1, context_lines=0)
        match = result["matches"][0]
        self.assertEqual(match["context_start_line"], 1)
        self.assertEqual(match["context_end_line"], 4)
        self.assertEqual(match["context"], "first\nMATCH alpha\nthird\nfourth")

        edge = engine.search("fifth", mode="bm25", capture_id=capture.capture_id, top_k=1, context_lines=1)
        self.assertEqual(edge["matches"][0]["context_start_line"], 2)
        self.assertEqual(edge["matches"][0]["context_end_line"], 5)
        self.assertEqual(edge["matches"][0]["context"], "MATCH alpha\nthird\nfourth\nfifth")

        self.assertEqual(engine.search("MATCH", context_lines=-1)["status"], "error")
        self.assertEqual(engine.search("MATCH", top_k=0)["status"], "error")

    def test_search_deduplicates_overlapping_contexts_before_top_k(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("\n".join(f"line {i}" for i in range(1, 11)), label="overlap-ranking")

        with patch.object(
            engine,
            "search_bm25",
            return_value=[(0, 1.0), (1, 0.9), (3, 0.8)],
        ):
            result = engine.search(
                "line",
                mode="bm25",
                capture_id=capture.capture_id,
                top_k=2,
                context_lines=0,
            )

        self.assertEqual([match["chunk_id"] for match in result["matches"]], [0, 3])

    def test_search_dedup_keeps_matches_that_reveal_new_lines(self):
        engine = EphemeralEngine(max_captures=1, semantic_chunk_lines=8)
        capture = engine.ingest("\n".join(f"line {i}" for i in range(1, 17)), label="dedup-visibility")
        # Semantic windows L1-8 and L9-16: contexts overlap, but the second reveals lines 12-16.
        with patch.object(engine, "search_semantic", return_value=[(0, 0.9), (1, 0.8)]):
            result = engine.search("line", mode="semantic", capture_id=capture.capture_id, top_k=5, context_lines=3)
        self.assertEqual([m["matched_range"] for m in result["matches"]], ["L1-L8", "L9-L16"])

        # Lexical windows L9-12, L13-16, L7-10: L7-10 overlaps the first match's lines, and
        # L13-16 is dropped only when the first match's context already shows all of it.
        with patch.object(engine, "search_bm25", return_value=[(4, 1.0), (6, 0.9), (3, 0.8)]):
            wide = engine.search("line", mode="bm25", capture_id=capture.capture_id, top_k=5, context_lines=4)
            narrow = engine.search("line", mode="bm25", capture_id=capture.capture_id, top_k=5, context_lines=1)
        self.assertEqual([m["matched_range"] for m in wide["matches"]], ["L9-L12"])
        self.assertEqual([m["matched_range"] for m in narrow["matches"]], ["L9-L12", "L13-L16"])

    def test_search_snippet_bounds_long_utf8_lines(self):
        engine = EphemeralEngine(max_captures=1)
        long_line = "MATCH " + "é" * (SEARCH_SNIPPET_MAX_BYTES * 2)
        capture = engine.ingest(long_line, label="long-search-line")

        result = engine.search("MATCH", mode="bm25", capture_id=capture.capture_id, top_k=1)
        snippet = result["matches"][0]["snippet"]

        self.assertLessEqual(len(snippet.encode("utf-8")), SEARCH_SNIPPET_MAX_BYTES)
        self.assertIn("search line truncated", snippet)

    def test_search_reader_defers_storage_close_during_eviction(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("target value\nsecond line", label="reader-lifetime")
        self.assertIs(engine.get_capture(), capture)
        entered = threading.Event()
        proceed = threading.Event()
        original_search = engine._search_capture

        def blocked_search(*args, **kwargs):
            entered.set()
            self.assertTrue(proceed.wait(timeout=2))
            return original_search(*args, **kwargs)

        with patch.object(engine, "_search_capture", side_effect=blocked_search):
            result_holder = {}
            search_thread = threading.Thread(
                target=lambda: result_holder.setdefault(
                    "result", engine.search("target", mode="bm25", capture_id=capture.capture_id)
                )
            )
            search_thread.start()
            self.assertTrue(entered.wait(timeout=2))

            engine.ingest("replacement capture", label="replacement")
            self.assertNotIn(capture.capture_id, engine.captures)
            self.assertTrue(capture.storage_close_pending)
            self.assertIsNotNone(capture.fts_conn)

            proceed.set()
            search_thread.join(timeout=2)

        self.assertFalse(search_thread.is_alive())
        self.assertEqual(result_holder["result"]["status"], "ok")
        self.assertEqual(result_holder["result"]["matches"][0]["context"], "target value\nsecond line")
        self.assertIsNone(capture.fts_conn)

    def test_lazy_semantic_search_survives_capture_eviction(self):
        class BlockingEmbedding:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def embed(self, texts):
                self.started.set()
                self.release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=1)
        embedding = BlockingEmbedding()
        engine.embedding_model = embedding
        capture = engine.ingest("semantic payload", label="lazy-eviction")
        result_holder = {}
        search_thread = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result",
                engine.search("payload", mode="semantic", capture_id=capture.capture_id),
            )
        )
        search_thread.start()
        self.assertTrue(embedding.started.wait(timeout=2))

        engine.ingest("replacement", label="evicts-lazy-search")
        self.assertNotIn(capture.capture_id, engine.captures)

        embedding.release.set()
        search_thread.join(timeout=2)
        self.assertFalse(search_thread.is_alive())
        self.assertEqual(result_holder["result"]["status"], "ok")
        self.assertEqual(result_holder["result"]["match_count"], 1)

    def test_summary_for_capture_survives_lru_eviction(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("first capture", label="first")

        engine.ingest("replacement capture", label="replacement")

        self.assertEqual(engine.get_summary(capture.capture_id)["status"], "error")
        summary = engine.get_summary_for_capture(capture, include_previews=False)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["capture_id"], capture.capture_id)
        self.assertEqual(summary["label"], "first")

    def test_04_slice_and_summary(self):
        lines = [f"Log line number {i}" for i in range(1, 101)]
        lines[49] = "FATAL: System ran out of file descriptors"
        cap = self.engine.ingest("\n".join(lines), label="100-lines-log")

        # Test summary
        summary = self.engine.get_summary(cap.capture_id)
        self.assertEqual(summary["total_lines"], 100)
        self.assertIn("error", summary["keyword_signals"])
        self.assertEqual(summary["keyword_signals"]["error"], 1)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["execution_status"], "captured")
        self.assertEqual(summary["partial"], False)
        self.assertIsInstance(summary["duration_ms"], float)
        self.assertGreater(summary["estimated_tokens"], 0)
        self.assertEqual(summary["errors"], [{"type": "error", "count": 1}])
        compact = self.engine.get_summary(cap.capture_id, include_previews=False)
        self.assertNotIn("head_preview", compact)
        self.assertNotIn("tail_preview", compact)

        # Test slice around line 50
        slice_res = self.engine.get_slice(48, 52, capture_id=cap.capture_id)
        self.assertEqual(slice_res["status"], "ok")
        self.assertIn("FATAL: System ran out of file descriptors", slice_res["content"])
        print("\n[Slice & Summary Test Passed]:")
        print(slice_res["content"])

    def test_04a_previews_are_utf8_bounded_without_changing_retained_content(self):
        long_line = "界" * 10_000
        cap = self.engine.ingest(long_line, label="long-unicode-line")

        summary = self.engine.get_summary(cap.capture_id)
        self.assertLessEqual(len(summary["head_preview"].encode("utf-8")), PREVIEW_MAX_BYTES)
        self.assertLessEqual(len(summary["tail_preview"].encode("utf-8")), PREVIEW_MAX_BYTES)
        self.assertIn("preview truncated", summary["head_preview"])
        self.assertIn("preview truncated", summary["tail_preview"])
        self.assertEqual(cap.raw_lines, [long_line])
        self.assertEqual(
            self.engine.get_slice(1, 1, capture_id=cap.capture_id)["content"],
            f"      1 | {long_line}",
        )

    def test_ingest_validates_and_preserves_structured_metrics(self):
        engine = EphemeralEngine(max_captures=2)
        capture = engine.ingest(
            "metric output",
            source="command",
            duration_ms=12.5,
            structured_metrics={"tests": {"passed": 2}},
        )
        summary = engine.get_summary(capture.capture_id)
        self.assertEqual(summary["source"], "command")
        self.assertEqual(summary["duration_ms"], 12.5)
        self.assertEqual(summary["structured_metrics"], {"tests": {"passed": 2}})

        with self.assertRaisesRegex(ValueError, "JSON object"):
            engine.ingest("invalid", structured_metrics=[])
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            engine.ingest("invalid", structured_metrics={"value": float("nan")})
        with self.assertRaisesRegex(ValueError, "exceeds"):
            engine.ingest("invalid", structured_metrics={"value": "x" * 17_000})
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            engine.ingest("invalid", duration_ms="12")
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            engine.ingest("invalid", duration_ms=-1)
        with self.assertRaisesRegex(ValueError, "source"):
            engine.ingest("invalid", source="")

    def test_ingest_preserves_positional_protected_capture_compatibility(self):
        engine = EphemeralEngine(max_captures=2)
        source = engine.ingest("source capture")
        combined = engine.ingest(
            "combined capture", "combined", "text", False, None, None, False,
            [source.capture_id],
        )
        self.assertEqual(combined.label, "combined")
        self.assertIn(source.capture_id, engine.captures)

    def test_05_lru_buffer_eviction(self):
        # max_captures is 3 for this focused eviction test
        self.engine.clear("all")
        self.assertEqual(len(self.engine.captures), 0)

        initial_captures = []
        for i in range(1, 4):
            initial_captures.append(self.engine.ingest(f"Content for run {i}\nDone {i}", label=f"Run {i}"))

        # Refresh Run 1 so Run 2 becomes the least recently used capture.
        self.engine.get_summary(initial_captures[0].capture_id)
        with self.assertLogs("ephemeral_buffer.engine", level="INFO") as events:
            self.engine.ingest("Content for run 4\nDone 4", label="Run 4")
            self.engine.ingest("Content for run 5\nDone 5", label="Run 5")

        active_caps = self.engine.list_captures()
        self.assertEqual(len(active_caps), 3)
        # Run 2 and then Run 3 should be evicted; recently used Run 1 remains.
        labels = [c["label"] for c in active_caps]
        self.assertIn("Run 5", labels)
        self.assertIn("Run 4", labels)
        self.assertIn("Run 1", labels)
        self.assertNotIn("Run 2", labels)
        self.assertNotIn("Run 3", labels)
        self.assertGreaterEqual(
            sum("capture_evicted" in event for event in events.output),
            2,
        )
        print("\n[LRU Eviction Passed] Least recently used captures evicted, memory bound maintained.")

    def test_06_non_log_text_does_not_emit_keyword_signals(self):
        source_text = """
        # Handle an error response and report a failure to the caller.
        def parse_result(value):
            return value
        """
        cap = self.engine.ingest(source_text.strip(), label="README and source excerpt")

        summary = self.engine.get_summary(cap.capture_id)
        self.assertEqual(cap.content_type, "text")
        self.assertEqual(summary["keyword_signals"], {})
        self.assertEqual(summary["signals_summary"], "None (non-log content)")

    def test_07_diff_parsing_and_file_mapping(self):
        diff_output = """diff --git a/tool-kernel/src/main.c b/tool-kernel/src/main.c
index 1234567..89abcdef 100644
--- a/tool-kernel/src/main.c
+++ b/tool-kernel/src/main.c
@@ -10,6 +10,8 @@ int main() {
     if (err != 0) {
+        char *msg = os.strerror(err);
+        except Exception as e:
         return -1;
     }
 }
diff --git a/tests/test_kernel.py b/tests/test_kernel.py
new file mode 100644
--- /dev/null
+++ b/tests/test_kernel.py
@@ -0,0 +1,5 @@
+def test_something():
+    # test error handling routines
+    assert True
"""
        cap = self.engine.ingest(diff_output.strip(), label="gh pr diff 68")
        self.assertEqual(cap.content_type, "diff")
        self.assertIsNotNone(cap.diff_meta)
        self.assertEqual(cap.diff_meta["total_files"], 2)
        self.assertEqual(cap.diff_meta["total_additions"], 5)
        self.assertEqual(cap.diff_meta["total_deletions"], 0)

        summary = self.engine.get_summary(cap.capture_id)
        # Verify false-positive signals on code keywords (error, exception, strerror) are suppressed!
        self.assertEqual(summary["signals_summary"], "None (Clean patch)")
        self.assertEqual(len(summary["keyword_signals"]), 0)
        self.assertIn("tool-kernel/src/main.c", summary["file_map"])
        self.assertIn("tests/test_kernel.py [ADDED]", summary["file_map"])

        # Verify buffer line mapping for second file
        f2 = cap.diff_meta["files"][1]
        slice_res = self.engine.get_slice(f2["start_line"], f2["end_line"], capture_id=cap.capture_id)
        self.assertIn("tests/test_kernel.py", slice_res["content"])
        self.assertIn("+def test_something():", slice_res["content"])
        print("\n[Diff Parsing & Signal Suppression Passed]:")
        print(summary["file_map"])

    def test_08_diff_conflict_detection(self):
        conflict_diff = """diff --git a/src/config.py b/src/config.py
--- a/src/config.py
+++ b/src/config.py
@@ -1,3 +1,7 @@
<<<<<<< HEAD
 PORT = 8080
=======
 PORT = 9090
>>>>>>> main
"""
        cap = self.engine.ingest(conflict_diff.strip(), label="git diff with conflicts")
        self.assertEqual(cap.content_type, "diff")
        self.assertTrue(cap.diff_meta["has_conflicts"])
        summary = self.engine.get_summary(cap.capture_id)
        self.assertIn("Conflict markers detected", summary["signals_summary"])
        print("\n[Diff Conflict Detection Passed] Correctly flagged merge conflict markers.")

    def test_diff_conflict_detection_interprets_hunk_prefixes(self):
        added_marker_diff = """diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1,3 +1,7 @@
+<<<<<<< HEAD
 PORT = 8080
+=======
 PORT = 9090
+>>>>>>> main
"""
        removed_marker_diff = """diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1,7 +1,3 @@
-<<<<<<< HEAD
-PORT = 8080
-=======
-PORT = 9090
->>>>>>> main
 PORT = 8080
"""

        added = self.engine.ingest(added_marker_diff.strip(), label="added-conflicts")
        removed = self.engine.ingest(removed_marker_diff.strip(), label="removed-conflicts")

        self.assertTrue(added.diff_meta["has_conflicts"])
        self.assertFalse(removed.diff_meta["has_conflicts"])

    def test_08_log_signal_filtering_and_benign_suppression(self):
        # 1. Clean run with benign zeros
        clean_log = """
============================= test session starts ==============================
collected 25 items
tests/test_api.py .........................                              [100%]
============================== 25 passed in 0.42s ==============================
passed: 25, failed: 0, errors: 0
no errors encountered.
"""
        cap_clean = self.engine.ingest(
            clean_log.strip(), label="pytest-clean-run", command_exit_code=0
        )
        summary_clean = self.engine.get_summary(cap_clean.capture_id)
        self.assertEqual(summary_clean["signals_summary"], "None (successful test run)")
        self.assertEqual(len(summary_clean["keyword_signals"]), 0)

        # 2. Failing run with actual errors
        failing_log = """
============================= test session starts ==============================
collected 25 items
tests/test_api.py ...........F.............                              [100%]
=================================== FAILURES ===================================
_________________________________ test_timeout _________________________________
E   ConnectionError: ERROR: Connection timed out after 10000ms
=========================== 1 failed, 24 passed in 1.12s ===========================
"""
        cap_fail = self.engine.ingest(failing_log.strip(), label="pytest-failing-run")
        summary_fail = self.engine.get_summary(cap_fail.capture_id)
        self.assertIn("failure", summary_fail["keyword_signals"])
        self.assertIn("error", summary_fail["keyword_signals"])
        self.assertIn("timeout", summary_fail["keyword_signals"])
        print("\n[Log Signal Scanner Passed] Benign zeros ignored, real errors/failures captured:")
        print(summary_fail["signals_summary"])

    def test_09_thread_safe_ingest_and_reads(self):
        self.engine.clear("all")

        def ingest_capture(index):
            return self.engine.ingest(f"Concurrent capture {index}\nDone", label=f"thread-{index}").capture_id

        with ThreadPoolExecutor(max_workers=8) as executor:
            capture_ids = list(executor.map(ingest_capture, range(12)))

        self.assertEqual(len(capture_ids), len(set(capture_ids)))
        self.assertEqual(len(self.engine.list_captures()), 3)

        latest_id = self.engine.list_captures()[0]["capture_id"]
        with ThreadPoolExecutor(max_workers=8) as executor:
            statuses = list(executor.map(
                lambda _: self.engine.get_summary(latest_id)["status"],
                range(16)
            ))
        self.assertEqual(statuses, ["ok"] * 16)
        print("\n[Thread Safety Passed] Concurrent ingest and reads remained consistent.")

    def test_10_byte_budget_and_buffer_stats(self):
        self.engine.clear("all")
        original_limit = self.engine.max_buffer_bytes
        self.engine.max_buffer_bytes = 50
        try:
            first = self.engine.ingest("a" * 20, label="a")
            second = self.engine.ingest("b" * 20, label="b")
            third = self.engine.ingest("c" * 20, label="c")

            stats = self.engine.get_buffer_stats()
            self.assertEqual(stats["capture_count"], 2)
            self.assertEqual(stats["total_bytes"], second.retained_byte_size + third.retained_byte_size)
            self.assertEqual(stats["max_buffer_bytes"], 50)
            self.assertNotIn(first.capture_id, self.engine.captures)

            with self.assertRaises(ValueError):
                self.engine.ingest("x" * 100, label="oversized")
        finally:
            self.engine.max_buffer_bytes = original_limit
            self.engine.clear("all")
        print("\n[Byte Budget Passed] Content byte budget evicted old captures and rejected oversized input.")

    def test_byte_budget_uses_actual_utf8_input_bytes(self):
        engine = EphemeralEngine(max_captures=3, max_buffer_bytes=7)
        try:
            no_trailing_newline = engine.ingest("abc", label="a")
            multibyte = engine.ingest("é", label="b")

            self.assertEqual(no_trailing_newline.byte_size, len("abc".encode("utf-8")))
            self.assertEqual(multibyte.byte_size, len("é".encode("utf-8")))
            self.assertEqual(engine.get_buffer_stats()["total_bytes"], 7)
            self.assertEqual(len(engine.captures), 2)

            with self.assertRaises(ValueError):
                engine.ingest("ééé", label="over")
        finally:
            engine.shutdown()

    def test_byte_budget_includes_utf8_label_bytes(self):
        engine = EphemeralEngine(max_captures=2, max_buffer_bytes=10)
        try:
            capture = engine.ingest("1234", label="ééé")
            self.assertEqual(capture.byte_size, 4)
            self.assertEqual(capture.label_byte_size, 6)
            self.assertEqual(capture.retained_byte_size, 10)
            self.assertEqual(engine.get_buffer_stats()["total_bytes"], 10)
            with self.assertRaisesRegex(ValueError, "content and label use 11 bytes"):
                engine.ingest("1234", label="éééx")
        finally:
            engine.clear("all")

    def test_11_buffer_stats_separate_accounted_and_process_memory(self):
        self.engine.ingest("stats payload", label="stats")

        stats = self.engine.get_buffer_stats()

        self.assertEqual(
            stats["accounted_bytes"],
            stats["total_bytes"] + stats["embedding_bytes"],
        )
        self.assertIn("process_rss_bytes", stats)
        self.assertIn("unaccounted_rss_bytes", stats)
        if stats["process_rss_bytes"] is not None:
            self.assertGreater(stats["process_rss_bytes"], 0)
            self.assertGreaterEqual(stats["unaccounted_rss_bytes"], 0)

    def test_index_chunk_budget_evicts_lru_and_rejects_oversized_capture(self):
        engine = EphemeralEngine(max_captures=3, max_indexed_chunks=3)
        first = engine.ingest("one\ntwo\nthree\nfour\nfive\nsix\nseven", label="first")
        second = engine.ingest("eight\nnine\nten\neleven\ntwelve\nthirteen\nfourteen", label="second")

        self.assertNotIn(first.capture_id, engine.captures)
        self.assertIn(second.capture_id, engine.captures)
        stats = engine.get_buffer_stats()
        self.assertEqual(stats["indexed_chunks"], 3)
        self.assertEqual(stats["remaining_indexed_chunks"], 0)

        rejecting = EphemeralEngine(max_captures=1, max_indexed_chunks=2)
        with self.assertRaisesRegex(ValueError, "indexed chunks"):
            rejecting.ingest("one\ntwo\nthree\nfour\nfive\nsix\nseven", label="too-large")
        self.assertEqual(rejecting.captures, {})

    def test_runtime_index_budget_decrease_evicts_oldest_captures(self):
        engine = EphemeralEngine(max_captures=3, max_indexed_chunks=6)
        first = engine.ingest("one\ntwo\nthree\nfour\nfive\nsix\nseven", label="first")
        second = engine.ingest("eight\nnine\nten\neleven\ntwelve\nthirteen\nfourteen", label="second")
        third = engine.ingest("fifteen\nsixteen\nseventeen\neighteen\nnineteen\ntwenty\ntwenty-one", label="third")

        result = engine.set_max_indexed_chunks(3)

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["effective"], 3)
        self.assertEqual(result["evicted_captures"], 1)
        self.assertNotIn(first.capture_id, engine.captures)
        self.assertNotIn(second.capture_id, engine.captures)
        self.assertIn(third.capture_id, engine.captures)
        self.assertEqual(engine.get_buffer_stats()["indexed_chunks"], 3)

    def test_runtime_index_budget_rejects_invalid_values(self):
        engine = EphemeralEngine(max_indexed_chunks=3)
        for value in (0, -1, True, "3"):
            with self.assertRaises(ValueError):
                engine.set_max_indexed_chunks(value)

    def test_eviction_of_missing_capture_is_reported_without_mutation(self):
        engine = EphemeralEngine(max_indexed_chunks=3, embedding_warmup=False)
        with engine._lock:
            self.assertFalse(engine._evict_capture_locked("missing"))

    def test_runtime_index_budget_adjustment_serializes_with_ingest(self):
        engine = EphemeralEngine(max_indexed_chunks=6, embedding_warmup=False)
        barrier = threading.Barrier(2)

        def ingest():
            barrier.wait()
            return engine.ingest("one\ntwo\nthree\nfour\nfive\nsix\nseven")

        def adjust():
            barrier.wait()
            return engine.set_max_indexed_chunks(3)

        with ThreadPoolExecutor(max_workers=2) as pool:
            ingest_future = pool.submit(ingest)
            adjust_future = pool.submit(adjust)
            ingest_future.result()
            adjust_future.result()
        engine.set_max_indexed_chunks(3)
        self.assertLessEqual(engine.get_buffer_stats()["indexed_chunks"], 3)

    def test_protected_ingest_evicts_only_unrelated_captures(self):
        engine = EphemeralEngine(max_captures=2)
        source = engine.ingest("source payload", label="source")
        unrelated = engine.ingest("unrelated payload", label="unrelated")

        admitted = engine.ingest(
            "consolidated payload",
            label="consolidated",
            protected_capture_ids=[source.capture_id],
        )

        self.assertIn(source.capture_id, engine.captures)
        self.assertNotIn(unrelated.capture_id, engine.captures)
        self.assertIn(admitted.capture_id, engine.captures)

    def test_protected_ingest_rejects_without_evicting_sources(self):
        engine = EphemeralEngine(max_captures=1)
        source = engine.ingest("source payload", label="source")

        with self.assertRaisesRegex(ValueError, "protected source captures"):
            engine.ingest(
                "consolidated payload",
                label="consolidated",
                protected_capture_ids=[source.capture_id],
            )

        self.assertEqual(list(engine.captures), [source.capture_id])
        self.assertEqual(engine.get_capture(source.capture_id).label, "source")

    def test_protected_ingest_rejects_missing_source_ids(self):
        engine = EphemeralEngine(max_captures=1)

        with self.assertRaisesRegex(ValueError, "unavailable source captures"):
            engine.ingest(
                "consolidated payload",
                label="consolidated",
                protected_capture_ids=["cap-missing"],
            )

        self.assertEqual(engine.captures, {})

    def test_protected_ingest_rejects_when_bytes_or_chunks_cannot_fit(self):
        byte_limited = EphemeralEngine(max_captures=2, max_buffer_bytes=25)
        byte_source = byte_limited.ingest("source payload", label="source")
        with self.assertRaisesRegex(ValueError, "protected source captures"):
            byte_limited.ingest(
                "consolidated payload",
                label="c",
                protected_capture_ids=[byte_source.capture_id],
            )
        self.assertIn(byte_source.capture_id, byte_limited.captures)

        chunk_limited = EphemeralEngine(max_captures=2, max_indexed_chunks=1)
        chunk_source = chunk_limited.ingest("source payload", label="source")
        with self.assertRaisesRegex(ValueError, "protected source captures"):
            chunk_limited.ingest(
                "consolidated payload",
                label="consolidated",
                protected_capture_ids=[chunk_source.capture_id],
            )
        self.assertIn(chunk_source.capture_id, chunk_limited.captures)

    def test_12_reads_are_not_blocked_by_embedding(self):
        class BlockingEmbedding:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def embed(self, texts):
                self.started.set()
                self.release.wait(timeout=2)
                return [[0.0] * 384 for _ in texts]

        original_model = self.engine.embedding_model
        blocker = BlockingEmbedding()
        self.engine.embedding_model = blocker
        capture = self.engine.ingest("blocked embedding")
        worker = threading.Thread(target=self.engine.search_semantic, args=(capture, "blocked"))
        try:
            worker.start()
            self.assertTrue(blocker.started.wait(timeout=1))
            started_at = time.monotonic()
            stats = self.engine.get_buffer_stats()
            self.assertLess(time.monotonic() - started_at, 0.5)
            self.assertIn("capture_count", stats)
            started_at = time.monotonic()
            concurrent_capture = self.engine.ingest("concurrent capture")
            self.assertLess(time.monotonic() - started_at, 0.5)
            self.assertIn(concurrent_capture.capture_id, self.engine.captures)
        finally:
            blocker.release.set()
            worker.join(timeout=2)
            self.engine.embedding_model = original_model

    def test_13_rejects_invalid_buffer_limits(self):
        with self.assertRaisesRegex(ValueError, "max_captures"):
            EphemeralEngine(max_captures=0)
        with self.assertRaisesRegex(ValueError, "max_buffer_bytes"):
            EphemeralEngine(max_buffer_bytes=0)
        with self.assertRaisesRegex(ValueError, "max_indexed_chunks"):
            EphemeralEngine(max_indexed_chunks=0)

    def test_ingest_rejects_invalid_original_byte_sizes_before_admission(self):
        engine = EphemeralEngine(max_captures=1)
        try:
            for invalid in (True, "12", -1):
                with self.subTest(original_byte_size=invalid):
                    with self.assertRaisesRegex(ValueError, "original_byte_size"):
                        engine.ingest("payload", original_byte_size=invalid)
            self.assertEqual(engine.captures, {})
            self.assertEqual(engine._next_id, 1)

            decoded = "\ufffd" * 171
            capture = engine.ingest(
                decoded,
                truncated=True,
                original_byte_size=300,
            )
            self.assertEqual(capture.original_byte_size, 300)
            self.assertGreater(capture.byte_size, capture.original_byte_size)
        finally:
            engine.shutdown()

    def test_14_empty_and_missing_capture_paths(self):
        empty = EphemeralEngine(max_captures=1)
        capture = empty.ingest("", label="empty")

        self.assertEqual(empty.search("anything")["status"], "ok")
        self.assertEqual(empty.search_bm25(capture, "anything"), [])
        self.assertEqual(empty.search_bm25(capture, '"'), [])
        self.assertEqual(empty.search_semantic(capture, "anything"), [])
        self.assertEqual(empty.get_slice(1, 1, capture_id="missing")["status"], "error")
        self.assertEqual(empty.get_slice(2, 1, capture_id=capture.capture_id)["status"], "error")
        self.assertEqual(empty.get_summary("missing")["status"], "error")

    def test_15_embedding_failure_is_not_cached_as_ready(self):
        class FailingEmbedding:
            def embed(self, _texts):
                raise RuntimeError("model unavailable")

        engine = EphemeralEngine(max_captures=1)
        engine.embedding_model = FailingEmbedding()
        capture = engine.ingest("payload", label="embedding-failure")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            engine.search_semantic(capture, "payload")
        self.assertIn(capture.capture_id, engine.captures)

    def test_lazy_embedding_cache_and_empty_embedding_paths(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("searchable payload", label="lazy-embedding")
        capture.embeddings = np.empty((0, 384), dtype=np.float32)
        self.assertEqual(engine.search_semantic(capture, "payload"), [])

        capture.embeddings = np.ones((1, 384), dtype=np.float32)
        engine._ensure_embeddings(capture)

        class EmbeddingPublishedWhileWaiting:
            def __enter__(self):
                capture.embeddings = np.ones((1, 384), dtype=np.float32)
                return self

            def __exit__(self, *_args):
                return False

        capture.embeddings = None
        original_lock = engine._embedding_lock
        engine._embedding_lock = EmbeddingPublishedWhileWaiting()
        try:
            engine._ensure_embeddings(capture)
        finally:
            engine._embedding_lock = original_lock

    def test_embedding_snapshot_aborts_when_capture_is_not_current(self):
        engine = EphemeralEngine(max_captures=1)
        engine._ensure_embeddings(SimpleNamespace(capture_id="missing", embeddings=None, chunks=[]))
        capture = engine.ingest("evicted before embedding", label="evicted")

        class EvictOnEnter:
            def __enter__(self):
                engine.captures.pop(capture.capture_id)
                return self

            def __exit__(self, *_args):
                return False

        original_lock = engine._embedding_lock
        engine._embedding_lock = EvictOnEnter()
        try:
            engine._ensure_embeddings(capture)
        finally:
            engine._embedding_lock = original_lock

    def test_async_semantic_prefetch_materializes_once_and_search_waits(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=2, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = BlockingEmbedding()
        try:
            capture = engine.ingest("prefetch payload", label="prefetch")
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(capture.semantic_index_state, "pending")
            self.assertIsNone(capture.embeddings)
            self.assertIn(capture.capture_id, engine._prefetch_running)
            engine._schedule_semantic_prefetch(capture)
            self.assertEqual(engine._prefetch_queue, {})
            stats = engine.get_buffer_stats()
            self.assertEqual((stats["semantic_prefetch_queued"], stats["semantic_prefetch_running"]), (0, 1))

            release.set()
            self.assertTrue(engine.search_semantic(capture, "payload"))
            self.assertEqual(capture.semantic_index_state, "ready")
            self.assertIsNotNone(capture.embeddings)
            engine._schedule_semantic_prefetch(capture)
            self.assertEqual(engine._prefetch_queue, {})
        finally:
            release.set()
            engine.shutdown()

    def test_async_semantic_prefetch_failure_is_reported_and_lazy_search_retries(self):
        class FailingEmbedding:
            def embed(self, _texts):
                raise RuntimeError("prefetch unavailable")

        engine = EphemeralEngine(max_captures=1, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = FailingEmbedding()
        try:
            capture = engine.ingest("prefetch failure", label="prefetch-failure")
            deadline = time.time() + 2
            while capture.semantic_index_state == "pending" and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(capture.semantic_index_state, "failed")
            self.assertEqual(engine.wait_for_semantic_index(capture), "failed")
            self.assertEqual(capture.semantic_index_state, "failed")
            with self.assertRaisesRegex(RuntimeError, "prefetch unavailable"):
                engine.search_semantic(capture, "failure")
            # A failed on-demand retry surfaces as a lexical fallback in hybrid mode.
            result = engine.search("failure", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(result["semantic_coverage"], "unavailable")
            self.assertEqual(result["semantic_fallback"], "RuntimeError")
            with self.assertRaisesRegex(RuntimeError, "prefetch unavailable"):
                engine.search("failure", mode="semantic", capture_id=capture.capture_id)
        finally:
            engine.shutdown()

    def test_semantic_index_metrics_distinguish_on_demand_completion(self):
        from metrics import LocalMetrics

        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            embedding_model_name="test",
            metrics=metrics,
        )
        engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[1.0] + [0.0] * 383 for _ in texts]},
        )()
        try:
            capture = engine.ingest("on demand payload", label="private")
            result = engine.search("payload", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(result["semantic_coverage"], "complete")

            source = metrics.snapshot()["semantic_index"]["on_demand"]
            self.assertEqual(source["queued"], 1)
            self.assertEqual(source["completed"], 1)
            self.assertEqual(source["failed"], 0)
            self.assertEqual(source["indexed_chunks"], len(capture.semantic_chunks))
            self.assertEqual(source["queue_wait_ms"]["count"], 1)
            self.assertEqual(source["indexing_duration_ms"]["count"], 1)
        finally:
            engine.shutdown()

        submit_metrics = LocalMetrics(enabled=True)
        submit_engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            metrics=submit_metrics,
        )
        try:
            capture = submit_engine.ingest("submit failure", label="private")
            with patch.object(
                submit_engine._on_demand_executor,
                "submit",
                side_effect=RuntimeError("executor closed"),
            ):
                result = submit_engine.search(
                    "failure",
                    mode="hybrid",
                    capture_id=capture.capture_id,
                )
            self.assertEqual(result["semantic_coverage"], "unavailable")
            self.assertEqual(
                submit_metrics.snapshot()["semantic_index"]["on_demand"]["failed"],
                1,
            )
        finally:
            submit_engine.shutdown()

    def test_async_semantic_jobs_stay_in_initiating_task_window(self):
        from metrics import LocalMetrics

        metrics = LocalMetrics(enabled=True)
        baseline = metrics.snapshot(include_snapshot_token=True)
        started = threading.Event()
        release = threading.Event()
        parent_ready = threading.Event()
        capture_holder = []

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=True,
            semantic_prefetch_workers=1,
            metrics=metrics,
        )
        engine.embedding_model = BlockingEmbedding()

        def capture_task():
            with metrics.measure("capture_text"):
                capture_holder.append(engine.ingest("prefetch payload", label="private"))
                parent_ready.set()
                started.wait(timeout=2)
                release.wait(timeout=2)

        capture_thread = threading.Thread(target=capture_task)
        capture_thread.start()
        try:
            self.assertTrue(parent_ready.wait(timeout=2))
            self.assertTrue(started.wait(timeout=2))
            prefetch_delta = metrics.snapshot(
                since_snapshot=baseline["snapshot_token"],
                include_snapshot_token=True,
            )
            self.assertEqual(prefetch_delta["semantic_index"]["prefetch"]["queued"], 0)
            self.assertEqual(prefetch_delta["semantic_index"]["prefetch"]["completed"], 0)

            with engine._lock:
                prefetch_done = engine._prefetch_running[capture_holder[0].capture_id]
            release.set()
            self.assertTrue(prefetch_done.wait(timeout=2))
            capture_thread.join(timeout=2)
            self.assertFalse(capture_thread.is_alive())
            prefetch_following = metrics.snapshot(
                since_snapshot=prefetch_delta["snapshot_token"],
            )
            self.assertEqual(prefetch_following["semantic_index"]["prefetch"]["queued"], 1)
            self.assertEqual(prefetch_following["semantic_index"]["prefetch"]["completed"], 1)
        finally:
            release.set()
            capture_thread.join(timeout=2)
            engine.shutdown()

        on_demand_metrics = LocalMetrics(enabled=True)
        on_demand_baseline = on_demand_metrics.snapshot(include_snapshot_token=True)
        on_demand_started = threading.Event()
        on_demand_release = threading.Event()
        on_demand_parent_ready = threading.Event()

        class BlockingOnDemandEmbedding:
            def embed(self, texts):
                on_demand_started.set()
                on_demand_release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        on_demand_engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            metrics=on_demand_metrics,
        )
        on_demand_engine.embedding_model = BlockingOnDemandEmbedding()
        capture = on_demand_engine.ingest("on demand payload", label="private")

        def search_task():
            with on_demand_metrics.measure("search_capture"):
                on_demand_parent_ready.set()
                on_demand_engine.search(
                    "payload",
                    mode="semantic",
                    capture_id=capture.capture_id,
                )

        search_thread = threading.Thread(target=search_task)
        search_thread.start()
        try:
            self.assertTrue(on_demand_parent_ready.wait(timeout=2))
            self.assertTrue(on_demand_started.wait(timeout=2))
            on_demand_delta = on_demand_metrics.snapshot(
                since_snapshot=on_demand_baseline["snapshot_token"],
                include_snapshot_token=True,
            )
            self.assertEqual(on_demand_delta["semantic_index"]["on_demand"]["queued"], 0)
            self.assertEqual(on_demand_delta["semantic_index"]["on_demand"]["completed"], 0)

            on_demand_release.set()
            search_thread.join(timeout=2)
            self.assertFalse(search_thread.is_alive())
            on_demand_following = on_demand_metrics.snapshot(
                since_snapshot=on_demand_delta["snapshot_token"],
            )
            self.assertEqual(on_demand_following["semantic_index"]["on_demand"]["queued"], 1)
            self.assertEqual(on_demand_following["semantic_index"]["on_demand"]["completed"], 1)
        finally:
            on_demand_release.set()
            search_thread.join(timeout=2)
            on_demand_engine.shutdown()

    def test_semantic_metrics_snapshot_callback_failure_does_not_break_metrics(self):
        from metrics import LocalMetrics

        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            metrics=metrics,
        )

        callback_lock_owned = []

        def failing_callback():
            callback_lock_owned.append(engine._lock._is_owned())
            raise RuntimeError("snapshot unavailable")

        engine.set_metrics_snapshot_callback(failing_callback)
        try:
            with self.assertLogs("ephemeral_buffer.engine", level="WARNING") as logs:
                with engine._lock:
                    engine._record_semantic_job_metrics_locked("on_demand", "failed")
                engine._flush_metrics_snapshot()
            self.assertIn("metrics_snapshot_callback_failed", "\n".join(logs.output))
            self.assertEqual(callback_lock_owned, [False])
            self.assertEqual(metrics.snapshot()["semantic_index"]["on_demand"]["failed"], 1)
        finally:
            engine.shutdown()

    def test_semantic_metrics_snapshot_flush_without_callback_is_safe(self):
        engine = EphemeralEngine(max_captures=1, semantic_prefetch=False)
        try:
            with engine._lock:
                engine._metrics_snapshot_pending = True
            engine._flush_metrics_snapshot()
            self.assertFalse(engine._metrics_snapshot_pending)
        finally:
            engine.shutdown()

    def test_semantic_index_metrics_count_pending_and_fallback_searches(self):
        from metrics import LocalMetrics

        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            semantic_wait_seconds=0.01,
            metrics=metrics,
        )
        engine.embedding_model = BlockingEmbedding()
        try:
            capture = engine.ingest("pending payload", label="private")
            result = engine.search("payload", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(result["semantic_coverage"], "pending")
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(
                metrics.snapshot()["semantic_index"]["search"]["pending_hybrid_responses"],
                1,
            )
        finally:
            release.set()
            engine.shutdown()

        fallback_metrics = LocalMetrics(enabled=True)
        fallback_engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            metrics=fallback_metrics,
        )

        class FailingEmbedding:
            def embed(self, _texts):
                raise RuntimeError("private embedding failure")

        fallback_engine.embedding_model = FailingEmbedding()
        try:
            capture = fallback_engine.ingest("fallback payload", label="private")
            result = fallback_engine.search(
                "payload",
                mode="hybrid",
                capture_id=capture.capture_id,
            )
            self.assertEqual(result["semantic_coverage"], "unavailable")
            self.assertEqual(
                fallback_metrics.snapshot()["semantic_index"]["search"]["semantic_fallbacks"],
                1,
            )
            self.assertEqual(
                fallback_metrics.snapshot()["semantic_index"]["on_demand"]["failed"],
                1,
            )
        finally:
            fallback_engine.shutdown()

    def test_semantic_index_metrics_count_evicted_and_cleared_jobs(self):
        from metrics import LocalMetrics

        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        metrics = LocalMetrics(enabled=True)
        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=True,
            semantic_prefetch_workers=1,
            metrics=metrics,
        )
        engine.embedding_model = BlockingEmbedding()
        try:
            first = engine.ingest("first payload", label="first")
            self.assertTrue(started.wait(timeout=2))
            second = engine.ingest("second payload", label="second")
            self.assertIn(second.capture_id, engine._prefetch_queue)
            self.assertEqual(engine.clear(second.capture_id), f"Cleared capture '{second.capture_id}'.")

            release.set()
            deadline = time.time() + 2
            while engine._prefetch_running and time.time() < deadline:
                time.sleep(0.01)

            source = metrics.snapshot()["semantic_index"]["prefetch"]
            self.assertEqual(source["queued"], 2)
            self.assertEqual(source["evicted"], 1)
            self.assertEqual(source["cleared"], 1)
            self.assertEqual(source["completed"], 0)
            self.assertEqual(first.semantic_index_state, "evicted")
        finally:
            release.set()
            engine.shutdown()

    def test_async_prefetch_burst_is_queued_newest_first_and_clear_drops_queued_work(self):
        started = threading.Event()
        release = threading.Event()
        embedded_order = []

        class BlockingEmbedding:
            def embed(self, texts):
                embedded_order.append(texts[0])
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=8, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = BlockingEmbedding()
        try:
            first = engine.ingest("first payload", label="first")
            self.assertTrue(started.wait(timeout=2))
            second = engine.ingest("second payload", label="second")
            third = engine.ingest("third payload", label="third")
            fourth = engine.ingest("fourth payload", label="fourth")
            # A burst never drops eligible captures: every one is queued behind the running job.
            self.assertEqual(
                list(engine._prefetch_queue),
                [second.capture_id, third.capture_id, fourth.capture_id],
            )
            self.assertEqual(engine.get_buffer_stats()["semantic_prefetch_queued"], 3)
            self.assertEqual(engine.clear(second.capture_id), "Cleared capture 'cap_2'.")
            self.assertNotIn(second.capture_id, engine._prefetch_queue)
            self.assertEqual(second.semantic_index_state, "evicted")
            self.assertEqual(first.semantic_index_state, "pending")

            release.set()
            deadline = time.time() + 2
            while (engine._prefetch_queue or engine._prefetch_running) and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(embedded_order, ["first payload", "fourth payload", "third payload"])
            self.assertEqual(
                [cap.semantic_index_state for cap in (first, third, fourth)],
                ["ready", "ready", "ready"],
            )
            self.assertEqual(engine._prefetch_workers_active, 0)
        finally:
            release.set()
            engine.shutdown()

    def test_search_on_queued_capture_dequeues_it_and_indexes_on_demand(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                if texts[0] == "first payload":
                    started.set()
                    release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=4, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = BlockingEmbedding()
        try:
            engine.ingest("first payload", label="first")
            self.assertTrue(started.wait(timeout=2))
            queued = engine.ingest("queued payload", label="queued")
            self.assertIn(queued.capture_id, engine._prefetch_queue)

            # The on-demand job is blocked behind the running prefetch job's
            # model lock, so a bounded wait reports pending without dequeuing twice.
            self.assertEqual(engine._await_semantic_index(queued, 0.05), "pending")
            self.assertNotIn(queued.capture_id, engine._prefetch_queue)
            self.assertIn(queued.capture_id, engine._on_demand_jobs)
            self.assertEqual(queued.semantic_index_state, "pending")
            engine._schedule_semantic_prefetch(queued)
            self.assertNotIn(queued.capture_id, engine._prefetch_queue)
            self.assertEqual(engine.get_buffer_stats()["semantic_index_on_demand_running"], 1)
            release.set()
            self.assertTrue(engine.search_semantic(queued, "payload"))
            self.assertEqual(queued.semantic_index_state, "ready")
            self.assertEqual(engine._on_demand_jobs, {})
        finally:
            release.set()
            engine.shutdown()

    def test_hybrid_search_answers_lexical_first_within_wait_budget(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=5)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=2, semantic_prefetch=False, semantic_wait_seconds=0.05
        )
        engine.embedding_model = BlockingEmbedding()
        try:
            capture = engine.ingest("alpha needle\nbeta line", label="budget")
            self.assertEqual(capture.semantic_index_state, "not-requested")

            started_at = time.monotonic()
            result = engine.search("needle", mode="hybrid", capture_id=capture.capture_id)
            self.assertLess(time.monotonic() - started_at, 1.0)
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["semantic_coverage"], "pending")
            self.assertEqual(result["semantic_index_state"], "pending")
            self.assertEqual(result["semantic_wait_seconds"], 0.05)
            self.assertIn("lexical (BM25) only", result["message"])
            self.assertNotIn("semantic_fallback", result)
            self.assertEqual([m["chunk_index"] for m in result["matches"]], ["lexical"])
            self.assertEqual(result["matches"][0]["matched_range"], "L1-L2")
            self.assertEqual(engine.get_buffer_stats()["semantic_index_on_demand_running"], 1)

            # BM25-only mode never touches the semantic index.
            lexical = engine.search("needle", mode="bm25", capture_id=capture.capture_id)
            self.assertEqual(lexical["semantic_coverage"], "not-requested")
            self.assertNotIn("message", lexical)

            # Once indexing finishes, the same query is fully hybrid and stable.
            release.set()
            self.assertEqual(engine.wait_for_semantic_index(capture, timeout=2), "ready")
            self.assertEqual(engine.wait_for_semantic_index(capture), "ready")
            complete = engine.search("needle", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(complete["semantic_coverage"], "complete")
            self.assertNotIn("message", complete)
            self.assertNotIn("semantic_wait_seconds", complete)
            self.assertEqual(
                complete["matches"][0]["matched_range"], result["matches"][0]["matched_range"]
            )
            self.assertGreater(complete["matches"][0]["score"], result["matches"][0]["score"])
        finally:
            release.set()
            engine.shutdown()

    def test_semantic_mode_waits_for_index_beyond_hybrid_budget(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=5)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=2, semantic_prefetch=False, semantic_wait_seconds=0)
        engine.embedding_model = BlockingEmbedding()
        try:
            capture = engine.ingest("semantic payload", label="semantic-wait")
            outcome = {}

            def run_semantic():
                outcome["result"] = engine.search("payload", mode="semantic", capture_id=capture.capture_id)

            worker = threading.Thread(target=run_semantic)
            worker.start()
            self.assertTrue(started.wait(timeout=2))
            worker.join(timeout=0.2)
            self.assertTrue(worker.is_alive())
            # A hybrid search issued meanwhile shares the job and answers immediately.
            hybrid = engine.search("payload", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(hybrid["semantic_coverage"], "pending")
            self.assertEqual(len(engine._on_demand_jobs), 1)
            release.set()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(outcome["result"]["semantic_coverage"], "complete")
            self.assertEqual(outcome["result"]["match_count"], 1)
        finally:
            release.set()
            engine.shutdown()

    def test_wait_budget_validation_and_unbounded_wait(self):
        for invalid in (-1, float("nan")):
            with self.assertRaisesRegex(ValueError, "semantic_wait_seconds"):
                EphemeralEngine(semantic_prefetch=False, semantic_wait_seconds=invalid)
        with patch.dict("os.environ", {"EPHEMERAL_SEMANTIC_WAIT_SECONDS": "2.5"}, clear=False):
            configured = EphemeralEngine(semantic_prefetch=False)
        try:
            self.assertEqual(configured.semantic_wait_seconds, 2.5)
            self.assertEqual(configured.get_buffer_stats()["semantic_wait_seconds"], 2.5)
        finally:
            configured.shutdown()

        unbounded = EphemeralEngine(
            max_captures=2, semantic_prefetch=False, semantic_wait_seconds=float("inf")
        )
        try:
            capture = unbounded.ingest("unbounded payload", label="unbounded")
            result = unbounded.search("payload", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(result["semantic_coverage"], "complete")
            self.assertEqual(capture.semantic_index_state, "ready")
            # Empty captures have no semantic windows to index, so they report
            # complete coverage for semantic and hybrid requests.
            empty = unbounded.ingest("", label="empty")
            for mode, coverage in (("hybrid", "complete"), ("semantic", "complete"), ("bm25", "not-requested")):
                result = unbounded.search("x", mode=mode, capture_id=empty.capture_id)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["matches"], [])
                self.assertEqual(result["semantic_coverage"], coverage)
                self.assertEqual(result["message"], "Capture is empty (0 lines).")
        finally:
            unbounded.shutdown()

    def test_await_semantic_index_recovers_from_unpublished_job_and_eviction(self):
        engine = EphemeralEngine(max_captures=1, semantic_prefetch=False, semantic_wait_seconds=0)
        try:
            capture = engine.ingest("recover payload", label="recover")
            # A finished prefetch job that could not publish leaves the capture
            # to inline indexing so the waiter still gets a ready index.
            finished = threading.Event()
            finished.set()
            engine._prefetch_running[capture.capture_id] = finished
            self.assertEqual(engine._await_semantic_index(capture, 1.0), "ready")
            self.assertIsNotNone(capture.embeddings)
            engine._prefetch_running.clear()

            # An on-demand job whose capture was evicted mid-flight leaves it lazily indexable.
            started = threading.Event()
            release = threading.Event()

            class BlockingEmbedding:
                def embed(self, texts):
                    started.set()
                    release.wait(timeout=2)
                    return [[1.0] + [0.0] * 383 for _ in texts]

            engine.embedding_model = BlockingEmbedding()
            stale = engine.ingest("stale payload", label="stale")
            self.assertEqual(engine._await_semantic_index(stale, 0), "pending")
            self.assertTrue(started.wait(timeout=2))
            engine.clear(stale.capture_id)
            self.assertEqual(stale.semantic_index_state, "evicted")
            stale.semantic_index_state = "pending"
            release.set()
            deadline = time.time() + 2
            while stale.capture_id in engine._on_demand_jobs and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(stale.semantic_index_state, "not-requested")
            self.assertIsNone(stale.embeddings)
            # A bounded wait on an evicted capture reports pending rather than
            # indexing inline; an unbounded one takes the lazy path, where
            # without a reader lease nothing indexes an evicted capture and
            # semantic search over it is simply empty.
            self.assertEqual(engine.wait_for_semantic_index(stale, timeout=2), "pending")
            self.assertEqual(engine.wait_for_semantic_index(stale), "ready")
            self.assertIsNone(stale.embeddings)
            self.assertEqual(engine.search_semantic(stale, "stale"), [])
        finally:
            engine.shutdown()

    def test_shutdown_joins_on_demand_index_threads(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=2, semantic_prefetch=False, semantic_prefetch_workers=1, semantic_wait_seconds=0
        )
        engine.embedding_model = BlockingEmbedding()
        other_pools = _semantic_index_threads()
        capture = engine.ingest("shutdown payload", label="on-demand-shutdown")
        self.assertEqual(engine.search("payload", mode="hybrid", capture_id=capture.capture_id)["semantic_coverage"], "pending")
        self.assertTrue(started.wait(timeout=2))
        # A second search queues behind the single pool worker.
        queued = engine.ingest("queued payload", label="queued-at-shutdown")
        self.assertEqual(engine.search("payload", mode="hybrid", capture_id=queued.capture_id)["semantic_coverage"], "pending")
        queued_job = engine._on_demand_jobs[queued.capture_id]
        self.assertFalse(queued_job.future.running())
        shutdown_thread = threading.Thread(target=engine.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=0.2)
        self.assertTrue(shutdown_thread.is_alive())
        # Shutdown cancels the queued job at once and only waits for the running one.
        self.assertTrue(queued_job.done.wait(timeout=2))
        self.assertTrue(queued_job.future.cancelled())
        self.assertEqual(queued.semantic_index_state, "not-requested")
        self.assertNotIn(queued.capture_id, engine._on_demand_jobs)
        release.set()
        shutdown_thread.join(timeout=2)
        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(_semantic_index_threads() - other_pools, set())
        self.assertEqual(engine._on_demand_jobs, {})
        self.assertEqual(capture.semantic_index_state, "ready")
        self.assertIsNone(queued.embeddings)
        # The cancelled capture is still buffered and indexes inline on demand.
        self.assertEqual(engine.search("payload", mode="hybrid", capture_id=queued.capture_id)["semantic_coverage"], "complete")
        self.assertEqual(queued.semantic_index_state, "ready")
        # After shutdown no thread may start, so an unindexed capture is indexed inline.
        late = engine.ingest("late payload", label="after-shutdown")
        self.assertEqual(late.semantic_index_state, "not-requested")
        result = engine.search("payload", mode="hybrid", capture_id=late.capture_id)
        self.assertEqual(result["semantic_coverage"], "complete")
        self.assertEqual(late.semantic_index_state, "ready")
        self.assertEqual(_semantic_index_threads() - other_pools, set())
        self.assertEqual(engine._on_demand_jobs, {})

    def test_on_demand_indexing_stays_bounded_under_capture_churn(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=5)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=1,
            semantic_prefetch=False,
            semantic_prefetch_workers=1,
            semantic_wait_seconds=0,
        )
        engine.embedding_model = BlockingEmbedding()
        other_pools = _semantic_index_threads()
        try:
            captures = []
            for index in range(20):
                capture = engine.ingest(f"churn payload {index}", label=f"churn-{index}")
                result = engine.search("payload", mode="hybrid", capture_id=capture.capture_id)
                self.assertEqual(result["semantic_coverage"], "pending")
                captures.append(capture)
                if index == 0:
                    # Pin the first job as the one holding the model before
                    # eviction can cancel it while still queued.
                    self.assertTrue(started.wait(timeout=2))

            # The pool, not the number of timed-out searches, bounds live
            # indexing threads; the buffer holds one capture, yet twenty
            # searches started indexing work.
            self.assertEqual(len(_semantic_index_threads() - other_pools), 1)
            # Only the job that already holds the model (its capture was
            # evicted mid-flight) and the live capture's job remain: every
            # other evicted capture's job was cancelled while still queued.
            self.assertEqual(
                set(engine._on_demand_jobs),
                {captures[0].capture_id, captures[-1].capture_id},
            )
            stats = engine.get_buffer_stats()
            self.assertEqual(stats["semantic_index_on_demand_running"], 1)
            self.assertEqual(stats["semantic_index_on_demand_queued"], 1)
            for stale in captures[:-1]:
                self.assertEqual(stale.semantic_index_state, "evicted")

            release.set()
            self.assertEqual(engine.wait_for_semantic_index(captures[-1], timeout=5), "ready")
            self.assertEqual(engine._on_demand_jobs, {})
            stats = engine.get_buffer_stats()
            self.assertEqual(stats["semantic_index_on_demand_running"], 0)
            self.assertEqual(stats["semantic_index_on_demand_queued"], 0)
            self.assertEqual(len(_semantic_index_threads() - other_pools), 1)
            # Neither the cancelled jobs nor the one that ran materialized
            # embeddings for evicted captures.
            for stale in captures[:-1]:
                self.assertIsNone(stale.embeddings)
            complete = engine.search("payload", mode="hybrid", capture_id=captures[-1].capture_id)
            self.assertEqual(complete["semantic_coverage"], "complete")
        finally:
            release.set()
            engine.shutdown()
        self.assertEqual(_semantic_index_threads() - other_pools, set())

    def test_cancelled_on_demand_job_releases_a_blocked_waiter(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=5)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=2, semantic_prefetch=False, semantic_prefetch_workers=1, semantic_wait_seconds=5
        )
        engine.embedding_model = BlockingEmbedding()
        try:
            blocker = engine.ingest("blocker payload", label="blocker")
            self.assertEqual(engine._await_semantic_index(blocker, 0), "pending")
            self.assertTrue(started.wait(timeout=2))
            queued = engine.ingest("queued payload", label="queued")
            outcome = {}

            def run_hybrid():
                outcome["hybrid"] = engine.search("payload", mode="hybrid", capture_id=queued.capture_id)

            def run_semantic():
                outcome["semantic"] = engine.search("payload", mode="semantic", capture_id=queued.capture_id)

            hybrid_waiter = threading.Thread(target=run_hybrid)
            semantic_waiter = threading.Thread(target=run_semantic)
            hybrid_waiter.start()
            semantic_waiter.start()
            deadline = time.time() + 2
            while queued.active_readers < 2 and time.time() < deadline:
                time.sleep(0.01)
            job = engine._on_demand_jobs[queued.capture_id]
            self.assertFalse(job.future.running())
            self.assertEqual(engine.get_buffer_stats()["semantic_index_on_demand_queued"], 1)

            # Clearing the capture cancels its queued job and wakes both
            # waiters.  The hybrid one is within its 5 s budget, so it answers
            # lexical-first at once rather than indexing the evicted capture
            # inline behind the model lock; the semantic one has no fallback
            # and indexes under its reader lease once the model is free.
            cleared_at = time.monotonic()
            engine.clear(queued.capture_id)
            self.assertTrue(job.done.wait(timeout=2))
            self.assertTrue(job.cancelled)
            self.assertTrue(job.future.cancelled())
            self.assertNotIn(queued.capture_id, engine._on_demand_jobs)
            hybrid_waiter.join(timeout=2)
            self.assertFalse(hybrid_waiter.is_alive())
            self.assertLess(time.monotonic() - cleared_at, 2)
            self.assertEqual(outcome["hybrid"]["semantic_coverage"], "pending")
            self.assertEqual([m["chunk_index"] for m in outcome["hybrid"]["matches"]], ["lexical"])
            self.assertIsNone(queued.embeddings)
            semantic_waiter.join(timeout=0.2)
            self.assertTrue(semantic_waiter.is_alive())
            release.set()
            semantic_waiter.join(timeout=5)
            self.assertFalse(semantic_waiter.is_alive())
            self.assertEqual(outcome["semantic"]["semantic_coverage"], "complete")
            self.assertEqual(outcome["semantic"]["match_count"], 1)
        finally:
            release.set()
            engine.shutdown()

    def test_queued_capture_that_fails_inline_is_marked_failed_and_retries_lazily(self):
        started = threading.Event()
        release = threading.Event()
        attempts = []

        class FlakyEmbedding:
            def embed(self, texts):
                if texts[0] == "first payload":
                    started.set()
                    release.wait(timeout=2)
                elif texts[0] == "queued payload":
                    attempts.append(texts[0])
                    if len(attempts) == 1:
                        raise RuntimeError("embedding backend unavailable")
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(
            max_captures=4, semantic_prefetch=True, semantic_prefetch_workers=1, semantic_wait_seconds=0
        )
        engine.embedding_model = FlakyEmbedding()
        try:
            engine.ingest("first payload", label="first")
            self.assertTrue(started.wait(timeout=2))
            queued = engine.ingest("queued payload", label="queued")
            self.assertIn(queued.capture_id, engine._prefetch_queue)
            self.assertEqual(queued.semantic_index_state, "pending")

            # A search pulls the capture out of the prefetch queue; its job is
            # blocked behind the running prefetch job's model lock.
            result = engine.search("payload", mode="hybrid", capture_id=queued.capture_id)
            self.assertEqual(result["semantic_coverage"], "pending")
            self.assertNotIn(queued.capture_id, engine._prefetch_queue)
            job = engine._on_demand_jobs[queued.capture_id]

            # When the dequeued job fails it is not left as a stale pending
            # entry: the capture is marked failed and no job or queue slot remains.
            release.set()
            self.assertTrue(job.done.wait(timeout=2))
            self.assertIsInstance(job.error, RuntimeError)
            self.assertEqual(queued.semantic_index_state, "failed")
            self.assertNotIn(queued.capture_id, engine._prefetch_queue)
            self.assertNotIn(queued.capture_id, engine._prefetch_running)
            self.assertNotIn(queued.capture_id, engine._on_demand_jobs)
            stats = engine.get_buffer_stats()
            self.assertEqual(stats["semantic_prefetch_failed"], 1)
            self.assertEqual(stats["semantic_prefetch_pending"], 0)

            # The failed capture retries through the normal lazy path.
            self.assertEqual(engine.wait_for_semantic_index(queued, timeout=2), "ready")
            self.assertEqual(attempts, ["queued payload", "queued payload"])
            self.assertEqual(queued.semantic_index_state, "ready")
            complete = engine.search("payload", mode="hybrid", capture_id=queued.capture_id)
            self.assertEqual(complete["semantic_coverage"], "complete")
        finally:
            release.set()
            engine.shutdown()

    def test_async_prefetch_lifecycle_error_and_shutdown_paths(self):
        with self.assertRaisesRegex(ValueError, "semantic_prefetch_workers"):
            EphemeralEngine(semantic_prefetch=True, semantic_prefetch_workers=0)

        submit_failure = EphemeralEngine(max_captures=2, semantic_prefetch=True, semantic_prefetch_workers=1)
        try:
            with patch.object(
                submit_failure._prefetch_executor,
                "submit",
                side_effect=RuntimeError("executor closed"),
            ):
                capture = submit_failure.ingest("submit failure", label="submit-failure")
                self.assertEqual(capture.semantic_index_state, "failed")
                self.assertEqual(submit_failure._prefetch_queue, {})
                # With a worker already active, a submit failure leaves the capture queued.
                submit_failure._prefetch_workers_active = 1
                queued = submit_failure.ingest("still queued", label="still-queued")
                self.assertEqual(queued.semantic_index_state, "pending")
                self.assertIn(queued.capture_id, submit_failure._prefetch_queue)
                submit_failure._prefetch_workers_active = 0
            submit_failure.captures.clear()
            submit_failure._schedule_semantic_prefetch(capture)
            self.assertNotIn(capture.capture_id, submit_failure._prefetch_queue)
        finally:
            submit_failure.shutdown()
            submit_failure.shutdown()
            self.assertEqual(queued.semantic_index_state, "not-requested")

        evicted_engine = EphemeralEngine(max_captures=1, semantic_prefetch=True, semantic_prefetch_workers=1)
        try:
            capture = SimpleNamespace(
                capture_id="cap-evicted",
                embeddings=None,
                semantic_index_state="pending",
                semantic_chunks=[Chunk(0, 1, 1, "x")],
                active_readers=0,
            )
            # A worker that finds its capture already evicted leaves it lazily indexable.
            evicted_engine._prefetch_queue[capture.capture_id] = capture
            evicted_engine._prefetch_workers_active = 1
            evicted_engine._prefetch_worker()
            self.assertEqual(capture.semantic_index_state, "not-requested")
            self.assertEqual(evicted_engine._prefetch_running, {})
            self.assertEqual(evicted_engine._prefetch_workers_active, 0)
        finally:
            evicted_engine.shutdown()

    def test_shutdown_drops_queued_prefetch_and_lets_running_work_finish(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=2, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = BlockingEmbedding()
        try:
            first = engine.ingest("first shutdown payload", label="shutdown-first")
            self.assertTrue(started.wait(timeout=2))
            second = engine.ingest("queued shutdown payload", label="shutdown-second")
            self.assertIn(second.capture_id, engine._prefetch_queue)

            shutdown_thread = threading.Thread(target=engine.shutdown)
            shutdown_thread.start()
            deadline = time.time() + 2
            while second.capture_id in engine._prefetch_queue and time.time() < deadline:
                time.sleep(0.01)
            self.assertNotIn(second.capture_id, engine._prefetch_queue)
            self.assertEqual(second.semantic_index_state, "not-requested")

            release.set()
            shutdown_thread.join(timeout=2)
            self.assertFalse(shutdown_thread.is_alive())
            self.assertTrue(engine._shutdown)
            self.assertEqual(engine._prefetch_queue, {})
            self.assertEqual(engine._prefetch_running, {})
            self.assertEqual(first.semantic_index_state, "ready")
        finally:
            release.set()
            engine.shutdown()

    def test_shutdown_waits_for_embedding_warmup(self):
        started = threading.Event()
        release = threading.Event()
        shutdown_entered = threading.Event()
        shutdown_done = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=1, embedding_warmup=True)
        engine.embedding_model = BlockingEmbedding()
        try:
            self.assertTrue(engine.start_embedding_warmup())
            self.assertTrue(started.wait(timeout=2))

            def shutdown():
                shutdown_entered.set()
                engine.shutdown()
                shutdown_done.set()

            shutdown_thread = threading.Thread(target=shutdown)
            shutdown_thread.start()
            self.assertTrue(shutdown_entered.wait(timeout=2))
            self.assertFalse(shutdown_done.wait(timeout=0.05))

            release.set()
            shutdown_thread.join(timeout=2)
            self.assertFalse(shutdown_thread.is_alive())
            self.assertTrue(shutdown_done.is_set())
            self.assertEqual(engine.embedding_warmup_state, "ready")
        finally:
            release.set()
            engine.shutdown()

    def test_bm25_invalid_query_and_sqlite_failure_return_no_matches(self):
        engine = EphemeralEngine(max_captures=1)
        capture = engine.ingest("searchable payload", label="search-errors")

        self.assertEqual(engine.search_bm25(capture, "!!!"), [])

        class BrokenCursor:
            def execute(self, *_args):
                raise sqlite3.OperationalError("fts unavailable")

        class BrokenConnection:
            def cursor(self):
                return BrokenCursor()

        capture.fts_conn = BrokenConnection()
        self.assertEqual(engine.search_bm25(capture, "searchable"), [])

    def test_python_lexical_fallback_preserves_token_matches(self):
        with patch("engine.sqlite_fts5_available", return_value=False):
            engine = EphemeralEngine(max_captures=1)
            capture = engine.ingest("noise line\nECONNREFUSED on port 5432\n", label="fallback")

        self.assertEqual(engine.lexical_backend, "python-fallback")
        self.assertIsNone(capture.fts_conn)
        results = engine.search_bm25(capture, "ECONNREFUSED")
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0)
        self.assertEqual(engine.search_bm25(capture, "missing-token"), [])
        self.assertEqual(engine.get_buffer_stats()["lexical_backend"], "python-fallback")

    def test_python_lexical_fallback_ignores_diacritics(self):
        with patch("engine.sqlite_fts5_available", return_value=False):
            engine = EphemeralEngine(max_captures=1)
            capture = engine.ingest("café connection failed", label="diacritics")

        self.assertEqual(engine.search_bm25(capture, "cafe")[0][0], 0)

    def test_lexical_backends_share_fts5_token_boundaries(self):
        backend_modes = [False]
        if sqlite_fts5_available():
            backend_modes.append(True)

        for fts5_available in backend_modes:
            with self.subTest(fts5_available=fts5_available):
                with patch("engine.sqlite_fts5_available", return_value=fts5_available):
                    engine = EphemeralEngine(max_captures=1)
                    capture = engine.ingest(
                        "database_connection café naïve v2.4 error-code",
                        label="token-boundaries",
                    )

                for query in ("connection", "cafe", "naive", "v2", "code"):
                    self.assertTrue(
                        engine.search_bm25(capture, query),
                        f"{query!r} should match with fts5_available={fts5_available}",
                    )

    def test_fallback_query_syntax_remains_tokenized_as_text(self):
        with patch("engine.sqlite_fts5_available", return_value=False):
            engine = EphemeralEngine(max_captures=1)
            capture = engine.ingest("database_connection", label="query-syntax")

        self.assertTrue(engine.search_bm25(capture, '"connection" OR missing'))

    def test_clear_single_and_missing_capture_paths(self):
        engine = EphemeralEngine(max_captures=2)
        capture = engine.ingest("cleanup payload", label="cleanup")

        self.assertEqual(
            engine.clear(capture.capture_id),
            f"Cleared capture '{capture.capture_id}'.",
        )
        self.assertEqual(engine.clear(capture.capture_id), f"Capture '{capture.capture_id}' not found.")

    def test_empty_capture_reader_returns_none(self):
        engine = EphemeralEngine(max_captures=1, embedding_model_name="test")
        try:
            self.assertIsNone(engine._acquire_capture_reader("latest"))
            self.assertIsNone(engine.get_capture())
        finally:
            engine.shutdown()

    def test_embedding_load_success_and_failure_diagnostics(self):
        class FakeEmbedding:
            pass

        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "0"}, clear=False), \
                patch("engine.TextEmbedding", return_value=FakeEmbedding()) as embedding, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            engine = EphemeralEngine(max_captures=1, embedding_cache_path="/cache")
            loaded = engine._get_embedding_model()

        self.assertIsInstance(loaded, FakeEmbedding)
        embedding.assert_called_once_with(model_name=engine.embedding_model_name, cache_dir="/cache")
        self.assertIn("Loading embedding model", stderr.getvalue())
        self.assertIn("Embedding model ready", stderr.getvalue())

        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "0"}, clear=False), \
                patch("engine.TextEmbedding", side_effect=RuntimeError("load failed")):
            failing = EphemeralEngine(max_captures=1)
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                failing._get_embedding_model()
        self.assertIsNone(failing.embedding_model)

    def test_embedding_warmup_succeeds_once_and_reports_readiness(self):
        class WarmEmbedding:
            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(list(texts))
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=1, embedding_warmup=True)
        embedding = WarmEmbedding()
        engine.embedding_model = embedding
        try:
            self.assertTrue(engine.start_embedding_warmup())
            self.assertEqual(engine.wait_for_embedding_warmup(timeout=2), "ready")
            self.assertFalse(engine.start_embedding_warmup())
            self.assertEqual(len(embedding.calls), 1)
            stats = engine.get_buffer_stats()
            self.assertEqual(stats["embedding_warmup_state"], "ready")
            self.assertIsNone(stats["embedding_warmup_failure"])
        finally:
            engine.shutdown()

    def test_embedding_warmup_failure_is_degraded_and_hybrid_falls_back(self):
        class FailingEmbedding:
            def embed(self, _texts):
                raise RuntimeError("model unavailable")

        engine = EphemeralEngine(max_captures=1, embedding_warmup=True)
        engine.embedding_model = FailingEmbedding()
        try:
            self.assertTrue(engine.start_embedding_warmup())
            self.assertEqual(engine.wait_for_embedding_warmup(timeout=2), "failed")
            stats = engine.get_buffer_stats()
            self.assertEqual(stats["embedding_warmup_failure"], "RuntimeError")

            capture = engine.ingest("lexical fallback marker", label="fallback")
            result = engine.search("marker", mode="hybrid", capture_id=capture.capture_id)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["semantic_fallback"], "RuntimeError")
            self.assertEqual(result["match_count"], 1)
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                engine.search("marker", mode="semantic", capture_id=capture.capture_id)
        finally:
            engine.shutdown()

    def test_embedding_warmup_can_be_disabled(self):
        engine = EphemeralEngine(max_captures=1, embedding_warmup=False)
        try:
            self.assertFalse(engine.start_embedding_warmup())
            self.assertEqual(engine.wait_for_embedding_warmup(timeout=0), "disabled")
        finally:
            engine.shutdown()

    def test_process_rss_falls_back_to_resource_and_can_be_unavailable(self):
        resource_module = SimpleNamespace(
            RUSAGE_SELF=object(),
            getrusage=lambda _kind: SimpleNamespace(ru_maxrss=4),
        )
        with patch("builtins.open", side_effect=OSError("statm unavailable")), \
                patch.dict(sys.modules, {"resource": resource_module}):
            self.assertEqual(process_rss_bytes(), 4096)

        with patch("builtins.open", side_effect=OSError("statm unavailable")), \
                patch.dict(sys.modules, {"resource": None}):
            self.assertIsNone(process_rss_bytes())


class TestEmbeddingStartup(unittest.TestCase):
    def test_embedding_model_loads_lazily(self):
        with patch("engine.TextEmbedding") as embedding:
            engine = EphemeralEngine(max_captures=1)

        embedding.assert_not_called()
        self.assertIsNone(engine.embedding_model)

    def test_embedding_warmup_thread_start_failure_is_degraded(self):
        engine = EphemeralEngine(max_captures=1, embedding_warmup=True)
        try:
            with patch.object(threading.Thread, "start", side_effect=RuntimeError("thread limit")):
                self.assertFalse(engine.start_embedding_warmup())
            self.assertEqual(engine.embedding_warmup_state, "failed")
            self.assertEqual(engine.embedding_warmup_failure, "RuntimeError")
        finally:
            engine.shutdown()

    def test_semantic_chunks_pack_lines_by_line_and_byte_caps(self):
        engine = EphemeralEngine(max_captures=1, semantic_chunk_lines=3, semantic_chunk_bytes=20)
        self.assertEqual(engine._semantic_chunk_lines([]), [])

        lines = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg"]
        chunks = engine._semantic_chunk_lines(lines)
        self.assertEqual([(c.chunk_id, c.start_line, c.end_line) for c in chunks], [(0, 1, 3), (1, 4, 6), (2, 7, 7)])
        self.assertEqual(chunks[0].text, "aaaa\nbbbb\ncccc")

        # The byte cap closes a window early, and an oversized line still gets its own chunk.
        wide = ["x" * 15, "y" * 10, "z" * 40, "w"]
        chunks = engine._semantic_chunk_lines(wide)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(1, 1), (2, 2), (3, 3), (4, 4)])

        overlapping = EphemeralEngine(max_captures=1, semantic_chunk_lines=4, semantic_chunk_overlap=2)
        chunks = overlapping._semantic_chunk_lines(lines)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(1, 4), (3, 6), (5, 7)])

        capture = engine.ingest("\n".join(lines), label="semantic-chunks")
        self.assertEqual(len(capture.chunks), 3)
        self.assertEqual(len(capture.semantic_chunks), 3)
        stats = engine.get_buffer_stats()
        self.assertEqual(stats["total_semantic_chunks"], 3)
        self.assertEqual(stats["semantic_chunk_lines"], 3)
        self.assertEqual(stats["semantic_chunk_bytes"], 20)
        self.assertEqual(stats["semantic_chunk_overlap"], 0)

    def test_semantic_chunk_settings_come_from_environment_and_are_validated(self):
        with patch.dict(
            "os.environ",
            {
                "EPHEMERAL_SEMANTIC_CHUNK_LINES": "6",
                "EPHEMERAL_SEMANTIC_CHUNK_BYTES": "512",
                "EPHEMERAL_SEMANTIC_CHUNK_OVERLAP": "1",
            },
            clear=False,
        ):
            engine = EphemeralEngine(max_captures=1)
        self.assertEqual(
            (engine.semantic_chunk_lines, engine.semantic_chunk_bytes, engine.semantic_chunk_overlap),
            (6, 512, 1),
        )
        with self.assertRaisesRegex(ValueError, "semantic_chunk_lines"):
            EphemeralEngine(max_captures=1, semantic_chunk_lines=0)
        with self.assertRaisesRegex(ValueError, "semantic_chunk_bytes"):
            EphemeralEngine(max_captures=1, semantic_chunk_bytes=0)
        with self.assertRaisesRegex(ValueError, "semantic_chunk_overlap"):
            EphemeralEngine(max_captures=1, semantic_chunk_lines=4, semantic_chunk_overlap=4)
        with self.assertRaisesRegex(ValueError, "semantic_chunk_overlap"):
            EphemeralEngine(max_captures=1, semantic_chunk_overlap=-1)

    def test_hybrid_fuses_semantic_windows_with_lexical_ranges(self):
        engine = EphemeralEngine(max_captures=1, semantic_chunk_lines=4)
        capture = engine.ingest("\n".join(f"line {i}" for i in range(1, 13)), label="fusion")
        # Lexical windows: 0=L1-4, 1=L3-6, 2=L5-8, 3=L7-10, 4=L9-12.
        # Semantic windows: 0=L1-4, 1=L5-8, 2=L9-12.
        with patch.object(engine, "search_bm25", return_value=[(4, 0.5), (0, 0.4)]), \
                patch.object(engine, "search_semantic", return_value=[(0, 0.9), (1, 0.8)]):
            result = engine.search("line", mode="hybrid", capture_id=capture.capture_id, top_k=3, context_lines=0)

        matches = result["matches"]
        # Semantic window 0 overlaps lexical window 0 and lifts it above the higher-ranked lexical hit.
        self.assertEqual([(m["chunk_index"], m["chunk_id"]) for m in matches], [
            ("lexical", 0), ("lexical", 4), ("semantic", 1),
        ])
        self.assertEqual(matches[0]["matched_range"], "L1-L4")
        self.assertEqual(matches[2]["matched_range"], "L5-L8")

        # A backend that reports the same chunk twice accumulates its score.
        with patch.object(engine, "search_bm25", return_value=[(2, 0.5), (2, 0.25)]):
            duplicated = engine.search("line", mode="bm25", capture_id=capture.capture_id, top_k=1)
        self.assertEqual(duplicated["matches"][0]["score"], 0.75)

        # A lexical window that merely straddles a semantic window is not a member of it.
        with patch.object(engine, "search_bm25", return_value=[(1, 0.5)]), \
                patch.object(engine, "search_semantic", return_value=[(1, 0.9)]):
            straddle = engine.search("line", mode="hybrid", capture_id=capture.capture_id, top_k=3, context_lines=0)
        # It gets no boost, and the semantic window is then redundant with the lexical match's lines.
        self.assertEqual(
            [(m["chunk_index"], m["chunk_id"], m["matched_range"], m["score"]) for m in straddle["matches"]],
            [("lexical", 1, "L3-L6", round(HYBRID_LEXICAL_WEIGHT / 61, 4))],
        )

        with patch.object(engine, "search_semantic", return_value=[(2, 0.9)]):
            semantic_only = engine.search("line", mode="semantic", capture_id=capture.capture_id, top_k=1)
        self.assertEqual(semantic_only["matches"][0]["chunk_index"], "semantic")
        self.assertEqual(semantic_only["matches"][0]["chunk_id"], 2)
        self.assertEqual(semantic_only["matches"][0]["matched_range"], "L9-L12")

    def test_embedding_threads_configuration_and_validation(self):
        with patch.dict("os.environ", {"EPHEMERAL_EMBEDDING_THREADS": "3"}, clear=False):
            configured = EphemeralEngine(max_captures=1)
        self.assertEqual(configured.embedding_threads, 3)
        self.assertEqual(configured.get_buffer_stats()["embedding_threads"], 3)

        explicit = EphemeralEngine(
            max_captures=1, embedding_threads=2, embedding_model_name=CATALOGUE_EMBEDDING_MODEL
        )
        self.assertEqual(explicit.embedding_threads, 2)
        with self.assertRaisesRegex(ValueError, "embedding_threads"):
            EphemeralEngine(max_captures=1, embedding_threads=0)

        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "0"}, clear=False), \
                patch("engine.TextEmbedding") as embedding, \
                patch("sys.stderr", new_callable=io.StringIO):
            explicit._get_embedding_model()
        # CI may set EPHEMERAL_FASTEMBED_CACHE_DIR, so only assert the settings under test.
        self.assertEqual(embedding.call_count, 1)
        self.assertEqual(embedding.call_args.kwargs["model_name"], explicit.embedding_model_name)
        self.assertEqual(embedding.call_args.kwargs["threads"], 2)
        embedding.add_custom_model.assert_not_called()

    def test_fp32_model_alias_is_registered_once_per_process(self):
        import engine as engine_module

        original_flag = engine_module._BUNDLED_MODELS_REGISTERED
        engine_module._BUNDLED_MODELS_REGISTERED = False
        try:
            with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "0"}, clear=False), \
                    patch("engine.TextEmbedding") as embedding, \
                    patch("sys.stderr", new_callable=io.StringIO):
                first = EphemeralEngine(max_captures=1, embedding_model_name=FP32_EMBEDDING_MODEL)
                first._get_embedding_model()
                second = EphemeralEngine(max_captures=1, embedding_model_name=FP32_EMBEDDING_MODEL)
                second._get_embedding_model()
            embedding.add_custom_model.assert_called_once()
            registration = embedding.add_custom_model.call_args.kwargs
            self.assertEqual(registration["model"], FP32_EMBEDDING_MODEL)
            self.assertEqual(registration["sources"].hf, "BAAI/bge-small-en-v1.5")
            self.assertEqual(registration["model_file"], "onnx/model.onnx")
            self.assertEqual(embedding.call_count, 2)

            engine_module._BUNDLED_MODELS_REGISTERED = False
            with patch("engine.TextEmbedding") as embedding:
                embedding.add_custom_model.side_effect = ValueError("already registered")
                register_bundled_embedding_models()
            self.assertTrue(engine_module._BUNDLED_MODELS_REGISTERED)
        finally:
            engine_module._BUNDLED_MODELS_REGISTERED = original_flag


if __name__ == "__main__":
    unittest.main()
