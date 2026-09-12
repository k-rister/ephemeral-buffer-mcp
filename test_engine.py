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
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
from engine import (
    EphemeralEngine,
    PREVIEW_MAX_BYTES,
    SEARCH_SNIPPET_MAX_BYTES,
    _bounded_preview,
    _decode_git_path,
    _parse_git_diff_paths,
    detect_content_type,
    detect_signals,
    parse_unified_diff,
    process_rss_bytes,
    sqlite_fts5_available,
)


class TestEngineClassification(unittest.TestCase):
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

    def test_04_slice_and_summary(self):
        lines = [f"Log line number {i}" for i in range(1, 101)]
        lines[49] = "FATAL: System ran out of file descriptors"
        cap = self.engine.ingest("\n".join(lines), label="100-lines-log")

        # Test summary
        summary = self.engine.get_summary(cap.capture_id)
        self.assertEqual(summary["total_lines"], 100)
        self.assertIn("error", summary["keyword_signals"])
        self.assertEqual(summary["keyword_signals"]["error"], 1)

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
            engine._schedule_semantic_prefetch(capture)
            self.assertEqual(len(engine._prefetch_futures), 1)

            release.set()
            self.assertTrue(engine.search_semantic(capture, "payload"))
            self.assertEqual(capture.semantic_index_state, "ready")
            self.assertIsNotNone(capture.embeddings)
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
            engine._wait_for_prefetch(capture)
            self.assertEqual(capture.semantic_index_state, "failed")
            with self.assertRaisesRegex(RuntimeError, "prefetch unavailable"):
                engine.search_semantic(capture, "failure")
        finally:
            engine.shutdown()

    def test_async_prefetch_is_bounded_and_clear_cancels_queued_work(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                started.set()
                release.wait(timeout=2)
                return [[1.0] + [0.0] * 383 for _ in texts]

        engine = EphemeralEngine(max_captures=4, semantic_prefetch=True, semantic_prefetch_workers=1)
        engine.embedding_model = BlockingEmbedding()
        try:
            first = engine.ingest("first payload", label="first")
            self.assertTrue(started.wait(timeout=2))
            second = engine.ingest("second payload", label="second")
            third = engine.ingest("third payload", label="third")
            self.assertLessEqual(len(engine._prefetch_futures), 2)
            self.assertEqual(engine.clear(second.capture_id), "Cleared capture 'cap_2'.")
            self.assertNotIn(second.capture_id, engine._prefetch_futures)
            self.assertEqual(first.semantic_index_state, "pending")
            release.set()
            engine.clear("all")
        finally:
            release.set()
            engine.shutdown()

    def test_async_prefetch_lifecycle_error_and_shutdown_paths(self):
        with self.assertRaisesRegex(ValueError, "semantic_prefetch_workers"):
            EphemeralEngine(semantic_prefetch=True, semantic_prefetch_workers=0)

        submit_failure = EphemeralEngine(max_captures=1, semantic_prefetch=True, semantic_prefetch_workers=1)
        try:
            with patch.object(
                submit_failure._prefetch_executor,
                "submit",
                side_effect=RuntimeError("executor closed"),
            ):
                capture = submit_failure.ingest("submit failure", label="submit-failure")
            self.assertEqual(capture.semantic_index_state, "failed")
            submit_failure.captures.clear()
            submit_failure._schedule_semantic_prefetch(capture)
        finally:
            submit_failure.shutdown()
            submit_failure.shutdown()

        callback_engine = EphemeralEngine(max_captures=1, semantic_prefetch=True, semantic_prefetch_workers=1)
        try:
            capture = SimpleNamespace(
                capture_id="cap-callback",
                embeddings=None,
                semantic_index_state="pending",
            )
            callback_engine.captures[capture.capture_id] = capture
            future = Future()
            callback_engine._prefetch_futures[capture.capture_id] = future
            future.cancel()
            callback_engine._prefetch_slots.acquire(blocking=False)
            callback_engine._prefetch_finished(capture.capture_id, future)
            self.assertEqual(capture.semantic_index_state, "not-requested")

            failed_future = Future()
            failed_future.set_exception(RuntimeError("background failure"))
            callback_engine._prefetch_futures[capture.capture_id] = failed_future
            callback_engine._wait_for_prefetch(capture)
        finally:
            callback_engine.shutdown()

    def test_shutdown_snapshots_prefetch_futures_before_cancellation(self):
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
            self.assertIn(second.capture_id, engine._prefetch_futures)

            shutdown_thread = threading.Thread(target=engine.shutdown)
            shutdown_thread.start()
            deadline = time.time() + 2
            while second.capture_id in engine._prefetch_futures and time.time() < deadline:
                time.sleep(0.01)
            self.assertNotIn(second.capture_id, engine._prefetch_futures)

            release.set()
            shutdown_thread.join(timeout=2)
            self.assertFalse(shutdown_thread.is_alive())
            self.assertTrue(engine._shutdown)
            self.assertEqual(engine._prefetch_futures, {})
            self.assertEqual(first.semantic_index_state, "ready")
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


if __name__ == "__main__":
    unittest.main()
