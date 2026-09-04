"""
Comprehensive unit & integration tests for EphemeralEngine and MCP Server tools.
"""

import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from engine import EphemeralEngine


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

    def test_05_lru_buffer_eviction(self):
        # max_captures is 3 for this focused eviction test
        self.engine.clear("all")
        self.assertEqual(len(self.engine.captures), 0)

        initial_captures = []
        for i in range(1, 4):
            initial_captures.append(self.engine.ingest(f"Content for run {i}\nDone {i}", label=f"Run {i}"))

        # Refresh Run 1 so Run 2 becomes the least recently used capture.
        self.engine.get_summary(initial_captures[0].capture_id)
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
        cap_clean = self.engine.ingest(clean_log.strip(), label="pytest-clean-run")
        summary_clean = self.engine.get_summary(cap_clean.capture_id)
        self.assertEqual(summary_clean["signals_summary"], "None detected")
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
            first = self.engine.ingest("a" * 20, label="byte-budget-1")
            second = self.engine.ingest("b" * 20, label="byte-budget-2")
            third = self.engine.ingest("c" * 20, label="byte-budget-3")

            stats = self.engine.get_buffer_stats()
            self.assertEqual(stats["capture_count"], 2)
            self.assertEqual(stats["total_bytes"], second.byte_size + third.byte_size)
            self.assertEqual(stats["max_buffer_bytes"], 50)
            self.assertNotIn(first.capture_id, self.engine.captures)

            with self.assertRaises(ValueError):
                self.engine.ingest("x" * 100, label="oversized")
        finally:
            self.engine.max_buffer_bytes = original_limit
            self.engine.clear("all")
        print("\n[Byte Budget Passed] Content byte budget evicted old captures and rejected oversized input.")

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
        worker = threading.Thread(target=self.engine.ingest, args=("blocked embedding",))
        try:
            worker.start()
            self.assertTrue(blocker.started.wait(timeout=1))
            stats = self.engine.get_buffer_stats()
            self.assertIn("capture_count", stats)
        finally:
            blocker.release.set()
            worker.join(timeout=2)
            self.engine.embedding_model = original_model


if __name__ == "__main__":
    unittest.main()
