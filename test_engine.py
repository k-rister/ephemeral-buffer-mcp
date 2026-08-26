"""
Comprehensive unit & integration tests for EphemeralEngine and MCP Server tools.
"""

import sys
import unittest
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
        self.assertIn("fatal", summary["keyword_signals"])
        self.assertEqual(summary["keyword_signals"]["fatal"], 1)

        # Test slice around line 50
        slice_res = self.engine.get_slice(48, 52, capture_id=cap.capture_id)
        self.assertEqual(slice_res["status"], "ok")
        self.assertIn("FATAL: System ran out of file descriptors", slice_res["content"])
        print("\n[Slice & Summary Test Passed]:")
        print(slice_res["content"])

    def test_05_ring_buffer_eviction(self):
        # max_captures is 3
        self.engine.clear("all")
        self.assertEqual(len(self.engine.captures), 0)

        for i in range(1, 6):
            self.engine.ingest(f"Content for run {i}\nDone {i}", label=f"Run {i}")

        active_caps = self.engine.list_captures()
        self.assertEqual(len(active_caps), 3)
        # Should contain Run 3, Run 4, Run 5 (Run 1 and 2 evicted)
        labels = [c["label"] for c in active_caps]
        self.assertIn("Run 5", labels)
        self.assertIn("Run 4", labels)
        self.assertIn("Run 3", labels)
        self.assertNotIn("Run 1", labels)
        print("\n[Ring Buffer Eviction Passed] Oldest captures safely evicted, memory bound maintained.")


if __name__ == "__main__":
    unittest.main()
