"""Unit tests for MCP tool validation, response formatting, and socket IPC."""

import asyncio
import io
import json
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import server


class FakeReader:
    def __init__(self, payload):
        self.payload = payload

    async def read(self, _limit):
        return self.payload


class FakeWriter:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, payload):
        self.writes.append(payload)

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class TestServerTools(unittest.TestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
        server.engine.clear("all")

    def test_capture_file_reports_missing_path(self):
        result = server.capture_file("/does/not/exist")

        self.assertIn("does not exist", result)

    def test_capture_file_rejects_limit_above_buffer_budget(self):
        server.engine.max_buffer_bytes = 16
        with tempfile.NamedTemporaryFile() as file_handle:
            result = server.capture_file(file_handle.name, max_bytes=17)

        self.assertIn("exceeds the configured buffer limit", result)

    def test_capture_file_rejects_non_positive_limit(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            result = server.capture_file(file_handle.name, max_bytes=0)

        self.assertIn("max_bytes must be at least 1", result)

    def test_capture_file_reports_read_failure(self):
        with tempfile.NamedTemporaryFile() as file_handle, \
                patch.object(server, "read_file_bounded", side_effect=OSError("permission denied")):
            result = server.capture_file(file_handle.name)

        self.assertIn("Error reading file", result)
        self.assertIn("permission denied", result)

    def test_execute_rejects_limit_above_buffer_budget(self):
        server.engine.max_buffer_bytes = 1024

        result = server.execute_and_capture("printf output", max_output_bytes=1025)

        self.assertIn("max_output_bytes", result)
        self.assertIn("exceeds the configured buffer limit", result)

    def test_execute_rejects_too_small_limit(self):
        result = server.execute_and_capture("printf output", max_output_bytes=100)

        self.assertIn("at least 512", result)

    def test_execute_reports_command_failure(self):
        with patch.object(server, "run_command_bounded", side_effect=OSError("unable to start command")):
            result = server.execute_and_capture("missing-command")

        self.assertIn("Error executing command", result)
        self.assertIn("unable to start command", result)

    def test_execute_reports_timeout(self):
        with patch.object(
            server,
            "run_command_bounded",
            return_value=("partial output", 124, False, 14, True),
        ) as run:
            result = server.execute_and_capture("sleep 10", timeout_seconds=0.5)

        self.assertIn("TIMED OUT after 0.5s", result)
        run.assert_called_once()

    def test_buffer_stats_formats_memory_metrics(self):
        server.capture_text("server stats payload", label="server-test")

        result = server.get_buffer_stats()

        self.assertIn("Captures:", result)
        self.assertIn("Embedding bytes:", result)
        self.assertIn("Embedding model:", result)
        self.assertIn("Process RSS:", result)
        self.assertIn("Unaccounted RSS bytes:", result)

    def test_capture_file_reads_and_labels_content(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "capture.log"
            file_path.write_text("file content", encoding="utf-8")

            result = server.capture_file(str(file_path))

        self.assertIn("Captured into ID", result)
        self.assertIn("capture.log", result)

    def test_capture_text_formats_diff_metadata(self):
        diff = """diff --git a/old.txt b/new.txt
--- a/old.txt
+++ b/new.txt
@@ -1 +1 @@
-old
+new
"""
        result = server.capture_text(diff, label="patch", content_type="diff")

        self.assertIn("Unified Diff", result)
        self.assertIn("new.txt", result)

    def test_execute_reports_failed_and_truncated_command(self):
        output = "x" * 700
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(output, 7, True, 700, False),
        ):
            result = server.execute_and_capture("failing-command", max_output_bytes=1024)

        self.assertIn("FAILED (Exit Code 7)", result)
        self.assertIn("truncated from 700 bytes", result)

    def test_search_and_read_tools_report_missing_and_empty_results(self):
        self.assertIn("Search Error", server.search_capture("query", capture_id="missing"))
        with patch.object(
            server.engine,
            "search",
            return_value={"status": "ok", "capture_id": "cap", "label": "label", "matches": []},
        ):
            self.assertIn("No matches found", server.search_capture("query"))
        self.assertIn("Error:", server.get_capture_slice(1, 1, capture_id="missing"))
        self.assertIn("Error:", server.get_capture_summary(capture_id="missing"))

    def test_search_and_read_tools_format_matches_and_diff_summary(self):
        with patch.object(
            server.engine,
            "search",
            return_value={
                "status": "ok",
                "mode": "bm25",
                "capture_id": "cap",
                "label": "label",
                "total_lines": 2,
                "matches": [{
                    "score": 1.0,
                    "matched_range": "1-1",
                    "context_range": "1-2",
                    "snippet": "match",
                }],
            },
        ):
            result = server.search_capture("query", mode="bm25")
        self.assertIn("Match #1", result)
        self.assertIn("match", result)

        with patch.object(
            server.engine,
            "get_slice",
            return_value={
                "status": "ok", "capture_id": "cap", "label": "label",
                "start_line": 1, "end_line": 1, "total_lines": 1, "content": "line",
            },
        ):
            self.assertIn("Lines 1 to 1", server.get_capture_slice(1, 1))
        with patch.object(
            server.engine,
            "get_summary",
            return_value={
                "status": "ok", "capture_id": "cap", "label": "diff",
                "content_type": "diff", "file_map": "new.txt [MODIFIED]",
                "diff_stats": "1 file", "timestamp": 1, "total_lines": 3,
                "byte_size": 10, "truncated": True, "original_byte_size": 20,
                "signals_summary": "None (Clean patch)",
            },
        ):
            summary = server.get_capture_summary("cap")
        self.assertIn("Modified Files Map", summary)
        self.assertIn("truncated from 20 bytes", summary)

    def test_empty_and_populated_capture_listing(self):
        self.assertIn("buffer is empty", server.list_captures())
        server.capture_text("one line", label="listed")
        self.assertIn("listed", server.list_captures())

    def test_buffer_stats_handles_unavailable_memory_metrics(self):
        with patch.object(
            server.engine,
            "get_buffer_stats",
            return_value={
                "capture_count": 0, "max_captures": 25, "total_bytes": 0,
                "max_buffer_bytes": 50, "total_lines": 0, "total_chunks": 0,
                "embedding_model": "model", "embedding_model_loaded": False,
                "embedding_cache_dir": None, "embedding_bytes": 0,
                "accounted_bytes": 0, "process_rss_bytes": None,
                "unaccounted_rss_bytes": None,
            },
        ):
            result = server.get_buffer_stats()
        self.assertIn("Process RSS: unavailable", result)
        self.assertIn("Unaccounted RSS bytes: unavailable", result)


class TestServerSocket(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes
        self.original_model = server.engine.embedding_model
        server.engine.max_buffer_bytes = 1024
        server.engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[0.0] * 384 for _ in texts]},
        )()

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
        server.engine.embedding_model = self.original_model
        server.engine.clear("all")

    async def run_handler(self, payload):
        writer = FakeWriter()
        server.handle_socket_client(FakeReader(payload), writer)
        for _ in range(100):
            if writer.closed or writer.writes:
                break
            await asyncio.sleep(0.01)
        return writer

    async def test_json_payload_returns_success_response(self):
        payload = json.dumps({"label": "socket-test", "text": "hello"}).encode()

        writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "socket-test")
        self.assertTrue(writer.closed)

    async def test_ingest_is_offloaded_from_event_loop(self):
        payload = json.dumps({"label": "offload-test", "text": "hello"}).encode()
        capture = SimpleNamespace(
            capture_id="cap_offload",
            label="offload-test",
            line_count=1,
            byte_size=5,
        )
        with patch.object(
            server.asyncio,
            "to_thread",
            new=AsyncMock(return_value=capture),
        ) as offload:
            await self.run_handler(payload)

        offload.assert_awaited_once()

    async def test_oversized_payload_returns_error_response(self):
        payload = b"x" * (server.engine.max_buffer_bytes + server.SOCKET_PAYLOAD_OVERHEAD)

        writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "error")
        self.assertIn("exceeds", response["message"])
        self.assertTrue(writer.closed)

    async def test_empty_payload_closes_without_response(self):
        writer = await self.run_handler(b"")

        self.assertEqual(writer.writes, [])
        self.assertTrue(writer.closed)

    async def test_malformed_payload_falls_back_to_plain_text(self):
        writer = await self.run_handler(b"not-json")

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "CLI pipe")

    async def test_ingest_failure_returns_error_response(self):
        with patch.object(server.engine, "ingest", side_effect=ValueError("invalid capture")):
            writer = await self.run_handler(json.dumps({"text": "payload"}).encode())

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "error")
        self.assertIn("invalid capture", response["message"])


class TestSocketServerStartup(unittest.TestCase):
    def test_live_socket_is_not_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(socket_path)
            except PermissionError as exc:
                listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            listener.listen()
            try:
                class FailingLoop:
                    def close(self):
                        pass

                    def run_until_complete(self, coroutine):
                        coroutine.close()
                        raise AssertionError("startup should stop before event loop execution")

                with patch.object(server, "SOCKET_PATH", socket_path), \
                        patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                        patch.object(server.asyncio, "set_event_loop"), \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    server.run_socket_server()
                self.assertIn("Socket already in use", stderr.getvalue())
            finally:
                listener.close()

            self.assertTrue(os.path.exists(socket_path))

    def test_stale_socket_is_removed_before_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                stale_listener.bind(socket_path)
            except PermissionError as exc:
                stale_listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            stale_listener.close()

            class FailingLoop:
                def close(self):
                    pass

                def run_until_complete(self, coroutine):
                    coroutine.close()
                    raise RuntimeError("socket unavailable")

            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

            self.assertFalse(os.path.exists(socket_path))

    def test_startup_failure_is_reported(self):
        class FailingLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                coroutine.close()
                raise RuntimeError("socket unavailable")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, "SOCKET_PATH", os.path.join(directory, "ephemeral.sock")), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        self.assertIn("Socket server error: socket unavailable", stderr.getvalue())

    def test_socket_probe_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "not-a-socket")
            Path(socket_path).write_text("occupied", encoding="utf-8")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        self.assertIn("Unable to verify existing socket", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
