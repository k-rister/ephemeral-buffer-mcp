"""Unit tests for MCP tool validation, response formatting, and socket IPC."""

import asyncio
import io
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class TestServerSocket(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes
        server.engine.max_buffer_bytes = 1024

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
        server.engine.clear("all")

    async def run_handler(self, payload):
        writer = FakeWriter()
        server.handle_socket_client(FakeReader(payload), writer)
        await asyncio.sleep(0.01)
        return writer

    async def test_json_payload_returns_success_response(self):
        payload = json.dumps({"label": "socket-test", "text": "hello"}).encode()

        writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "socket-test")
        self.assertTrue(writer.closed)

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
            listener.bind(socket_path)
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
            stale_listener.bind(socket_path)
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


if __name__ == "__main__":
    unittest.main()
