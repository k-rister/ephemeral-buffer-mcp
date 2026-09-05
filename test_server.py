"""Unit tests for MCP tool validation, response formatting, and socket IPC."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

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

    def test_execute_rejects_limit_above_buffer_budget(self):
        server.engine.max_buffer_bytes = 1024

        result = server.execute_and_capture("printf output", max_output_bytes=1025)

        self.assertIn("max_output_bytes", result)
        self.assertIn("exceeds the configured buffer limit", result)

    def test_execute_rejects_too_small_limit(self):
        result = server.execute_and_capture("printf output", max_output_bytes=100)

        self.assertIn("at least 512", result)

    def test_buffer_stats_formats_memory_metrics(self):
        server.capture_text("server stats payload", label="server-test")

        result = server.get_buffer_stats()

        self.assertIn("Captures:", result)
        self.assertIn("Embedding bytes:", result)
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


if __name__ == "__main__":
    unittest.main()
