"""Unit tests for MCP tool validation and response formatting."""

import tempfile
import unittest
from pathlib import Path

import server


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


if __name__ == "__main__":
    unittest.main()
