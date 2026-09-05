"""CLI configuration regression tests."""

import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cli

CLI_PATH = Path(__file__).with_name("cli.py")


class TestCliConfiguration(unittest.TestCase):
    def test_invalid_buffer_environment_does_not_break_cli(self):
        environment = os.environ.copy()
        environment["EPHEMERAL_MAX_BUFFER_BYTES"] = "not-an-integer"
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--max-output-bytes", result.stdout)
        self.assertIn("Ignoring invalid EPHEMERAL_MAX_BUFFER_BYTES", result.stderr)

    def test_send_to_mcp_reports_missing_socket(self):
        with patch.object(cli, "SOCKET_PATH", "/tmp/does-not-exist.sock"):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("socket not found", result["message"])

    def test_send_to_mcp_serializes_payload_and_reads_response(self):
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.shutdown_mode = None
                self.closed = False
                self.responses = [b'{"status":"ok"}', b""]

            def connect(self, path):
                self.path = path

            def sendall(self, payload):
                self.sent = payload

            def shutdown(self, mode):
                self.shutdown_mode = mode

            def recv(self, _size):
                return self.responses.pop(0)

            def close(self):
                self.closed = True

        fake_socket = FakeSocket()
        with patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"), \
                patch.object(cli.os.path, "exists", return_value=True), \
                patch.object(cli.socket, "socket", return_value=fake_socket):
            result = cli.send_to_mcp(
                "output", label="build", content_type="log", truncated=True, original_byte_size=100
            )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(fake_socket.path, "/tmp/ephemeral.sock")
        self.assertEqual(fake_socket.shutdown_mode, cli.socket.SHUT_WR)
        self.assertTrue(fake_socket.closed)
        self.assertEqual(
            json.loads(fake_socket.sent),
            {
                "label": "build",
                "text": "output",
                "content_type": "log",
                "truncated": True,
                "original_byte_size": 100,
            },
        )

    def test_send_to_mcp_reports_transport_failure(self):
        with patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"), \
                patch.object(cli.os.path, "exists", return_value=True), \
                patch.object(cli.socket, "socket", side_effect=OSError("connection refused")):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("connection refused", result["message"])

    def test_stdin_capture_forwards_truncation_metadata(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "Piped STDIN"}
        with patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--max-output-bytes", "1024"]), \
                patch.object(sys, "stdin", io.StringIO("x" * 2000)):
            cli.main()

        payload = send.call_args.args[0]
        kwargs = send.call_args.kwargs
        self.assertTrue(kwargs["truncated"])
        self.assertEqual(kwargs["original_byte_size"], 2000)
        self.assertLessEqual(len(payload.encode("utf-8")), 1024)

    def test_wrapped_command_forwards_exit_code_and_label(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "build"}
        with patch.object(cli, "run_command_bounded", return_value=("command output", 3, False, 14)) as run, \
                patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--label", "build", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 3)
        run.assert_called_once_with("echo ok", None, cli.DEFAULT_MAX_OUTPUT_BYTES)
        self.assertEqual(send.call_args.kwargs["label"], "build")

    def test_wrapped_command_reports_capture_warning(self):
        response = {"status": "error", "message": "socket unavailable"}
        with patch.object(cli, "run_command_bounded", return_value=("output", 0, False, 6)), \
                patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 0)
        self.assertIn("socket unavailable", stderr.getvalue())

    def test_wrapped_command_reports_invalid_output_limit(self):
        with patch.object(cli, "run_command_bounded", side_effect=ValueError("limit too small")), \
                patch.object(sys, "argv", ["cli.py", "--", "echo", "ok"]):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
