"""CLI configuration regression tests."""

import io
import json
import os
import runpy
import socket
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cli

CLI_PATH = Path(__file__).with_name("cli.py")


class TestCliConfiguration(unittest.TestCase):
    def _launcher_fixture(self, environment_name):
        root = Path(tempfile.mkdtemp())
        launcher = root / "run.sh"
        shutil.copy(Path(__file__).with_name("run.sh"), launcher)
        launcher.chmod(0o755)
        interpreter = root / environment_name / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text('#!/bin/sh\nprintf \'%s\' "$MARKER"\n', encoding="utf-8")
        interpreter.chmod(0o755)
        return root, launcher

    def test_launcher_prefers_documented_dot_venv(self):
        root, launcher = self._launcher_fixture(".venv")
        legacy = root / "venv" / "bin" / "python"
        legacy.parent.mkdir(parents=True)
        shutil.copy(root / ".venv/bin/python", legacy)
        marker = root / "selected"
        try:
            result = subprocess.run(
                [str(launcher)],
                env={**os.environ, "MARKER": str(marker), "PYTHON": "/invalid/python"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, str(marker))
        finally:
            shutil.rmtree(root)

    def test_launcher_retains_legacy_venv_compatibility(self):
        root, launcher = self._launcher_fixture("venv")
        marker = root / "selected"
        try:
            result = subprocess.run(
                [str(launcher)],
                env={**os.environ, "MARKER": str(marker), "PYTHON": "/invalid/python"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, str(marker))
        finally:
            shutil.rmtree(root)

    def test_launcher_rejects_invalid_explicit_python(self):
        root = Path(tempfile.mkdtemp())
        launcher = root / "run.sh"
        shutil.copy(Path(__file__).with_name("run.sh"), launcher)
        launcher.chmod(0o755)
        try:
            result = subprocess.run(
                [str(launcher)],
                env={**os.environ, "PYTHON": "/invalid/python"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 127)
            self.assertIn("Configured PYTHON interpreter was not found", result.stderr)
        finally:
            shutil.rmtree(root)

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

    def test_send_to_mcp_rejects_unisolated_strict_mode(self):
        with patch.dict(os.environ, {"EPHEMERAL_REQUIRE_ISOLATION": "1"}, clear=True), \
                patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("Socket isolation is required", result["message"])

    def test_send_to_mcp_serializes_payload_and_reads_response(self):
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.shutdown_mode = None
                self.closed = False
                self.responses = [b'{"status":"ok"}', b""]

            def settimeout(self, value):
                self.timeout = value

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
                "command_exit_code": None,
                "timed_out": False,
            },
        )

    def test_send_to_mcp_reports_transport_failure(self):
        with patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"), \
                patch.object(cli.os.path, "exists", return_value=True), \
                patch.object(cli.socket, "socket", side_effect=OSError("connection refused")):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("connection refused", result["message"])

    def test_send_to_mcp_applies_socket_timeout_and_closes_on_timeout(self):
        class StalledSocket:
            def settimeout(self, value):
                self.timeout = value

            def connect(self, _path):
                pass

            def sendall(self, _payload):
                pass

            def shutdown(self, _mode):
                pass

            def recv(self, _size):
                raise socket.timeout()

            def close(self):
                self.closed = True

        stalled = StalledSocket()
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_TIMEOUT_SECONDS": "1.5"}, clear=True), \
                patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"), \
                patch.object(cli.os.path, "exists", return_value=True), \
                patch.object(cli.socket, "socket", return_value=stalled):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("Timed out communicating", result["message"])
        self.assertEqual(stalled.timeout, 1.5)
        self.assertTrue(stalled.closed)

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

    def test_binary_stdin_capture_uses_buffer(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_binary", "label": "Piped STDIN"}
        stdin = SimpleNamespace(isatty=lambda: False, buffer=io.BytesIO(b"binary input"))
        with patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py"]), \
                patch.object(sys, "stdin", stdin):
            cli.main()

        self.assertEqual(send.call_args.args[0], "binary input")

    def test_wrapped_command_reports_timeout(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_timeout", "label": "sleep"}
        with patch.object(cli, "run_command_bounded", return_value=("partial", 124, False, 7, True)), \
                patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py", "--timeout-seconds", "0.5", "--", "sleep", "10"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 124)
        self.assertIn("timed out after 0.5s", stderr.getvalue())

    def test_stdin_capture_reports_warning(self):
        response = {"status": "error", "message": "socket unavailable"}
        with patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py"]), \
                patch.object(sys, "stdin", io.StringIO("input")), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 1)
        self.assertIn("socket unavailable", stderr.getvalue())

    def test_tty_without_command_prints_help(self):
        stdin = SimpleNamespace(isatty=lambda: True)
        with patch.object(sys, "argv", ["cli.py"]), \
                patch.object(sys, "stdin", stdin), \
                patch.object(sys, "stdout", io.StringIO()) as stdout:
            cli.main()

        self.assertIn("Pipe output into Ephemeral Buffer", stdout.getvalue())

    def test_module_entrypoint_runs_main(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", [str(CLI_PATH)]), \
                patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)), \
                patch.object(sys, "stdout", stdout):
            runpy.run_path(str(CLI_PATH), run_name="__main__")

        self.assertIn("Pipe output into Ephemeral Buffer", stdout.getvalue())

    def test_wrapped_command_forwards_exit_code_and_label(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "build"}
        with patch.object(cli, "run_command_bounded", return_value=("command output", 3, False, 14, False)) as run, \
                patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--label", "build", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 3)
        run.assert_called_once_with("echo ok", None, cli.DEFAULT_MAX_OUTPUT_BYTES, None)
        self.assertEqual(send.call_args.kwargs["label"], "build")

    def test_wrapped_command_preserves_argument_boundaries(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "build"}
        command = [sys.executable, "-c", "import sys; print(repr(sys.argv[1:]));", "two words", "$(printf unsafe)"]
        with patch.object(cli, "run_command_bounded", return_value=("['two words', '$(printf unsafe)']\n", 0, False, 37, False)) as run, \
                patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py", "--", *command]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 0)
        run.assert_called_once_with(shlex.join(command), None, cli.DEFAULT_MAX_OUTPUT_BYTES, None)

    def test_wrapped_command_does_not_reinterpret_literal_command_substitution(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "build"}
        command = [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(printf unsafe)"]
        with patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py", "--", *command]), \
                patch.object(sys, "stdout", io.StringIO()) as stdout, \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 0)
        self.assertIn("$(printf unsafe)", stdout.getvalue())

    def test_wrapped_command_reports_capture_warning(self):
        response = {"status": "error", "message": "socket unavailable"}
        with patch.object(cli, "run_command_bounded", return_value=("output", 0, False, 6, False)), \
                patch.object(cli, "send_to_mcp", return_value=response), \
                patch.object(sys, "argv", ["cli.py", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 1)
        self.assertIn("socket unavailable", stderr.getvalue())

    def test_wrapped_command_reports_invalid_output_limit(self):
        with patch.object(cli, "run_command_bounded", side_effect=ValueError("limit too small")), \
                patch.object(sys, "argv", ["cli.py", "--", "echo", "ok"]):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
