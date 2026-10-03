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
from socket_protocol import FRAME_HEADER_SIZE, decode_header, encode_frame
import socket_protocol
from config import load_settings

import cli

CLI_PATH = Path(__file__).with_name("cli.py")
TEST_ENV_WRAPPER = Path(__file__).with_name("scripts") / "with-test-env.sh"
SESSION_ENV_HELPER = Path(__file__).with_name("ephemeral-session-env")


class TestCliConfiguration(unittest.TestCase):
    def test_test_environment_wrapper_overrides_active_session_paths(self):
        probe = (
            "import cli, json, os, server; "
            "from config import execution_state_dir, socket_path; "
            "print(json.dumps({"
            "'session_id': os.environ['EPHEMERAL_SESSION_ID'], "
            "'socket_path': socket_path(), 'cli_socket_path': cli.SOCKET_PATH, "
            "'server_socket_path': server.SOCKET_PATH, "
            "'execution_state_dir': execution_state_dir(), "
            "'server_execution_state_dir': str(server.execution_manager.store.state_dir), "
            "'metrics_file': os.environ['EPHEMERAL_METRICS_FILE'], "
            "'log_file': os.environ['EPHEMERAL_LOG_FILE'], "
            "'require_isolation': os.environ['EPHEMERAL_REQUIRE_ISOLATION']}))"
        )
        environment = {
            **os.environ,
            "EPHEMERAL_SESSION_ID": "active-session",
            "EPHEMERAL_SOCKET_PATH": "/tmp/active-session.sock",
            "EPHEMERAL_EXECUTION_STATE_DIR": "/tmp/active-session-executions",
            "EPHEMERAL_METRICS_FILE": "/tmp/active-session-metrics.json",
            "EPHEMERAL_LOG_FILE": "/tmp/active-session.jsonl",
            "EPHEMERAL_REQUIRE_ISOLATION": "0",
        }
        result = subprocess.run(
            [str(TEST_ENV_WRAPPER), sys.executable, "-c", probe],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)
        runtime_dir = Path(paths["socket_path"]).parent
        self.assertEqual(paths["socket_path"], paths["cli_socket_path"])
        self.assertEqual(paths["socket_path"], paths["server_socket_path"])
        self.assertTrue(runtime_dir.name.startswith("ephemeral-buffer-test."))
        self.assertTrue(paths["session_id"].startswith("test-"))
        self.assertEqual(Path(paths["execution_state_dir"]).parent, runtime_dir)
        self.assertEqual(Path(paths["server_execution_state_dir"]).parent, runtime_dir)
        self.assertEqual(Path(paths["metrics_file"]).parent, runtime_dir)
        self.assertEqual(Path(paths["log_file"]).parent, runtime_dir)
        self.assertEqual(paths["require_isolation"], "1")
        self.assertFalse(runtime_dir.exists())

    def test_session_helper_keeps_explicit_socket_identity_without_minting_session(self):
        environment = os.environ.copy()
        environment.pop("EPHEMERAL_SESSION_ID", None)
        environment.pop("EPHEMERAL_EXECUTION_STATE_DIR", None)
        environment.pop("EPHEMERAL_LOG_FILE", None)
        environment["EPHEMERAL_SOCKET_PATH"] = "relative/fixed-eb-socket.sock"
        expected_socket = os.path.abspath(environment["EPHEMERAL_SOCKET_PATH"])
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\n%s\\n%s\\n" "${EPHEMERAL_SESSION_ID:-<unset>}" "$EPHEMERAL_SOCKET_PATH" "$EPHEMERAL_LOG_FILE"',
                "bash",
                str(SESSION_ENV_HELPER),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        session_id, socket_path, log_file = result.stdout.splitlines()
        self.assertEqual(session_id, "<unset>")
        self.assertEqual(socket_path, expected_socket)
        self.assertEqual(log_file, f"{expected_socket.removesuffix('.sock')}.jsonl")

    def test_recv_exact_rejects_truncated_response(self):
        class EmptySocket:
            def recv(self, _size):
                return b""

        with self.assertRaisesRegex(ValueError, "truncated socket response"):
            cli._recv_exact(EmptySocket(), 1)

    def test_socket_frame_round_trip_and_validation(self):
        payload = b"fragmented payload"
        frame = encode_frame(payload)
        self.assertEqual(decode_header(frame[:FRAME_HEADER_SIZE]), len(payload))
        with self.assertRaisesRegex(ValueError, "truncated frame header"):
            decode_header(frame[:2])
        with self.assertRaisesRegex(ValueError, "invalid frame magic"):
            decode_header(b"XXXX" + frame[4:FRAME_HEADER_SIZE])
        with self.assertRaisesRegex(ValueError, "unsupported frame version"):
            decode_header(frame[:4] + b"\x02" + frame[5:FRAME_HEADER_SIZE])

    def test_socket_frame_rejects_invalid_payloads(self):
        with self.assertRaises(TypeError):
            encode_frame("text")
        with patch.object(socket_protocol, "MAX_FRAME_LENGTH", 1):
            with self.assertRaisesRegex(ValueError, "too large"):
                encode_frame(b"12")

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
        with patch.object(cli, "SETTINGS", load_settings({"EPHEMERAL_REQUIRE_ISOLATION": "1"})), \
                patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"):
            result = cli.send_to_mcp("output")

        self.assertEqual(result["status"], "error")
        self.assertIn("Socket isolation is required", result["message"])

    def test_send_to_mcp_serializes_payload_and_reads_response(self):
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.closed = False
                self.response = bytearray(encode_frame(b'{"status":"ok"}'))

            def settimeout(self, value):
                self.timeout = value

            def connect(self, path):
                self.path = path

            def sendall(self, payload):
                self.sent = payload

            def recv(self, size):
                chunk = bytes(self.response[:size])
                del self.response[:size]
                return chunk

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
        self.assertTrue(fake_socket.closed)
        sent_length = decode_header(fake_socket.sent[:FRAME_HEADER_SIZE])
        self.assertEqual(
            json.loads(fake_socket.sent[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + sent_length]),
            {
                "label": "build",
                "text": "output",
                "content_type": "log",
                "truncated": True,
                "original_byte_size": 100,
                "command_exit_code": None,
                "timed_out": False,
                "duration_ms": None,
            },
        )

    def test_large_socket_frame_stops_upload_when_busy_response_arrives_first(self):
        client_socket, server_socket = socket.socketpair()
        busy_frame = encode_frame(
            b'{"status":"error","code":"server_busy","message":"retry"}'
        )
        try:
            server_socket.sendall(busy_frame)

            class ConnectedSocket:
                def __init__(self, wrapped):
                    self.wrapped = wrapped
                    self.sent = 0

                def __getattr__(self, name):
                    return getattr(self.wrapped, name)

                def connect(self, _path):
                    return None

                def send(self, payload):
                    self.sent += len(payload)
                    return self.wrapped.send(payload)

            connected_socket = ConnectedSocket(client_socket)
            with patch.object(cli, "SOCKET_PATH", "/tmp/ephemeral.sock"), \
                    patch.object(cli.os.path, "exists", return_value=True), \
                    patch.object(cli.socket, "socket", return_value=connected_socket):
                result = cli.send_to_mcp("x" * 4096)

            self.assertEqual(result["code"], "server_busy")
            self.assertEqual(connected_socket.sent, 0)
        finally:
            client_socket.close()
            server_socket.close()

    def test_large_frame_selector_handles_partial_io_and_blocking(self):
        class FakeSelector:
            def __init__(self):
                self.events = [
                    cli.selectors.EVENT_READ | cli.selectors.EVENT_WRITE,
                    cli.selectors.EVENT_WRITE,
                    cli.selectors.EVENT_READ,
                    cli.selectors.EVENT_READ,
                    cli.selectors.EVENT_READ,
                ]
                self.modified = []

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return False

            def register(self, *_args):
                return None

            def modify(self, _fileobj, events):
                self.modified.append(events)

            def select(self, _timeout):
                mask = self.events.pop(0)
                return [(None, mask)]

        response_frame = encode_frame(b'{"status":"ok"}')

        class FakeSocket:
            def __init__(self):
                self.recv_chunks = [
                    BlockingIOError(),
                    response_frame[:3],
                    response_frame[3:FRAME_HEADER_SIZE + 2],
                    response_frame[FRAME_HEADER_SIZE + 2:],
                ]
                self.send_calls = 0
                self.sent = bytearray()

            def setblocking(self, _value):
                return None

            def recv(self, _size):
                chunk = self.recv_chunks.pop(0)
                if isinstance(chunk, Exception):
                    raise chunk
                return chunk

            def send(self, payload):
                self.send_calls += 1
                if self.send_calls == 1:
                    raise BlockingIOError
                self.sent.extend(payload)
                return len(payload)

        fake_selector = FakeSelector()
        fake_socket = FakeSocket()
        with patch.object(cli.selectors, "DefaultSelector", return_value=fake_selector):
            result = cli._send_large_frame_with_early_response(fake_socket, b"z" * 2048)

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(fake_socket.sent, b"z" * 2048)
        self.assertEqual(fake_selector.modified, [cli.selectors.EVENT_READ])

    def test_large_frame_selector_handles_early_response_eof_and_timeout(self):
        class FakeSelector:
            def __init__(self, events):
                self.events = events
                self.select_calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return False

            def register(self, *_args):
                return None

            def modify(self, *_args):
                return None

            def select(self, _timeout):
                self.select_calls += 1
                if not self.events:
                    return []
                return [(None, self.events.pop(0))]

        class FakeSocket:
            def __init__(self, chunks):
                self.chunks = list(chunks)
                self.sent = 0

            def setblocking(self, _value):
                return None

            def recv(self, _size):
                chunk = self.chunks.pop(0)
                if isinstance(chunk, Exception):
                    raise chunk
                return chunk

            def send(self, payload):
                self.sent += len(payload)
                return len(payload)

        busy_frame = encode_frame(b'{"status":"error","code":"server_busy"}')
        selector = FakeSelector([cli.selectors.EVENT_READ | cli.selectors.EVENT_WRITE])
        early_socket = FakeSocket([busy_frame])
        with patch.object(cli.selectors, "DefaultSelector", return_value=selector):
            early = cli._send_large_frame_with_early_response(early_socket, b"z" * 2048)
        self.assertEqual(early["code"], "server_busy")
        self.assertEqual(early_socket.sent, 0)

        eof_selector = FakeSelector([cli.selectors.EVENT_READ])
        with patch.object(cli.selectors, "DefaultSelector", return_value=eof_selector):
            with self.assertRaisesRegex(ValueError, "truncated socket response"):
                cli._send_large_frame_with_early_response(FakeSocket([b""]), b"z" * 2048)

        timeout_selector = FakeSelector([])
        with patch.object(cli.selectors, "DefaultSelector", return_value=timeout_selector):
            with self.assertRaises(socket.timeout):
                cli._send_large_frame_with_early_response(FakeSocket([]), b"z" * 2048)

        expired_selector = FakeSelector([cli.selectors.EVENT_WRITE])
        with patch.object(cli.selectors, "DefaultSelector", return_value=expired_selector), \
                patch.object(
                    cli, "SETTINGS", load_settings({"EPHEMERAL_SOCKET_TIMEOUT_SECONDS": "1"})
                ), \
                patch.object(cli.time, "monotonic", side_effect=[10.0, 11.1]):
            with self.assertRaises(socket.timeout):
                cli._send_large_frame_with_early_response(FakeSocket([]), b"z" * 2048)
        self.assertEqual(expired_selector.select_calls, 0)

    def test_large_frame_decoder_waits_for_header_and_complete_payload(self):
        frame = encode_frame(b'{"status":"ok"}')
        self.assertIsNone(cli._decode_available_response(bytearray(frame[:3])))
        self.assertIsNone(cli._decode_available_response(bytearray(frame[:-1])))
        self.assertEqual(cli._decode_available_response(bytearray(frame)), {"status": "ok"})

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
        with patch.object(
                cli, "SETTINGS", load_settings({"EPHEMERAL_SOCKET_TIMEOUT_SECONDS": "1.5"})
        ), \
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

    def test_invalid_binary_stdin_is_bounded_after_decoding(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_binary", "label": "Piped STDIN"}
        stdin = SimpleNamespace(isatty=lambda: False, buffer=io.BytesIO(b"\xff" * 600))
        with patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--max-output-bytes", "512"]), \
                patch.object(sys, "stdin", stdin):
            cli.main()

        payload = send.call_args.args[0]
        self.assertLessEqual(len(payload.encode("utf-8")), 512)
        self.assertTrue(send.call_args.kwargs["truncated"])
        self.assertEqual(send.call_args.kwargs["original_byte_size"], 600)

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
        response = {
            "status": "ok",
            "line_count": 1,
            "capture_id": "cap_test",
            "label": "build",
            "summary": {"schema_version": 1, "execution_status": "captured"},
        }
        with patch.object(cli, "run_command_bounded", return_value=("command output", 3, False, 14, False)) as run, \
                patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--label", "build", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 3)
        run.assert_called_once_with("echo ok", None, cli.SETTINGS.max_buffer_bytes.value, None)
        self.assertEqual(send.call_args.kwargs["label"], "build")
        self.assertIsInstance(send.call_args.kwargs["duration_ms"], float)
        self.assertIn('"schema_version":1', stderr.getvalue())

    def test_wrapped_command_forwards_execution_duration(self):
        response = {"status": "ok", "line_count": 1, "capture_id": "cap_test", "label": "build"}
        with patch.object(cli, "run_command_bounded", return_value=("output", 0, False, 6, False)), \
                patch.object(cli.time, "perf_counter", side_effect=[10.0, 10.25]), \
                patch.object(cli, "send_to_mcp", return_value=response) as send, \
                patch.object(sys, "argv", ["cli.py", "--", "echo", "ok"]), \
                patch.object(sys, "stdout", io.StringIO()), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 0)
        self.assertEqual(send.call_args.kwargs["duration_ms"], 250.0)

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
        run.assert_called_once_with(
            shlex.join(command), None, cli.SETTINGS.max_buffer_bytes.value, None
        )

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

    def test_stdin_reports_invalid_output_limit_without_traceback(self):
        with patch.object(
            sys,
            "argv",
            ["cli.py", "--max-output-bytes", "511"],
        ), patch.object(sys, "stdin", io.StringIO("input")), \
                patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_result:
                cli.main()

        self.assertEqual(exit_result.exception.code, 2)
        self.assertIn("must be at least 512", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
