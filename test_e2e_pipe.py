"""
Test the end-to-end socket IPC and CLI piping against the running server.
"""

import os
import socket
import sys
import tempfile
import time
import subprocess
import unittest
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from server import SOCKET_JSON_MAX_EXPANSION, SOCKET_PAYLOAD_OVERHEAD
from socket_protocol import FRAME_HEADER_SIZE, decode_header, encode_frame

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"
CLI_PATH = PROJECT_ROOT / "cli.py"


def recv_exact(sock, size):
    chunks = []
    while sum(len(chunk) for chunk in chunks) < size:
        chunk = sock.recv(size - sum(len(part) for part in chunks))
        if not chunk:
            raise AssertionError("truncated framed response")
        chunks.append(chunk)
    return b"".join(chunks)


class TestEndToEndPipe(unittest.TestCase):
    @contextmanager
    def _isolated_server(self):
        with tempfile.TemporaryDirectory(prefix="ephemeral-buffer-e2e-") as temp_dir:
            temp_root = Path(temp_dir)
            isolated_socket_path = temp_root / "ephemeral-buffer.sock"
            execution_state_dir = temp_root / "executions"
            test_env = os.environ.copy()
            test_env.update({
                "EPHEMERAL_SOCKET_PATH": str(isolated_socket_path),
                "EPHEMERAL_EXECUTION_STATE_DIR": str(execution_state_dir),
                "EPHEMERAL_MAX_BUFFER_BYTES": "1024",
            })

            proc = subprocess.Popen(
                [sys.executable, str(SERVER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=test_env,
            )
            try:
                for _ in range(50):
                    if isolated_socket_path.exists():
                        break
                    time.sleep(0.1)

                self.assertTrue(
                    isolated_socket_path.exists(),
                    "Socket should be created by server",
                )
                yield str(isolated_socket_path), test_env
            finally:
                proc.terminate()
                try:
                    proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()

    def _exercise_pipe_to_socket(self):
        with self._isolated_server() as (isolated_socket_path, test_env):
            simulated_log = (
                "Running test suite: OrderService\n"
                "✓ test_create_order passed (12ms)\n"
                "✓ test_cancel_order passed (8ms)\n"
                "✗ test_refund_order failed:\n"
                "  Error: GatewayTimeout at PaymentProcessor.processRefund (payments.ts:89:15)\n"
                "  Caused by: upstream proxy 10.0.4.12 returned 504 Gateway Timeout\n"
                "Tests completed: 2 passed, 1 failed.\n"
            )

            pipe_proc = subprocess.run(
                [sys.executable, str(CLI_PATH), "--label", "order-tests"],
                input=simulated_log,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=test_env,
            )
            self.assertIn("Successfully captured", pipe_proc.stderr)
            self.assertIn("order-tests", pipe_proc.stderr)

            bounded_proc = subprocess.run(
                [
                    sys.executable,
                    CLI_PATH,
                    "--label",
                    "bounded-pipe",
                    "--max-output-bytes",
                    "1024",
                ],
                input="HEAD" * 1000 + "TAIL",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=test_env,
            )
            self.assertEqual(bounded_proc.returncode, 0)
            self.assertIn("Successfully captured", bounded_proc.stderr)
            self.assertIn("bounded-pipe", bounded_proc.stderr)

            oversized_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                oversized_socket.connect(isolated_socket_path)
                try:
                    oversized_socket.sendall(encode_frame(
                        b"x" * (
                            1024 * SOCKET_JSON_MAX_EXPANSION
                            + SOCKET_PAYLOAD_OVERHEAD
                            + 1
                        )
                    ))
                except BrokenPipeError:
                    # The server may close as soon as it observes the bounded read.
                    pass
                header = recv_exact(oversized_socket, FRAME_HEADER_SIZE)
                response_length = decode_header(header)
                response = recv_exact(oversized_socket, response_length).decode("utf-8")
            finally:
                oversized_socket.close()
            self.assertIn('"status": "error"', response)
            self.assertIn("exceed", response)

    def test_pipe_to_socket_preserves_inherited_live_socket(self):
        with tempfile.TemporaryDirectory(prefix="ephemeral-buffer-inherited-") as temp_dir:
            inherited_socket_path = Path(temp_dir) / "inherited.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as inherited_listener:
                inherited_listener.bind(str(inherited_socket_path))
                inherited_listener.listen()
                with patch.dict(
                    os.environ,
                    {"EPHEMERAL_SOCKET_PATH": str(inherited_socket_path)},
                ):
                    self._exercise_pipe_to_socket()

                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.connect(str(inherited_socket_path))
                finally:
                    probe.close()

    def test_pipe_to_socket_preserves_inherited_regular_file(self):
        original_contents = b"leave this inherited file alone\n"
        with tempfile.TemporaryDirectory(prefix="ephemeral-buffer-inherited-") as temp_dir:
            inherited_file_path = Path(temp_dir) / "inherited.sock"
            inherited_file_path.write_bytes(original_contents)
            with patch.dict(
                os.environ,
                {"EPHEMERAL_SOCKET_PATH": str(inherited_file_path)},
            ):
                self._exercise_pipe_to_socket()

            self.assertEqual(inherited_file_path.read_bytes(), original_contents)


if __name__ == "__main__":
    unittest.main()
