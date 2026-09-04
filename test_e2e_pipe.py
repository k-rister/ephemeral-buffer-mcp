"""
Test the end-to-end socket IPC and CLI piping against the running server.
"""

import os
import socket
import sys
import time
import subprocess
import unittest
import json
from pathlib import Path
from config import socket_path

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"
CLI_PATH = PROJECT_ROOT / "cli.py"
SOCKET_PATH = socket_path()


class TestEndToEndPipe(unittest.TestCase):
    def test_pipe_to_socket(self):
        # Start server as a subprocess
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "EPHEMERAL_MAX_BUFFER_BYTES": "1024"},
        )

        try:
            # Wait for socket to appear
            for _ in range(50):
                if os.path.exists(SOCKET_PATH):
                    break
                time.sleep(0.1)

            self.assertTrue(os.path.exists(SOCKET_PATH), "Socket should be created by server")

            # Pipe a simulated big test output to agy-cap
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
                text=True
            )

            print("\nCLI output:", pipe_proc.stderr)
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
                text=True
            )
            self.assertEqual(bounded_proc.returncode, 0)
            self.assertIn("Successfully captured", bounded_proc.stderr)
            self.assertIn("bounded-pipe", bounded_proc.stderr)

            oversized_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            oversized_socket.connect(SOCKET_PATH)
            try:
                oversized_socket.sendall(b"x" * (1024 + 64 * 1024 + 1))
            except BrokenPipeError:
                # The server may close as soon as it observes the bounded read.
                pass
            response = oversized_socket.recv(4096).decode("utf-8")
            oversized_socket.close()
            self.assertIn('"status": "error"', response)
            self.assertIn("exceeds", response)

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            if os.path.exists(SOCKET_PATH):
                try:
                    os.unlink(SOCKET_PATH)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
