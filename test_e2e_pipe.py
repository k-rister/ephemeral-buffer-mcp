"""
Test the end-to-end socket IPC and CLI piping against the running server.
"""

import os
import sys
import time
import subprocess
import unittest
import json

SERVER_PATH = "/home/krister/antigravity/ephemeral-buffer-mcp/run.sh"
CLI_PATH = "/home/krister/antigravity/ephemeral-buffer-mcp/cli.py"
SOCKET_PATH = "/tmp/ephemeral_buffer.sock"


class TestEndToEndPipe(unittest.TestCase):
    def test_pipe_to_socket(self):
        # Start server as a subprocess
        proc = subprocess.Popen(
            [SERVER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
                [sys.executable, CLI_PATH, "--label", "order-tests"],
                input=simulated_log,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print("\nCLI output:", pipe_proc.stderr)
            self.assertIn("Successfully captured", pipe_proc.stderr)
            self.assertIn("order-tests", pipe_proc.stderr)

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
