"""CLI configuration regression tests."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
