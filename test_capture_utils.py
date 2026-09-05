"""Tests for bounded command-output capture."""

import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from capture_utils import bound_chunks, read_file_bounded, run_command_bounded


class TestBoundedCommandCapture(unittest.TestCase):
    def test_large_output_keeps_head_and_tail(self):
        command = (
            f"{shlex.quote(sys.executable)} -c "
            "\"import sys; sys.stdout.write('HEAD' * 1000); sys.stdout.write('TAIL')\""
        )
        output, exit_code, truncated, original_size, timed_out = run_command_bounded(command, None, 1024)

        self.assertEqual(exit_code, 0)
        self.assertTrue(truncated)
        self.assertFalse(timed_out)
        self.assertGreater(original_size, len(output.encode("utf-8")))
        self.assertLessEqual(len(output.encode("utf-8")), 1024)
        self.assertIn("output truncated", output)
        self.assertIn("TAIL", output)

    def test_small_output_is_complete(self):
        command = f"{shlex.quote(sys.executable)} -c \"print('complete')\""
        output, exit_code, truncated, original_size, timed_out = run_command_bounded(command, None, 1024)

        self.assertEqual(exit_code, 0)
        self.assertFalse(truncated)
        self.assertFalse(timed_out)
        self.assertEqual(original_size, len(output.encode("utf-8")))
        self.assertEqual(output, "complete\n")

    def test_command_timeout_terminates_process_group(self):
        command = f"{shlex.quote(sys.executable)} -c \"import time; print('started', flush=True); time.sleep(10)\""
        output, exit_code, truncated, original_size, timed_out = run_command_bounded(
            command, None, 1024, timeout_seconds=0.1
        )

        self.assertEqual(exit_code, 124)
        self.assertTrue(timed_out)
        self.assertIn("started", output)
        self.assertFalse(truncated)
        self.assertGreater(original_size, 0)

    def test_timeout_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "timeout_seconds"):
            run_command_bounded("true", None, 1024, timeout_seconds=0)

    def test_invalid_output_limit_does_not_start_process(self):
        with patch("capture_utils.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "max_output_bytes"):
                run_command_bounded("echo should-not-run", None, 100)

        popen.assert_not_called()

    def test_stream_chunks_are_bounded(self):
        output, truncated, original_size = bound_chunks(
            [b"A" * 700, b"B" * 700],
            1024,
        )

        self.assertTrue(truncated)
        self.assertEqual(original_size, 1400)
        self.assertLessEqual(len(output.encode("utf-8")), 1024)
        self.assertIn("output truncated", output)

    def test_file_read_rejects_oversized_content(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            file_handle.write(b"x" * 1024)
            file_handle.flush()
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_file_bounded(file_handle.name, 512)


if __name__ == "__main__":
    unittest.main()
