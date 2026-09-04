"""Tests for bounded command-output capture."""

import shlex
import sys
import unittest

from capture_utils import bound_chunks, run_command_bounded


class TestBoundedCommandCapture(unittest.TestCase):
    def test_large_output_keeps_head_and_tail(self):
        command = (
            f"{shlex.quote(sys.executable)} -c "
            "\"import sys; sys.stdout.write('HEAD' * 1000); sys.stdout.write('TAIL')\""
        )
        output, exit_code, truncated, original_size = run_command_bounded(command, None, 1024)

        self.assertEqual(exit_code, 0)
        self.assertTrue(truncated)
        self.assertGreater(original_size, len(output.encode("utf-8")))
        self.assertLessEqual(len(output.encode("utf-8")), 1024)
        self.assertIn("output truncated", output)
        self.assertIn("TAIL", output)

    def test_small_output_is_complete(self):
        command = f"{shlex.quote(sys.executable)} -c \"print('complete')\""
        output, exit_code, truncated, original_size = run_command_bounded(command, None, 1024)

        self.assertEqual(exit_code, 0)
        self.assertFalse(truncated)
        self.assertEqual(original_size, len(output.encode("utf-8")))
        self.assertEqual(output, "complete\n")

    def test_stream_chunks_are_bounded(self):
        output, truncated, original_size = bound_chunks(
            [b"A" * 700, b"B" * 700],
            1024,
        )

        self.assertTrue(truncated)
        self.assertEqual(original_size, 1400)
        self.assertLessEqual(len(output.encode("utf-8")), 1024)
        self.assertIn("output truncated", output)


if __name__ == "__main__":
    unittest.main()
