"""Tests for bounded command-output capture."""

import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from capture_utils import (
    _BoundedCapture,
    _run_command_bounded,
    _terminate_process_group,
    bound_chunks,
    read_file_bounded,
    run_command_bounded,
)


class TestBoundedCommandCapture(unittest.TestCase):
    def test_empty_and_followup_chunks_update_bounded_capture(self):
        capture = _BoundedCapture(512)

        capture.add(b"")
        capture.add(b"A" * 300)
        capture.add(b"B" * 300)

        output, truncated, total_bytes = capture.finish()

        self.assertTrue(truncated)
        self.assertEqual(total_bytes, 600)
        self.assertLessEqual(len(output.encode("utf-8")), 512)
        self.assertIn("B", output)

    def test_truncation_marker_is_bounded(self):
        capture = _BoundedCapture(512)
        capture.add(b"x" * 300)
        capture.total_bytes = 10**300

        output, truncated, _ = capture.finish()

        self.assertTrue(truncated)
        self.assertLessEqual(len(output.encode("utf-8")), 512)

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
        with self.assertLogs("ephemeral_buffer.capture", level="WARNING") as events:
            output, exit_code, truncated, original_size, timed_out = run_command_bounded(
                command, None, 1024, timeout_seconds=0.1
            )

        self.assertEqual(exit_code, 124)
        self.assertTrue(timed_out)
        self.assertIn("started", output)
        self.assertFalse(truncated)
        self.assertGreater(original_size, 0)
        self.assertTrue(any("command_timeout" in event for event in events.output))
        self.assertTrue(all(command not in event for event in events.output))

    def test_timeout_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "timeout_seconds"):
            run_command_bounded("true", None, 1024, timeout_seconds=0)

    def test_selector_timeout_marks_command_timed_out(self):
        class FakeSelector:
            def register(self, _stream, _event):
                pass

            def get_map(self):
                return {"stdout": object()}

            def select(self, _timeout):
                return []

            def close(self):
                pass

        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            stdout = FakeStream()
            returncode = 0

            def wait(self, timeout=None):
                return None

            def kill(self):
                pass

        with patch("capture_utils.selectors.DefaultSelector", return_value=FakeSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=FakeProcess()), \
                patch("capture_utils._terminate_process_group") as terminate:
            result = _run_command_bounded("ignored", None, 1024, timeout_seconds=None)

        self.assertEqual(result[1], 124)
        self.assertTrue(result[4])
        terminate.assert_called_once()

    def test_expired_deadline_and_wait_timeout_force_cleanup(self):
        class FakeSelector:
            def register(self, _stream, _event):
                pass

            def get_map(self):
                return {"stdout": object()}

            def select(self, _timeout):
                self.fail("expired deadline should not poll the selector")

            def close(self):
                pass

        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            stdout = FakeStream()
            returncode = 0

            def __init__(self):
                self.wait_calls = 0
                self.killed = False

            def wait(self, timeout=None):
                self.wait_calls += 1
                if timeout is not None:
                    raise subprocess.TimeoutExpired("ignored", timeout)

            def kill(self):
                self.killed = True

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=FakeSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.time.monotonic", side_effect=[0, 2]), \
                patch("capture_utils._terminate_process_group") as terminate:
            result = _run_command_bounded("ignored", None, 1024, timeout_seconds=1)

        self.assertEqual(result[1], 124)
        self.assertTrue(result[4])
        self.assertEqual(process.wait_calls, 2)
        self.assertTrue(process.killed)
        terminate.assert_called_once_with(process)

    def test_process_group_cleanup_falls_back_to_process_methods(self):
        class FakeProcess:
            pid = 42

            def __init__(self):
                self.terminated = False
                self.killed = False
                self.wait_calls = 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("ignored", timeout)

        process = FakeProcess()
        with patch("capture_utils.os.killpg", side_effect=OSError("unavailable")):
            _terminate_process_group(process)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

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

    def test_file_read_rejects_non_positive_limit(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                read_file_bounded(file_handle.name, 0)

    def test_bound_chunks_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "max_output_bytes"):
            bound_chunks([b"payload"], 100)


if __name__ == "__main__":
    unittest.main()
