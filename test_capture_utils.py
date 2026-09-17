"""Tests for bounded command-output capture."""

import shlex
import errno
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import capture_utils
from capture_utils import (
    _BoundedCapture,
    _run_command_bounded,
    _terminate_process_group,
    bound_chunks,
    read_file_bounded,
    run_command_bounded,
)


class TestBoundedCommandCapture(unittest.TestCase):
    def test_cleanup_status_attachment_ignores_immutable_exception(self):
        capture_utils._attach_cleanup_status(object(), False)

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

    def test_truncation_marker_can_be_bounded_to_zero_bytes(self):
        self.assertEqual(_BoundedCapture._truncate_text_to_bytes("text", 0), "")

    def test_oversized_truncation_marker_is_bounded(self):
        capture = _BoundedCapture(512)
        capture.add(b"x" * 300)
        capture.total_bytes = 10**1000

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

    def test_process_marker_is_passed_to_child_environment(self):
        command = (
            f"{shlex.quote(sys.executable)} -c "
            "\"import os; print(os.environ['EPHEMERAL_EXECUTION_PROCESS_MARKER'])\""
        )
        output, exit_code, truncated, _original_size, timed_out = run_command_bounded(
            command,
            None,
            1024,
            process_marker="marker-for-recovery",
        )

        self.assertEqual((output, exit_code, truncated, timed_out), ("marker-for-recovery\n", 0, False, False))

    def test_supervisor_terminates_detached_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "detached-pid"
            script = (
                "import subprocess, sys; "
                "child = subprocess.Popen(['setsid', 'sleep', '60'], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "open(sys.argv[1], 'w', encoding='ascii').write(str(child.pid))"
            )
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(str(pid_path))}"
            result = run_command_bounded(
                command, None, 1024, process_marker="detached-descendant"
            )
            detached_pid = int(pid_path.read_text(encoding="ascii"))
            for _ in range(20):
                try:
                    os.kill(detached_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                self.fail("detached descendant survived supervisor cleanup")
            self.assertTrue(result.cleanup_confirmed)

    def test_supervisor_timeout_waits_for_detached_cleanup_handshake(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "detached-pid"
            child_script = (
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w', encoding='ascii').write(str(__import__('os').getpid())); "
                "time.sleep(60)"
            )
            parent_script = (
                "import subprocess, sys, time; "
                f"subprocess.Popen(['setsid', sys.executable, '-c', {child_script!r}, "
                f"sys.argv[1]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "time.sleep(60)"
            )
            command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_script)} "
                f"{shlex.quote(str(pid_path))}"
            )
            result = run_command_bounded(
                command,
                None,
                1024,
                timeout_seconds=0.1,
                process_marker="detached-timeout-handshake",
            )
            detached_pid = int(pid_path.read_text(encoding="ascii"))
            try:
                for _ in range(300):
                    try:
                        os.kill(detached_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
                else:
                    self.fail("detached descendant survived timeout cleanup")
            finally:
                try:
                    os.kill(detached_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.assertTrue(result.cleanup_confirmed)

    def test_supervised_cleanup_and_marker_scan_fail_closed(self):
        self.assertFalse(
            capture_utils._terminate_supervised_process(Mock(pid=None), None, "marker")
        )
        process = Mock(pid=42, returncode=0)
        with patch.object(capture_utils.os, "killpg", side_effect=ProcessLookupError()), \
                patch.object(capture_utils, "_marker_processes", return_value=set()):
            self.assertTrue(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        with patch.object(capture_utils.os, "killpg", side_effect=OSError("term failed")), \
                patch.object(process, "terminate", side_effect=OSError("fallback failed")):
            self.assertFalse(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        with patch.object(capture_utils.os, "killpg", side_effect=OSError("term failed")), \
                patch.object(capture_utils, "_marker_processes", return_value=set()):
            self.assertTrue(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        process.wait.side_effect = [subprocess.TimeoutExpired("command", 3), None]
        with patch.object(capture_utils.os, "killpg", return_value=None), \
                patch.object(capture_utils, "_marker_processes", return_value=set()):
            self.assertFalse(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        process.wait.side_effect = [subprocess.TimeoutExpired("command", 3), None]
        with patch.object(
            capture_utils.os,
            "killpg",
            side_effect=[None, ProcessLookupError()],
        ), patch.object(capture_utils, "_marker_processes", return_value=set()):
            self.assertFalse(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        process.wait.side_effect = [subprocess.TimeoutExpired("command", 3), None]
        with patch.object(capture_utils.os, "killpg", side_effect=[None, OSError("kill failed")]), \
                patch.object(process, "kill", side_effect=OSError("fallback failed")):
            self.assertFalse(capture_utils._terminate_supervised_process(process, 42, "marker"))
        process = Mock(pid=42, returncode=0)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("command", 3),
            subprocess.TimeoutExpired("command", 1),
        ]
        with patch.object(capture_utils.os, "killpg", return_value=None):
            self.assertFalse(capture_utils._terminate_supervised_process(process, 42, "marker"))

        with patch.object(capture_utils.sys, "platform", "win32"):
            self.assertIsNone(capture_utils._marker_processes("marker"))
        with patch.object(capture_utils.os, "listdir", side_effect=OSError("proc missing")):
            self.assertIsNone(capture_utils._marker_processes("marker"))
        stat = type("Stat", (), {"st_uid": 10})()
        with patch.object(capture_utils.os, "listdir", return_value=["not-a-pid", "123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", return_value=type("Stat", (), {"st_uid": 11})()):
            self.assertEqual(capture_utils._marker_processes("marker"), set())
        with patch.object(capture_utils.os, "listdir", return_value=["123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", side_effect=FileNotFoundError()):
            self.assertEqual(capture_utils._marker_processes("marker"), set())
        with patch.object(capture_utils.os, "listdir", return_value=["123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", side_effect=OSError("stat failed")):
            self.assertIsNone(capture_utils._marker_processes("marker"))
        with patch.object(capture_utils.os, "listdir", return_value=["123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", return_value=stat), \
                patch.object(capture_utils.Path, "read_bytes", side_effect=FileNotFoundError()):
            self.assertEqual(capture_utils._marker_processes("marker"), set())
        with patch.object(capture_utils.os, "listdir", return_value=["123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", return_value=stat), \
                patch.object(capture_utils.Path, "read_bytes", side_effect=OSError("read failed")):
            self.assertIsNone(capture_utils._marker_processes("marker"))
        with patch.object(capture_utils.os, "listdir", return_value=["123", "456"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", return_value=stat), \
                patch.object(
                    capture_utils.Path,
                    "read_bytes",
                    side_effect=[PermissionError("protected"), b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0"],
                ):
            self.assertEqual(capture_utils._marker_processes("marker"), {456})
        with patch.object(capture_utils.os, "listdir", return_value=["123"]), \
                patch.object(capture_utils.os, "getuid", return_value=10), \
                patch.object(capture_utils.os, "stat", return_value=stat), \
                patch.object(
                    capture_utils.Path,
                    "read_bytes",
                    return_value=b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0",
                ):
            self.assertEqual(capture_utils._marker_processes("marker"), {123})

    def test_marker_process_absence_waits_for_proc_settle(self):
        with patch.object(
            capture_utils,
            "_marker_processes",
            side_effect=[{123}, set()],
        ), patch.object(
            capture_utils.time,
            "monotonic",
            side_effect=[0, 0.1],
        ), patch.object(capture_utils.time, "sleep") as sleep:
            self.assertTrue(capture_utils._marker_processes_absent("marker"))
        sleep.assert_called_once_with(0.01)

        with patch.object(capture_utils, "_marker_processes", side_effect=[None, set()]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 0.1]), \
                patch.object(capture_utils.time, "sleep") as sleep:
            self.assertTrue(capture_utils._marker_processes_absent("marker"))
        sleep.assert_called_once_with(0.01)

        with patch.object(capture_utils, "_marker_processes", return_value=None), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 1]):
            self.assertIsNone(capture_utils._marker_processes_absent("marker"))

        with patch.object(capture_utils, "_marker_processes", return_value={123}), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 1]), \
                patch.object(capture_utils.time, "sleep") as sleep:
            self.assertFalse(capture_utils._marker_processes_absent("marker"))
        sleep.assert_not_called()

    def test_supervisor_helpers_fail_closed_and_reap_descendants(self):
        with patch.object(capture_utils.sys, "platform", "win32"):
            self.assertFalse(capture_utils._enable_subreaper())
        fake_prctl = Mock(return_value=0)
        fake_libc = type("Libc", (), {"prctl": fake_prctl})()
        with patch.object(capture_utils.sys, "platform", "linux"), \
                patch.object(capture_utils.ctypes, "CDLL", return_value=fake_libc):
            self.assertTrue(capture_utils._enable_subreaper())
            fake_prctl.return_value = 1
            self.assertFalse(capture_utils._enable_subreaper())
        with patch.object(capture_utils.ctypes, "CDLL", side_effect=AttributeError("missing")):
            self.assertFalse(capture_utils._enable_subreaper())

        with patch.object(capture_utils.os, "listdir", side_effect=OSError("proc missing")):
            self.assertIsNone(capture_utils._supervisor_descendants(1))
        stat_lines = {
            1: "1 (root) S 0 0",
            2: "2 (child) S 1 0",
            3: "3 (grandchild) S 2 0",
            4: "4 (short)",
            5: "5 (invalid) S not-an-int",
        }
        with patch.object(capture_utils.os, "listdir", return_value=["not-a-pid", *[str(pid) for pid in stat_lines]]), \
                patch.object(capture_utils.Path, "read_text", side_effect=list(stat_lines.values())):
            self.assertEqual(capture_utils._supervisor_descendants(1), {2, 3})

        with patch.object(capture_utils.os, "waitpid", side_effect=[(1, 0), (0, 0)]):
            self.assertFalse(capture_utils._supervisor_reap())
        with patch.object(capture_utils.os, "waitpid", side_effect=ChildProcessError):
            self.assertTrue(capture_utils._supervisor_reap())
        with patch.object(capture_utils.os, "waitpid", side_effect=OSError("wait failed")):
            self.assertFalse(capture_utils._supervisor_reap())

        with patch.object(capture_utils, "_supervisor_descendants", return_value=None):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, set()]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", side_effect=[True, True]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 0.1, 0.2]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertTrue(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", return_value=set()), \
                patch.object(capture_utils, "_supervisor_reap", return_value=False):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, None]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", return_value=True), \
                patch.object(capture_utils.time, "monotonic", return_value=0), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, None]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", return_value=True), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 0.1]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", return_value={2}), \
                patch.object(capture_utils, "_supervisor_signal_descendants", side_effect=[True, True]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 0.1, 2, 2, 0.1, 4]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", return_value={2}), \
                patch.object(capture_utils, "_supervisor_signal_descendants", side_effect=[True, False, False]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 2, 2, 4]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", return_value={2}), \
                patch.object(capture_utils, "_supervisor_signal_descendants", side_effect=[True, False, False]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 2, 2, 4]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, None]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", return_value=True), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 2, 2, 0.1]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, None]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", return_value=False):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_descendants", side_effect=[{2}, None]), \
                patch.object(capture_utils, "_supervisor_signal_descendants", side_effect=[True, False]), \
                patch.object(capture_utils.time, "monotonic", side_effect=[0, 2]), \
                patch.object(capture_utils, "_supervisor_reap"):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
    def test_supervisor_signal_descendants_pins_process_generations(self):
        self.assertTrue(capture_utils._supervisor_signal_descendants(set(), signal.SIGTERM))
        with patch.object(capture_utils.os, "pidfd_open", None):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", return_value=None), \
                patch.object(capture_utils, "_supervisor_process_absent", return_value=True):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", return_value=None), \
                patch.object(capture_utils, "_supervisor_process_absent", return_value=False):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        missing = OSError(errno.ESRCH, "gone")
        with patch.object(capture_utils, "_supervisor_process_identity", return_value=("start", "boot")), \
                patch.object(capture_utils.os, "pidfd_open", side_effect=missing):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", return_value=("start", "boot")), \
                patch.object(capture_utils.os, "pidfd_open", side_effect=OSError("open failed")):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), None, None, None]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils, "_supervisor_pidfd_terminated", return_value=True), \
                patch.object(capture_utils.signal, "pidfd_send_signal") as send, \
                patch.object(capture_utils.os, "close"):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
            send.assert_not_called()
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), None, None, None]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils, "_supervisor_pidfd_terminated", return_value=False), \
                patch.object(capture_utils.os, "close"):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), ("new", "boot")]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils.os, "close"):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), ("start", "boot")]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils.signal, "pidfd_send_signal", side_effect=[ProcessLookupError(), OSError("send failed")]), \
                patch.object(capture_utils.os, "close"):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), ("start", "boot")]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils.signal, "pidfd_send_signal", side_effect=OSError("send failed")), \
                patch.object(capture_utils.os, "close"):
            self.assertFalse(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), ("start", "boot")]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils.signal, "pidfd_send_signal", side_effect=OSError("gone")), \
                patch.object(capture_utils, "_supervisor_pidfd_terminated", return_value=True), \
                patch.object(capture_utils.os, "close"):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))

    def test_supervisor_identity_and_pidfd_helpers_fail_closed(self):
        with patch.object(capture_utils.Path, "read_text", side_effect=OSError("proc missing")):
            self.assertIsNone(capture_utils._supervisor_process_identity(123))
        with patch.object(capture_utils.Path, "read_text", side_effect=["", "stat"]):
            self.assertIsNone(capture_utils._supervisor_process_identity(123))
        with patch.object(capture_utils.Path, "read_text", side_effect=["boot", "1 (name) S"]):
            self.assertIsNone(capture_utils._supervisor_process_identity(123))
        stat_line = "1 (name) S " + " ".join(["0"] * 18 + ["123"])
        with patch.object(capture_utils.Path, "read_text", side_effect=["boot", stat_line]):
            self.assertEqual(capture_utils._supervisor_process_identity(123), ("123", "boot"))
        with patch.object(capture_utils.os, "kill", side_effect=ProcessLookupError()):
            self.assertTrue(capture_utils._supervisor_process_absent(123))
        with patch.object(capture_utils.os, "kill", return_value=None), \
                patch.object(capture_utils.Path, "read_text", return_value="123 (zombie) Z 0 0"):
            self.assertTrue(capture_utils._supervisor_process_absent(123))
        with patch.object(capture_utils.os, "kill", side_effect=OSError("permission denied")):
            self.assertFalse(capture_utils._supervisor_process_absent(123))
        with patch.object(capture_utils.os, "kill", return_value=None):
            self.assertFalse(capture_utils._supervisor_process_absent(123))
        with patch.object(capture_utils.select, "select", return_value=([9], [], [])):
            self.assertTrue(capture_utils._supervisor_pidfd_terminated(9))
        with patch.object(capture_utils.select, "select", return_value=([], [], [])):
            self.assertFalse(capture_utils._supervisor_pidfd_terminated(9))
        with patch.object(capture_utils.select, "select", side_effect=[OSError("bad fd"), ValueError("bad fd")]):
            self.assertFalse(capture_utils._supervisor_pidfd_terminated(9))
            self.assertFalse(capture_utils._supervisor_pidfd_terminated(9))
        with patch.object(capture_utils, "_supervisor_descendants", return_value={2}), \
                patch.object(capture_utils, "_supervisor_signal_descendants", return_value=False):
            self.assertFalse(capture_utils._supervisor_cleanup(1))
        with patch.object(capture_utils, "_supervisor_process_identity", side_effect=[("start", "boot"), ("start", "boot")]), \
                patch.object(capture_utils.os, "pidfd_open", return_value=9), \
                patch.object(capture_utils.signal, "pidfd_send_signal"), \
                patch.object(capture_utils.os, "close", side_effect=OSError("close failed")):
            self.assertTrue(capture_utils._supervisor_signal_descendants({123}, signal.SIGTERM))

        with patch.object(capture_utils, "_enable_subreaper", return_value=False):
            self.assertEqual(
                capture_utils._supervisor_main("true"),
                capture_utils._SUPERVISOR_CLEANUP_FAILURE,
            )
        handlers = {}

        def install(signum, handler):
            handlers[signum] = handler

        child = Mock()
        child.wait.return_value = 7
        with patch.object(capture_utils, "_enable_subreaper", return_value=True), \
                patch.object(capture_utils.signal, "signal", side_effect=install), \
                patch.object(capture_utils.subprocess, "Popen", return_value=child), \
                patch.object(capture_utils, "_supervisor_cleanup", return_value=True):
            self.assertEqual(capture_utils._supervisor_main("true"), 7)
        with patch.object(capture_utils, "_enable_subreaper", return_value=True), \
                patch.object(capture_utils.signal, "signal", side_effect=install), \
                patch.object(capture_utils.subprocess, "Popen", return_value=child), \
                patch.object(capture_utils, "_supervisor_cleanup", return_value=False):
            self.assertEqual(
                capture_utils._supervisor_main("true"),
                capture_utils._SUPERVISOR_CLEANUP_FAILURE,
            )
        with patch.object(capture_utils, "_enable_subreaper", return_value=True), \
                patch.object(capture_utils.signal, "signal", side_effect=install), \
                patch.object(capture_utils.subprocess, "Popen", side_effect=OSError("launch failed")), \
                patch.object(capture_utils, "_supervisor_cleanup", return_value=True):
            self.assertEqual(
                capture_utils._supervisor_main("true"),
                capture_utils._SUPERVISOR_CLEANUP_FAILURE,
            )
        with patch.object(capture_utils, "_enable_subreaper", return_value=True), \
                patch.object(capture_utils.signal, "signal", side_effect=install), \
                patch.object(capture_utils.subprocess, "Popen", side_effect=OSError("launch failed")), \
                patch.object(capture_utils, "_supervisor_cleanup", side_effect=OSError("cleanup failed")):
            self.assertEqual(
                capture_utils._supervisor_main("true"),
                capture_utils._SUPERVISOR_CLEANUP_FAILURE,
            )
        with patch.object(capture_utils, "_supervisor_cleanup", return_value=True), \
                patch.object(capture_utils.os, "_exit", side_effect=RuntimeError("exited")):
            with self.assertRaisesRegex(RuntimeError, "exited"):
                handlers[capture_utils.signal.SIGTERM](15, None)

    def test_process_started_callback_receives_process_identity(self):
        identities = []
        result = run_command_bounded(
            "true", None, 1024,
            process_started=lambda process_id, process_group_id: identities.append(
                (process_id, process_group_id)
            ),
        )
        self.assertEqual(result[1], 0)
        self.assertEqual(len(identities), 1)
        self.assertGreater(identities[0][0], 0)
        self.assertGreater(identities[0][1], 0)

    def test_process_started_callback_failure_terminates_process(self):
        def fail(*_identity):
            raise RuntimeError("cannot persist process identity")

        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "cannot persist"):
            run_command_bounded(
                f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(5)\"",
                None,
                1024,
                process_started=fail,
            )
        self.assertLess(time.monotonic() - started, 3)

    def test_early_eof_waits_for_process_within_deadline(self):
        script = "import os, time; os.close(1); time.sleep(0.2)"
        command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        output, exit_code, truncated, original_size, timed_out = run_command_bounded(
            command, None, 1024, timeout_seconds=5
        )

        self.assertEqual((output, exit_code, truncated, original_size, timed_out), ("", 0, False, 0, False))

    def test_early_eof_without_deadline_does_not_invent_timeout(self):
        script = "import os, time; os.close(1); time.sleep(0.2)"
        command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        output, exit_code, truncated, original_size, timed_out = run_command_bounded(
            command, None, 1024
        )

        self.assertEqual((output, exit_code, truncated, original_size, timed_out), ("", 0, False, 0, False))

    def test_early_eof_still_honors_expired_deadline(self):
        script = "import os, time; os.close(1); time.sleep(2)"
        command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        output, exit_code, truncated, original_size, timed_out = run_command_bounded(
            command, None, 1024, timeout_seconds=0.1
        )

        self.assertEqual((output, exit_code, truncated, original_size, timed_out), ("", 124, False, 0, True))

    def test_post_eof_wait_timeout_uses_timeout_cleanup(self):
        class FakeStream:
            def read1(self, _size):
                return b""

            def close(self):
                pass

        stream = FakeStream()

        class FakeSelector:
            def __init__(self):
                self.active = True

            def register(self, _stream, _event):
                pass

            def get_map(self):
                return {"stdout": object()} if self.active else {}

            def select(self, _timeout):
                return [(type("Key", (), {"fileobj": stream})(), None)]

            def unregister(self, _stream):
                self.active = False

            def close(self):
                pass

        class FakeProcess:
            stdout = stream
            returncode = 0

            def __init__(self):
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("ignored", timeout)

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=FakeSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.time.monotonic", side_effect=[0, 0, 0]), \
                patch("capture_utils._terminate_process_group") as terminate:
            result = _run_command_bounded("ignored", None, 1024, timeout_seconds=1)

        self.assertEqual(result[1], 124)
        self.assertTrue(result[4])
        terminate.assert_called_once_with(process)

    def test_timeout_cleanup_failure_is_raised_without_prior_error(self):
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

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=FakeSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch(
                    "capture_utils._terminate_process_group",
                    side_effect=RuntimeError("cleanup failed"),
                ) as terminate:
            with self.assertRaisesRegex(RuntimeError, "cleanup failed") as raised:
                _run_command_bounded("ignored", None, 1024, timeout_seconds=None)

        terminate.assert_called_once_with(process)
        self.assertFalse(raised.exception.cleanup_confirmed)

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

    def test_timeout_kills_descendant_after_group_leader_exits(self):
        child_script = "import time; time.sleep(10)"
        parent_script = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child_script!r}])"
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_script)}"
        started = time.monotonic()
        output, exit_code, _truncated, _original_size, timed_out = run_command_bounded(
            command, None, 1024, timeout_seconds=0.1
        )

        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(output, "")
        self.assertEqual(exit_code, 124)
        self.assertTrue(timed_out)

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
                patch("capture_utils._terminate_process_group", return_value=False) as terminate:
            result = _run_command_bounded("ignored", None, 1024, timeout_seconds=None)

        self.assertEqual(result[1], 124)
        self.assertTrue(result[4])
        self.assertFalse(result.cleanup_confirmed)
        terminate.assert_called_once()

    def test_keyboard_interrupt_terminates_the_process_group(self):
        class InterruptingSelector:
            def register(self, _stream, _event):
                pass

            def get_map(self):
                return {"stdout": object()}

            def select(self, _timeout):
                raise KeyboardInterrupt

            def close(self):
                pass

        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            pid = 42
            stdout = FakeStream()
            returncode = 130

            def wait(self, timeout=None):
                return None

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=InterruptingSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.os.getpgid", return_value=42), \
                patch("capture_utils._terminate_process_group", return_value=False) as terminate:
            with self.assertRaises(KeyboardInterrupt) as raised:
                _run_command_bounded("ignored", None, 1024, timeout_seconds=None)
        terminate.assert_called_once_with(process, 42)
        self.assertFalse(raised.exception.cleanup_confirmed)

    def test_selector_construction_failure_terminates_the_process_group(self):
        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            pid = 42
            stdout = FakeStream()
            returncode = 130

            def wait(self, timeout=None):
                pass

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", side_effect=KeyboardInterrupt), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.os.getpgid", return_value=42), \
                patch("capture_utils._terminate_process_group") as terminate:
            with self.assertRaises(KeyboardInterrupt):
                _run_command_bounded("ignored", None, 1024, timeout_seconds=None)
        terminate.assert_called_once_with(process, 42)

    def test_cleanup_failure_is_attached_to_original_interrupt(self):
        class InterruptingSelector:
            def register(self, _stream, _event):
                raise KeyboardInterrupt

            def close(self):
                pass

        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            pid = 42
            stdout = FakeStream()
            returncode = 130

            def wait(self, timeout=None):
                pass

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=InterruptingSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.os.getpgid", return_value=42), \
                patch(
                    "capture_utils._terminate_process_group",
                    side_effect=RuntimeError("cleanup failed"),
                ) as terminate:
            with self.assertRaises(KeyboardInterrupt) as raised:
                _run_command_bounded("ignored", None, 1024, timeout_seconds=None)
        terminate.assert_called_once_with(process, 42)
        self.assertFalse(raised.exception.cleanup_confirmed)

    def test_process_start_cleanup_failure_is_attached_to_callback_error(self):
        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            pid = 42
            stdout = FakeStream()

        process = FakeProcess()

        def fail(*_identity):
            raise RuntimeError("checkpoint failed")

        with patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.os.getpgid", return_value=42), \
                patch(
                    "capture_utils._terminate_process_group",
                    side_effect=RuntimeError("cleanup failed"),
                ) as terminate:
            with self.assertRaisesRegex(RuntimeError, "checkpoint failed") as raised:
                _run_command_bounded(
                    "ignored", None, 1024, timeout_seconds=None, process_started=fail
                )
        terminate.assert_called_once_with(process, 42)
        self.assertFalse(raised.exception.cleanup_confirmed)

    def test_selector_registration_failure_terminates_the_process_group(self):
        class FailingSelector:
            def register(self, _stream, _event):
                raise RuntimeError("selector unavailable")

            def close(self):
                pass

        class FakeStream:
            def close(self):
                pass

        class FakeProcess:
            pid = 42
            stdout = FakeStream()
            returncode = 1

            def wait(self, timeout=None):
                pass

        process = FakeProcess()
        with patch("capture_utils.selectors.DefaultSelector", return_value=FailingSelector()), \
                patch("capture_utils.subprocess.Popen", return_value=process), \
                patch("capture_utils.os.getpgid", return_value=42), \
                patch("capture_utils._terminate_process_group") as terminate:
            with self.assertRaisesRegex(RuntimeError, "selector unavailable"):
                _run_command_bounded("ignored", None, 1024, timeout_seconds=None)
        terminate.assert_called_once_with(process, 42)

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

    def test_process_group_cleanup_uses_group_termination_when_available(self):
        process = type("Process", (), {"pid": 42, "wait": lambda _self, timeout=None: None})()

        with patch("capture_utils.os.killpg") as killpg:
            _terminate_process_group(process)

        self.assertEqual(killpg.call_count, 3)
        self.assertEqual(killpg.call_args_list[0].args, (42, 15))

    def test_process_group_cleanup_confirms_group_absence_after_kill(self):
        process = type("Process", (), {"pid": 42, "wait": lambda _self, timeout=None: None})()

        with patch(
            "capture_utils.os.killpg",
            side_effect=[None, None, ProcessLookupError()],
        ) as killpg:
            self.assertTrue(_terminate_process_group(process))
        self.assertEqual(killpg.call_count, 3)

    def test_process_group_cleanup_accepts_missing_group_after_unreaped_wait(self):
        class UnreapedProcess:
            pid = 42

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("ignored", timeout)

        process = UnreapedProcess()
        with patch(
            "capture_utils.os.killpg",
            side_effect=[None, ProcessLookupError()],
        ) as killpg:
            self.assertTrue(_terminate_process_group(process))
        self.assertEqual(killpg.call_count, 2)

    def test_process_group_cleanup_escalates_to_group_kill_after_wait_timeout(self):
        class SlowProcess:
            pid = 42

            def __init__(self):
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("ignored", timeout)

            def kill(self):
                self.killed = True

        process = SlowProcess()
        with patch("capture_utils.os.killpg") as killpg:
            _terminate_process_group(process)

        self.assertEqual(killpg.call_count, 3)
        self.assertEqual(killpg.call_args_list[1].args, (42, 9))
        self.assertGreaterEqual(process.wait_calls, 2)

    def test_process_group_cleanup_retries_kill_when_reap_stays_blocked(self):
        class UnkillableProcess:
            pid = 42

            def __init__(self):
                self.wait_calls = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                raise subprocess.TimeoutExpired("ignored", timeout)

            def kill(self):
                raise OSError("already gone")

        process = UnkillableProcess()
        with patch(
            "capture_utils.os.killpg",
            side_effect=[None, None, ProcessLookupError()],
        ) as killpg:
            self.assertTrue(_terminate_process_group(process))
        self.assertEqual(killpg.call_count, 3)
        self.assertEqual(process.wait_calls, 3)

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

    def test_invalid_utf8_replacement_stays_within_output_budget(self):
        output, truncated, original_size = bound_chunks([b"\xff" * 600], 512)

        self.assertTrue(truncated)
        self.assertEqual(original_size, 600)
        self.assertLessEqual(len(output.encode("utf-8")), 512)
        self.assertIn("output truncated", output)

    def test_complete_invalid_utf8_that_expands_is_bounded(self):
        capture = _BoundedCapture(512)
        capture.add(b"\xff" * 200)

        output, truncated, original_size = capture.finish()

        self.assertTrue(truncated)
        self.assertEqual(original_size, 200)
        self.assertLessEqual(len(output.encode("utf-8")), 512)

    def test_file_read_rejects_oversized_content(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            file_handle.write(b"x" * 1024)
            file_handle.flush()
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_file_bounded(file_handle.name, 512)

    def test_file_read_returns_utf8_content(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            file_handle.write("héllo".encode("utf-8"))
            file_handle.flush()
            self.assertEqual(read_file_bounded(file_handle.name, 512), "héllo")

    def test_file_read_rejects_non_positive_limit(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                read_file_bounded(file_handle.name, 0)

    def test_bound_chunks_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "max_output_bytes"):
            bound_chunks([b"payload"], 100)


if __name__ == "__main__":
    unittest.main()
