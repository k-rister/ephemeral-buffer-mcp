"""Tests for durable phase-level execution and resume behavior."""

import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import execution
from capture_utils import BoundedCommandResult
from execution import (
    ExecutionBusyError,
    ExecutionStore,
    MAX_EXECUTION_PHASES,
    MAX_PHASE_NAME_BYTES,
    MAX_PHASE_ATTEMPTS,
    PhaseExecutionManager,
)


class Runner:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def __call__(self, command, cwd, max_output_bytes, timeout_seconds):
        self.calls.append((command, cwd, max_output_bytes, timeout_seconds))
        result = self.results.get(command, (f"output:{command}", 0, False, len(command), False))
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(command, cwd, max_output_bytes, timeout_seconds)
        return result


class TestPhaseExecutionManager(unittest.TestCase):
    def manager(self, runner=None, max_output_bytes=4096, process_cleanup=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return PhaseExecutionManager(
            directory.name,
            max_output_bytes=max_output_bytes,
            command_runner=runner or Runner(),
            process_cleanup=process_cleanup,
        )

    @staticmethod
    def phase(name, command, **overrides):
        value = {"name": name, "command": command}
        value.update(overrides)
        return value

    def test_success_persists_output_and_fresh_manager_skips_completed_phases(self):
        runner = Runner()
        manager = self.manager(runner)
        result = manager.start(
            [self.phase("prepare", "prepare"), self.phase("verify", "verify")],
            execution_id="successful-execution",
            label="release preparation",
            cwd="/tmp",
            timeout_seconds=4,
            max_output_bytes=1024,
        )

        self.assertEqual(result["execution_status"], "completed")
        self.assertFalse(result["partial"])
        self.assertEqual(result["completed_phase_count"], 2)
        self.assertEqual([call[0] for call in runner.calls], ["prepare", "verify"])
        self.assertEqual(
            [event["status"] for event in result["phases"][0]["events"]],
            ["started", "completed"],
        )
        self.assertEqual(result["phases"][0]["output_bytes"], len("output:prepare"))
        self.assertEqual(result["resume"]["available"], False)

        fresh_runner = Runner({"prepare": AssertionError("completed phase reran")})
        fresh = PhaseExecutionManager(manager.store.state_dir, command_runner=fresh_runner)
        self.assertEqual(fresh.public("successful-execution")["execution_status"], "completed")
        self.assertEqual(fresh.output("successful-execution", "prepare")["phases"][0]["output"], "output:prepare")
        self.assertEqual(fresh_runner.calls, [])
        self.assertEqual(fresh.get("successful-execution")["label"], "release preparation")

    def test_failed_phase_requires_explicit_retry_and_later_phases_wait(self):
        attempts = {"tests": 0}

        def test_result(*_args):
            attempts["tests"] += 1
            return ("failure" if attempts["tests"] == 1 else "pass", 1 if attempts["tests"] == 1 else 0, False, 7, False)

        runner = Runner({"tests": test_result})
        manager = self.manager(runner)
        phases = [self.phase("tests", "tests"), self.phase("deploy", "deploy", side_effects="unsafe")]
        first = manager.start(phases, execution_id="retry-execution")
        self.assertEqual(first["execution_status"], "partial")
        self.assertEqual([phase["status"] for phase in first["phases"]], ["failed", "pending"])

        unchanged = manager.resume("retry-execution")
        self.assertEqual(unchanged["phases"][0]["attempts"], 1)
        self.assertTrue(unchanged["resume"]["retry_required"])
        self.assertEqual(len(runner.calls), 1)

        completed = manager.resume("retry-execution", retry_failed=True)
        self.assertEqual(completed["execution_status"], "completed")
        self.assertEqual([call[0] for call in runner.calls], ["tests", "tests", "deploy"])
        self.assertEqual(len(completed["phases"][0]["attempt_results"]), 2)
        self.assertEqual(completed["phases"][0]["attempt_results"][0]["exit_code"], 1)

    def test_timed_out_unsafe_phase_requires_confirmation(self):
        attempts = {"deploy": 0}

        def timeout_then_success(*_args):
            attempts["deploy"] += 1
            return ("partial deploy", 124 if attempts["deploy"] == 1 else 0, False, 13, attempts["deploy"] == 1)

        runner = Runner({"deploy": timeout_then_success})
        manager = self.manager(runner, process_cleanup=lambda: None)
        initial = manager.start(
            [self.phase("deploy", "deploy", unsafe_side_effects=True, idempotency_key="deploy-v1")],
            execution_id="unsafe-timeout",
        )
        self.assertEqual(initial["phases"][0]["status"], "timed_out")
        self.assertTrue(initial["resume"]["unsafe_confirmation_required"])

        blocked = manager.resume("unsafe-timeout", retry_failed=True)
        self.assertEqual(blocked["phases"][0]["attempts"], 1)
        self.assertEqual(len(runner.calls), 1)

        completed = manager.resume("unsafe-timeout", retry_failed=True, confirm_unsafe=True)
        self.assertEqual(completed["execution_status"], "completed")
        self.assertEqual(completed["phases"][0]["attempts"], 2)

    def test_attempt_limit_is_terminal_and_does_not_grow_retry_history(self):
        manager = self.manager()
        manager.create([self.phase("limited", "limited")], execution_id="terminal-limit")
        record = manager.get("terminal-limit")
        record["phases"][0]["status"] = "failed"
        record["phases"][0]["attempts"] = MAX_PHASE_ATTEMPTS
        manager.store.save(record)
        observed = manager.public("terminal-limit")
        self.assertTrue(observed["resume"]["attempt_limit_reached"])
        self.assertFalse(observed["resume"]["available"])
        self.assertFalse(observed["resume"]["retry_required"])
        first = manager.resume("terminal-limit", retry_failed=True)
        event_count = len(first["phases"][0]["events"])
        attempt_count = len(first["phases"][0]["attempt_results"])
        second = manager.resume("terminal-limit", retry_failed=True)
        self.assertEqual(len(second["phases"][0]["events"]), event_count)
        self.assertEqual(len(second["phases"][0]["attempt_results"]), attempt_count)
        self.assertFalse(second["resume"]["retry_required"])
        self.assertFalse(second["resume"]["available"])

    def test_concurrent_manager_cannot_rerun_live_phase_and_status_remains_visible(self):
        started = threading.Event()
        release = threading.Event()

        def blocking_runner(*_args):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return ("done", 0, False, 4, False)

        first_runner = Runner({"live": blocking_runner})
        first = self.manager(first_runner)
        second = PhaseExecutionManager(first.store.state_dir, command_runner=Runner())
        outcome = []

        def run_first():
            outcome.append(first.start([self.phase("live", "live", side_effects="unsafe")], execution_id="live-execution"))

        worker = threading.Thread(target=run_first)
        worker.start()
        self.assertTrue(started.wait(timeout=5))
        observed = second.public("live-execution")
        self.assertEqual(observed["execution_status"], "running")
        self.assertFalse(observed["resume"]["available"])
        with self.assertRaises(ExecutionBusyError):
            second.resume("live-execution", retry_failed=True, confirm_unsafe=True)
        release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome[0]["execution_status"], "completed")

    def test_cross_process_lease_blocks_duplicate_execution(self):
        manager = self.manager()
        marker = Path(manager.store.state_dir) / "child-ready"
        code = (
            "import sys, time; "
            "from pathlib import Path; "
            "from execution import ExecutionStore; "
            "store = ExecutionStore(sys.argv[1]); "
            "lease = store.lease('cross-process'); "
            "lease.__enter__(); "
            "Path(sys.argv[2]).write_text('ready', encoding='ascii'); "
            "time.sleep(5); "
            "lease.__exit__(None, None, None)"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(manager.store.state_dir), str(marker)]
        )
        self.addCleanup(lambda: child.poll() is None and child.terminate())
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.05)
        self.assertTrue(marker.exists())
        with self.assertRaises(ExecutionBusyError):
            with manager.store.lease("cross-process"):
                pass
        child.terminate()
        child.wait(timeout=5)

    def test_allow_unsafe_policy_allows_explicit_retry_without_confirmation(self):
        runner = Runner({"publish": ("no", 2, False, 2, False)})
        manager = self.manager(runner)
        failed = manager.start(
            [self.phase("publish", "publish", side_effects="unsafe")],
            execution_id="allow-unsafe",
            resume_policy="allow-unsafe",
        )
        self.assertEqual(failed["resume"]["unsafe_confirmation_required"], False)
        runner.results["publish"] = ("yes", 0, False, 3, False)
        retried = manager.resume("allow-unsafe", retry_failed=True)
        self.assertEqual(retried["execution_status"], "completed")

    def test_process_restart_recovers_started_phase_as_interrupted(self):
        runner = Runner()
        manager = self.manager(runner)
        manager.create([self.phase("long", "long")], execution_id="interrupted-execution")
        record = manager.get("interrupted-execution")
        record["phases"][0]["status"] = "started"
        record["phases"][0]["attempts"] = 1
        record["phases"][0]["process_id"] = 123
        record["phases"][0]["process_group_id"] = 456
        record["phases"][0]["process_start_time"] = "start"
        record["phases"][0]["process_boot_id"] = "boot"
        record["phases"][0]["process_containment"] = execution.PROCESS_CONTAINMENT_SUBREAPER
        manager.store.save(record)

        fresh_runner = Runner()
        fresh = PhaseExecutionManager(manager.store.state_dir, command_runner=fresh_runner)
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True):
            recovered = fresh.public("interrupted-execution")
        self.assertEqual(recovered["execution_status"], "interrupted")
        self.assertEqual(recovered["phases"][0]["status"], "interrupted")
        self.assertIn("process restart recovery", recovered["phases"][0]["events"][-1]["reason"])
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True):
            resumed = fresh.resume("interrupted-execution")
        self.assertEqual(resumed["execution_status"], "completed")
        self.assertEqual([event["status"] for event in resumed["phases"][0]["events"]], ["interrupted", "started", "completed"])

    def test_recovery_terminates_stale_process_group_before_resume(self):
        manager = self.manager()
        manager.create([self.phase("stale", "stale")], execution_id="stale-process")
        record = manager.get("stale-process")
        phase = record["phases"][0]
        phase["status"] = "started"
        phase["attempts"] = 1
        phase["process_id"] = 123
        phase["process_group_id"] = 456
        phase["process_start_time"] = "start"
        phase["process_boot_id"] = "boot"
        phase["process_containment"] = execution.PROCESS_CONTAINMENT_SUBREAPER
        manager.store.save(record)
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True) as terminate:
            recovered = PhaseExecutionManager(
                manager.store.state_dir, command_runner=Runner()
            ).public("stale-process")
        self.assertEqual(recovered["phases"][0]["status"], "interrupted")
        terminate.assert_called_once_with(
            123, expected_identity=("start", "boot"), expected_group=456
        )
        stale = {
            "process_id": 123,
            "process_group_id": 456,
            "process_start_time": "start",
            "process_boot_id": "boot",
            "process_containment": execution.PROCESS_CONTAINMENT_SUBREAPER,
        }
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True) as terminate:
            self.assertTrue(execution._terminate_stale_process(stale))
        terminate.assert_called_once_with(
            123, expected_identity=("start", "boot"), expected_group=456
        )
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=False):
            self.assertFalse(execution._terminate_stale_process(stale))
        with patch.object(execution.os, "killpg") as killpg:
            self.assertFalse(execution._terminate_stale_process({"process_id": 0, "process_group_id": 456}))
        self.assertEqual(killpg.call_count, 1)
        killpg.assert_called_with(456, 0)

    def _assert_recovery_cleans_marked_descendant(self, *, detached):
        with tempfile.TemporaryDirectory() as directory:
            marker = f"restart-recovery-{os.getpid()}-{time.monotonic_ns()}"
            pid_path = Path(directory) / "child-pid"
            child_script = (
                "import os, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w', encoding='ascii').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            supervisor_script = (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
                "start_new_session=bool(int(sys.argv[3])), "
                "stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL); "
                "time.sleep(60)"
            )
            environment = os.environ.copy()
            environment[execution.PROCESS_MARKER_ENV] = marker
            supervisor = None
            child_pid = None
            child_reaper = None
            reaper_result = []
            libc = ctypes.CDLL(None, use_errno=True)
            previous_subreaper = ctypes.c_int()
            self.assertEqual(
                libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0), 0,
                "PR_GET_CHILD_SUBREAPER failed",
            )
            self.assertEqual(
                libc.prctl(36, 1, 0, 0, 0), 0,
                "PR_SET_CHILD_SUBREAPER failed",
            )
            try:
                supervisor = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        supervisor_script,
                        child_script,
                        str(pid_path),
                        "1" if detached else "0",
                    ],
                    env=environment,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for _ in range(200):
                    if pid_path.exists():
                        try:
                            child_pid = int(pid_path.read_text(encoding="ascii"))
                        except (OSError, ValueError):
                            pass
                        else:
                            break
                    if supervisor.poll() is not None:
                        self.fail("test supervisor exited before creating its phase child")
                    time.sleep(0.01)
                self.assertIsNotNone(child_pid)
                process_start, process_boot = execution._proc_identity(supervisor.pid)
                self.assertIsNotNone(process_start)
                self.assertIsNotNone(process_boot)
                supervisor_group = os.getpgid(supervisor.pid)

                manager = PhaseExecutionManager(
                    Path(directory) / "state", command_runner=Runner()
                )
                manager.create(
                    [self.phase("detached", "detached")],
                    execution_id="reaped-supervisor",
                )
                record = manager.store.load("reaped-supervisor", recover=False)
                phase = record["phases"][0]
                phase.update(
                    {
                        "status": "started",
                        "attempts": 1,
                        "process_id": supervisor.pid,
                        "process_group_id": supervisor_group,
                        "process_start_time": process_start,
                        "process_boot_id": process_boot,
                        "process_containment": execution.PROCESS_CONTAINMENT_SUBREAPER,
                        "process_launch_token": marker,
                        "process_fence_pending": True,
                    }
                )
                manager.store.save(record)

                terminate_pidfd = execution._terminate_pidfd

                def terminate_and_reap(process_id, **kwargs):
                    nonlocal child_reaper
                    if process_id == supervisor.pid:
                        os.kill(process_id, signal.SIGTERM)
                        supervisor.wait(timeout=5)
                        child_reaper = threading.Thread(
                            target=lambda: reaper_result.append(
                                os.waitpid(child_pid, 0)
                            ),
                            daemon=True,
                        )
                        child_reaper.start()
                        if "expected_group" in kwargs:
                            return execution._process_group_is_absent(
                                kwargs["expected_group"]
                            )
                        return True
                    result = terminate_pidfd(process_id, **kwargs)
                    if process_id == child_pid and child_reaper is not None:
                        child_reaper.join(timeout=5)
                        self.assertFalse(child_reaper.is_alive())
                    return result

                with patch.object(
                    execution, "_terminate_pidfd", side_effect=terminate_and_reap
                ), patch.object(
                    execution, "_marker_processes", side_effect=[{child_pid}, set()]
                ):
                    recovered = PhaseExecutionManager(
                        manager.store.state_dir, command_runner=Runner()
                    ).public("reaped-supervisor")

                self.assertEqual(recovered["phases"][0]["status"], "interrupted")
                self.assertTrue(recovered["resume"]["available"], recovered)
                self.assertEqual(len(reaper_result), 1)
                self.assertEqual(reaper_result[0][0], child_pid)
            finally:
                if supervisor is not None and supervisor.poll() is None:
                    supervisor.terminate()
                if supervisor is not None:
                    try:
                        supervisor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        supervisor.kill()
                        supervisor.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if child_reaper is not None:
                    child_reaper.join(timeout=5)
                elif child_pid is not None:
                    try:
                        os.waitpid(child_pid, 0)
                    except ChildProcessError:
                        pass
                libc.prctl(36, previous_subreaper.value, 0, 0, 0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "process recovery requires Linux")
    def test_recovery_cleans_marked_detached_descendant_after_supervisor_is_reaped(self):
        self._assert_recovery_cleans_marked_descendant(detached=True)

    @unittest.skipUnless(sys.platform.startswith("linux"), "process recovery requires Linux")
    def test_recovery_cleans_marked_same_group_descendant_after_supervisor_is_reaped(self):
        self._assert_recovery_cleans_marked_descendant(detached=False)

    def test_recovery_refuses_a_reused_process_id(self):
        stale = {
            "process_id": 123,
            "process_group_id": 456,
            "process_start_time": "old-start",
            "process_boot_id": "boot",
            "process_containment": execution.PROCESS_CONTAINMENT_SUBREAPER,
        }
        with patch.object(execution, "_proc_identity", return_value=("new-start", "boot")):
            with patch.object(execution.os, "killpg", return_value=None):
                self.assertFalse(execution._terminate_stale_process(stale))

        with patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution.os, "killpg", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertTrue(execution._terminate_stale_process(stale))

        self.assertTrue(
            execution._terminate_stale_process(
                {"process_id": 123, "process_group_id": 456, "process_fence_pending": False}
            )
        )
        with patch.object(execution, "_proc_identity", return_value=("old-start", "new-boot")):
            self.assertTrue(execution._terminate_stale_process(stale))

    def test_process_identity_handles_missing_proc_metadata(self):
        with patch.object(execution.Path, "read_text", side_effect=OSError("missing")):
            self.assertEqual(execution._proc_identity(123), (None, None))

    def test_pidfd_termination_is_pinned_and_fails_closed(self):
        with patch.object(execution.select, "select", side_effect=OSError("select failed")):
            self.assertFalse(execution._pidfd_terminated(9))
        with patch.object(execution.select, "select", return_value=([9], [], [])):
            self.assertTrue(execution._pidfd_terminated(9))

        with patch.object(execution.os, "pidfd_open", None):
            self.assertFalse(execution._terminate_pidfd(123))
        with patch.object(execution.os, "getpid", return_value=123):
            self.assertFalse(execution._terminate_pidfd(123))
        with patch.object(execution.os, "pidfd_open", side_effect=OSError(errno.EPERM, "denied")):
            self.assertFalse(execution._terminate_pidfd(123))
        with patch.object(execution.os, "pidfd_open", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertTrue(execution._terminate_pidfd(123))
        with patch.object(execution.os, "pidfd_open", side_effect=OSError(errno.ESRCH, "gone")), \
                patch.object(execution, "_process_group_is_absent", return_value=True):
            self.assertTrue(execution._terminate_pidfd(123, expected_group=456))
        with patch.object(execution.os, "pidfd_open", side_effect=OSError(errno.ESRCH, "gone")), \
                patch.object(execution, "_process_group_is_absent", return_value=True) as group_absent:
            self.assertTrue(
                execution._terminate_pidfd(
                    123,
                    expected_identity=("start", "boot"),
                    expected_group=456,
                )
            )
        group_absent.assert_called_once_with(
            456, expected_leader=(123, ("start", "boot"))
        )

        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=("new", "boot")), \
                patch.object(execution.os, "close"):
            self.assertFalse(
                execution._terminate_pidfd(123, expected_identity=("old", "boot"))
            )
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution, "_pidfd_terminated", return_value=True), \
                patch.object(execution.os, "close"):
            self.assertTrue(
                execution._terminate_pidfd(123, expected_identity=("old", "boot"))
            )
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution, "_pidfd_terminated", return_value=True), \
                patch.object(execution, "_process_group_is_absent", return_value=True), \
                patch.object(execution.os, "close"):
            self.assertTrue(
                execution._terminate_pidfd(
                    123, expected_identity=("old", "boot"), expected_group=456
                )
            )

        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=("old", "boot")), \
                patch.object(execution.os, "getpgid", return_value=789), \
                patch.object(execution.os, "close"):
            self.assertFalse(
                execution._terminate_pidfd(
                    123, expected_identity=("old", "boot"), expected_group=456
                )
            )
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=("old", "boot")), \
                patch.object(execution.os, "getpgid", side_effect=OSError(errno.EPERM, "denied")), \
                patch.object(execution.os, "close"):
            self.assertFalse(
                execution._terminate_pidfd(
                    123, expected_identity=("old", "boot"), expected_group=456
                )
            )
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution, "_proc_identity", return_value=("old", "boot")), \
                patch.object(execution.os, "getpgid", side_effect=OSError(errno.ESRCH, "gone")), \
                patch.object(execution, "_pidfd_terminated", return_value=True), \
                patch.object(execution, "_process_group_is_absent", return_value=True), \
                patch.object(execution.os, "close"):
            self.assertTrue(
                execution._terminate_pidfd(
                    123, expected_identity=("old", "boot"), expected_group=456
                )
            )

        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution.signal, "pidfd_send_signal", side_effect=ProcessLookupError()), \
                patch.object(execution.os, "close"):
            self.assertTrue(execution._terminate_pidfd(123))
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution.signal, "pidfd_send_signal", side_effect=ProcessLookupError()), \
                patch.object(execution, "_pidfd_terminated", return_value=True), \
                patch.object(execution, "_process_group_is_absent", return_value=True), \
                patch.object(execution.os, "close"):
            self.assertTrue(execution._terminate_pidfd(123, expected_group=456))
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution.signal, "pidfd_send_signal"), \
                patch.object(execution, "_pidfd_terminated", side_effect=[False, True]), \
                patch.object(execution.os, "close"):
            self.assertTrue(execution._terminate_pidfd(123))
        with patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution.signal, "pidfd_send_signal", side_effect=[None, ProcessLookupError()]), \
                patch.object(execution, "_pidfd_terminated", side_effect=[False, False]), \
                patch.object(execution.os, "close"):
            self.assertFalse(execution._terminate_pidfd(123))

    def test_process_group_termination_proves_group_absence(self):
        with patch.object(execution.os, "getpgrp", return_value=456), \
                patch.object(execution.os, "killpg") as killpg:
            self.assertFalse(execution._terminate_process_group(456))
        killpg.assert_not_called()
        with patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg", side_effect=OSError(errno.ESRCH, "gone")) as killpg:
            self.assertTrue(execution._terminate_process_group(456))
        self.assertEqual(killpg.call_count, 1)
        for side_effect, expected_calls in (
            ([None, OSError(errno.ESRCH, "gone")], 2),
            ([None, None, OSError(errno.ESRCH, "gone")], 3),
        ):
            with self.subTest(side_effect=side_effect), \
                    patch.object(execution.os, "getpgrp", return_value=1), \
                    patch.object(execution.os, "killpg", side_effect=side_effect) as killpg:
                self.assertTrue(execution._terminate_process_group(456))
            self.assertEqual(killpg.call_count, expected_calls)
        for side_effect, expected in (
            ([None, None, None, OSError(errno.ESRCH, "gone")], True),
            ([None, None, None, OSError(errno.EPERM, "denied")], False),
            ([None, None, None, None], False),
        ):
            with self.subTest(side_effect=side_effect), \
                    patch.object(execution.os, "getpgrp", return_value=1), \
                    patch.object(execution.os, "killpg", side_effect=side_effect):
                self.assertEqual(execution._terminate_process_group(456), expected)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "waitid"),
        "unreaped process-group recovery requires Linux waitid",
    )
    def test_process_group_with_only_unreaped_zombies_is_terminated(self):
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            os.setsid()
            os.write(write_fd, b"1")
            os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        child_reaped = False
        try:
            self.assertEqual(os.read(read_fd, 1), b"1")
            for _ in range(200):
                result = os.waitid(
                    os.P_PID,
                    child_pid,
                    os.WEXITED | os.WNOWAIT | os.WNOHANG,
                )
                if result is not None and result.si_pid == child_pid:
                    break
                time.sleep(0.01)
            else:
                self.fail("test child did not exit before the timeout")

            expected_identity = execution._proc_identity(child_pid)
            self.assertTrue(all(expected_identity))
            os.killpg(child_pid, 0)
            self.assertTrue(
                execution._process_group_is_absent(
                    child_pid,
                    expected_leader=(child_pid, expected_identity),
                )
            )
            child_reaped = True
        finally:
            os.close(read_fd)
            if not child_reaped:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError:
                    pass

    def test_process_group_reaping_fails_closed_on_wait_errors(self):
        for wait_result in (
            ChildProcessError(),
            OSError(errno.ECHILD, "no children"),
        ):
            with self.subTest(wait_result=wait_result), \
                    patch.object(execution, "_proc_identity", return_value=("leader", "boot")), \
                    patch.object(execution.os, "getpgrp", return_value=1), \
                    patch.object(execution.os, "killpg", side_effect=[None, None]), \
                    patch.object(execution.os, "waitpid", side_effect=wait_result) as waitpid:
                self.assertFalse(
                    execution._process_group_is_absent(
                        456, expected_leader=(456, ("leader", "boot"))
                    )
                )
                waitpid.assert_called_once_with(-456, os.WNOHANG)

        with patch.object(execution, "_proc_identity", return_value=("leader", "boot")), \
                patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg") as killpg, \
                patch.object(execution.os, "waitpid", side_effect=OSError(errno.EPERM, "denied")):
            self.assertFalse(
                execution._process_group_is_absent(
                    456, expected_leader=(456, ("leader", "boot"))
                )
            )
        killpg.assert_called_once_with(456, 0)

        with patch.object(execution.sys, "platform", "linux"), \
                patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(
                    execution.os,
                    "killpg",
                    side_effect=[None, OSError(errno.ESRCH, "group gone")],
                ) as killpg, \
                patch.object(
                    execution.os,
                    "kill",
                    side_effect=OSError(errno.ESRCH, "leader gone"),
                ) as kill, \
                patch.object(execution.os, "waitpid", side_effect=ChildProcessError()) as waitpid:
            self.assertTrue(
                execution._process_group_is_absent(
                    456, expected_leader=(456, ("leader", "boot"))
                )
            )
        kill.assert_called_once_with(456, 0)
        waitpid.assert_called_once_with(-456, os.WNOHANG)
        self.assertEqual(killpg.call_count, 2)

        with patch.object(execution.sys, "platform", "linux"), \
                patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg") as killpg, \
                patch.object(
                    execution.os, "kill", side_effect=OSError(errno.EPERM, "denied")
                ), \
                patch.object(execution.os, "waitpid") as waitpid:
            self.assertFalse(
                execution._process_group_is_absent(
                    456, expected_leader=(456, ("leader", "boot"))
                )
            )
        killpg.assert_called_once_with(456, 0)
        waitpid.assert_not_called()

        with patch.object(execution, "_proc_identity", return_value=("leader", "boot")), \
                patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg", side_effect=[None, None]), \
                patch.object(execution.os, "waitpid", return_value=(0, 0)):
            self.assertFalse(
                execution._process_group_is_absent(
                    456, expected_leader=(456, ("leader", "boot"))
                )
            )

        with patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg", return_value=None), \
                patch.object(execution.os, "waitpid") as waitpid:
            self.assertFalse(
                execution._process_group_is_absent(
                    456, expected_leader=(456, (None, None))
                )
            )
        waitpid.assert_not_called()

        with patch.object(execution.sys, "platform", "win32"), \
                patch.object(execution.os, "getpgrp") as getpgrp, \
                patch.object(execution.os, "killpg") as killpg, \
                patch.object(execution.os, "waitpid") as waitpid:
            self.assertFalse(execution._process_group_is_absent(456))
        waitpid.assert_not_called()
        getpgrp.assert_not_called()
        killpg.assert_not_called()

        with patch.object(execution.os, "getpgrp", return_value=456), \
                patch.object(execution.os, "killpg") as killpg, \
                patch.object(execution.os, "waitpid") as waitpid:
            self.assertFalse(execution._process_group_is_absent(456))
        waitpid.assert_not_called()
        killpg.assert_not_called()

    def test_proc_visibility_detection_fails_closed_for_hidepid_or_unknown_mounts(self):
        for mounts, restricted in (
            ("proc /proc proc rw,nosuid,nodev 0 0\n", False),
            ("proc /proc proc rw,hidepid=off 0 0\n", False),
            ("proc /proc proc rw,hidepid=1 0 0\n", False),
            ("proc /proc proc rw,hidepid=noaccess 0 0\n", False),
            ("proc /proc proc rw,hidepid=2,gid=1000 0 0\n", True),
            ("proc /proc proc rw,hidepid=invisible 0 0\n", True),
            ("proc /proc proc rw,hidepid=4 0 0\n", True),
            ("proc /proc proc rw,hidepid=ptraceable 0 0\n", True),
            ("sysfs /sys sysfs rw 0 0\n", True),
        ):
            with self.subTest(mounts=mounts), \
                    patch.object(execution.Path, "read_text", return_value=mounts), \
                    patch.object(execution.os, "getgroups", return_value=[3000]), \
                    patch.object(execution.os, "getgid", return_value=2000), \
                    patch.object(execution.os, "getegid", return_value=2001):
                self.assertEqual(execution._proc_visibility_restricted(), restricted)
        for mounts in (
            "proc /proc proc rw,hidepid=2,hidepid=4 0 0\n",
            "proc /proc proc rw,hidepid=unknown 0 0\n",
            "proc /proc proc rw,hidepid=3 0 0\n",
            "proc /proc proc rw,hidepid=2,gid=invalid 0 0\n",
        ):
            with self.subTest(mounts=mounts), \
                    patch.object(execution.Path, "read_text", return_value=mounts), \
                    patch.object(execution.os, "getgroups", return_value=[1000]), \
                    patch.object(execution.os, "getgid", return_value=2000), \
                    patch.object(execution.os, "getegid", return_value=2001):
                self.assertTrue(execution._proc_visibility_restricted())
        with patch.object(
            execution.Path,
            "read_text",
            return_value="proc /proc proc rw,hidepid=2,gid=1000 0 0\n",
        ), patch.object(execution.os, "getgroups", side_effect=OSError("groups unavailable")):
            self.assertTrue(execution._proc_visibility_restricted())
        for groups, real_gid, effective_gid in (
            ([1000], 2000, 2001),
            ([3000], 1000, 2001),
            ([3000], 2000, 1000),
        ):
            with self.subTest(groups=groups, real_gid=real_gid, effective_gid=effective_gid), \
                    patch.object(
                        execution.Path,
                        "read_text",
                        return_value="proc /proc proc rw,hidepid=2,gid=1000 0 0\n",
                    ), \
                    patch.object(execution.os, "getgroups", return_value=groups), \
                    patch.object(execution.os, "getgid", return_value=real_gid), \
                    patch.object(execution.os, "getegid", return_value=effective_gid):
                self.assertFalse(execution._proc_visibility_restricted())
        with patch.object(execution.Path, "read_text", side_effect=OSError("mounts unavailable")):
            self.assertTrue(execution._proc_visibility_restricted())

    def test_process_recovery_capability_probe_and_startup_gate(self):
        with patch.object(execution.sys, "platform", "win32"):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution, "fcntl", None):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution.os, "pidfd_open", None):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution.signal, "pidfd_send_signal", None):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution.Path, "is_dir", return_value=False):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution.Path, "is_dir", return_value=True), \
                patch.object(execution.Path, "read_text", return_value="boot"), \
                patch.object(execution, "_proc_identity", return_value=(None, None)):
            self.assertFalse(execution._process_recovery_supported())
        with patch.object(execution.Path, "is_dir", return_value=True), \
                patch.object(execution.Path, "read_text", side_effect=OSError("boot id missing")):
            self.assertFalse(execution._process_recovery_supported())

        manager = self.manager(runner=execution.run_command_bounded)
        with patch.object(execution, "_process_recovery_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "pidfd"):
                manager.start(
                    [self.phase("blocked", "blocked")],
                    execution_id="unsupported-recovery",
                )

        with patch.object(execution.Path, "is_dir", return_value=True), \
                patch.object(execution.Path, "read_text", return_value="boot"), \
                patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution.os, "pidfd_open", return_value=9), \
                patch.object(execution.signal, "pidfd_send_signal"), \
                patch.object(execution.select, "select", return_value=([], [], [])), \
                patch.object(execution.os, "close"):
            self.assertTrue(execution._process_recovery_supported())

    def test_marker_recovery_fences_process_group_without_checkpointed_identity(self):
        phase = {
            "process_id": None,
            "process_group_id": None,
            "process_launch_token": "marker",
        }
        with patch.object(
            execution,
            "_marker_process_identities",
            side_effect=[{123: ("start", "boot")}, {}],
        ), \
                patch.object(execution, "_terminate_pidfd", return_value=True) as terminate:
            self.assertTrue(execution._terminate_stale_process(phase))
        terminate.assert_called_once_with(123, expected_identity=("start", "boot"))
        with patch.object(execution, "_marker_process_identities", return_value={}):
            self.assertTrue(execution._terminate_stale_process(phase))
        with patch.object(execution, "_marker_process_identities", return_value={123: ("start", "boot")}), \
                patch.object(execution, "_terminate_pidfd", return_value=False):
            self.assertFalse(execution._terminate_stale_process(phase))
        phase["process_group_id"] = 456
        with patch.object(
            execution,
            "_marker_process_identities",
            side_effect=[{123: ("start", "boot")}, {}],
        ), \
                patch.object(execution, "_terminate_pidfd", return_value=True) as terminate, \
                patch.object(execution, "_process_group_is_absent", return_value=True):
            self.assertTrue(execution._terminate_stale_process(phase))
        terminate.assert_called_once_with(123, expected_identity=("start", "boot"))
        with patch.object(execution, "_marker_process_identities", return_value={123: ("start", "boot")}), \
                patch.object(execution, "_terminate_pidfd", return_value=False), \
                patch.object(execution, "_process_group_is_absent", return_value=True):
            self.assertFalse(execution._terminate_stale_process(phase))
        with patch.object(execution.os, "killpg", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertTrue(execution._terminate_stale_process({"process_id": 0, "process_group_id": 456}))

    def test_recovery_keeps_fence_when_unmarked_group_member_survives_marker_cleanup(self):
        phase = {
            "process_id": 123,
            "process_group_id": 456,
            "process_start_time": "start",
            "process_boot_id": "boot",
            "process_launch_token": "marker",
        }
        with patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(
                    execution,
                    "_marker_process_identities",
                    side_effect=[{789: ("child-start", "boot")}, {}],
                ), \
                patch.object(execution, "_terminate_pidfd", return_value=True) as terminate, \
                patch.object(execution, "_process_group_is_absent", return_value=False) as group_absent:
            self.assertFalse(execution._terminate_stale_process(phase))
        terminate.assert_called_once_with(789, expected_identity=("child-start", "boot"))
        group_absent.assert_called_once_with(
            456, expected_leader=(123, ("start", "boot"))
        )

    @patch.object(execution, "_proc_visibility_restricted", return_value=False)
    def test_marker_process_scan_fails_closed_for_unavailable_proc_metadata(
        self, _proc_visibility
    ):
        with patch.object(execution.sys, "platform", "win32"):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution, "_proc_visibility_restricted", return_value=True):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", side_effect=OSError("proc unavailable")):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["not-a-pid", "123"]), \
                patch.object(execution.Path, "read_bytes", side_effect=FileNotFoundError()):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.Path, "read_bytes", return_value=b"OTHER=value\0"):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(
                    execution.Path,
                    "read_bytes",
                    return_value=b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0",
                ):
            self.assertEqual(execution._marker_processes("marker"), {123})
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.Path, "read_bytes", side_effect=FileNotFoundError()):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(
                    execution.Path,
                    "read_bytes",
                    side_effect=OSError(errno.EIO, "environ read failed"),
                ):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(
                    execution.Path,
                    "read_bytes",
                    side_effect=PermissionError(errno.EACCES, "environ unavailable"),
                ):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.Path, "read_bytes", return_value=b"OTHER=value\0"):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.Path, "read_bytes", return_value=b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0"), \
                patch.object(execution.os, "getpgid", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.Path, "read_bytes", return_value=b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0"), \
                patch.object(execution.os, "getpgid", side_effect=OSError(errno.EPERM, "denied")):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution, "_marker_process_identities", return_value=None):
            self.assertFalse(
                execution._terminate_stale_process(
                    {"process_launch_token": "marker"}
                )
            )

        with patch.object(execution, "_marker_process_identities", return_value=None):
            self.assertFalse(
                execution._terminate_stale_process(
                    {"process_group_id": 456, "process_launch_token": "marker"}
                )
            )
        with patch.object(execution, "_marker_process_identities", return_value={}), \
                patch.object(execution, "_process_group_is_absent", return_value=True):
            self.assertTrue(
                execution._terminate_stale_process(
                    {"process_group_id": 456, "process_launch_token": "marker"}
                )
            )
        with patch.object(execution, "_marker_process_identities", return_value={}), \
                patch.object(execution, "_process_group_is_absent", return_value=False):
            self.assertFalse(
                execution._terminate_stale_process(
                    {
                        "process_id": 123,
                        "process_group_id": 456,
                        "process_launch_token": "marker",
                    }
                )
            )
        with patch.object(execution, "_proc_identity", return_value=("other", "boot")), \
                patch.object(execution, "_marker_process_identities", return_value={}), \
                patch.object(execution, "_process_group_is_absent", return_value=True):
            self.assertTrue(
                execution._terminate_stale_process(
                    {
                        "process_id": 123,
                        "process_group_id": 456,
                        "process_start_time": "start",
                        "process_boot_id": "boot",
                        "process_launch_token": "marker",
                    }
                )
            )

    def test_marker_process_identities_pin_scan_results(self):
        with patch.object(execution, "_marker_processes", return_value=None):
            self.assertIsNone(execution._marker_process_identities("marker"))
        with patch.object(execution, "_marker_processes", return_value=set()):
            self.assertEqual(execution._marker_process_identities("marker"), {})
        with patch.object(execution, "_marker_processes", return_value={123}), \
                patch.object(execution, "_proc_identity", return_value=("start", "boot")):
            self.assertEqual(
                execution._marker_process_identities("marker"),
                {123: ("start", "boot")},
            )
        with patch.object(execution, "_marker_processes", return_value={123}), \
                patch.object(execution, "_proc_identity", return_value=(None, None)):
            self.assertIsNone(execution._marker_process_identities("marker"))

    def test_recovery_fences_when_durable_marker_is_not_observable(self):
        manager = self.manager()
        manager.create([self.phase("launching", "launching")], execution_id="marker-gone")
        record = manager.get("marker-gone")
        phase = record["phases"][0]
        phase["status"] = "started"
        phase["attempts"] = 1
        phase["process_launch_token"] = "marker"
        phase["process_fence_pending"] = True
        manager.store.save(record)
        with patch.object(execution, "_marker_process_identities", return_value=None):
            recovered = manager.public("marker-gone")
        self.assertEqual(recovered["phases"][0]["status"], "interrupted")
        self.assertTrue(manager.get("marker-gone")["phases"][0]["process_fence_pending"])
        self.assertFalse(recovered["resume"]["available"])

    def test_recovery_reports_unconfirmed_group_fence(self):
        stale = {
            "process_id": 123,
            "process_group_id": 456,
            "process_start_time": "start",
            "process_boot_id": "boot",
        }
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution.os, "getpgrp", return_value=1), \
                patch.object(execution.os, "killpg", side_effect=[None, None, None, OSError("alive")]):
            self.assertFalse(execution._terminate_stale_process(stale))

    def test_recovery_requires_identity_and_distinguishes_permission_from_absence(self):
        stale = {
            "process_id": 123,
            "process_group_id": 456,
            "process_start_time": "start",
            "process_boot_id": "boot",
        }
        for identity in ((None, None), ("start", None), (None, "boot")):
            with self.subTest(identity=identity), \
                    patch.object(execution, "_proc_identity", return_value=identity), \
                    patch.object(execution.os, "killpg", return_value=None):
                self.assertFalse(execution._terminate_stale_process(stale))
        for missing in ({}, {"process_start_time": "start"}, {"process_boot_id": "boot"}):
            incomplete = {key: stale[key] for key in ("process_id", "process_group_id")}
            incomplete.update(missing)
            with self.subTest(missing=missing), patch.object(execution.os, "killpg", return_value=None):
                self.assertFalse(execution._terminate_stale_process(incomplete))
        with patch.object(execution, "_proc_identity", return_value=(None, None)), \
                patch.object(execution.os, "killpg", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertTrue(execution._terminate_stale_process(stale))

    def test_crash_before_process_checkpoint_blocks_recovery(self):
        manager = self.manager()
        manager.create([self.phase("launching", "launching")], execution_id="launch-window")
        record = manager.get("launch-window")
        phase = record["phases"][0]
        phase["status"] = "started"
        phase["attempts"] = 1
        phase["process_fence_pending"] = True
        manager.store.save(record)
        recovered = manager.public("launch-window")
        self.assertFalse(recovered["resume"]["available"])
        self.assertTrue(manager.get("launch-window")["phases"][0]["process_fence_pending"])

    def test_legacy_started_record_is_recovered_fail_closed(self):
        manager = self.manager()
        manager.create([self.phase("legacy", "legacy")], execution_id="legacy-started")
        record = manager.get("legacy-started")
        phase = record["phases"][0]
        phase["status"] = "started"
        phase["attempts"] = 1
        phase.pop("process_fence_pending", None)
        manager.store.save(record)
        recovered = manager.public("legacy-started")
        self.assertFalse(recovered["resume"]["available"])
        self.assertTrue(manager.get("legacy-started")["phases"][0]["process_fence_pending"])

    def test_pending_process_fence_blocks_resume(self):
        manager = self.manager()
        manager.create([self.phase("pending", "pending")], execution_id="pending-fence")
        record = manager.get("pending-fence")
        phase = record["phases"][0]
        phase["status"] = "started"
        phase["attempts"] = 1
        phase["process_id"] = 123
        phase["process_group_id"] = 456
        phase["process_fence_pending"] = True
        manager.store.save(record)
        with patch.object(execution, "_terminate_stale_process", return_value=False):
            recovered = manager.resume("pending-fence")
        self.assertFalse(recovered["resume"]["available"])
        self.assertTrue(manager.get("pending-fence")["phases"][0]["process_fence_pending"])
        self.assertEqual(manager.command_runner.calls, [])
        event_count = len(manager.get("pending-fence")["phases"][0]["events"])
        with patch.object(execution, "_terminate_stale_process", return_value=False):
            manager.resume("pending-fence")
        self.assertEqual(
            len(manager.get("pending-fence")["phases"][0]["events"]),
            event_count,
        )

    def test_default_runner_persists_and_clears_process_identity(self):
        base = self.manager()
        manager = PhaseExecutionManager(
            base.store.state_dir,
            command_runner=execution.run_command_bounded,
        )
        result = manager.start(
            [self.phase("identity", "printf identity")],
            execution_id="process-identity",
        )
        self.assertEqual(result["execution_status"], "completed")
        persisted = manager.get("process-identity")
        self.assertIsNone(persisted["phases"][0]["process_id"])
        self.assertIsNone(persisted["phases"][0]["process_group_id"])

    def test_default_runner_cleanup_failure_keeps_interruption_fenced(self):
        base = self.manager()
        interrupted = KeyboardInterrupt("interrupted")
        interrupted.cleanup_confirmed = False
        def interrupted_runner(*_args, **kwargs):
            kwargs["process_started"](123, 456)
            raise interrupted

        with patch.object(
            execution,
            "run_command_bounded",
            side_effect=interrupted_runner,
        ), patch.object(execution, "_proc_identity", return_value=("start", "boot")):
            manager = PhaseExecutionManager(
                base.store.state_dir,
                command_runner=execution.run_command_bounded,
            )
            with self.assertRaises(KeyboardInterrupt):
                manager.start([self.phase("unsafe", "unsafe")], execution_id="default-fence")
        record = manager.get("default-fence")
        phase = record["phases"][0]
        self.assertTrue(phase["process_fence_pending"])
        self.assertEqual(
            (phase["process_id"], phase["process_group_id"], phase["process_start_time"], phase["process_boot_id"]),
            (123, 456, "start", "boot"),
        )
        self.assertFalse(manager.public("default-fence")["resume"]["available"])

    def test_default_runner_exception_cleanup_failure_keeps_phase_fenced(self):
        base = self.manager()
        failure = RuntimeError("runner failed")
        failure.cleanup_confirmed = False
        def failed_runner(*_args, **kwargs):
            kwargs["process_started"](123, 456)
            raise failure

        with patch.object(
            execution,
            "run_command_bounded",
            side_effect=failed_runner,
        ), patch.object(execution, "_proc_identity", return_value=("start", "boot")):
            manager = PhaseExecutionManager(
                base.store.state_dir,
                command_runner=execution.run_command_bounded,
            )
            result = manager.start([self.phase("failed", "failed")], execution_id="default-error-fence")
        self.assertEqual(result["phases"][0]["status"], "failed")
        phase = manager.get("default-error-fence")["phases"][0]
        self.assertTrue(phase["process_fence_pending"])
        self.assertEqual(
            (phase["process_id"], phase["process_group_id"], phase["process_start_time"], phase["process_boot_id"]),
            (123, 456, "start", "boot"),
        )

    def test_default_runner_non_timeout_cleanup_failure_keeps_phase_fenced(self):
        base = self.manager()
        result = BoundedCommandResult(
            "partial",
            1,
            False,
            7,
            False,
            cleanup_confirmed=False,
        )
        def failed_runner(*_args, **kwargs):
            kwargs["process_started"](123, 456)
            return result

        with patch.object(execution, "run_command_bounded", side_effect=failed_runner) as runner, \
                patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=False):
            manager = PhaseExecutionManager(base.store.state_dir)
            first = manager.start(
                [self.phase("failed", "failed")],
                execution_id="default-non-timeout-fence",
            )
            self.assertEqual(first["phases"][0]["status"], "failed")
            self.assertTrue(
                manager.get("default-non-timeout-fence")["phases"][0]["process_fence_pending"]
            )
            blocked = manager.resume(
                "default-non-timeout-fence",
                retry_failed=True,
            )
        self.assertTrue(
            manager.get("default-non-timeout-fence")["phases"][0]["process_fence_pending"]
        )
        self.assertEqual(runner.call_count, 1)

    def test_unsafe_started_phase_is_recovered_before_resume_confirmation(self):
        runner = Runner()
        manager = self.manager(runner)
        manager.create(
            [self.phase("publish", "publish", side_effects="unsafe")],
            execution_id="unsafe-started",
        )
        record = manager.get("unsafe-started")
        record["phases"][0]["status"] = "started"
        record["phases"][0]["attempts"] = 1
        record["phases"][0]["process_id"] = 123
        record["phases"][0]["process_group_id"] = 456
        record["phases"][0]["process_start_time"] = "start"
        record["phases"][0]["process_boot_id"] = "boot"
        record["phases"][0]["process_containment"] = execution.PROCESS_CONTAINMENT_SUBREAPER
        manager.store.save(record)
        fresh = PhaseExecutionManager(manager.store.state_dir, command_runner=runner)
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True):
            blocked = fresh.resume("unsafe-started")
        self.assertEqual(blocked["phases"][0]["status"], "interrupted")
        self.assertTrue(blocked["resume"]["unsafe_confirmation_required"])
        self.assertEqual(runner.calls, [])
        with patch.object(execution, "_proc_identity", return_value=("start", "boot")), \
                patch.object(execution, "_terminate_pidfd", return_value=True):
            completed = fresh.resume("unsafe-started", confirm_unsafe=True)
        self.assertEqual(completed["execution_status"], "completed")

    def test_runner_error_is_persisted_and_can_be_retried(self):
        runner = Runner({"broken": RuntimeError("runner unavailable"), "long-error": RuntimeError("x" * 5000)})
        manager = self.manager(runner, process_cleanup=lambda: None)
        failed = manager.start([self.phase("broken", "broken")], execution_id="runner-error")
        self.assertEqual(failed["phases"][0]["status"], "failed")
        self.assertEqual(failed["phases"][0]["result"]["error_type"], "RuntimeError")
        runner.results["broken"] = ("fixed", 0, False, 5, False)
        recovered = manager.resume("runner-error", retry_failed=True)
        self.assertEqual(recovered["execution_status"], "completed")
        long_error = manager.start([self.phase("long-error", "long-error")], execution_id="long-error")
        self.assertEqual(len(long_error["phases"][0]["error"].encode("utf-8")), 4096)

    def test_runner_interrupt_is_persisted_and_reraised(self):
        runner = Runner({"interrupt": KeyboardInterrupt()})
        manager = self.manager(runner)
        with self.assertRaises(KeyboardInterrupt):
            manager.start([self.phase("interrupt", "interrupt")], execution_id="keyboard-interrupt")
        record = manager.get("keyboard-interrupt")
        self.assertEqual(record["phases"][0]["status"], "interrupted")
        self.assertTrue(record["phases"][0]["process_fence_pending"])
        self.assertFalse(manager.public("keyboard-interrupt")["resume"]["available"])

    def test_runner_interrupt_uses_cleanup_hook_before_allowing_resume(self):
        runner = Runner({"interrupt": KeyboardInterrupt()})
        cleanup_calls = []
        manager = self.manager(
            runner,
            process_cleanup=lambda: cleanup_calls.append("cleaned"),
        )
        with self.assertRaises(KeyboardInterrupt):
            manager.start([self.phase("interrupt", "interrupt")], execution_id="clean-interrupt")
        self.assertEqual(cleanup_calls, ["cleaned"])
        record = manager.get("clean-interrupt")
        self.assertEqual(record["phases"][0]["status"], "interrupted")
        self.assertFalse(record["phases"][0]["process_fence_pending"])
        self.assertTrue(manager.public("clean-interrupt")["resume"]["available"])

    def test_failed_opaque_cleanup_is_retried_before_runner_resume(self):
        attempts = {"runner": 0, "cleanup": 0}

        def runner(*_args):
            attempts["runner"] += 1
            if attempts["runner"] == 1:
                raise KeyboardInterrupt()
            return ("clean", 0, False, 5, False)

        def cleanup():
            attempts["cleanup"] += 1
            if attempts["cleanup"] == 1:
                raise RuntimeError("cleanup unavailable")

        manager = self.manager(runner, process_cleanup=cleanup)
        with self.assertRaises(KeyboardInterrupt):
            manager.start([self.phase("opaque", "opaque")], execution_id="opaque-retry")
        self.assertTrue(manager.get("opaque-retry")["phases"][0]["process_fence_pending"])
        resumed = manager.resume("opaque-retry")
        self.assertEqual(resumed["execution_status"], "completed")
        self.assertEqual(attempts, {"runner": 2, "cleanup": 2})

    def test_timeout_without_cleanup_confirmation_blocks_retry(self):
        runner = Runner({"timeout": ("partial", 124, False, 7, True)})
        manager = self.manager(runner)
        result = manager.start([self.phase("timeout", "timeout")], execution_id="timeout-fence")
        self.assertEqual(result["phases"][0]["status"], "timed_out")
        self.assertTrue(manager.get("timeout-fence")["phases"][0]["process_fence_pending"])
        blocked = manager.resume("timeout-fence", retry_failed=True)
        self.assertEqual(blocked["phases"][0]["attempts"], 1)
        self.assertEqual([call[0] for call in runner.calls], ["timeout"])
        self.assertFalse(manager.list_public()[0]["resume"]["available"])

    def test_recovery_preserves_failed_and_timed_out_status_for_retry_policy(self):
        for status in ("failed", "timed_out"):
            with self.subTest(status=status):
                runner = Runner()
                manager = self.manager(runner)
                execution_id = f"pending-{status}"
                manager.create([self.phase(status, status)], execution_id=execution_id)
                record = manager.get(execution_id)
                phase = record["phases"][0]
                phase["status"] = status
                phase["attempts"] = 1
                phase["process_id"] = 123
                phase["process_group_id"] = 456
                phase["process_start_time"] = "start"
                phase["process_boot_id"] = "boot"
                phase["process_containment"] = execution.PROCESS_CONTAINMENT_SUBREAPER
                phase["process_fence_pending"] = True
                manager.store.save(record)

                with patch.object(execution, "_terminate_stale_process", return_value=True):
                    recovered = manager.resume(execution_id)

                self.assertEqual(recovered["phases"][0]["status"], status)
                self.assertEqual(runner.calls, [])

    def test_interruption_fence_handles_default_and_failed_custom_cleanup(self):
        base = self.manager()
        default = PhaseExecutionManager(
            base.store.state_dir,
            command_runner=execution.run_command_bounded,
        )
        self.assertTrue(default._fence_interrupted_runner(True))
        self.assertFalse(default._fence_interrupted_runner(False))

        def failing_cleanup():
            raise RuntimeError("cleanup unavailable")

        custom = PhaseExecutionManager(
            base.store.state_dir,
            command_runner=Runner(),
            process_cleanup=failing_cleanup,
        )
        self.assertFalse(custom._fence_interrupted_runner())

    def test_output_handler_and_structured_metrics_are_retained(self):
        runner = Runner()
        manager = self.manager(runner)
        seen = []

        def handler(phase, output, result):
            seen.append((phase["name"], output, result["exit_code"]))
            return "cap_phase"

        result = manager.start(
            [self.phase("metric", "metric", structured_metrics={"items": 3})],
            execution_id="metrics-execution",
            output_handler=handler,
        )
        self.assertEqual(seen, [("metric", "output:metric", 0)])
        self.assertEqual(result["phases"][0]["result"]["capture_id"], "cap_phase")
        self.assertEqual(result["phases"][0]["result"]["structured_metrics"], {"items": 3})
        self.assertEqual(result["phases"][0]["structured_metrics"], {"items": 3})
        self.assertNotIn("structured_metrics", result["phases"][0]["attempt_results"][0])

    def test_output_handler_failure_does_not_lose_completed_result(self):
        manager = self.manager()

        def handler(*_args):
            raise OSError("capture unavailable")

        result = manager.start([self.phase("captured", "captured")], execution_id="handler-error", output_handler=handler)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["phases"][0]["result"]["capture_error"], "capture unavailable")

    def test_retry_runner_error_clears_previous_attempt_output(self):
        attempts = {"retry": 0}

        def results(*_args):
            attempts["retry"] += 1
            if attempts["retry"] == 1:
                return ("old output", 1, False, 10, False)
            raise RuntimeError("retry failed before output")

        manager = self.manager(Runner({"retry": results}))
        manager.start([self.phase("retry", "retry")], execution_id="stale-output")
        retried = manager.resume("stale-output", retry_failed=True)
        self.assertEqual(retried["phases"][0]["status"], "failed")
        self.assertEqual(manager.output("stale-output")["phases"][0]["output"], "")

    def test_paths_flags_and_resource_bounds_are_explicit(self):
        manager = self.manager(max_output_bytes=4096)
        relative = manager.create(
            [self.phase("path", "path", cwd=".", unsafe_side_effects=False)],
            execution_id="absolute-path",
        )
        self.assertTrue(Path(relative["phases"][0]["cwd"]).is_absolute())
        with self.assertRaisesRegex(ValueError, "contradictory"):
            manager.create(
                [self.phase("conflict", "conflict", side_effects="unsafe", unsafe_side_effects=False)],
                execution_id="conflicting-safety",
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            manager.create(
                [self.phase("bad-flag", "bad-flag", unsafe_side_effects="false")],
                execution_id="bad-safety-type",
            )
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            manager.create(
                [self.phase("bad-metrics", "bad-metrics", structured_metrics=object())],
                execution_id="bad-metrics",
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            manager.create(
                [self.phase("large-metrics", "large-metrics", structured_metrics={"x": "y" * 20000})],
                execution_id="large-metrics",
            )
        with self.assertRaisesRegex(ValueError, "at most"):
            manager.create(
                [self.phase(str(index), "noop") for index in range(MAX_EXECUTION_PHASES + 1)],
                execution_id="too-many-phases",
            )
        with patch.object(execution, "MAX_EXECUTION_STATE_BYTES", 1024 * 1024 + 6 * 1024):
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                manager.create(
                    [self.phase("a", "a", max_output_bytes=1024), self.phase("b", "b")],
                    execution_id="aggregate-output-limit",
                )
        self.assertEqual(manager.list_public(limit=1, offset=1), [])
        with self.assertRaisesRegex(ValueError, "between"):
            manager.list_public(limit=101)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            manager.list_public(offset=-1)
        with self.assertRaisesRegex(ValueError, "integer"):
            manager.list_public(limit=True)

    def test_lease_and_store_defensive_limits(self):
        manager = self.manager()
        with patch.object(execution.sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Linux platform"):
                with manager.store.lease("unsupported-platform"):
                    pass
        with patch.object(execution, "fcntl", None):
            with self.assertRaisesRegex(RuntimeError, "Linux file-locking"):
                with manager.store.lease("unsupported"):
                    pass
            self.assertFalse(manager.store.is_locked("unsupported"))
            manager.store.save({"execution_id": "unsupported-save"})

        with patch.object(execution.fcntl, "flock", side_effect=OSError(1, "not permitted")):
            with self.assertRaisesRegex(OSError, "not permitted"):
                with manager.store.lease("lease-error"):
                    pass
        lock_path = manager.store._lock_path("probe-error")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()
        with patch.object(execution.fcntl, "flock", side_effect=OSError(1, "not permitted")):
            with self.assertRaisesRegex(OSError, "not permitted"):
                manager.store.is_locked("probe-error")
        self.assertFalse(manager.store.is_locked("missing-lock"))
        with patch.object(execution.fcntl, "flock", side_effect=BlockingIOError()):
            self.assertTrue(manager.store.is_locked("probe-error"))
        with patch.object(execution.fcntl, "flock", return_value=None):
            self.assertFalse(manager.store.is_locked("probe-error"))

        with patch.object(execution, "MAX_EXECUTION_STATE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "state exceeds"):
                manager.store.save({"execution_id": "too-large"})
        manager.store.save({"execution_id": "record-one"})
        with patch.object(execution, "MAX_EXECUTION_RECORDS", 3):
            manager.store.save({"execution_id": "record-two"})
        with patch.object(execution, "MAX_EXECUTION_RECORDS", 3):
            with self.assertRaisesRegex(ValueError, "maximum"):
                manager.store.save({"execution_id": "record-three"})

        manager.create([self.phase("limited", "limited")], execution_id="attempt-limit")
        limited = manager.get("attempt-limit")
        limited["phases"][0]["status"] = "failed"
        limited["phases"][0]["attempts"] = MAX_PHASE_ATTEMPTS
        manager.store.save(limited)
        result = manager.resume("attempt-limit", retry_failed=True)
        self.assertIn("attempt limit", result["phases"][0]["error"])
        record_lock = manager.store.state_dir / ".records.lock"
        record_lock.unlink()
        lock_target = manager.store.state_dir.parent / "external-record-lock"
        lock_target.touch()
        record_lock.symlink_to(lock_target)
        with self.assertRaisesRegex(ValueError, "record-count lock file"):
            manager.store.save({"execution_id": "lock-symlink"})

    def test_validation_and_duplicate_boundaries(self):
        with self.assertRaisesRegex(ValueError, "at least 512"):
            self.manager(max_output_bytes=511)
        manager = self.manager(max_output_bytes=1024)
        cases = [
            ([], "non-empty list"),
            (["phase"], "must be an object"),
            ([{}], "name must be"),
            ([{"name": "x"}], "command must be"),
            ([self.phase("x", "x", cwd=1)], "cwd must be"),
            ([self.phase("x", "x", timeout_seconds=0)], "positive number"),
            ([self.phase("x", "x", timeout_seconds=True)], "positive number"),
            ([self.phase("x", "x", max_output_bytes=511)], "at least 512"),
            ([self.phase("x", "x", max_output_bytes=2048)], "cannot exceed"),
            ([self.phase("x", "x", side_effects="maybe")], "side_effects"),
            ([self.phase("x", "x", idempotency_key=1)], "idempotency_key"),
            ([self.phase("x", "x", structured_metrics=[])], "structured_metrics"),
        ]
        cases.append(([self.phase("x", "x", structured_metrics=object())], "JSON-compatible"))
        for index, (phases, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                manager.create(phases, execution_id=f"invalid-{index}")
        with self.assertRaisesRegex(ValueError, "unique"):
            manager.create([self.phase("same", "one"), self.phase("same", "two")], execution_id="duplicate")
        with self.assertRaisesRegex(ValueError, "resume_policy"):
            manager.create([self.phase("x", "x")], execution_id="bad-policy", resume_policy="always")
        with self.assertRaisesRegex(ValueError, "label"):
            manager.create([self.phase("x", "x")], execution_id="label", label="x" * 1025)
        with self.assertRaisesRegex(ValueError, "execution_id"):
            manager.create([self.phase("x", "x")], execution_id="x" * 257)
        generated = manager.create([self.phase("generated", "generated")])
        self.assertTrue(generated["execution_id"].startswith("exec_"))
        manager.create([self.phase("x", "x")], execution_id="duplicate-id")
        with self.assertRaisesRegex(ValueError, "already exists"):
            manager.create([self.phase("x", "x")], execution_id="duplicate-id")

    def test_store_missing_corrupt_and_list_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExecutionStore(directory)
            with self.assertRaisesRegex(KeyError, "not found"):
                store.load("missing")
            self.assertEqual(store.list(), [])
            with tempfile.TemporaryDirectory() as missing_directory:
                self.assertEqual(ExecutionStore(Path(missing_directory) / "missing").list(), [])
            corrupt = Path(directory) / "corrupt.json"
            corrupt.write_text("not json", encoding="utf-8")
            valid = Path(directory) / "valid.json"
            valid.write_text("[]", encoding="utf-8")
            self.assertEqual(store.list(), [])
            manager = PhaseExecutionManager(directory, command_runner=Runner())
            manager.create([self.phase("x", "x")], execution_id="listed")
            self.assertEqual(store.list()[0]["execution_id"], "listed")
            self.assertEqual(store.state_dir.stat().st_mode & 0o777, 0o700)
            summary_path = next(store.state_dir.glob("*.summary.json"))
            self.assertNotIn('"output"', summary_path.read_text(encoding="utf-8"))
            reconciled = manager.get("listed")
            reconciled["label"] = "reconciled"
            replace_calls = 0
            original_replace = execution.os.replace

            def replace_main_only(source, destination):
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 2:
                    raise OSError("summary replace failed")
                return original_replace(source, destination)

            with patch("execution.os.replace", side_effect=replace_main_only):
                with self.assertRaisesRegex(OSError, "summary replace failed"):
                    store.save(reconciled)
            main_path = store._path("listed")
            os_mtime = summary_path.stat().st_mtime + 1
            os.utime(main_path, (os_mtime, os_mtime))
            listed_after_partial_save = {
                item["execution_id"]: item for item in store.list()
            }
            self.assertEqual(listed_after_partial_save["listed"]["label"], "reconciled")
            invalid_state = store._path("invalid-state")
            invalid_state.write_text(json.dumps({"execution_id": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid state"):
                store.load("invalid-state")
            unreadable = store._path("unreadable")
            unreadable.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unreadable state"):
                store.load("unreadable")
            started = manager.get("listed")
            started["phases"][0]["status"] = "started"
            started["phases"][0]["attempts"] = 1
            store.save(started)
            listed_records = {item["execution_id"]: item for item in store.list()}
            self.assertEqual(listed_records["listed"]["execution_status"], "interrupted")
            self.assertEqual(store.list(offset=100), [])
            orphan_summary = store._summary_path("orphan")
            orphan_summary.write_text(
                json.dumps({
                    "execution_id": "orphan",
                    "updated_at": "9999-01-01T00:00:00Z",
                    "phases": [{"status": "started"}],
                }),
                encoding="utf-8",
            )
            self.assertEqual(store.list(limit=1), [])
            with patch("execution.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    store.save({"execution_id": "replace-failure"})
            outside = Path(directory).parent / "outside-record.json"
            outside.write_text(json.dumps({"execution_id": "outside"}), encoding="utf-8")
            record_path = store._path("listed")
            record_path.unlink()
            record_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "record file"):
                store.load("listed")
            self.assertFalse(store._path("replace-failure").exists())

    def test_public_resume_and_output_errors(self):
        manager = self.manager()
        with self.assertRaisesRegex(KeyError, "not found"):
            manager.public("missing")
        with self.assertRaisesRegex(KeyError, "not found"):
            manager.resume("missing")
        self.assertFalse(manager.store._lock_path("missing").exists())
        manager.create([self.phase("x", "x")], execution_id="lookup")
        self.assertEqual(manager.public("lookup", include_output=True)["phases"][0]["output"], "")
        pending = manager.get("lookup")
        pending["phases"][0]["output"] = "x" * 1024
        manager.store.save(pending)
        paged = manager.output("lookup", max_bytes=512)
        self.assertEqual(len(paged["phases"][0]["output"]), 512)
        pending = manager.get("lookup")
        pending["phases"][0]["output"] = None
        manager.store.save(pending)
        self.assertEqual(manager.public("lookup")["phases"][0]["output_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            manager.output("lookup", offset=-1)
        with self.assertRaisesRegex(ValueError, "integer"):
            manager.output("lookup", max_bytes="512")
        with self.assertRaisesRegex(ValueError, "between"):
            manager.output("lookup", max_bytes=511)
        with self.assertRaisesRegex(ValueError, "requires phase_name"):
            manager.output("lookup", offset=1)
        with self.assertRaisesRegex(KeyError, "Phase"):
            manager.output("lookup", "missing-phase")
        with self.assertRaisesRegex(ValueError, "phase_name"):
            manager.output("lookup", "x" * (MAX_PHASE_NAME_BYTES + 1))
        listed = manager.list_public()[0]
        self.assertEqual(listed["execution_id"], "lookup")
        self.assertNotIn("events", listed["phases"][0])

    def test_state_directory_rejects_symlink_redirection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "state"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable on this platform")
            manager = PhaseExecutionManager(link, command_runner=Runner())
            with self.assertRaisesRegex(ValueError, "symlink"):
                manager.create([self.phase("x", "x")], execution_id="symlink-state")
            with self.assertRaisesRegex(ValueError, "symlink"):
                manager.store.list()
            with self.assertRaisesRegex(ValueError, "state directory"):
                manager.store.is_locked("symlink-state")

            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_link = root / "parent-link"
            parent_link.symlink_to(real_parent, target_is_directory=True)
            parent_manager = PhaseExecutionManager(
                parent_link / "state", command_runner=Runner()
            )
            with self.assertRaisesRegex(ValueError, "path component"):
                parent_manager.create([self.phase("x", "x")], execution_id="parent-link")

    def test_state_directory_rejects_non_directory_after_creation(self):
        manager = self.manager()
        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaisesRegex(ValueError, "not a private directory"):
                manager.store._validate_state_dir()

    def test_state_directory_rejects_foreign_owner(self):
        manager = self.manager()
        manager.create([self.phase("owned", "owned")], execution_id="owned-state")
        current_uid = os.getuid()
        with patch.object(os, "getuid", return_value=current_uid + 1):
            with self.assertRaisesRegex(ValueError, "not owned"):
                manager.store._ensure_state_dir()
            with self.assertRaisesRegex(ValueError, "not owned"):
                manager.public("owned-state")
            with self.assertRaisesRegex(ValueError, "not owned"):
                manager.store.list()

    def test_state_directory_security_rejects_missing_and_non_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = ExecutionStore(Path(directory) / "missing")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                missing._validate_state_dir()
            os.chmod(manager_path := Path(directory), 0o755)
            try:
                with self.assertRaisesRegex(ValueError, "not private"):
                    ExecutionStore(manager_path)._validate_state_dir()
            finally:
                os.chmod(manager_path, 0o700)

        with tempfile.TemporaryDirectory() as directory:
            absent = ExecutionStore(Path(directory) / "absent")
            self.assertFalse(absent.is_locked("missing"))

    def test_capacity_reports_pairs_and_does_not_run_recovery(self):
        manager = self.manager()
        manager.create([self.phase("inspect", "inspect")], execution_id="capacity-record")
        record = manager.get("capacity-record")
        record["phases"][0]["status"] = "started"
        record["phases"][0]["process_fence_pending"] = True
        manager.store.save(record)
        extra = manager.store.state_dir / "unmanaged.bin"
        extra.write_bytes(b"other")

        with patch("execution._terminate_stale_process", side_effect=AssertionError("capacity must not recover")):
            capacity = manager.capacity()

        self.assertEqual(capacity["record_count"], 1)
        self.assertEqual(capacity["summary_count"], 1)
        self.assertEqual(capacity["paired_record_count"], 1)
        self.assertEqual(capacity["fence_pending_record_count"], 1)
        self.assertEqual(capacity["other_file_bytes"], 5)
        self.assertEqual(capacity["committed_bytes"], capacity["record_bytes"] + capacity["summary_bytes"])
        self.assertTrue(capacity["capacity_confident"])

    def test_capacity_reports_missing_paths_and_uncertain_filesystem_or_leases(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = ExecutionStore(Path(directory) / "missing")
            capacity = missing.capacity()
            self.assertFalse(capacity["state_directory_exists"])
            self.assertEqual(capacity["record_count"], 0)

        manager = self.manager()
        manager.create([self.phase("capacity", "capacity")], execution_id="capacity-lock")
        with patch("execution.shutil.disk_usage", side_effect=OSError("no statvfs")), \
                patch.object(manager.store, "is_locked", side_effect=OSError("lock error")):
            capacity = manager.capacity()
        self.assertIsNone(capacity["filesystem"]["free_bytes"])
        self.assertGreater(capacity["uncertain_lock_count"], 0)
        self.assertFalse(capacity["capacity_confident"])
        self.assertEqual(capacity["available_bytes"], 0)

    def test_invalid_checkpoint_reservation_fails_capacity_closed(self):
        manager = self.manager()
        (manager.store.state_dir / ".checkpoint-invalid.json").write_text(
            "not json", encoding="utf-8"
        )
        capacity = manager.capacity()
        self.assertEqual(capacity["invalid_reservation_count"], 1)
        self.assertFalse(capacity["capacity_confident"])
        self.assertEqual(capacity["available_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "reservation is unreadable"):
            manager.store.save({"execution_id": "quota-fail-closed", "phases": []})

    def test_capacity_reports_symlink_checkpoint_reservation(self):
        manager = self.manager()
        target = manager.store.state_dir / "reservation-target"
        target.write_text("{}", encoding="utf-8")
        reservation = manager.store.state_dir / ".checkpoint-symlink.json"
        try:
            reservation.symlink_to(target)
        except OSError:
            self.skipTest("symlinks are unavailable on this platform")

        capacity = manager.capacity()
        self.assertEqual(capacity["symlink_count"], 1)
        self.assertEqual(capacity["invalid_reservation_count"], 1)
        self.assertFalse(capacity["capacity_confident"])
        self.assertEqual(capacity["available_bytes"], 0)

    def test_aggregate_quota_and_checkpoint_reservation_are_accounted(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "MAX_EXECUTION_STATE_BYTES", 4096
        ):
            store = ExecutionStore(
                directory,
                quota_bytes=10 * 1024,
                checkpoint_reserve_bytes=4096,
            )
            store.save({"execution_id": "quota-a", "payload": "a" * 1100})
            store.save({"execution_id": "quota-b", "payload": "b" * 1100})
            with self.assertRaisesRegex(ValueError, "aggregate quota"):
                store.save({"execution_id": "quota-c", "payload": "c" * 1100})

        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "MAX_EXECUTION_STATE_BYTES", 1024
        ):
            store = ExecutionStore(
                directory,
                quota_bytes=4096,
                checkpoint_reserve_bytes=1024,
            )
            store.save({"execution_id": "reserved", "phases": []})
            with store.lease("reserved"):
                store.reserve_checkpoint("reserved", 512)
                capacity = store.capacity()
                self.assertEqual(capacity["active_reserved_bytes"], 512)
                with self.assertRaisesRegex(ValueError, "exceeded its reserved growth"):
                    store.save({
                        "execution_id": "reserved",
                        "phases": [],
                        "large": "x" * 800,
                    })
                store.save({"execution_id": "reserved", "phases": [], "updated": True})
                store.release_checkpoint("reserved")
                store.reserve_checkpoint("reserved", 512)
                store.save(
                    {"execution_id": "reserved", "phases": [], "updated": True, "result": "done"},
                    consume_checkpoint_reservation=True,
                )
                store.release_checkpoint("reserved")
            self.assertEqual(store.capacity()["active_reserved_bytes"], 0)

            with store.lease("reserved"):
                store.reserve_checkpoint("reserved", 512)
            self.assertEqual(store.capacity()["stale_reserved_bytes"], 512)
            store.save({"execution_id": "after-stale", "phases": []})
            self.assertEqual(store.capacity()["stale_reserved_bytes"], 0)

            with store.lease("reserved"):
                with self.assertRaisesRegex(ValueError, "insufficient checkpoint headroom"):
                    store.reserve_checkpoint("reserved", 3000)

            with store.lease("reserved"), patch(
                "execution.shutil.disk_usage",
                return_value=SimpleNamespace(total=4096, used=4096, free=0),
            ):
                with self.assertRaisesRegex(ValueError, "filesystem has insufficient free space"):
                    store.reserve_checkpoint("reserved", 512)

    def test_storage_limits_and_checkpoint_reservation_validate_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "MAX_EXECUTION_STATE_BYTES", 1024
        ):
            with self.assertRaisesRegex(ValueError, "quota must be a positive integer"):
                ExecutionStore(directory, quota_bytes=0, checkpoint_reserve_bytes=1024)
            with self.assertRaisesRegex(ValueError, "at least 1,024 bytes"):
                ExecutionStore(directory, quota_bytes=4096, checkpoint_reserve_bytes=512)
            with self.assertRaisesRegex(ValueError, "smaller than the quota"):
                ExecutionStore(directory, quota_bytes=1024, checkpoint_reserve_bytes=1024)

            store = ExecutionStore(
                directory,
                quota_bytes=4096,
                checkpoint_reserve_bytes=1024,
            )
            with self.assertRaisesRegex(RuntimeError, "require the execution lease"):
                store.reserve_checkpoint("no-lease", 100)
            with store.lease("no-lease"):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    store.reserve_checkpoint("no-lease", True)
            store.release_checkpoint("no-reservation")

    def test_retirement_dry_run_and_archive_preserve_record_pair(self):
        manager = self.manager()
        manager.create([self.phase("archive", "archive")], execution_id="archive-me", label="archive label")
        record_path = manager.store._path("archive-me")
        summary_path = manager.store._summary_path("archive-me")
        expected_bytes = record_path.stat().st_size + summary_path.stat().st_size

        preview = manager.retire(["archive-me"])
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["eligible_count"], 1)
        self.assertEqual(preview["projected_reclaimed_bytes"], expected_bytes)
        self.assertTrue(record_path.exists())
        self.assertTrue(summary_path.exists())

        with tempfile.TemporaryDirectory() as archive_directory:
            archive_path = Path(archive_directory) / "execution-records.tar"
            result = manager.retire(
                ["archive-me"],
                archive_path=str(archive_path),
                dry_run=False,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["projected_reclaimed_bytes"], expected_bytes)
            self.assertFalse(record_path.exists())
            self.assertFalse(summary_path.exists())
            self.assertEqual(archive_path.stat().st_mode & 0o777, 0o600)
            with tarfile.open(archive_path, "r") as archive:
                manifest = json.load(archive.extractfile("manifest.json"))
                names = set(archive.getnames())
            self.assertEqual(manifest["executions"][0]["execution_id"], "archive-me")
            self.assertIn(f"records/{manager.store._filename('archive-me')}", names)
            self.assertIn(
                f"records/{manager.store._filename('archive-me')[:-5]}.summary.json",
                names,
            )

    def test_retirement_refuses_active_fence_pending_and_in_directory_archives(self):
        manager = self.manager()
        manager.create([self.phase("protected", "protected")], execution_id="protected")
        with manager.store.lease("protected"):
            preview = manager.retire(["protected"])
            self.assertFalse(preview["items"][0]["eligible"])
            self.assertIn("active lease", preview["items"][0]["reason"])

        record = manager.get("protected")
        record["phases"][0]["process_fence_pending"] = True
        manager.store.save(record)
        preview = manager.retire(["protected"])
        self.assertFalse(preview["items"][0]["eligible"])
        self.assertIn("fence-pending", preview["items"][0]["reason"])

        with self.assertRaisesRegex(ValueError, "outside"):
            manager.retire(
                ["protected"],
                archive_path=str(manager.store.state_dir / "archive.tar"),
                dry_run=False,
            )

    def test_retirement_validates_ids_and_archive_destination(self):
        manager = self.manager()
        manager.create([self.phase("archive", "archive")], execution_id="archive-validation")
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            manager.retire([])
        with self.assertRaisesRegex(ValueError, "at most"):
            manager.retire([
                f"retire-{index}"
                for index in range(execution.MAX_EXECUTION_RETIRE_BATCH + 1)
            ])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            manager.retire(["archive-validation", "archive-validation"])
        with self.assertRaisesRegex(ValueError, "execution_id"):
            manager.retire([""])
        with self.assertRaisesRegex(ValueError, "boolean"):
            manager.retire(["archive-validation"], dry_run=1)
        with self.assertRaisesRegex(ValueError, "archive_path is required"):
            manager.retire(["archive-validation"], dry_run=False)

        with tempfile.TemporaryDirectory() as archive_directory:
            with self.assertRaisesRegex(ValueError, "non-empty"):
                manager.retire(["archive-validation"], archive_path="", dry_run=False)
            with self.assertRaisesRegex(ValueError, "too long"):
                manager.retire(["archive-validation"], archive_path="x" * 4097, dry_run=False)
            with self.assertRaisesRegex(ValueError, "already exist"):
                manager.retire(
                    ["archive-validation"],
                    archive_path=str(Path(archive_directory) / "missing" / "archive.tar"),
                    dry_run=False,
                )
            existing = Path(archive_directory) / "existing.tar"
            existing.touch()
            with self.assertRaisesRegex(ValueError, "already exists"):
                manager.retire(
                    ["archive-validation"], archive_path=str(existing), dry_run=False
                )
            symlink = Path(archive_directory) / "archive-link.tar"
            try:
                symlink.symlink_to(existing)
            except OSError:
                pass
            else:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    manager.retire(
                        ["archive-validation"], archive_path=str(symlink), dry_run=False
                    )
                real_parent = Path(archive_directory) / "real-parent"
                real_parent.mkdir()
                parent_link = Path(archive_directory) / "parent-link"
                parent_link.symlink_to(real_parent, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    manager.retire(
                        ["archive-validation"],
                        archive_path=str(parent_link / "archive.tar"),
                        dry_run=False,
                    )

            with patch("execution.os.access", return_value=False):
                with self.assertRaisesRegex(ValueError, "not writable"):
                    manager.retire(
                        ["archive-validation"],
                        archive_path=str(Path(archive_directory) / "unwritable.tar"),
                        dry_run=False,
                    )

            parent_mode = os.stat(archive_directory).st_mode & 0o777
            try:
                os.chmod(archive_directory, 0o777)
                with self.assertRaisesRegex(ValueError, "group/world writable"):
                    manager.retire(
                        ["archive-validation"],
                        archive_path=str(Path(archive_directory) / "unsafe.tar"),
                        dry_run=False,
                    )
            finally:
                os.chmod(archive_directory, parent_mode)

            foreign_uid = os.stat(archive_directory).st_uid + 1
            with patch("execution.os.getuid", return_value=foreign_uid):
                with self.assertRaisesRegex(ValueError, "owned by the current user"):
                    manager.retire(
                        ["archive-validation"],
                        archive_path=str(Path(archive_directory) / "foreign.tar"),
                        dry_run=False,
                    )

    def test_retirement_preview_reports_missing_and_unpaired_records(self):
        manager = self.manager()
        missing = manager.retire(["missing-record"])
        self.assertFalse(missing["items"][0]["eligible"])
        self.assertIn("missing", missing["items"][0]["reason"])

        manager.create([self.phase("unpaired", "unpaired")], execution_id="unpaired-record")
        manager.store._summary_path("unpaired-record").unlink()
        unpaired = manager.retire(["unpaired-record"])
        self.assertFalse(unpaired["items"][0]["eligible"])
        self.assertEqual(unpaired["projected_reclaimed_bytes"], 0)
        capacity = manager.capacity()
        self.assertEqual(capacity["unpaired_record_count"], 1)
        (manager.store.state_dir / "invalid.json").write_text("not json", encoding="utf-8")
        self.assertEqual(manager.capacity()["invalid_record_count"], 1)

        manager.create([self.phase("corrupt", "corrupt")], execution_id="corrupt-pair")
        manager.store._summary_path("corrupt-pair").write_text("not json", encoding="utf-8")
        corrupt = manager.retire(["corrupt-pair"])
        self.assertFalse(corrupt["items"][0]["eligible"])
        self.assertIn("unreadable", corrupt["items"][0]["reason"])

    def test_retirement_revalidates_races_and_leaves_sources_on_archive_failure(self):
        manager = self.manager()
        manager.create([self.phase("race", "race")], execution_id="retire-race")
        store = manager.store
        with tempfile.TemporaryDirectory() as archive_directory:
            archive_path = Path(archive_directory) / "raced.tar"
            original_writer = store._write_archive_temp

            def change_record_after_archiving(target, snapshots, *, source_digests=None):
                temporary = original_writer(
                    target,
                    snapshots,
                    source_digests=source_digests,
                )
                record_path = snapshots[0][2]
                record_path.write_bytes(record_path.read_bytes().replace(b"race", b"racy", 1))
                return temporary

            with patch.object(store, "_write_archive_temp", side_effect=change_record_after_archiving):
                with self.assertRaisesRegex(ValueError, "state changed"):
                    manager.retire(
                        ["retire-race"],
                        archive_path=str(archive_path),
                        dry_run=False,
                    )
            self.assertTrue(store._path("retire-race").exists())
            self.assertFalse(archive_path.exists())

        manager.create([self.phase("busy", "busy")], execution_id="retire-busy")
        with tempfile.TemporaryDirectory() as archive_directory:
            archive_path = Path(archive_directory) / "busy.tar"
            with patch.object(store, "lease", side_effect=ExecutionBusyError("busy")):
                refused = manager.retire(
                    ["retire-busy"],
                    archive_path=str(archive_path),
                    dry_run=False,
                )
            self.assertEqual(refused["status"], "refused")
            self.assertIn("became active", refused["reason"])
            self.assertFalse(archive_path.exists())

        manager.create([self.phase("archive-failure", "archive-failure")], execution_id="archive-failure")
        with tempfile.TemporaryDirectory() as archive_directory:
            archive_path = Path(archive_directory) / "failed.tar"
            with patch.object(store, "_write_archive_temp", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    manager.retire(
                        ["archive-failure"],
                        archive_path=str(archive_path),
                        dry_run=False,
                    )
            self.assertTrue(store._path("archive-failure").exists())
            self.assertFalse(archive_path.exists())

    def test_archive_temp_cleanup_when_a_source_disappears(self):
        manager = self.manager()
        manager.create([self.phase("archive-temp", "archive-temp")], execution_id="archive-temp")
        item, record, record_path, summary_path = manager.store._retirement_snapshot("archive-temp")
        record_path.unlink()
        with tempfile.TemporaryDirectory() as archive_directory:
            archive_path = Path(archive_directory) / "temporary.tar"
            with self.assertRaises(FileNotFoundError):
                manager.store._write_archive_temp(
                    archive_path,
                    [(item, record, record_path, summary_path)],
                )
            self.assertEqual(list(Path(archive_directory).glob(".ephemeral-executions-archive-*")), [])

    def test_managed_file_inventory_handles_disappearing_and_unmanaged_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = ExecutionStore(Path(directory) / "missing")
            self.assertEqual(missing._managed_file_sizes(), (0, 0, 0, 0))

            missing.state_dir.mkdir(mode=0o700)
            state_dir = missing.state_dir
            (state_dir / "record.json").write_bytes(b"rec")
            (state_dir / "record.summary.json").write_bytes(b"sum!")
            (state_dir / "tmp-crash-leftover").write_bytes(b"temp")
            (state_dir / "readme.txt").write_bytes(b"ignored")
            vanished = state_dir / "vanished.json"
            vanished.write_bytes(b"gone")
            original_stat = Path.stat
            vanished_stat_calls = 0

            def stat_with_disappearing_file(path, *args, **kwargs):
                nonlocal vanished_stat_calls
                if path == vanished:
                    vanished_stat_calls += 1
                    if vanished_stat_calls > 1:
                        raise FileNotFoundError(path)
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", stat_with_disappearing_file):
                self.assertEqual(missing._managed_file_sizes(), (3, 4, 1, 4))

            (state_dir / "not-a-record.json").mkdir()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                missing._managed_file_sizes()

    def test_reservation_scans_and_checkpoint_cleanup_handle_races(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExecutionStore(Path(directory) / "state")
            self.assertEqual(store._reservation_entries(), [])
            store.state_dir.mkdir(mode=0o700)

            fifo = store.state_dir / ".checkpoint-fifo.json"
            try:
                os.mkfifo(fifo)
            except (AttributeError, OSError):
                self.skipTest("FIFO creation is unavailable on this platform")
            fifo_entries = store._reservation_entries()
            self.assertEqual(len(fifo_entries), 1)
            self.assertTrue(fifo_entries[0]["invalid"])
            fifo.unlink()

            execution_id = "reservation-scan"
            reservation = store._reservation_path(execution_id)
            reservation.write_text(
                json.dumps({"execution_id": execution_id, "reserved_bytes": 9}),
                encoding="utf-8",
            )
            with patch("execution.stat.S_ISREG", side_effect=[True, False]):
                scanned = store._reservation_entries()
            self.assertTrue(scanned[0]["invalid"])
            reservation.write_text(
                json.dumps({"execution_id": "different-id", "reserved_bytes": 9}),
                encoding="utf-8",
            )
            self.assertTrue(store._reservation_entries()[0]["invalid"])
            reservation.unlink()

            store._write_reservation_locked("stale-reservation", 11)
            stale_path = store._reservation_path("stale-reservation")
            original_unlink = Path.unlink

            def stale_reservation_disappears(path, *args, **kwargs):
                if path == stale_path:
                    raise FileNotFoundError(path)
                return original_unlink(path, *args, **kwargs)

            with patch.object(store, "is_locked", return_value=False), patch.object(
                Path, "unlink", stale_reservation_disappears
            ):
                self.assertEqual(
                    store._reserved_checkpoint_bytes_locked(clean_stale=True), 0
                )

            store._write_reservation_locked("counted-reservation", 37)
            self.assertEqual(
                store._reserved_checkpoint_bytes_locked(
                    exclude_execution_id="stale-reservation"
                ),
                37,
            )

            store._write_reservation_locked("own-stale-reservation", 17)
            store.save({"execution_id": "own-stale-reservation", "phases": []})
            self.assertFalse(store._reservation_path("own-stale-reservation").exists())

            replacement_failure = store._reservation_path("replace-failure")
            with patch("execution.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    store._write_reservation_locked("replace-failure", 1)
            self.assertFalse(replacement_failure.exists())
            self.assertEqual(list(store.state_dir.glob("tmp*")), [])

    def test_consumed_reservation_disappearing_during_unlink_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExecutionStore(directory)
            execution_id = "consume-race"
            store.save({"execution_id": execution_id, "phases": []})
            with store.lease(execution_id):
                store.reserve_checkpoint(execution_id, 256)
                reservation = store._reservation_path(execution_id)
                original_unlink = Path.unlink

                def consume_reservation_race(path, *args, **kwargs):
                    if path == reservation:
                        original_unlink(path, *args, **kwargs)
                        raise FileNotFoundError(path)
                    return original_unlink(path, *args, **kwargs)

                with patch.object(Path, "unlink", consume_reservation_race):
                    store.save(
                        {"execution_id": execution_id, "phases": [], "saved": True},
                        consume_checkpoint_reservation=True,
                    )
            self.assertFalse(reservation.exists())

    def test_capacity_skips_vanished_nonregular_and_temporary_entries(self):
        manager = self.manager()
        manager.create([self.phase("capacity-race", "capacity-race")], execution_id="capacity-race")
        state_dir = manager.store.state_dir
        (state_dir / "not-a-file").mkdir()
        (state_dir / ".diagnostic-note").write_bytes(b"hidden")
        (state_dir / "tmp-orphan").write_bytes(b"orphan")
        (state_dir / "mismatched.json").write_text(
            json.dumps({"execution_id": "wrong-id", "phases": []}), encoding="utf-8"
        )
        vanished = state_dir / "vanished-during-scan.json"
        vanished.write_text("{}", encoding="utf-8")
        manager.store._write_reservation_locked("uncertain-reservation", 73)
        original_stat = Path.stat

        def stat_with_disappearing_file(path, *args, **kwargs):
            if path == vanished:
                raise FileNotFoundError(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", stat_with_disappearing_file), patch.object(
            manager.store, "is_locked", side_effect=OSError("lease state unavailable")
        ):
            capacity = manager.capacity()

        self.assertEqual(capacity["temporary_file_bytes"], len(b"orphan"))
        self.assertEqual(capacity["other_file_bytes"], len(b"hidden"))
        self.assertGreaterEqual(capacity["invalid_record_count"], 1)
        self.assertGreaterEqual(capacity["uncertain_lock_count"], 1)
        self.assertFalse(capacity["capacity_confident"])

    def test_retirement_snapshot_normalizes_metadata_and_rejects_bad_state(self):
        manager = self.manager()
        store = manager.store
        store._ensure_state_dir()

        invalid_phases_id = "retirement-invalid-phases"
        phases_path = store._path(invalid_phases_id)
        phases_path.write_text(
            json.dumps({"execution_id": invalid_phases_id, "phases": None}),
            encoding="utf-8",
        )
        store._summary_path(invalid_phases_id).write_text("{}", encoding="utf-8")
        invalid_phases = store._retirement_snapshot(invalid_phases_id)[0]
        self.assertIn("ValueError", invalid_phases["reason"])

        invalid_summary_id = "retirement-invalid-summary"
        manager.create([self.phase("summary", "summary")], execution_id=invalid_summary_id)
        store._summary_path(invalid_summary_id).write_text("[]", encoding="utf-8")
        invalid_summary = store._retirement_snapshot(invalid_summary_id)[0]
        self.assertIn("unreadable", invalid_summary["reason"])

        metadata_id = "retirement-metadata"
        store.save({
            "execution_id": metadata_id,
            "execution_status": None,
            "label": None,
            "updated_at": 123,
            "phases": [],
        })
        metadata, *_ = store._retirement_snapshot(metadata_id)
        self.assertEqual(metadata["execution_status"], "unknown")
        self.assertEqual(metadata["label"], "")
        self.assertIsNone(metadata["updated_at"])

        with patch.object(store, "is_locked", side_effect=OSError("uncertain lease")):
            uncertain = store._retirement_snapshot(metadata_id)[0]
        self.assertIn("lease state is uncertain", uncertain["reason"])

    def test_retirement_refusal_revalidation_and_removal_failures(self):
        manager = self.manager()
        store = manager.store
        with tempfile.TemporaryDirectory() as archive_directory:
            archive_root = Path(archive_directory)

            manager.create([self.phase("fenced", "fenced")], execution_id="retire-fenced")
            fenced = manager.get("retire-fenced")
            fenced["phases"][0]["process_fence_pending"] = True
            store.save(fenced)
            refused = manager.retire(
                ["retire-fenced"],
                archive_path=str(archive_root / "fenced.tar"),
                dry_run=False,
            )
            self.assertEqual(refused["status"], "refused")

            refresh_id = "retire-refresh"
            manager.create([self.phase("refresh", "refresh")], execution_id=refresh_id)
            original_snapshot = store._retirement_snapshot
            snapshot_calls = 0

            def become_fence_pending(execution_id, *, ignore_active_lease=False):
                nonlocal snapshot_calls
                snapshot_calls += 1
                if snapshot_calls == 2:
                    record = store._read_record(execution_id)
                    record["phases"][0]["process_fence_pending"] = True
                    store.save(record)
                return original_snapshot(
                    execution_id, ignore_active_lease=ignore_active_lease
                )

            with patch.object(store, "_retirement_snapshot", side_effect=become_fence_pending):
                changed = manager.retire(
                    [refresh_id],
                    archive_path=str(archive_root / "refresh.tar"),
                    dry_run=False,
                )
            self.assertEqual(changed["status"], "refused")
            self.assertIn("no longer eligible", changed["reason"])

            nonregular_id = "retire-nonregular-reservation"
            manager.create([self.phase("reservation", "reservation")], execution_id=nonregular_id)
            store._reservation_path(nonregular_id).mkdir()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                manager.retire(
                    [nonregular_id],
                    archive_path=str(archive_root / "nonregular.tar"),
                    dry_run=False,
                )
            self.assertTrue(store._path(nonregular_id).exists())
            store._reservation_path(nonregular_id).rmdir()

            unlink_id = "retire-record-unlink-failure"
            manager.create([self.phase("unlink", "unlink")], execution_id=unlink_id)
            record_path = store._path(unlink_id)
            original_unlink = Path.unlink

            def fail_record_unlink(path, *args, **kwargs):
                if path == record_path:
                    raise PermissionError("record removal denied")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", fail_record_unlink):
                removal = manager.retire(
                    [unlink_id],
                    archive_path=str(archive_root / "unlink.tar"),
                    dry_run=False,
                )
            self.assertEqual(removal["status"], "partial")
            self.assertEqual(removal["retired_count"], 0)
            self.assertTrue(record_path.exists())

            reservation_id = "retire-reservation-unlink-failure"
            manager.create([self.phase("reservation-remove", "reservation-remove")], execution_id=reservation_id)
            reservation_path = store._reservation_path(reservation_id)
            store._write_reservation_locked(reservation_id, 32)

            def fail_reservation_unlink(path, *args, **kwargs):
                if path == reservation_path:
                    raise PermissionError("reservation removal denied")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", fail_reservation_unlink):
                reservation_removal = manager.retire(
                    [reservation_id],
                    archive_path=str(archive_root / "reservation.tar"),
                    dry_run=False,
                )
            self.assertEqual(reservation_removal["status"], "partial")
            self.assertEqual(reservation_removal["retired_count"], 1)
            self.assertTrue(reservation_path.exists())

            fsync_id = "retire-directory-fsync-failure"
            manager.create([self.phase("fsync", "fsync")], execution_id=fsync_id)
            original_fsync_directory = store._fsync_directory

            def fail_state_directory_fsync(path):
                if path == store.state_dir:
                    raise OSError("state directory fsync failed")
                return original_fsync_directory(path)

            with patch.object(store, "_fsync_directory", side_effect=fail_state_directory_fsync):
                fsync_result = manager.retire(
                    [fsync_id],
                    archive_path=str(archive_root / "fsync.tar"),
                    dry_run=False,
                )
            self.assertEqual(fsync_result["status"], "partial")
            self.assertEqual(fsync_result["retired_count"], 1)

    def test_retirement_archive_temp_cleanup_tolerates_missing_temp_files(self):
        manager = self.manager()
        store = manager.store
        manager.create([self.phase("archive-temp-race", "archive-temp-race")], execution_id="archive-temp-race")
        item, metadata, record_path, summary_path = store._retirement_snapshot("archive-temp-race")
        record_path.unlink()
        original_unlink = Path.unlink

        def archive_temp_disappears(path, *args, **kwargs):
            if path.name.startswith(".ephemeral-executions-archive-"):
                original_unlink(path, *args, **kwargs)
                raise FileNotFoundError(path)
            return original_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as archive_directory:
            with patch.object(Path, "unlink", archive_temp_disappears):
                with self.assertRaises(FileNotFoundError):
                    store._write_archive_temp(
                        Path(archive_directory) / "missing-source.tar",
                        [(item, metadata, record_path, summary_path)],
                    )

            manager.create([self.phase("retire-temp-race", "retire-temp-race")], execution_id="retire-temp-race")
            original_writer = store._write_archive_temp
            temporary_archives = []

            def remember_archive_temp(*args, **kwargs):
                temporary = original_writer(*args, **kwargs)
                temporary_archives.append(temporary)
                return temporary

            def final_archive_temp_disappears(path, *args, **kwargs):
                if temporary_archives and path == temporary_archives[0]:
                    original_unlink(path, *args, **kwargs)
                    raise FileNotFoundError(path)
                return original_unlink(path, *args, **kwargs)

            with patch.object(store, "_write_archive_temp", side_effect=remember_archive_temp), \
                    patch.object(store, "_file_digest", side_effect=OSError("digest failed")), \
                    patch.object(Path, "unlink", final_archive_temp_disappears):
                with self.assertRaisesRegex(OSError, "digest failed"):
                    manager.retire(
                        ["retire-temp-race"],
                        archive_path=str(Path(archive_directory) / "retire-temp.tar"),
                        dry_run=False,
                    )

    def test_start_releases_checkpoint_reservation_when_save_fails(self):
        manager = self.manager(Runner())
        initial_id = "initial-save-failure"
        initial_reservation = manager.store._reservation_path(initial_id)
        with patch.object(manager.store, "save", side_effect=OSError("initial save failed")):
            with self.assertRaisesRegex(OSError, "initial save failed"):
                manager.start([self.phase("initial", "initial")], execution_id=initial_id)
        self.assertFalse(initial_reservation.exists())

        started_id = "started-save-failure"
        started_reservation = manager.store._reservation_path(started_id)
        original_save = manager.store.save
        save_calls = 0

        def fail_started_save(record, **kwargs):
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                raise OSError("started save failed")
            return original_save(record, **kwargs)

        with patch.object(manager.store, "save", side_effect=fail_started_save):
            with self.assertRaisesRegex(OSError, "started save failed"):
                manager.start([self.phase("started", "started")], execution_id=started_id)
        self.assertFalse(started_reservation.exists())


if __name__ == "__main__":
    unittest.main()
