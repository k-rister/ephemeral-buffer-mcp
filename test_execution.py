"""Tests for durable phase-level execution and resume behavior."""

import errno
import json
import os
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from pathlib import Path
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
        killpg.assert_called_once_with(456, 0)

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

    def test_marker_process_scan_fails_closed_for_unavailable_proc_metadata(self):
        with patch.object(execution.sys, "platform", "win32"):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", side_effect=OSError("proc unavailable")):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["not-a-pid", "123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", side_effect=FileNotFoundError()):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 11})()):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", side_effect=OSError("stat unavailable")):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 10})()), \
                patch.object(execution.Path, "read_bytes", side_effect=FileNotFoundError()):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 10})()), \
                patch.object(execution.Path, "read_bytes", side_effect=OSError("environ unavailable")):
            self.assertIsNone(execution._marker_process_groups("marker"))
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 10})()), \
                patch.object(execution.Path, "read_bytes", return_value=b"OTHER=value\0"):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 10})()), \
                patch.object(execution.Path, "read_bytes", return_value=b"EPHEMERAL_EXECUTION_PROCESS_MARKER=marker\0"), \
                patch.object(execution.os, "getpgid", side_effect=OSError(errno.ESRCH, "gone")):
            self.assertEqual(execution._marker_process_groups("marker"), set())
        with patch.object(execution.os, "listdir", return_value=["123"]), \
                patch.object(execution.os, "getuid", return_value=10), \
                patch.object(execution.os, "stat", return_value=type("Stat", (), {"st_uid": 10})()), \
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


if __name__ == "__main__":
    unittest.main()
