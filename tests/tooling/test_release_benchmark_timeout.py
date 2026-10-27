"""Tests for release-benchmark command timeout handling."""

import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_with_timeout as timeout_runner
from tests.paths import REPOSITORY_ROOT


TIMEOUT_RUNNER = REPOSITORY_ROOT / "scripts" / "run_with_timeout.py"
RELEASE_BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "collect-release-benchmarks.sh"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _process_state(pid: int) -> str | None:
    """Return a Linux process state, or None after the process is reaped."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return stat.rsplit(")", 1)[1].strip().split(maxsplit=1)[0]


class TestReleaseBenchmarkTimeoutRunner(unittest.TestCase):
    def test_normal_completion_preserves_exit_code(self):
        result = subprocess.run(
            [
                sys.executable,
                str(TIMEOUT_RUNNER),
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stderr, "")

    def test_timeout_kills_child_that_ignores_sigterm(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            ready_file = Path(temporary_directory) / "child-ready"
            child_program = (
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w').close(); time.sleep(60)"
            )
            parent_program = (
                "import os, subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {child_program!r}, {str(ready_file)!r}])\n"
                f"ready = {str(ready_file)!r}\n"
                "deadline = time.monotonic() + 5\n"
                "while not os.path.exists(ready) and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(60)\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(TIMEOUT_RUNNER),
                    "--timeout",
                    "1",
                    "--",
                    sys.executable,
                    "-c",
                    parent_program,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=12,
            )

            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertIn("command timed out after 1 seconds", result.stderr)
            self.assertIn("sending SIGKILL", result.stderr)
            child_pid = int(result.stdout.strip())
            deadline = time.monotonic() + 3
            state = _process_state(child_pid)
            while state not in (None, "Z") and time.monotonic() < deadline:
                time.sleep(0.05)
                state = _process_state(child_pid)
            self.assertIn(state, (None, "Z"), f"child process {child_pid} is still running")

    def test_cancellation_signals_clean_up_the_benchmark_process_group(self):
        for requested_signal, expected_return_code in (
            (signal.SIGINT, 128 + signal.SIGINT),
            (signal.SIGTERM, 128 + signal.SIGTERM),
        ):
            with self.subTest(signal=requested_signal), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                ready_file = root / "child-ready"
                pid_file = root / "process-pids"
                child_program = (
                    "import signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "open(sys.argv[1], 'w').close(); time.sleep(60)"
                )
                parent_program = (
                    "import os, pathlib, subprocess, sys, time\n"
                    f"child = subprocess.Popen([sys.executable, '-c', {child_program!r}, sys.argv[1]])\n"
                    "while not pathlib.Path(sys.argv[1]).exists():\n"
                    "    time.sleep(0.01)\n"
                    "pathlib.Path(sys.argv[2]).write_text(f'{os.getpid()} {child.pid}', encoding='utf-8')\n"
                    "time.sleep(60)\n"
                )
                helper = subprocess.Popen(
                    [
                        sys.executable,
                        str(TIMEOUT_RUNNER),
                        "--timeout",
                        "60",
                        "--",
                        sys.executable,
                        "-c",
                        parent_program,
                        str(ready_file),
                        str(pid_file),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                process_ids: list[int] = []
                try:
                    deadline = time.monotonic() + 5
                    while not pid_file.exists() and helper.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pid_file.exists(), "benchmark did not report its process IDs")
                    process_ids = [int(pid) for pid in pid_file.read_text(encoding="utf-8").split()]

                    os.kill(helper.pid, requested_signal)
                    _stdout, stderr = helper.communicate(timeout=12)

                    self.assertEqual(helper.returncode, expected_return_code, stderr)
                    signal_name = signal.Signals(requested_signal).name
                    self.assertIn(f"runner received {signal_name}", stderr)
                    self.assertIn("sending SIGKILL", stderr)
                    deadline = time.monotonic() + 3
                    states = [_process_state(pid) for pid in process_ids]
                    while any(state not in (None, "Z") for state in states) and time.monotonic() < deadline:
                        time.sleep(0.05)
                        states = [_process_state(pid) for pid in process_ids]
                    self.assertTrue(
                        all(state in (None, "Z") for state in states),
                        f"benchmark process group is still running: {states}",
                    )
                finally:
                    if helper.poll() is None:
                        helper.kill()
                        helper.wait(timeout=3)
                    if process_ids:
                        try:
                            os.killpg(process_ids[0], signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_signal_during_process_launch_is_deferred_until_cleanup_can_reap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ready_file = root / "child-ready"
            pid_file = root / "process-pids"
            child_program = (
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w').close(); time.sleep(60)"
            )
            parent_program = (
                "import os, pathlib, subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {child_program!r}, sys.argv[1]])\n"
                "while not pathlib.Path(sys.argv[1]).exists():\n"
                "    time.sleep(0.01)\n"
                "pathlib.Path(sys.argv[2]).write_text(f'{os.getpid()} {child.pid}', encoding='utf-8')\n"
                "time.sleep(60)\n"
            )
            command = [sys.executable, "-c", parent_program, str(ready_file), str(pid_file)]
            original_popen = subprocess.Popen
            launched_processes: list[subprocess.Popen[str]] = []

            def launch_then_signal(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                launched_processes.append(process)
                deadline = time.monotonic() + 5
                while not pid_file.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not pid_file.exists():
                    raise AssertionError("benchmark process IDs were not reported during launch")
                os.kill(os.getpid(), signal.SIGTERM)
                return process

            process_ids: list[int] = []
            try:
                with patch.object(timeout_runner.subprocess, "Popen", side_effect=launch_then_signal), patch(
                    "sys.stderr", new_callable=io.StringIO
                ) as stderr:
                    return_code = timeout_runner.run(command, timeout_seconds=60)

                self.assertEqual(return_code, 128 + signal.SIGTERM)
                self.assertIn("runner received SIGTERM", stderr.getvalue())
                self.assertIn("sending SIGKILL", stderr.getvalue())
                process_ids = [int(pid) for pid in pid_file.read_text(encoding="utf-8").split()]
                deadline = time.monotonic() + 3
                states = [_process_state(pid) for pid in process_ids]
                while any(state not in (None, "Z") for state in states) and time.monotonic() < deadline:
                    time.sleep(0.05)
                    states = [_process_state(pid) for pid in process_ids]
                self.assertTrue(
                    all(state in (None, "Z") for state in states),
                    f"benchmark process group is still running: {states}",
                )
                if launched_processes:
                    self.assertIsNotNone(launched_processes[0].returncode)
            finally:
                for process in launched_processes:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if process.poll() is None:
                        process.wait(timeout=3)


class TestReleaseBenchmarkTimeoutCollection(unittest.TestCase):
    def test_timeout_is_recorded_and_collection_continues_without_changing_agent_timeout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "repository"
            scripts = repository / "scripts"
            benchmarks = repository / "benchmarks"
            data = benchmarks / "data"
            scripts.mkdir(parents=True)
            data.mkdir(parents=True)
            shutil.copyfile(RELEASE_BENCHMARK_SCRIPT, scripts / RELEASE_BENCHMARK_SCRIPT.name)
            shutil.copyfile(TIMEOUT_RUNNER, scripts / TIMEOUT_RUNNER.name)
            (benchmarks / "__init__.py").write_text("", encoding="utf-8")
            (benchmarks / "benchmark_concurrency.py").write_text(
                "import time\ntime.sleep(60)\n",
                encoding="utf-8",
            )
            (benchmarks / "benchmark_admission.py").write_text(
                "print('admission benchmark completed')\n",
                encoding="utf-8",
            )
            (benchmarks / "run_codex_agent_ab.py").write_text(
                "import json, os, sys\n"
                "with open(os.environ['RELEASE_TEST_AGENT_ARGS_LOG'], 'a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            (data / "benchmark_baseline.json").write_text("{}\n", encoding="utf-8")

            _git(repository, "init", "--quiet")
            _git(repository, "config", "user.name", "Release benchmark test")
            _git(repository, "config", "user.email", "release-test@example.invalid")
            _git(repository, "add", ".")
            _git(repository, "commit", "--quiet", "-m", "fixture")
            baseline_revision = _git(repository, "rev-parse", "HEAD")

            codex_home = root / "codex-home"
            binary_directory = root / "bin"
            codex_home.mkdir()
            binary_directory.mkdir()
            fake_codex = binary_directory / "codex"
            fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_codex.chmod(0o700)
            agent_args_log = root / "agent-args.jsonl"
            output_directory = root / "bundle"
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "CODEX_HOME": str(codex_home),
                    "PATH": str(binary_directory) + os.pathsep + environment.get("PATH", ""),
                    "RELEASE_TEST_AGENT_ARGS_LOG": str(agent_args_log),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    str(scripts / RELEASE_BENCHMARK_SCRIPT.name),
                    "--baseline-ref",
                    baseline_revision,
                    "--model",
                    "test-model",
                    "--output-dir",
                    str(output_directory),
                    "--samples",
                    "1",
                    "--semantic-samples",
                    "1",
                    "--repetitions",
                    "1",
                    "--agent-ab-repetitions",
                    "1",
                    "--agent-timeout",
                    "37",
                    "--benchmark-timeout",
                    "1",
                ],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            status_lines = (output_directory / "STATUS.tsv").read_text(encoding="utf-8").splitlines()
            statuses = {line.split("\t")[1]: line.split("\t") for line in status_lines[1:]}
            self.assertEqual(statuses["baseline-concurrency"][0:3], ["failed", "baseline-concurrency", "124"])
            self.assertEqual(statuses["candidate-concurrency"][0:3], ["failed", "candidate-concurrency", "124"])
            self.assertEqual(statuses["baseline-admission"][0:3], ["success", "baseline-admission", "0"])
            self.assertEqual(statuses["candidate-admission"][0:3], ["success", "candidate-admission", "0"])
            self.assertIn(
                "command timed out after 1 seconds",
                (output_directory / "logs" / "candidate-concurrency.log").read_text(encoding="utf-8"),
            )
            manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["benchmark_timeout_seconds"], 1)
            self.assertEqual(manifest["configuration"]["agent_timeout_seconds"], 37)

            agent_invocations = [
                json.loads(line) for line in agent_args_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(agent_invocations), 2)
            for invocation in agent_invocations:
                timeout_index = invocation.index("--timeout")
                self.assertEqual(invocation[timeout_index + 1], "37")


if __name__ == "__main__":
    unittest.main()
