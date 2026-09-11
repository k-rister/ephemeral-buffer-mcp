#!/usr/bin/env python3
"""Run the agent-level A/B schedule through the Codex CLI.

The runner deliberately keeps prompts, transcripts, commands, and captures out
of the records envelope.  Task prompts are supplied separately in a local
manifest and are never copied to the output records.
"""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_agent_ab import MODES, RECORDS_SCHEMA_VERSION, TASKS, _read_json, validate_records


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT = 900
EXCLUDED_FIXTURE_NAMES = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    payload = _read_json(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("task manifest must contain a tasks object")
    known = {task["id"] for task in TASKS}
    manifest = {}
    for task_id, task in tasks.items():
        if task_id not in known or not isinstance(task, dict):
            raise ValueError(f"task manifest contains an unknown task: {task_id}")
        prompt = task.get("prompt")
        marker = task.get("signal_marker", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id} requires a non-empty prompt")
        if not isinstance(marker, str):
            raise ValueError(f"task {task_id} signal_marker must be a string")
        manifest[task_id] = {"prompt": prompt, "signal_marker": marker}
    missing = known - set(manifest)
    if missing:
        raise ValueError(f"task manifest is missing task: {sorted(missing)[0]}")
    return manifest


def _copy_fixture(source: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED_FIXTURE_NAMES}

    shutil.copytree(source, destination, ignore=ignore, symlinks=False)


def _event_objects(output: str) -> list[dict[str, Any]]:
    events = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _terminate_process_group(
    process: subprocess.Popen[str], process_group_id: int | None = None
) -> None:
    """Terminate Codex and any MCP children that inherited its output pipes."""
    group_id = process_group_id if process_group_id is not None else process.pid
    if os.name == "posix":
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - Windows process-group behavior is platform-specific.
        process.kill()


def _status_peak_rss_bytes(status: str) -> int:
    """Parse Linux /proc status peak RSS, returning zero when unavailable."""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except ValueError:
                    return 0
    return 0


def _process_peak_rss_bytes(pid: int) -> int:
    """Read the current invocation's peak RSS from its process record."""
    if sys.platform != "linux":
        return 0
    try:
        return _status_peak_rss_bytes(Path(f"/proc/{pid}/status").read_text(encoding="ascii"))
    except (OSError, UnicodeError):
        return 0


def _run_codex_process(
    command: list[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run Codex with bounded cleanup for descendants holding output pipes."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        start_new_session=os.name == "posix",
    )
    process_group_id = process.pid
    if os.name == "posix":
        try:
            process_group_id = os.getpgid(process.pid)
        except OSError:
            pass
    stop_monitor = threading.Event()
    peak_rss = [0]

    def monitor_rss() -> None:
        while not stop_monitor.is_set():
            peak_rss[0] = max(peak_rss[0], _process_peak_rss_bytes(process.pid))
            if process.poll() is not None:
                return
            stop_monitor.wait(0.01)

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, process_group_id)
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired as drain_timeout:
            _terminate_process_group(process, process_group_id)
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired as final_timeout:
                stdout = _as_text(final_timeout.output or drain_timeout.output)
                stderr = _as_text(final_timeout.stderr or drain_timeout.stderr)
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    completed.peak_rss_bytes = peak_rss[0]
    return completed


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _stable_tool_id(item: dict[str, Any]) -> str | None:
    """Return a lifecycle-stable identifier when the event provides one."""
    for key in ("id", "item_id", "itemId", "call_id", "callId"):
        value = item.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return None


def _event_metrics(output: str) -> tuple[int, int, int, int | None, int | None, list[int | float], list[int | float]]:
    """Return tool, MCP, duplicate-command, and usage metrics from JSONL.

    Lifecycle snapshots with a stable item identifier count once. Events without
    an identifier remain independent because identical commands can represent
    distinct invocations when the producer supplies no correlation key.
    """
    tool_calls = 0
    mcp_tool_calls = 0
    commands: list[str] = []
    identified_tools: dict[str, dict[str, Any]] = {}
    anonymous_tools: list[dict[str, Any]] = []
    input_tokens = None
    output_tokens = None
    input_token_samples = []
    output_token_samples = []
    for event in _event_objects(output):
        for item in _walk_dicts(event):
            event_type = item.get("type")
            is_tool = isinstance(event_type, str) and (
                "tool_call" in event_type or event_type in {"command_execution", "function_call"}
            )
            stable_id = _stable_tool_id(item) if is_tool else None
            is_mcp_tool = isinstance(event_type, str) and "mcp" in event_type.lower() and (
                "call" in event_type.lower() or "tool" in event_type.lower()
            )
            if is_tool:
                if stable_id is None:
                    anonymous_tools.append({"item": item, "is_mcp": is_mcp_tool})
                else:
                    logical = identified_tools.setdefault(
                        stable_id, {"item": {}, "is_mcp": False}
                    )
                    logical["item"].update(item)
                    logical["is_mcp"] = logical["is_mcp"] or is_mcp_tool
            usage = item.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("input_tokens"), (int, float)):
                    input_tokens = usage["input_tokens"]
                    input_token_samples.append(usage["input_tokens"])
                if isinstance(usage.get("output_tokens"), (int, float)):
                    output_tokens = usage["output_tokens"]
                    output_token_samples.append(usage["output_tokens"])
    for logical in [*identified_tools.values(), *anonymous_tools]:
        tool_calls += 1
        if logical["is_mcp"]:
            mcp_tool_calls += 1
        item = logical["item"]
        for key in ("command", "cmd", "shell_command"):
            command = item.get(key)
            if isinstance(command, str) and command.strip():
                commands.append(command.strip())
    counts = Counter(commands)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return (
        tool_calls,
        mcp_tool_calls,
        repeated,
        input_tokens,
        output_tokens,
        input_token_samples,
        output_token_samples,
    )


def _codex_command(
    *,
    codex: str,
    model: str,
    mode: str,
    fixture: Path,
    mcp_python: str,
    mcp_module: str,
    mcp_server_script: str | None,
    allow_mcp_approvals: bool,
    sandbox: str,
    timeout: int,
    mcp_env: dict[str, str] | None = None,
) -> list[str]:
    command = [codex]
    if mode == "mcp" and allow_mcp_approvals:
        command.append("--approve-for-me")
    command.append("exec")
    command.extend([
        "--model",
        model,
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write" if mode == "mcp" and allow_mcp_approvals else sandbox,
        "--cd",
        str(fixture),
        "--config",
        "model=" + json.dumps(model),
    ])
    if mode == "mcp":
        mcp_args = [mcp_server_script] if mcp_server_script else ["-m", mcp_module]
        command.extend([
            "--config",
            "mcp_servers.ephemeral-buffer.command=" + json.dumps(mcp_python),
            "--config",
            "mcp_servers.ephemeral-buffer.args=" + json.dumps(mcp_args),
        ])
        if mcp_env:
            env_config = "{" + ", ".join(
                f"{key}={json.dumps(value)}" for key, value in sorted(mcp_env.items())
            ) + "}"
            command.extend([
                "--config",
                "mcp_servers.ephemeral-buffer.env=" + env_config,
            ])
    return command


def _run_one(
    item: dict[str, Any],
    task: dict[str, str],
    *,
    args: argparse.Namespace,
    repository: Path,
    scratch: Path,
) -> dict[str, Any]:
    fixture = scratch / f"{item['sequence']}-{item['mode']}"
    _copy_fixture(repository, fixture)
    diagnostic_log = None
    diagnostic_log_dir = getattr(args, "diagnostic_log_dir", None)
    if item["mode"] == "mcp" and diagnostic_log_dir:
        diagnostic_log = Path(diagnostic_log_dir) / f"{item['sequence']:03d}-{item['mode']}.jsonl"
        diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
    mcp_env = None
    if diagnostic_log is not None:
        mcp_env = {
            "EPHEMERAL_LOG_LEVEL": "INFO",
            "EPHEMERAL_LOG_FILE": str(diagnostic_log),
        }
    if item["mode"] == "mcp":
        if os.environ.get("EPHEMERAL_TEST_EMBEDDINGS"):
            if mcp_env is None:
                mcp_env = {}
            mcp_env["EPHEMERAL_TEST_EMBEDDINGS"] = os.environ["EPHEMERAL_TEST_EMBEDDINGS"]
        if mcp_env is None:
            mcp_env = {}
        mcp_env["EPHEMERAL_DISABLE_SOCKET_SERVER"] = "1"
    command = _codex_command(
        codex=args.codex,
        model=args.model,
        mode=item["mode"],
        fixture=fixture,
        mcp_python=args.mcp_python,
        mcp_module=args.mcp_module,
        mcp_server_script=args.mcp_server_script,
        allow_mcp_approvals=getattr(args, "allow_mcp_approvals", False),
        sandbox=args.sandbox,
        timeout=args.timeout,
        mcp_env=mcp_env,
    )
    started = time.monotonic()
    output = ""
    success = False
    exit_code = None
    failure_reason = None
    peak_rss_bytes = 0
    require_mcp_calls = item["mode"] == "mcp" and getattr(args, "require_mcp_calls", False)
    prompt = task["prompt"]
    if require_mcp_calls:
        prompt = (
            "This is the MCP treatment. You must use the configured ephemeral-buffer MCP tools "
            "for the capture and search workflow when they are available. Do not substitute direct "
            "shell filtering for those MCP operations.\n\n"
            + prompt
        )
    try:
        completed = _run_codex_process(
            [*command, prompt], cwd=fixture, timeout=args.timeout
        )
        output = completed.stdout + completed.stderr
        success = completed.returncode == 0
        exit_code = completed.returncode
        peak_rss_bytes = getattr(completed, "peak_rss_bytes", 0)
        if not success:
            failure_reason = "codex_exit_nonzero"
    except subprocess.TimeoutExpired as exc:
        output = _as_text(exc.stdout) + _as_text(exc.stderr)
        success = False
        failure_reason = "timeout"
    duration = time.monotonic() - started
    (
        tool_calls,
        mcp_tool_calls,
        repeated_commands,
        input_tokens,
        output_tokens,
        input_token_samples,
        output_token_samples,
    ) = _event_metrics(output)
    if require_mcp_calls and success and mcp_tool_calls == 0:
        success = False
        failure_reason = "mcp_not_used"
    marker = task["signal_marker"]
    return {
        "task_id": item["task_id"],
        "repetition": item["repetition"],
        "mode": item["mode"],
        "completed": success,
        "signal_retrieved": bool(marker and marker in output),
        "duration_seconds": duration,
        "tool_calls": tool_calls,
        "repeated_commands": repeated_commands,
        # Codex CLI does not expose context bytes; this is the observable
        # prompt/event envelope, kept as a comparable proxy between modes.
        "context_bytes_proxy": len(prompt.encode()) + len(output.encode()),
        "prompt_bytes_proxy": len(prompt.encode()),
        "output_bytes_proxy": len(output.encode()),
        "peak_rss_bytes": peak_rss_bytes,
        "exit_code": exit_code,
        "failure_reason": failure_reason,
        "mcp_tool_calls": mcp_tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_token_samples": input_token_samples,
        "output_token_samples": output_token_samples,
    }


def run_schedule(schedule: dict[str, Any], manifest: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    if schedule.get("benchmark") != "agent-ab":
        raise ValueError("schedule benchmark must be agent-ab")
    repository = Path(args.repository).resolve()
    if not repository.is_dir():
        raise ValueError(f"repository fixture is not a directory: {repository}")
    runs = []
    with tempfile.TemporaryDirectory(prefix="codex-agent-ab-") as temporary:
        scratch = Path(temporary)
        for item in schedule.get("schedule", []):
            if item["mode"] not in MODES or item["task_id"] not in manifest:
                raise ValueError(f"schedule item is not covered by task manifest: {item}")
            if args.dry_run:
                print(json.dumps({"sequence": item["sequence"], "mode": item["mode"], "task_id": item["task_id"]}))
                continue
            runs.append(_run_one(item, manifest[item["task_id"]], args=args, repository=repository, scratch=scratch))
    if args.dry_run:
        return {}
    protocol = {
        "model_config": args.model,
        "repository_fixture": args.repository_fixture,
        "environment": args.environment,
        "reset_policy": "fresh-copy-per-run",
        "agent_adapter": "codex-cli",
        "approval_policy": "automatic-review-mcp" if getattr(args, "allow_mcp_approvals", False) else "read-only-sandbox",
        "mcp_usage_policy": "required" if getattr(args, "require_mcp_calls", False) else "opportunistic",
    }
    payload = {
        "schema_version": schedule["schema_version"],
        "benchmark": "agent-ab",
        "records_schema_version": RECORDS_SCHEMA_VERSION,
        "task_fixture_version": schedule["task_fixture_version"],
        "protocol": protocol,
        "schedule": schedule,
        "runs": runs,
        "privacy": "records contain metadata only; prompts, transcripts, commands, captures, and user content are excluded",
    }
    validate_records(payload, schedule)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="Private local task prompt manifest; never commit it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--repository-fixture", default="local-fixture")
    parser.add_argument("--environment", default="local")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--mcp-python", default=sys.executable)
    parser.add_argument("--mcp-module", default="server")
    parser.add_argument("--mcp-server-script", help="Absolute server script when the fixture does not provide the MCP module")
    parser.add_argument(
        "--allow-mcp-approvals",
        action="store_true",
        help="Use Codex automatic review and workspace-write isolation for MCP runs",
    )
    parser.add_argument(
        "--require-mcp-calls",
        action="store_true",
        help="Require at least one MCP tool call in every MCP run",
    )
    parser.add_argument("--sandbox", choices=("read-only", "workspace-write"), default="read-only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--diagnostic-log-dir",
        type=Path,
        help="Optional directory for content-free MCP lifecycle logs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    schedule = _read_json(args.schedule)
    manifest = _load_manifest(args.tasks)
    payload = run_schedule(schedule, manifest, args)
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(payload["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
