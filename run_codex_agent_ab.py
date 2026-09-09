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


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate Codex and any MCP children that inherited its output pipes."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:  # pragma: no cover - Windows process-group behavior is platform-specific.
        process.kill()


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
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _event_metrics(output: str) -> tuple[int, int, int, int | None, int | None]:
    """Return tool, MCP, duplicate-command, and usage metrics from JSONL."""
    tool_calls = 0
    mcp_tool_calls = 0
    commands: list[str] = []
    input_tokens = None
    output_tokens = None
    for event in _event_objects(output):
        for item in _walk_dicts(event):
            event_type = item.get("type")
            if isinstance(event_type, str) and (
                "tool_call" in event_type or event_type in {"command_execution", "function_call"}
            ):
                tool_calls += 1
            if isinstance(event_type, str) and "mcp" in event_type.lower() and (
                "call" in event_type.lower() or "tool" in event_type.lower()
            ):
                mcp_tool_calls += 1
            usage = item.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("input_tokens"), (int, float)):
                    input_tokens = usage["input_tokens"]
                if isinstance(usage.get("output_tokens"), (int, float)):
                    output_tokens = usage["output_tokens"]
            for key in ("command", "cmd", "shell_command"):
                command = item.get(key)
                if isinstance(command, str) and command.strip():
                    commands.append(command.strip())
    counts = Counter(commands)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return tool_calls, mcp_tool_calls, repeated, input_tokens, output_tokens


def _peak_rss_bytes() -> int:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (ImportError, AttributeError, OSError):
        return 0
    # Linux reports KiB; macOS reports bytes.
    return int(value * 1024 if sys.platform != "darwin" else value)


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
    exit_code = None
    failure_reason = None
    try:
        completed = _run_codex_process(
            [*command, task["prompt"]], cwd=fixture, timeout=args.timeout
        )
        output = completed.stdout + completed.stderr
        success = completed.returncode == 0
        exit_code = completed.returncode
        if not success:
            failure_reason = "codex_exit_nonzero"
    except subprocess.TimeoutExpired as exc:
        output = _as_text(exc.stdout) + _as_text(exc.stderr)
        success = False
        failure_reason = "timeout"
    duration = time.monotonic() - started
    tool_calls, mcp_tool_calls, repeated_commands, input_tokens, output_tokens = _event_metrics(output)
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
        "context_bytes_proxy": len(task["prompt"].encode()) + len(output.encode()),
        "peak_rss_bytes": _peak_rss_bytes(),
        "exit_code": exit_code,
        "failure_reason": failure_reason,
        "mcp_tool_calls": mcp_tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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
