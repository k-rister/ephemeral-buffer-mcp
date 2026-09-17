"""Durable, phase-level command execution with explicit resume boundaries.

Execution records are deliberately separate from the in-memory capture ring.
The ring is optimized for search during one server session; this module keeps
the bounded output and phase metadata needed to recover a long-running task
after a process restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import tempfile
import threading
import time
import uuid
import errno
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from capture_utils import run_command_bounded
from config import DEFAULT_MAX_OUTPUT_BYTES

try:
    import fcntl
except ImportError:  # pragma: no cover - the server requires Unix sockets.
    fcntl = None


EXECUTION_SCHEMA_VERSION = 1
PHASE_STATUSES = ("pending", "started", "completed", "failed", "interrupted", "timed_out")
RESUME_POLICIES = ("safe", "allow-unsafe")
MAX_EXECUTION_ID_BYTES = 256
MAX_PHASE_NAME_BYTES = 256
MAX_ERROR_BYTES = 4096
MAX_STRUCTURED_METRICS_BYTES = 16 * 1024
MAX_EXECUTION_PHASES = 64
MAX_PHASE_ATTEMPTS = 32
MAX_EXECUTION_STATE_BYTES = 64 * 1024 * 1024
EXECUTION_METADATA_RESERVE_BYTES = 4 * 1024 * 1024
MAX_EXECUTION_RECORDS = 1000
DEFAULT_EXECUTION_LIST_LIMIT = 20
MAX_EXECUTION_LIST_LIMIT = 100
MAX_EXECUTION_OUTPUT_CHUNK_BYTES = 8 * 1024
JSON_OUTPUT_EXPANSION_BOUND = 6


CommandRunner = Callable[[str, Optional[str], int, Optional[float]], Tuple[str, int, bool, int, bool]]
OutputHandler = Callable[[Dict[str, Any], str, Dict[str, Any]], Optional[str]]


class ExecutionBusyError(RuntimeError):
    """Raised when another process currently owns an execution lease."""


def _now() -> str:
    """Return a stable UTC timestamp for persisted metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bounded_error(value: Any) -> str:
    """Keep persisted exception text useful without allowing unbounded state."""
    text = str(value)
    if len(text.encode("utf-8")) <= MAX_ERROR_BYTES:
        return text
    encoded = text.encode("utf-8")[:MAX_ERROR_BYTES]
    return encoded.decode("utf-8", errors="ignore")


def _json_copy(value: Any, field_name: str) -> Any:
    """Validate and detach JSON-compatible caller metadata."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-compatible values") from exc
    if len(encoded) > MAX_STRUCTURED_METRICS_BYTES:
        raise ValueError(
            f"{field_name} exceeds the {MAX_STRUCTURED_METRICS_BYTES:,}-byte limit"
        )
    return json.loads(encoded.decode("utf-8"))


def _validate_positive_number(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number or null")
    parsed = float(value)
    if parsed <= 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a positive number or null")
    return parsed


def _validate_max_output(value: Any, field_name: str, limit: int) -> int:
    if value is None:
        return limit
    if isinstance(value, bool) or not isinstance(value, int) or value < 512:
        raise ValueError(f"{field_name} must be an integer of at least 512 bytes")
    if value > limit:
        raise ValueError(f"{field_name} cannot exceed the configured {limit:,}-byte limit")
    return value


def _validate_text(value: Any, field_name: str, max_bytes: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        expected = "a non-empty string" if not allow_empty else "a string"
        raise ValueError(f"{field_name} must be {expected}")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes:,}-byte limit")
    return value


def _phase_event(phase: Dict[str, Any], status: str, **details: Any) -> None:
    event = {"status": status, "timestamp": _now(), "attempt": phase["attempts"]}
    event.update(details)
    phase["events"].append(event)


def _proc_identity(process_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Return Linux process start and boot identities when available."""
    boot_id = None
    start_time = None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        stat_line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        remainder = stat_line[stat_line.rfind(") ") + 2 :].split()
        if len(remainder) > 19:
            start_time = remainder[19]
    except (OSError, UnicodeError, ValueError):
        return None, None
    return start_time, boot_id


def _process_group_absent(error: OSError) -> bool:
    """Return true only when the kernel proved that a process group is gone."""
    return getattr(error, "errno", None) == errno.ESRCH


def _terminate_stale_process(phase: Dict[str, Any]) -> bool:
    """Stop a command left behind by a process that died during execution."""
    process_id = phase.get("process_id")
    process_group_id = phase.get("process_group_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
    ):
        return phase.get("process_fence_pending") is False
    expected_start = phase.get("process_start_time")
    expected_boot = phase.get("process_boot_id")
    if (
        not isinstance(expected_start, str)
        or not expected_start
        or not isinstance(expected_boot, str)
        or not expected_boot
    ):
        return False
    current_start, current_boot = _proc_identity(process_id)
    if current_start is not None or current_boot is not None:
        if (
            not isinstance(current_start, str)
            or not current_start
            or not isinstance(current_boot, str)
            or not current_boot
            or current_start != expected_start
            or current_boot != expected_boot
        ):
            return False
    else:
        try:
            os.kill(process_id, 0)
        except OSError as exc:
            if not _process_group_absent(exc):
                return False
        else:
            return False
    try:
        current_process_group_id = os.getpgid(process_id)
    except OSError as exc:
        if not _process_group_absent(exc):
            return False
    else:
        if current_process_group_id != process_group_id:
            return False
    try:
        if os.getpgrp() == process_group_id:
            return True
        os.killpg(process_group_id, 0)
    except OSError as exc:
        return _process_group_absent(exc)
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except OSError as exc:
        return _process_group_absent(exc)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError as exc:
        return _process_group_absent(exc)
    try:
        os.killpg(process_group_id, 0)
    except OSError as exc:
        return _process_group_absent(exc)
    return False


class ExecutionStore:
    """Atomic JSON-file storage for resumable execution records."""

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.state_dir = Path(os.path.abspath(os.path.expanduser(os.fspath(state_dir))))
        self._lock = threading.RLock()

    @staticmethod
    def _filename(execution_id: str) -> str:
        digest = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()
        return f"{digest}.json"

    def _path(self, execution_id: str) -> Path:
        return self.state_dir / self._filename(execution_id)

    def _lock_path(self, execution_id: str) -> Path:
        return self.state_dir / f".{self._filename(execution_id)}.lock"

    def _summary_path(self, execution_id: str) -> Path:
        return self.state_dir / f"{self._filename(execution_id)[:-5]}.summary.json"

    def _ensure_state_dir(self) -> None:
        """Create the state directory with owner-only permissions."""
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._validate_state_dir(require_private_mode=False)
        os.chmod(self.state_dir, 0o700)

    def _validate_state_dir(
        self,
        *,
        require_exists: bool = True,
        require_private_mode: bool = True,
    ) -> bool:
        """Validate state-directory identity, ownership, and permissions."""
        self._reject_symlink_components(self.state_dir)
        self._reject_symlink(self.state_dir, "state directory")
        if not self.state_dir.exists():
            if require_exists:
                raise ValueError(f"execution state directory does not exist: {self.state_dir}")
            return False
        if not self.state_dir.is_dir():
            raise ValueError(f"execution state directory is not a private directory: {self.state_dir}")
        state_stat = self.state_dir.stat()
        current_uid = getattr(os, "getuid", lambda: None)()
        if current_uid is not None and state_stat.st_uid != current_uid:
            raise ValueError(f"execution state directory is not owned by the current user: {self.state_dir}")
        if require_private_mode and stat.S_IMODE(state_stat.st_mode) & 0o077:
            raise ValueError(f"execution state directory is not private: {self.state_dir}")
        return True

    @staticmethod
    def _reject_symlink(path: Path, description: str) -> None:
        if path.is_symlink():
            raise ValueError(f"execution {description} must not be a symlink: {path}")

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        """Reject symlinked directory components before opening state paths."""
        absolute = Path(os.path.abspath(os.fspath(path)))
        current = Path(absolute.anchor)
        for component in absolute.parts[:-1]:
            if component == absolute.anchor:
                continue
            current /= component
            if current.is_symlink():
                raise ValueError(
                    f"execution state path component must not be a symlink: {current}"
                )

    @contextmanager
    def _record_lease(self):
        """Serialize record-count checks across processes sharing this directory."""
        if fcntl is None:
            yield
            return
        self._ensure_state_dir()
        lock_path = self.state_dir / ".records.lock"
        self._reject_symlink(lock_path, "record-count lock file")
        lock_stream = lock_path.open("a+", encoding="ascii")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                lock_stream.close()

    @contextmanager
    def lease(self, execution_id: str):
        """Hold an inter-process lease while one execution runs a phase."""
        if fcntl is None:
            raise RuntimeError("durable execution leases require a Unix file-locking platform")
        self._ensure_state_dir()
        lock_path = self._lock_path(execution_id)
        self._reject_symlink(lock_path, "lock file")
        lock_stream = lock_path.open("a+", encoding="ascii")
        try:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {
                    errno.EACCES, errno.EAGAIN,
                }:
                    raise ExecutionBusyError(
                        f"Execution '{execution_id}' is already running"
                    ) from exc
                raise
            yield
        finally:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                lock_stream.close()

    def is_locked(self, execution_id: str) -> bool:
        """Return whether another process currently owns an execution lease."""
        if fcntl is None:
            return False
        if not self._validate_state_dir(require_exists=False):
            return False
        lock_path = self._lock_path(execution_id)
        if not lock_path.exists():
            self._reject_symlink(lock_path, "lock file")
            return False
        self._reject_symlink(lock_path, "lock file")
        lock_stream = lock_path.open("a+", encoding="ascii")
        try:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {
                    errno.EACCES, errno.EAGAIN,
                }:
                    return True
                raise
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            return False
        finally:
            lock_stream.close()

    def save(self, record: Dict[str, Any]) -> None:
        """Persist one record with a replace, so readers never see a half-file."""
        with self._lock:
            with self._record_lease():
                self._ensure_state_dir()
                path = self._path(record["execution_id"])
                encoded = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
                if len(encoded) > MAX_EXECUTION_STATE_BYTES:
                    raise ValueError(
                        f"execution state exceeds the {MAX_EXECUTION_STATE_BYTES:,}-byte limit"
                    )
                record_paths = [
                    candidate
                    for candidate in self.state_dir.glob("*.json")
                    if not candidate.name.endswith(".summary.json")
                ]
                if not path.exists() and len(record_paths) >= MAX_EXECUTION_RECORDS:
                    raise ValueError(
                        f"execution state contains the maximum of {MAX_EXECUTION_RECORDS:,} records"
                    )
                summary_record = json.loads(encoded.decode("utf-8"))
                summary_record["phases"] = [
                    {
                        key: phase.get(key)
                        for key in (
                            "name", "status", "side_effects", "unsafe_side_effects",
                            "attempts", "error",
                        )
                    }
                    for phase in summary_record.get("phases", [])
                ]
                summary_encoded = json.dumps(
                    summary_record, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                for destination, payload in (
                    (path, encoded),
                    (self._summary_path(record["execution_id"]), summary_encoded),
                ):
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="wb", dir=self.state_dir, delete=False
                        ) as stream:
                            temporary = Path(stream.name)
                            stream.write(payload)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, destination)
                    finally:
                        if temporary is not None and temporary.exists():
                            temporary.unlink()
                directory_fd = os.open(
                    self.state_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

    def _read_record(self, execution_id: str) -> Dict[str, Any]:
        with self._lock:
            self._validate_state_dir(require_exists=False)
            path = self._path(execution_id)
            self._reject_symlink(path, "record file")
            try:
                with path.open("r", encoding="utf-8") as stream:
                    record = json.load(stream)
            except FileNotFoundError as exc:
                raise KeyError(f"Execution '{execution_id}' was not found") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Execution '{execution_id}' has unreadable state") from exc
            if not isinstance(record, dict) or record.get("execution_id") != execution_id:
                raise ValueError(f"Execution '{execution_id}' has invalid state")
            return record

    def load(self, execution_id: str, recover: bool = True) -> Dict[str, Any]:
        record = self._read_record(execution_id)
        if not recover or not any(
            phase.get("status") == "started" for phase in record.get("phases", [])
        ):
            return record
        try:
            with self.lease(execution_id):
                record = self._read_record(execution_id)
                if self._recover_started(record):
                    record["updated_at"] = _now()
                    self.save(record)
                return record
        except ExecutionBusyError:
            return self._read_record(execution_id)

    @staticmethod
    def _recover_started(record: Dict[str, Any]) -> bool:
        changed = False
        for phase in record.get("phases", []):
            if phase.get("status") != "started" and not phase.get("process_fence_pending"):
                continue
            if phase.get("status") == "started" and "process_fence_pending" not in phase:
                phase["process_fence_pending"] = True
            already_pending = (
                phase.get("status") == "interrupted"
                and phase.get("process_fence_pending") is True
            )
            fenced = _terminate_stale_process(phase)
            phase["status"] = "interrupted"
            phase["process_fence_pending"] = not fenced
            phase["error"] = (
                "process termination is pending before the phase can resume"
                if not fenced
                else "process terminated before the phase completed"
            )
            if not already_pending:
                _phase_event(phase, "interrupted", reason="process restart recovery")
            changed = True
        if changed:
            PhaseExecutionManager._refresh_overall_status(record)
        return changed

    def list(
        self,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._validate_state_dir(require_exists=False):
                return []
            main_paths = {
                path.name: path
                for path in self.state_dir.glob("*.json")
                if not path.name.endswith(".summary.json") and not path.is_symlink()
            }
            summary_paths = {
                path.name.removesuffix(".summary.json") + ".json": path
                for path in self.state_dir.glob("*.summary.json")
                if not path.is_symlink()
            }
            paths = []
            for name in sorted(set(main_paths) | set(summary_paths)):
                main_path = main_paths.get(name)
                summary_path = summary_paths.get(name)
                if summary_path is not None and (
                    main_path is None
                    or summary_path.stat().st_mtime_ns >= main_path.stat().st_mtime_ns
                ):
                    paths.append(summary_path)
                elif main_path is not None:
                    paths.append(main_path)
            records = []
            for path in paths:
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        record = json.load(stream)
                    if (
                        isinstance(record, dict)
                        and isinstance(record.get("execution_id"), str)
                        and isinstance(record.get("phases"), list)
                    ):
                        records.append(record)
                except (KeyError, ValueError, OSError, json.JSONDecodeError):
                    continue
            records = sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)
            if limit is not None:
                records = records[offset:offset + limit]
            elif offset:
                records = records[offset:]
            selected = []
            for record in records:
                try:
                    if any(phase.get("status") == "started" for phase in record.get("phases", [])):
                        record = self.load(record["execution_id"])
                    selected.append(record)
                except (KeyError, ValueError, OSError, json.JSONDecodeError):
                    continue
            return selected


class PhaseExecutionManager:
    """Create, run, inspect, and resume durable sequential phase executions."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str],
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        command_runner: Optional[CommandRunner] = None,
    ):
        if max_output_bytes < 512:
            raise ValueError("max_output_bytes must be at least 512")
        self.store = ExecutionStore(state_dir)
        self.max_output_bytes = max_output_bytes
        self.command_runner = command_runner or run_command_bounded
        self._lock = threading.RLock()

    @staticmethod
    def _refresh_overall_status(record: Dict[str, Any]) -> None:
        statuses = [phase["status"] for phase in record["phases"]]
        if statuses and all(status == "completed" for status in statuses):
            status = "completed"
        elif any(status == "started" for status in statuses):
            status = "running"
        elif any(status == "interrupted" for status in statuses):
            status = "interrupted"
        elif any(status in {"failed", "timed_out"} for status in statuses):
            status = "partial"
        else:
            status = "pending"
        record["execution_status"] = status
        record["partial"] = status != "completed"

    def _validate_phase(
        self,
        raw_phase: Any,
        index: int,
        default_cwd: str,
        default_timeout: Optional[float],
        phase_output_limit: int,
    ) -> Dict[str, Any]:
        if not isinstance(raw_phase, dict):
            raise ValueError(f"phases[{index}] must be an object")
        name = _validate_text(raw_phase.get("name"), f"phases[{index}].name", MAX_PHASE_NAME_BYTES)
        command = _validate_text(raw_phase.get("command"), f"phases[{index}].command", 16 * 1024)
        cwd = raw_phase.get("cwd") or default_cwd
        cwd = _validate_text(cwd, f"phases[{index}].cwd", 4096)
        cwd = str(Path(cwd).expanduser().resolve(strict=False))
        timeout = raw_phase.get("timeout_seconds", default_timeout)
        timeout = _validate_positive_number(timeout, f"phases[{index}].timeout_seconds")
        max_output = _validate_max_output(
            raw_phase.get("max_output_bytes", phase_output_limit),
            f"phases[{index}].max_output_bytes",
            phase_output_limit,
        )
        side_effects = raw_phase.get("side_effects", "none")
        if "unsafe_side_effects" in raw_phase:
            unsafe_side_effects = raw_phase["unsafe_side_effects"]
            if not isinstance(unsafe_side_effects, bool):
                raise ValueError(f"phases[{index}].unsafe_side_effects must be a boolean")
            alias_side_effects = "unsafe" if unsafe_side_effects else "none"
            if "side_effects" in raw_phase and side_effects != alias_side_effects:
                raise ValueError(
                    f"phases[{index}] has contradictory side_effects and unsafe_side_effects"
                )
            side_effects = alias_side_effects
        if side_effects not in {"none", "unsafe"}:
            raise ValueError(f"phases[{index}].side_effects must be 'none' or 'unsafe'")
        idempotency_key = raw_phase.get("idempotency_key")
        if idempotency_key is not None:
            idempotency_key = _validate_text(
                idempotency_key, f"phases[{index}].idempotency_key", MAX_PHASE_NAME_BYTES
            )
        metrics = _json_copy(raw_phase.get("structured_metrics", {}), f"phases[{index}].structured_metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"phases[{index}].structured_metrics must be an object")
        return {
            "name": name,
            "command": command,
            "cwd": cwd,
            "timeout_seconds": timeout,
            "max_output_bytes": max_output,
            "side_effects": side_effects,
            "unsafe_side_effects": side_effects == "unsafe",
            "idempotency_key": idempotency_key,
            "structured_metrics": metrics,
            "status": "pending",
            "attempts": 0,
            "events": [],
            "attempt_results": [],
            "output": "",
            "result": None,
            "error": None,
            "process_id": None,
            "process_group_id": None,
            "process_start_time": None,
            "process_boot_id": None,
            "process_fence_pending": False,
        }

    def _new_record(
        self,
        phases: List[Dict[str, Any]],
        execution_id: Optional[str],
        label: str,
        resume_policy: str,
        cwd: Optional[str],
        timeout_seconds: Optional[float],
        max_output_bytes: Optional[int],
    ) -> Dict[str, Any]:
        if execution_id is None:
            execution_id = f"exec_{uuid.uuid4().hex}"
        execution_id = _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
        if resume_policy not in RESUME_POLICIES:
            raise ValueError("resume_policy must be 'safe' or 'allow-unsafe'")
        label = _validate_text(label or execution_id, "label", 1024)
        default_cwd = _validate_text(cwd or os.getcwd(), "cwd", 4096)
        default_timeout = _validate_positive_number(timeout_seconds, "timeout_seconds")
        default_output = _validate_max_output(max_output_bytes, "max_output_bytes", self.max_output_bytes)
        if not isinstance(phases, list) or not phases:
            raise ValueError("phases must be a non-empty list")
        if len(phases) > MAX_EXECUTION_PHASES:
            raise ValueError(f"phases must contain at most {MAX_EXECUTION_PHASES} items")
        state_output_budget = max(
            512,
            (MAX_EXECUTION_STATE_BYTES - EXECUTION_METADATA_RESERVE_BYTES)
            // JSON_OUTPUT_EXPANSION_BOUND,
        )
        phase_output_limit = min(default_output, state_output_budget // len(phases))
        normalized = [
            self._validate_phase(item, index, default_cwd, default_timeout, phase_output_limit)
            for index, item in enumerate(phases)
        ]
        names = [phase["name"] for phase in normalized]
        if len(names) != len(set(names)):
            raise ValueError("phase names must be unique within an execution")
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "execution_id": execution_id,
            "label": label,
            "created_at": _now(),
            "updated_at": _now(),
            "execution_status": "pending",
            "partial": True,
            "resume_policy": resume_policy,
            "phases": normalized,
        }

    def _load(self, execution_id: str) -> Dict[str, Any]:
        return self.store.load(_validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES))

    def _public(
        self,
        record: Dict[str, Any],
        include_output: bool = False,
        compact: bool = False,
        execution_in_progress: Optional[bool] = None,
    ) -> Dict[str, Any]:
        phases = []
        for phase in record["phases"]:
            keys = (
                ("name", "status", "side_effects", "unsafe_side_effects", "attempts", "error")
                if compact
                else (
                    "name", "status", "cwd", "timeout_seconds", "max_output_bytes",
                    "side_effects", "unsafe_side_effects", "idempotency_key",
                    "structured_metrics", "attempts", "events", "attempt_results",
                    "result", "error",
                )
            )
            item = {
                key: phase.get(key)
                for key in keys
            }
            if include_output and not compact:
                item["output"] = phase.get("output", "")
            elif not compact:
                output = phase.get("output")
                if output is not None:
                    item["output_bytes"] = len(output.encode("utf-8"))
                else:
                    item["output_bytes"] = (phase.get("result") or {}).get(
                        "output_byte_size", 0
                    )
            phases.append(item)
        completed = sum(phase["status"] == "completed" for phase in record["phases"])
        next_phase = next(
            (phase["name"] for phase in record["phases"] if phase["status"] != "completed"),
            None,
        )
        next_record = next(
            (phase for phase in record["phases"] if phase["status"] != "completed"),
            None,
        )
        if execution_in_progress is None:
            execution_in_progress = any(
                phase["status"] == "started" for phase in record["phases"]
            )
        attempt_limit_reached = bool(
            next_record
            and next_record.get("attempts", 0) >= MAX_PHASE_ATTEMPTS
        )
        retry_required = bool(
            next_record
            and next_record["status"] in {"failed", "timed_out"}
            and not attempt_limit_reached
        )
        confirmation_required = bool(
            next_record
            and next_record["unsafe_side_effects"]
            and next_record["status"] in {"failed", "timed_out", "interrupted"}
            and not attempt_limit_reached
            and record["resume_policy"] != "allow-unsafe"
        )
        execution_status = record["execution_status"]
        summary = (
            f"Execution '{record['execution_id']}' is {execution_status}; "
            f"{completed}/{len(record['phases'])} phases completed"
        )
        if next_phase:
            summary += f"; next phase: {next_phase}"
        return {
            "status": "ok",
            "schema_version": record["schema_version"],
            "execution_id": record["execution_id"],
            "label": record["label"],
            "execution_status": execution_status,
            "partial": record["partial"],
            "summary": summary,
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "resume": {
                "available": (
                    next_phase is not None
                    and not attempt_limit_reached
                    and not execution_in_progress
                    and not bool(next_record and next_record.get("process_fence_pending"))
                ),
                "retry_required": retry_required,
                "unsafe_confirmation_required": confirmation_required,
                "attempt_limit_reached": attempt_limit_reached,
                "next_phase": next_phase,
                "policy": record["resume_policy"],
            },
            "completed_phase_count": completed,
            "phase_count": len(record["phases"]),
            "phases": phases,
        }

    def create(
        self,
        phases: List[Dict[str, Any]],
        *,
        execution_id: Optional[str] = None,
        label: str = "",
        resume_policy: str = "safe",
        cwd: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_output_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            with self._create_and_lease(
                phases, execution_id, label, resume_policy, cwd,
                timeout_seconds, max_output_bytes,
            ) as record:
                return record

    @contextmanager
    def _create_and_lease(
        self,
        phases: List[Dict[str, Any]],
        execution_id: Optional[str],
        label: str,
        resume_policy: str,
        cwd: Optional[str],
        timeout_seconds: Optional[float],
        max_output_bytes: Optional[int],
    ):
        """Create a record and retain its lease through the caller's work."""
        record = self._new_record(
            phases, execution_id, label, resume_policy, cwd,
            timeout_seconds, max_output_bytes,
        )
        with self.store.lease(record["execution_id"]):
            try:
                self.store.load(record["execution_id"], recover=False)
            except KeyError:
                self.store.save(record)
                yield record
                return
            raise ValueError(f"Execution '{record['execution_id']}' already exists")

    @staticmethod
    def _first_incomplete(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return next((phase for phase in record["phases"] if phase["status"] != "completed"), None)

    @staticmethod
    def _attempt_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
        """Keep retry history bounded without repeating phase-level metrics."""
        snapshot = dict(result)
        snapshot.pop("structured_metrics", None)
        return snapshot

    def _record_process_identity(
        self,
        record: Dict[str, Any],
        phase: Dict[str, Any],
        process_id: int,
        process_group_id: Optional[int],
    ) -> None:
        """Persist the child process identity before command output is consumed."""
        phase["process_id"] = process_id
        phase["process_group_id"] = process_group_id
        phase["process_start_time"], phase["process_boot_id"] = _proc_identity(process_id)
        phase["process_fence_pending"] = False
        record["updated_at"] = _now()
        self.store.save(record)

    @staticmethod
    def _clear_process_identity(phase: Dict[str, Any]) -> None:
        phase["process_id"] = None
        phase["process_group_id"] = None
        phase["process_start_time"] = None
        phase["process_boot_id"] = None
        phase["process_fence_pending"] = False

    def _run(
        self,
        record: Dict[str, Any],
        *,
        retry_failed: bool,
        confirm_unsafe: bool,
        output_handler: Optional[OutputHandler],
    ) -> Dict[str, Any]:
        while True:
            phase = self._first_incomplete(record)
            if phase is None:
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                return record
            if phase.get("process_fence_pending"):
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                return record
            if phase["status"] in {"failed", "timed_out"}:
                if not retry_failed:
                    self._refresh_overall_status(record)
                    record["updated_at"] = _now()
                    self.store.save(record)
                    return record
            if phase["status"] in {"failed", "timed_out", "interrupted"}:
                if (
                    phase["unsafe_side_effects"]
                    and not confirm_unsafe
                    and record["resume_policy"] != "allow-unsafe"
                ):
                    self._refresh_overall_status(record)
                    record["updated_at"] = _now()
                    self.store.save(record)
                    return record

            if phase["attempts"] >= MAX_PHASE_ATTEMPTS:
                if (phase.get("result") or {}).get("error_type") == "attempt_limit":
                    return record
                phase["error"] = f"phase exceeded the {MAX_PHASE_ATTEMPTS}-attempt limit"
                phase["result"] = {"error_type": "attempt_limit"}
                _phase_event(phase, "failed", error_type="attempt_limit")
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                return record

            phase["attempts"] += 1
            phase["status"] = "started"
            phase["error"] = None
            phase["output"] = ""
            phase["result"] = None
            phase["process_id"] = None
            phase["process_group_id"] = None
            phase["process_start_time"] = None
            phase["process_boot_id"] = None
            phase["process_fence_pending"] = True
            _phase_event(phase, "started")
            self._refresh_overall_status(record)
            record["updated_at"] = _now()
            self.store.save(record)
            started = time.perf_counter()
            try:
                if self.command_runner is run_command_bounded:
                    output, exit_code, truncated, original_byte_size, timed_out = run_command_bounded(
                        phase["command"],
                        phase["cwd"],
                        phase["max_output_bytes"],
                        phase["timeout_seconds"],
                        process_started=lambda process_id, process_group_id: self._record_process_identity(
                            record, phase, process_id, process_group_id
                        ),
                    )
                else:
                    output, exit_code, truncated, original_byte_size, timed_out = self.command_runner(
                        phase["command"],
                        phase["cwd"],
                        phase["max_output_bytes"],
                        phase["timeout_seconds"],
                    )
            except (KeyboardInterrupt, SystemExit):
                self._clear_process_identity(phase)
                phase["status"] = "interrupted"
                phase["error"] = "phase interrupted before a result was available"
                _phase_event(phase, "interrupted", reason="runner interruption")
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                raise
            except Exception as exc:
                self._clear_process_identity(phase)
                phase["status"] = "failed"
                phase["error"] = _bounded_error(exc)
                phase["result"] = {
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                }
                phase["attempt_results"].append(self._attempt_snapshot(phase["result"]))
                _phase_event(phase, "failed", error_type=type(exc).__name__)
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                return record

            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self._clear_process_identity(phase)
            phase["output"] = output
            phase["result"] = {
                "duration_ms": duration_ms,
                "exit_code": exit_code,
                "timed_out": bool(timed_out),
                "truncated": bool(truncated),
                "output_byte_size": len(output.encode("utf-8")),
                "original_byte_size": original_byte_size,
                "structured_metrics": phase["structured_metrics"],
            }
            if output_handler is not None:
                try:
                    capture_id = output_handler(phase, output, phase["result"])
                    if capture_id is not None:
                        phase["result"]["capture_id"] = capture_id
                except Exception as exc:
                    phase["result"]["capture_error"] = _bounded_error(exc)
            if timed_out:
                phase["status"] = "timed_out"
                event_status = "timed_out"
            elif exit_code == 0:
                phase["status"] = "completed"
                event_status = "completed"
            else:
                phase["status"] = "failed"
                event_status = "failed"
            phase["attempt_results"].append(self._attempt_snapshot(phase["result"]))
            _phase_event(phase, event_status, exit_code=exit_code)
            self._refresh_overall_status(record)
            record["updated_at"] = _now()
            self.store.save(record)
            if phase["status"] != "completed":
                return record

    def start(
        self,
        phases: List[Dict[str, Any]],
        *,
        execution_id: Optional[str] = None,
        label: str = "",
        resume_policy: str = "safe",
        cwd: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_output_bytes: Optional[int] = None,
        output_handler: Optional[OutputHandler] = None,
    ) -> Dict[str, Any]:
        with self._create_and_lease(
            phases, execution_id, label, resume_policy, cwd,
            timeout_seconds, max_output_bytes,
        ) as record:
            record = self.store.load(record["execution_id"], recover=False)
            completed = self._run(
                record,
                retry_failed=False,
                confirm_unsafe=False,
                output_handler=output_handler,
            )
            return self._public(completed)

    def resume(
        self,
        execution_id: str,
        *,
        retry_failed: bool = False,
        confirm_unsafe: bool = False,
        output_handler: Optional[OutputHandler] = None,
    ) -> Dict[str, Any]:
        execution_id = _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
        # Avoid creating a durable lease file for a request that cannot resolve
        # to an execution record. The second read under the lease closes the
        # normal create/resume race without leaking files for typos.
        self.store.load(execution_id, recover=False)
        with self.store.lease(execution_id):
            record = self.store.load(execution_id, recover=False)
            if self.store._recover_started(record):
                record["updated_at"] = _now()
                self.store.save(record)
            completed = self._run(
                record,
                retry_failed=retry_failed,
                confirm_unsafe=confirm_unsafe,
                output_handler=output_handler,
            )
            return self._public(completed)

    def get(self, execution_id: str, *, include_output: bool = False) -> Dict[str, Any]:
        with self._lock:
            return self._load(execution_id)

    def public(self, execution_id: str, *, include_output: bool = False) -> Dict[str, Any]:
        with self._lock:
            record = self._load(execution_id)
            return self._public(
                record,
                include_output=include_output,
                execution_in_progress=self.store.is_locked(record["execution_id"]),
            )

    def output(
        self,
        execution_id: str,
        phase_name: Optional[str] = None,
        *,
        offset: int = 0,
        max_bytes: int = MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
    ) -> Dict[str, Any]:
        with self._lock:
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("offset must be a non-negative integer")
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
                raise ValueError("max_bytes must be an integer")
            if max_bytes < 512 or max_bytes > MAX_EXECUTION_OUTPUT_CHUNK_BYTES:
                raise ValueError(
                    f"max_bytes must be between 512 and {MAX_EXECUTION_OUTPUT_CHUNK_BYTES}"
                )
            if phase_name is None and offset:
                raise ValueError("offset requires phase_name")
            record = self._load(execution_id)
            phases = record["phases"]
            if phase_name is not None:
                phase_name = _validate_text(phase_name, "phase_name", MAX_PHASE_NAME_BYTES)
                matches = [phase for phase in phases if phase["name"] == phase_name]
                if not matches:
                    raise KeyError(f"Phase '{phase_name}' was not found in execution '{execution_id}'")
                phases = matches
            output_phases = []
            remaining = max_bytes
            for phase in phases:
                full_output = phase.get("output", "")
                phase_offset = offset if phase_name is not None else 0
                source = full_output[phase_offset:]
                retained = source.encode("utf-8")[:remaining]
                output = retained.decode("utf-8", errors="ignore")
                next_offset = phase_offset + len(output)
                result = dict(phase.get("result") or {})
                result.pop("structured_metrics", None)
                output_phases.append({
                    "name": phase["name"],
                    "status": phase["status"],
                    "output": output,
                    "output_byte_size": len(full_output.encode("utf-8")),
                    "offset": phase_offset,
                    "next_offset": next_offset,
                    "truncated": next_offset < len(full_output),
                    "result": result,
                })
                remaining -= len(output.encode("utf-8"))
                if remaining <= 0:
                    remaining = 0
            return {
                "status": "ok",
                "execution_id": record["execution_id"],
                "execution_status": record["execution_status"],
                "partial": record["partial"],
                "max_bytes": max_bytes,
                "phases": output_phases,
            }

    def list_public(
        self,
        *,
        limit: int = DEFAULT_EXECUTION_LIST_LIMIT,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit < 1 or limit > MAX_EXECUTION_LIST_LIMIT:
                raise ValueError(
                    f"limit must be between 1 and {MAX_EXECUTION_LIST_LIMIT}"
                )
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("offset must be a non-negative integer")
            records = self.store.list(limit=limit, offset=offset)
            return [
                self._public(
                    record,
                    compact=True,
                    execution_in_progress=self.store.is_locked(record["execution_id"]),
                )
                for record in records
            ]
