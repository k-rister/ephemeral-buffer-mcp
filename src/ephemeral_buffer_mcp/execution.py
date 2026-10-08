"""Durable, phase-level command execution with explicit resume boundaries.

Execution records are deliberately separate from the in-memory capture ring.
The ring is optimized for search during one server session; this module keeps
the bounded output and phase metadata needed to recover a long-running task
after a process restart.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import select
import signal
import shutil
import stat
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import errno
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .capture_utils import PROCESS_MARKER_ENV, run_command_bounded
from .config import (
    DEFAULT_MAX_OUTPUT_BYTES,
    execution_checkpoint_reserve_bytes,
    execution_state_quota_bytes,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - durable execution requires Linux.
    fcntl = None


EXECUTION_SCHEMA_VERSION = 2
LEGACY_EXECUTION_SCHEMA_VERSION = 1
PHASE_STATUSES = ("pending", "started", "completed", "failed", "interrupted", "timed_out")
RESUME_POLICIES = ("safe", "allow-unsafe")
MAX_EXECUTION_ID_BYTES = 256
MAX_PHASE_NAME_BYTES = 256
MAX_ERROR_BYTES = 4096
MAX_STRUCTURED_METRICS_BYTES = 16 * 1024
MAX_EXECUTION_PHASES = 64
MAX_PHASE_ATTEMPTS = 32
# Durable execution recovery is supported on Linux, where pid_t is signed 32-bit.
MAX_PROCESS_ID = (1 << 31) - 1
MAX_EXECUTION_STATE_BYTES = 64 * 1024 * 1024
EXECUTION_METADATA_RESERVE_BYTES = 4 * 1024 * 1024
MAX_EXECUTION_RECORDS = 1000
MAX_BACKGROUND_EXECUTIONS = 8
DEFAULT_EXECUTION_LIST_LIMIT = 20
MAX_EXECUTION_LIST_LIMIT = 100
MAX_EXECUTION_OUTPUT_CHUNK_BYTES = 8 * 1024
MAX_EXECUTION_RECORD_PREVIEW_BYTES = 8 * 1024
MAX_EXECUTION_RETIRE_BATCH = 20
JSON_OUTPUT_EXPANSION_BOUND = 6
PROCESS_CONTAINMENT_SUBREAPER = "linux-subreaper"


CommandRunner = Callable[[str, Optional[str], int, Optional[float]], Tuple[str, int, bool, int, bool]]
ProcessCleanup = Callable[[], None]
OutputHandler = Callable[[Dict[str, Any], str, Dict[str, Any]], Optional[str]]


class ExecutionBusyError(RuntimeError):
    """Raised when another process currently owns an execution lease."""


class ExecutionRecordError(ValueError):
    """A durable record cannot safely enter normal execution or recovery."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "MALFORMED_RECORD",
        schema_version: Any = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.schema_version = schema_version


@dataclass(frozen=True)
class CleanupOutcome:
    """Whether process cleanup was confirmed and why a fence remains."""

    confirmed: bool
    blocked_reason_code: Optional[str] = None


class ExecutionListDiagnostic(dict):
    """A diagnostic mapping that remains compatible with dict consumers."""

    def __init__(self, response: Dict[str, Any]) -> None:
        super().__init__(response)

    @property
    def response(self) -> Dict[str, Any]:
        """Return this diagnostic mapping as a plain dictionary response."""
        return self


BLOCKED_REASON_REMEDIES = {
    "PROCESS_IDENTITY_INCOMPLETE": (
        "Inspect the saved process identity and host state; keep the execution fenced "
        "until the process group is proven stopped."
    ),
    "PROCESS_STATE_UNVERIFIABLE": (
        "Restore process visibility or inspect the host process state, then request "
        "recovery again."
    ),
    "PROCESS_NOT_CONFIRMED_GONE": (
        "Confirm that no process from this execution remains, then request recovery "
        "again."
    ),
    "PROCESS_CLEANUP_UNCONFIRMED": (
        "Check process recovery permissions and the execution process group before "
        "requesting recovery again."
    ),
    "CLEANUP_HOOK_UNAVAILABLE": (
        "Configure the runner cleanup hook and confirm that it stops all child "
        "processes before resuming."
    ),
    "CLEANUP_HOOK_FAILED": (
        "Repair the runner cleanup hook and confirm that it stops all child processes "
        "before resuming."
    ),
}


def _blocked_reason(code: Optional[str]) -> Optional[Dict[str, str]]:
    if code is None:
        return None
    return {
        "code": code,
        "remedy": BLOCKED_REASON_REMEDIES.get(
            code,
            "Inspect the execution record and host process state before requesting recovery again.",
        ),
    }


def _record_error(
    execution_id: str,
    detail: str,
    *,
    reason_code: str = "MALFORMED_RECORD",
    schema_version: Any = None,
) -> ExecutionRecordError:
    return ExecutionRecordError(
        f"Execution '{execution_id}' has invalid state: {detail}",
        reason_code=reason_code,
        schema_version=schema_version,
    )


def _max_phase_output_bytes(phase_count: int) -> int:
    state_output_budget = max(
        512,
        (MAX_EXECUTION_STATE_BYTES - EXECUTION_METADATA_RESERVE_BYTES)
        // JSON_OUTPUT_EXPANSION_BOUND,
    )
    return state_output_budget // phase_count


def _normalize_v1_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the known v1 layout into the current in-memory v2 shape."""
    normalized = dict(record)
    phases = normalized.get("phases")
    if isinstance(phases, list):
        normalized_phases = []
        for phase in phases:
            if not isinstance(phase, dict):
                normalized_phases.append(phase)
                continue
            item = dict(phase)
            unsafe_side_effects = item.get("unsafe_side_effects", False)
            item.setdefault(
                "side_effects",
                "unsafe" if unsafe_side_effects is True else "none",
            )
            item.setdefault(
                "unsafe_side_effects",
                item.get("side_effects") == "unsafe",
            )
            item.setdefault("timeout_seconds", None)
            for key in (
                "structured_metrics", "events", "attempt_results", "output",
                "result", "error", "process_id", "process_group_id",
                "process_start_time", "process_boot_id", "process_launch_token",
                "process_containment",
            ):
                default = {} if key == "structured_metrics" else [] if key in {
                    "events", "attempt_results"
                } else "" if key == "output" else None
                item.setdefault(key, default)
            item.setdefault(
                "process_fence_pending",
                item.get("status") == "started",
            )
            item.setdefault("blocked_reason_code", None)
            normalized_phases.append(item)
        normalized["phases"] = normalized_phases
    normalized["schema_version"] = EXECUTION_SCHEMA_VERSION
    return normalized


def _validate_execution_record(
    record: Any,
    execution_id: str,
) -> Dict[str, Any]:
    """Validate and normalize a record before public interpretation or recovery."""
    if not isinstance(record, dict):
        raise _record_error(execution_id, "record must be an object")
    version = record.get("schema_version")
    if type(version) is not int:
        raise _record_error(
            execution_id,
            "schema_version must be an integer",
            schema_version=version,
        )
    if version == LEGACY_EXECUTION_SCHEMA_VERSION:
        record = _normalize_v1_record(record)
    elif version != EXECUTION_SCHEMA_VERSION:
        raise _record_error(
            execution_id,
            f"unsupported schema version {version}",
            reason_code="UNSUPPORTED_SCHEMA_VERSION",
            schema_version=version,
        )

    def require_text(value: Any, field: str, max_bytes: int) -> None:
        if not isinstance(value, str) or not value:
            raise _record_error(execution_id, f"{field} must be a bounded non-empty string")
        try:
            too_long = len(value.encode("utf-8")) > max_bytes
        except UnicodeEncodeError:
            too_long = True
        if too_long:
            raise _record_error(execution_id, f"{field} must be a bounded non-empty string")

    if record.get("execution_id") != execution_id:
        raise _record_error(execution_id, "execution_id does not match its record path")
    require_text(record.get("execution_id"), "execution_id", MAX_EXECUTION_ID_BYTES)
    if record.get("background_error") is not None:
        require_text(record["background_error"], "background_error", MAX_ERROR_BYTES)
    require_text(record.get("label"), "label", 1024)
    require_text(record.get("created_at"), "created_at", 64)
    require_text(record.get("updated_at"), "updated_at", 64)
    if record.get("execution_status") not in (
        "pending", "running", "completed", "partial", "interrupted"
    ):
        raise _record_error(execution_id, "execution_status is invalid")
    if not isinstance(record.get("partial"), bool):
        raise _record_error(execution_id, "partial must be a boolean")
    if record.get("resume_policy") not in RESUME_POLICIES:
        raise _record_error(execution_id, "resume_policy is invalid")
    phases = record.get("phases")
    if not isinstance(phases, list) or not phases or len(phases) > MAX_EXECUTION_PHASES:
        raise _record_error(execution_id, "phases must be a non-empty bounded list")
    max_phase_output_bytes = _max_phase_output_bytes(len(phases))

    phase_names = set()
    for index, phase in enumerate(phases):
        prefix = f"phases[{index}]"
        if not isinstance(phase, dict):
            raise _record_error(execution_id, f"{prefix} must be an object")
        require_text(phase.get("name"), f"{prefix}.name", MAX_PHASE_NAME_BYTES)
        require_text(phase.get("command"), f"{prefix}.command", 16 * 1024)
        require_text(phase.get("cwd"), f"{prefix}.cwd", 4096)
        if phase["name"] in phase_names:
            raise _record_error(execution_id, "phase names must be unique")
        phase_names.add(phase["name"])
        if phase.get("status") not in PHASE_STATUSES:
            raise _record_error(execution_id, f"{prefix}.status is invalid")
        attempts = phase.get("attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or attempts > MAX_PHASE_ATTEMPTS
        ):
            raise _record_error(execution_id, f"{prefix}.attempts is invalid")
        if not isinstance(phase.get("process_fence_pending"), bool):
            raise _record_error(execution_id, f"{prefix}.process_fence_pending must be a boolean")
        if phase.get("side_effects") not in ("none", "unsafe"):
            raise _record_error(execution_id, f"{prefix}.side_effects is invalid")
        if not isinstance(phase.get("unsafe_side_effects"), bool):
            raise _record_error(execution_id, f"{prefix}.unsafe_side_effects must be a boolean")
        if phase["unsafe_side_effects"] != (phase["side_effects"] == "unsafe"):
            raise _record_error(execution_id, f"{prefix} side-effect fields disagree")
        if not isinstance(phase.get("events"), list) or not all(
            isinstance(event, dict) for event in phase["events"]
        ):
            raise _record_error(execution_id, f"{prefix}.events must be a list of objects")
        if not isinstance(phase.get("attempt_results"), list) or not all(
            isinstance(result, dict) for result in phase["attempt_results"]
        ):
            raise _record_error(
                execution_id,
                f"{prefix}.attempt_results must be a list of objects",
            )
        if phase.get("result") is not None and not isinstance(phase.get("result"), dict):
            raise _record_error(execution_id, f"{prefix}.result must be an object or null")
        if phase.get("output") is not None and not isinstance(phase.get("output"), str):
            raise _record_error(execution_id, f"{prefix}.output must be a string or null")
        if phase.get("error") is not None and not isinstance(phase.get("error"), str):
            raise _record_error(execution_id, f"{prefix}.error must be a string or null")
        if not isinstance(phase.get("structured_metrics"), dict):
            raise _record_error(execution_id, f"{prefix}.structured_metrics must be an object")
        try:
            _json_copy(phase["structured_metrics"], f"{prefix}.structured_metrics")
        except ValueError as exc:
            raise _record_error(
                execution_id,
                f"{prefix}.structured_metrics is invalid",
            ) from exc
        try:
            if "timeout_seconds" not in phase:
                raise ValueError("timeout_seconds is missing")
            _validate_positive_number(phase["timeout_seconds"], f"{prefix}.timeout_seconds")
        except ValueError as exc:
            raise _record_error(execution_id, f"{prefix}.timeout_seconds is invalid") from exc
        max_output_bytes = phase.get("max_output_bytes")
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes < 512
            or max_output_bytes > max_phase_output_bytes
        ):
            raise _record_error(execution_id, f"{prefix}.max_output_bytes is invalid")
        idempotency_key = phase.get("idempotency_key")
        if idempotency_key is not None:
            require_text(idempotency_key, f"{prefix}.idempotency_key", MAX_PHASE_NAME_BYTES)
        for key in ("process_id", "process_group_id"):
            value = phase.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > MAX_PROCESS_ID
            ):
                raise _record_error(execution_id, f"{prefix}.{key} is invalid")
        for key in ("process_start_time", "process_boot_id", "process_launch_token"):
            value = phase.get(key)
            if value is not None:
                require_text(value, f"{prefix}.{key}", 256)
                if key == "process_launch_token":
                    try:
                        value.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise _record_error(
                            execution_id,
                            f"{prefix}.process_launch_token must contain only ASCII characters",
                        ) from exc
        containment = phase.get("process_containment")
        if containment is not None:
            require_text(containment, f"{prefix}.process_containment", 128)
        blocked_code = phase.get("blocked_reason_code")
        if blocked_code is not None:
            require_text(blocked_code, f"{prefix}.blocked_reason_code", 64)

    try:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _record_error(execution_id, "record contains invalid JSON values") from exc
    if len(encoded) > MAX_EXECUTION_STATE_BYTES:
        raise _record_error(execution_id, "record exceeds the size limit")
    return record


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
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number or null") from exc
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


class _DigestingReader:
    """Hash the exact bytes copied from a file into a tar member."""

    def __init__(self, source: Any):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the source and hash the exact returned chunk."""
        chunk = self.source.read(size)
        self.digest.update(chunk)
        return chunk

    def hexdigest(self) -> str:
        """Return the SHA-256 digest of all bytes read so far."""
        return self.digest.hexdigest()


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


def _proc_visibility_restricted() -> bool:
    """Return true when procfs may hide process entries needed for recovery."""
    try:
        mounts = Path("/proc/mounts").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return True
    proc_mount_found = False
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[1] != "/proc" or fields[2] != "proc":
            continue
        proc_mount_found = True
        options = fields[3].split(",")
        hidepid_values = [option.partition("=")[2] for option in options
                          if option.startswith("hidepid=")]
        if not hidepid_values:
            continue
        if len(hidepid_values) != 1:
            return True
        hidepid_value = hidepid_values[0]
        hidepid_aliases = {
            "off": 0,
            "noaccess": 1,
            "invisible": 2,
            "ptraceable": 4,
        }
        if hidepid_value in hidepid_aliases:
            hidepid = hidepid_aliases[hidepid_value]
        else:
            try:
                hidepid = int(hidepid_value)
            except ValueError:
                return True
        # hidepid=1 keeps PID directories visible while restricting their
        # contents. Higher modes hide process entries unless the service is in
        # the procfs gid= group that is allowed to inspect restricted entries.
        if hidepid in (0, 1):
            continue
        if hidepid not in (2, 4):
            return True
        gid_values = [option.partition("=")[2] for option in options
                      if option.startswith("gid=")]
        if len(gid_values) != 1:
            return True
        try:
            proc_gid = int(gid_values[0])
            service_gids = set(os.getgroups())
            service_gids.update((os.getgid(), os.getegid()))
        except (OSError, ValueError):
            return True
        if proc_gid < 0 or proc_gid not in service_gids:
            return True
    return not proc_mount_found


def _process_group_is_absent(
    process_group_id: int,
    *,
    expected_leader: Optional[Tuple[int, Tuple[str, str]]] = None,
) -> bool:
    """Return true when a process group is gone after reaping owned zombies."""
    if not sys.platform.startswith("linux"):
        return False
    if os.getpgrp() == process_group_id:
        return False
    try:
        os.killpg(process_group_id, 0)
    except OSError as exc:
        return _process_group_absent(exc)

    can_reap_children = False
    if expected_leader is not None:
        leader_id, leader_identity = expected_leader
        if (
            leader_id == process_group_id
            and isinstance(leader_identity, tuple)
            and len(leader_identity) == 2
            and all(isinstance(value, str) and value for value in leader_identity)
        ):
            current_identity = _proc_identity(leader_id)
            if current_identity == leader_identity:
                can_reap_children = True
            elif current_identity == (None, None):
                try:
                    os.kill(leader_id, 0)
                except OSError as exc:
                    can_reap_children = _process_group_absent(exc)

    if not can_reap_children:
        return False
    while True:
        try:
            child_pid, _status = os.waitpid(-process_group_id, os.WNOHANG)
        except ChildProcessError:
            break
        except OSError:
            return False
        if child_pid == 0:
            break

    # Recheck with the kernel after reaping only children owned by this process.
    # Any surviving member, including one hidden by procfs permissions, keeps
    # recovery fenced.
    try:
        os.killpg(process_group_id, 0)
    except OSError as exc:
        return _process_group_absent(exc)
    return False


def _pidfd_terminated(pidfd: int, timeout: float = 0.0) -> bool:
    try:
        ready, _write, _error = select.select([pidfd], [], [], timeout)
    except (OSError, ValueError):
        return False
    return bool(ready)


def _process_recovery_supported() -> bool:
    """Probe every backend required to recover built-in command processes."""
    if not sys.platform.startswith("linux") or fcntl is None:
        return False
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    select_fn = getattr(select, "select", None)
    if not all(callable(value) for value in (pidfd_open, pidfd_send_signal, select_fn)):
        return False
    try:
        if not Path("/proc").is_dir():
            return False
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if not boot_id or _proc_identity(os.getpid()) == (None, None):
            return False
        pidfd = pidfd_open(os.getpid())
        try:
            pidfd_send_signal(pidfd, 0)
            select_fn([pidfd], [], [], 0)
        finally:
            os.close(pidfd)
    except (OSError, ValueError, UnicodeError, TypeError):
        return False
    return True


def _terminate_pidfd(
    process_id: int,
    *,
    expected_identity: Optional[Tuple[str, str]] = None,
    expected_group: Optional[int] = None,
) -> bool:
    """Signal a pinned process without exposing a reusable numeric PID race."""
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is None or pidfd_send_signal is None or process_id == os.getpid():
        return False
    try:
        pidfd = pidfd_open(process_id)
    except OSError as exc:
        if not _process_group_absent(exc):
            return False
        return expected_group is None or _process_group_is_absent(
            expected_group,
            expected_leader=(process_id, expected_identity)
            if expected_identity is not None
            else None,
        )
    try:
        if expected_identity is not None:
            current_identity = _proc_identity(process_id)
            if current_identity != expected_identity:
                if current_identity == (None, None) and _pidfd_terminated(pidfd):
                    return expected_group is None or _process_group_is_absent(
                        expected_group,
                        expected_leader=(process_id, expected_identity),
                    )
                return False
        if expected_group is not None:
            try:
                if os.getpgid(process_id) != expected_group:
                    return False
            except OSError as exc:
                if not _process_group_absent(exc):
                    return False
                return _pidfd_terminated(pidfd) and _process_group_is_absent(
                    expected_group,
                    expected_leader=(process_id, expected_identity)
                    if expected_identity is not None
                    else None,
                )
        try:
            pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            return True if expected_group is None else (
                _pidfd_terminated(pidfd) and _process_group_is_absent(
                    expected_group,
                    expected_leader=(process_id, expected_identity)
                    if expected_identity is not None
                    else None,
                )
            )
        if not _pidfd_terminated(pidfd, 1):
            try:
                pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not _pidfd_terminated(pidfd, 1):
                return False
        return expected_group is None or _process_group_is_absent(
            expected_group,
            expected_leader=(process_id, expected_identity)
            if expected_identity is not None
            else None,
        )
    finally:
        os.close(pidfd)


def _terminate_process_group(process_group_id: int) -> bool:
    """Terminate a verified process group and prove that it disappeared."""
    try:
        if os.getpgrp() == process_group_id:
            return False
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


def _marker_processes(marker: str) -> Optional[set[int]]:
    """Find Linux processes carrying a durable execution marker."""
    if not sys.platform.startswith("linux") or _proc_visibility_restricted():
        return None
    marker_bytes = f"{PROCESS_MARKER_ENV}={marker}".encode("ascii")
    try:
        process_names = os.listdir("/proc")
    except OSError:
        return None
    processes: set[int] = set()
    for process_name in process_names:
        if not process_name.isdigit():
            continue
        process_id = int(process_name)
        try:
            environment = Path(f"/proc/{process_id}/environ").read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            # Without reading this entry, recovery cannot prove that it is
            # unrelated to the execution being recovered.
            return None
        if marker_bytes not in environment.split(b"\0"):
            continue
        processes.add(process_id)
    return processes


def _marker_process_groups(marker: str) -> Optional[set[int]]:
    """Find Linux process groups carrying a durable execution marker."""
    processes = _marker_processes(marker)
    if processes is None:
        return None
    groups: set[int] = set()
    for process_id in processes:
        try:
            groups.add(os.getpgid(process_id))
        except OSError as exc:
            if _process_group_absent(exc):
                continue
            return None
    return groups


def _marker_process_identities(marker: str) -> Optional[Dict[int, Tuple[str, str]]]:
    """Pin marker-bearing processes to their current kernel identities."""
    processes = _marker_processes(marker)
    if processes is None:
        return None
    identities: Dict[int, Tuple[str, str]] = {}
    for process_id in processes:
        start_time, boot_id = _proc_identity(process_id)
        if not start_time or not boot_id:
            return None
        identities[process_id] = (start_time, boot_id)
    return identities


def _terminate_marker_processes(
    marker: str,
    identities: Optional[Dict[int, Tuple[str, str]]] = None,
) -> bool:
    """Terminate marker-bearing processes through pinned process descriptors."""
    identities = _marker_process_identities(marker) if identities is None else identities
    if identities is None:
        return False
    if not identities:
        return True
    if not all(
        _terminate_pidfd(process_id, expected_identity=identity)
        for process_id, identity in identities.items()
    ):
        return False
    return _marker_process_identities(marker) == {}


def _terminate_marker_processes_or_confirm_group_absent(
    marker: str,
    process_group_id: int,
    *,
    expected_leader: Optional[Tuple[int, Tuple[str, str]]] = None,
) -> bool:
    """Clean marked descendants and confirm no live group member remains."""
    identities = _marker_process_identities(marker)
    if identities is None:
        return False
    if identities:
        return _terminate_marker_processes(
            marker, identities
        ) and _process_group_is_absent(
            process_group_id, expected_leader=expected_leader
        )
    return _process_group_is_absent(
        process_group_id, expected_leader=expected_leader
    )


def _terminate_stale_process_outcome(phase: Dict[str, Any]) -> CleanupOutcome:
    """Stop a stale command and retain a stable reason if its fence remains."""

    def result(confirmed: bool, reason_code: str) -> CleanupOutcome:
        return CleanupOutcome(confirmed, None if confirmed else reason_code)

    process_id = phase.get("process_id")
    process_group_id = phase.get("process_group_id")
    valid_process_id = (
        not isinstance(process_id, bool)
        and isinstance(process_id, int)
        and process_id > 0
    )
    valid_process_group_id = (
        not isinstance(process_group_id, bool)
        and isinstance(process_group_id, int)
        and process_group_id > 0
    )
    marker = phase.get("process_launch_token")
    if not valid_process_id or not valid_process_group_id:
        if isinstance(marker, str) and marker:
            # A process checkpoint can be interrupted before start-time
            # metadata is written. Pin marker-bearing PIDs individually; a
            # numeric process group is not safe to signal in that window.
            if valid_process_group_id:
                return result(
                    _terminate_marker_processes_or_confirm_group_absent(
                        marker, process_group_id
                    ),
                    "PROCESS_STATE_UNVERIFIABLE",
                )
            return result(
                _terminate_marker_processes(marker),
                "PROCESS_STATE_UNVERIFIABLE",
            )
        if valid_process_group_id:
            return result(
                _process_group_is_absent(process_group_id),
                "PROCESS_IDENTITY_INCOMPLETE",
            )
        # Never treat incomplete identity metadata as proof that an unknown
        # process group is gone. Legacy records without a launch marker remain
        # fenced until an operator resolves them.
        return CleanupOutcome(False, "PROCESS_IDENTITY_INCOMPLETE")
    expected_start = phase.get("process_start_time")
    expected_boot = phase.get("process_boot_id")
    if (
        not isinstance(expected_start, str)
        or not expected_start
        or not isinstance(expected_boot, str)
        or not expected_boot
    ):
        if isinstance(marker, str) and marker:
            return result(
                _terminate_marker_processes_or_confirm_group_absent(
                    marker, process_group_id
                ),
                "PROCESS_STATE_UNVERIFIABLE",
            )
        return result(
            _process_group_is_absent(process_group_id),
            "PROCESS_IDENTITY_INCOMPLETE",
        )
    current_start, current_boot = _proc_identity(process_id)
    if (
        not isinstance(current_start, str)
        or not current_start
        or not isinstance(current_boot, str)
        or not current_boot
        or current_start != expected_start
        or current_boot != expected_boot
    ):
        # A missing /proc identity is not proof that the original leader is
        # inactive, unless the kernel also proves the whole persisted group
        # has no live members.
        if isinstance(marker, str) and marker:
            return result(
                _terminate_marker_processes_or_confirm_group_absent(
                    marker,
                    process_group_id,
                    expected_leader=(process_id, (expected_start, expected_boot)),
                ),
                "PROCESS_STATE_UNVERIFIABLE",
            )
        return result(
            _process_group_is_absent(process_group_id),
            "PROCESS_STATE_UNVERIFIABLE",
        )
    if phase.get("process_containment") != PROCESS_CONTAINMENT_SUBREAPER:
        # Legacy records have no pinned supervisor.  Never signal a numeric
        # group whose membership can have changed since the checkpoint.
        return result(
            _process_group_is_absent(process_group_id),
            "PROCESS_IDENTITY_INCOMPLETE",
        )
    if isinstance(marker, str) and marker:
        # Stop the pinned supervisor, clean its marked descendants, and then
        # require the original process group to have no live members.
        if not _terminate_pidfd(
            process_id,
            expected_identity=(expected_start, expected_boot),
        ):
            return CleanupOutcome(False, "PROCESS_CLEANUP_UNCONFIRMED")
        return result(
            _terminate_marker_processes(marker) and _process_group_is_absent(
                process_group_id,
                expected_leader=(process_id, (expected_start, expected_boot)),
            ),
            "PROCESS_NOT_CONFIRMED_GONE",
        )
    terminated = _terminate_pidfd(
        process_id,
        expected_identity=(expected_start, expected_boot),
        expected_group=process_group_id,
    )
    if not terminated:
        return CleanupOutcome(False, "PROCESS_CLEANUP_UNCONFIRMED")
    return CleanupOutcome(True)


def _terminate_stale_process(phase: Dict[str, Any]) -> bool:
    """Backward-compatible boolean wrapper for stale process cleanup."""
    return _terminate_stale_process_outcome(phase).confirmed


class ExecutionStore:
    """Atomic JSON-file storage for resumable execution records."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str],
        *,
        quota_bytes: Optional[int] = None,
        checkpoint_reserve_bytes: Optional[int] = None,
    ):
        self.state_dir = Path(os.path.abspath(os.path.expanduser(os.fspath(state_dir))))
        self.quota_bytes = (
            execution_state_quota_bytes() if quota_bytes is None else quota_bytes
        )
        self.checkpoint_reserve_bytes = (
            execution_checkpoint_reserve_bytes()
            if checkpoint_reserve_bytes is None
            else checkpoint_reserve_bytes
        )
        if (
            isinstance(self.quota_bytes, bool)
            or not isinstance(self.quota_bytes, int)
            or self.quota_bytes < 1
        ):
            raise ValueError("execution state quota must be a positive integer")
        if (
            isinstance(self.checkpoint_reserve_bytes, bool)
            or not isinstance(self.checkpoint_reserve_bytes, int)
            or self.checkpoint_reserve_bytes < MAX_EXECUTION_STATE_BYTES
        ):
            raise ValueError(
                "execution checkpoint reserve must be at least "
                f"{MAX_EXECUTION_STATE_BYTES:,} bytes"
            )
        if self.checkpoint_reserve_bytes >= self.quota_bytes:
            raise ValueError("execution checkpoint reserve must be smaller than the quota")
        self._lock = threading.RLock()

    @staticmethod
    def _filename(execution_id: str) -> str:
        digest = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()
        return f"{digest}.json"

    @staticmethod
    def _is_record_digest(value: str) -> bool:
        return len(value) == 64 and all(char in "0123456789abcdef" for char in value)

    @classmethod
    def _matches_record_filename(cls, execution_id: str, filename: str) -> bool:
        try:
            return cls._filename(execution_id) == filename
        except UnicodeEncodeError:
            return False

    def _path(self, execution_id: str) -> Path:
        return self.state_dir / self._filename(execution_id)

    def _lock_path(self, execution_id: str) -> Path:
        return self.state_dir / f".{self._filename(execution_id)}.lock"

    def _summary_path(self, execution_id: str) -> Path:
        return self.state_dir / f"{self._filename(execution_id)[:-5]}.summary.json"

    @staticmethod
    def _compact_listing_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only fields needed to build a compact public list entry."""
        compact = {
            key: record.get(key)
            for key in (
                "schema_version", "execution_id", "label", "created_at",
                "updated_at", "execution_status", "partial", "resume_policy",
            )
        }
        if record.get("background_error") is not None:
            compact["background_error"] = record["background_error"]
        compact["phases"] = [
            {
                key: phase.get(key)
                for key in (
                    "name", "status", "side_effects", "unsafe_side_effects",
                    "attempts", "error", "process_fence_pending",
                    "blocked_reason_code",
                )
            }
            for phase in record.get("phases", [])
        ]
        return compact

    @staticmethod
    def _listing_file_diagnostic(
        path: Path,
        *,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe an unreadable record without returning any stored contents."""
        try:
            info = path.stat(follow_symlinks=False)
            record_bytes = info.st_size if stat.S_ISREG(info.st_mode) else 0
        except OSError:
            record_bytes = 0
        return {
            "status": "invalid_record",
            "execution_id": execution_id,
            "record_file_id": path.name.removesuffix(".json"),
            "schema_version": None,
            "read_only": True,
            "reason_code": "MALFORMED_RECORD",
            "message": "Execution record file is unreadable or malformed",
            "record_bytes": record_bytes,
            "record_preview": "[preview omitted because it could not be safely sanitized]",
            "record_preview_sanitized": True,
            "record_preview_redacted": False,
            "record_preview_truncated": True,
        }

    @staticmethod
    def _listing_file_sort_time(path: Path) -> str:
        try:
            return time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(path.stat(follow_symlinks=False).st_mtime),
            )
        except OSError:
            return ""

    def _reservation_path(self, execution_id: str) -> Path:
        digest = self._filename(execution_id)[:-5]
        return self.state_dir / f".checkpoint-{digest}.json"

    def _managed_file_sizes(self) -> Tuple[int, int, int, int]:
        """Return record, summary, record-count, and atomic-temp byte totals."""
        record_bytes = 0
        summary_bytes = 0
        record_count = 0
        temporary_bytes = 0
        if not self._validate_state_dir(require_exists=False):
            return record_bytes, summary_bytes, record_count, temporary_bytes
        for path in self.state_dir.iterdir():
            if path.name.startswith("."):
                continue
            is_summary = path.name.endswith(".summary.json")
            is_record = path.name.endswith(".json") and not is_summary
            is_temporary = path.name.startswith("tmp")
            if not (is_record or is_summary or is_temporary):
                continue
            self._reject_symlink(path, "state file")
            try:
                info = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"execution state file is not a regular file: {path}")
            if is_summary:
                summary_bytes += info.st_size
            elif is_temporary:
                temporary_bytes += info.st_size
            else:
                record_bytes += info.st_size
                record_count += 1
        return record_bytes, summary_bytes, record_count, temporary_bytes

    def _reservation_entries(self) -> List[Dict[str, Any]]:
        entries = []
        if not self._validate_state_dir(require_exists=False):
            return entries
        for path in self.state_dir.glob(".checkpoint-*.json"):
            try:
                path_info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(path_info.st_mode):
                    raise ValueError("checkpoint reservation is not a regular file")
                open_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                open_flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, open_flags)
                try:
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise ValueError("checkpoint reservation is not a regular file")
                    stream = os.fdopen(descriptor, "r", encoding="utf-8")
                    descriptor = -1
                    with stream:
                        reservation = json.load(stream)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                execution_id = reservation.get("execution_id")
                reserved_bytes = reservation.get("reserved_bytes")
                if (
                    not isinstance(reservation, dict)
                    or not isinstance(execution_id, str)
                    or self._reservation_path(execution_id).name != path.name
                    or isinstance(reserved_bytes, bool)
                    or not isinstance(reserved_bytes, int)
                    or reserved_bytes < 0
                ):
                    raise ValueError("invalid checkpoint reservation")
            except (OSError, json.JSONDecodeError, UnicodeError, AttributeError, ValueError):
                entries.append({
                    "path": path,
                    "execution_id": None,
                    "reserved_bytes": 0,
                    "invalid": True,
                })
                continue
            entries.append({
                "path": path,
                "execution_id": execution_id,
                "reserved_bytes": reserved_bytes,
                "invalid": False,
            })
        return entries

    def _reserved_checkpoint_bytes_locked(
        self,
        *,
        exclude_execution_id: Optional[str] = None,
        clean_stale: bool = False,
    ) -> int:
        reserved_bytes = 0
        for entry in self._reservation_entries():
            execution_id = entry["execution_id"]
            if entry["invalid"] or execution_id is None:
                raise ValueError(
                    "execution checkpoint reservation is unreadable; inspect the state directory"
                )
            if clean_stale and not self.is_locked(execution_id):
                try:
                    entry["path"].unlink()
                except FileNotFoundError:
                    pass
                continue
            if execution_id != exclude_execution_id:
                reserved_bytes += entry["reserved_bytes"]
        return reserved_bytes

    def _write_reservation_locked(self, execution_id: str, reserved_bytes: int) -> None:
        path = self._reservation_path(execution_id)
        self._reject_symlink(path, "checkpoint reservation")
        encoded = json.dumps(
            {"execution_id": execution_id, "reserved_bytes": reserved_bytes},
            sort_keys=True,
        ).encode("utf-8")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.state_dir, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._fsync_directory(self.state_dir)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def reserve_checkpoint(self, execution_id: str, reserved_bytes: int) -> None:
        """Reserve bounded record growth before starting a durable phase."""
        if isinstance(reserved_bytes, bool) or not isinstance(reserved_bytes, int) or reserved_bytes < 0:
            raise ValueError("checkpoint reservation must be a non-negative integer")
        with self._lock:
            with self._record_lease():
                self._ensure_state_dir()
                if not self.is_locked(execution_id):
                    raise RuntimeError("checkpoint reservations require the execution lease")
                record_bytes, summary_bytes, _, temporary_bytes = self._managed_file_sizes()
                other_reserved = self._reserved_checkpoint_bytes_locked(
                    exclude_execution_id=execution_id,
                    clean_stale=True,
                )
                if (
                    record_bytes
                    + summary_bytes
                    + temporary_bytes
                    + other_reserved
                    + reserved_bytes
                    > self.quota_bytes - self.checkpoint_reserve_bytes
                ):
                    raise ValueError(
                        "execution state quota has insufficient checkpoint headroom; "
                        "retire durable records or increase EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES"
                    )
                required_free_bytes = (
                    other_reserved + reserved_bytes + self.checkpoint_reserve_bytes
                )
                if shutil.disk_usage(self.state_dir).free < required_free_bytes:
                    raise ValueError(
                        "filesystem has insufficient free space for the next checkpoint"
                    )
                self._write_reservation_locked(execution_id, reserved_bytes)

    def release_checkpoint(self, execution_id: str) -> None:
        """Release a phase reservation after save, rollback, or pre-start cancel."""
        with self._lock:
            with self._record_lease():
                path = self._reservation_path(execution_id)
                self._reject_symlink(path, "checkpoint reservation")
                try:
                    path.unlink()
                except FileNotFoundError:
                    return
                self._fsync_directory(self.state_dir)

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
        if not sys.platform.startswith("linux"):
            raise RuntimeError("durable execution leases require a Linux platform")
        if fcntl is None:
            raise RuntimeError("durable execution leases require Linux file-locking support")
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

    def save(
        self,
        record: Dict[str, Any],
        *,
        consume_checkpoint_reservation: bool = False,
    ) -> None:
        """Persist one record with a replace, so readers never see a half-file."""
        with self._lock:
            with self._record_lease():
                self._ensure_state_dir()
                path = self._path(record["execution_id"])
                summary_path = self._summary_path(record["execution_id"])
                self._reject_symlink(path, "record file")
                self._reject_symlink(summary_path, "summary file")
                encoded = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
                if len(encoded) > MAX_EXECUTION_STATE_BYTES:
                    raise ValueError(
                        f"execution state exceeds the {MAX_EXECUTION_STATE_BYTES:,}-byte limit"
                    )
                record_bytes, summary_bytes, record_count, temporary_bytes = (
                    self._managed_file_sizes()
                )
                if not path.exists() and record_count >= MAX_EXECUTION_RECORDS:
                    raise ValueError(
                        f"execution state contains the maximum of {MAX_EXECUTION_RECORDS:,} records"
                    )
                summary_record = json.loads(encoded.decode("utf-8"))
                summary_record["phases"] = [
                    {
                        key: phase.get(key)
                        for key in (
                            "name", "status", "side_effects", "unsafe_side_effects",
                            "attempts", "error", "process_fence_pending",
                            "blocked_reason_code",
                        )
                    }
                    for phase in summary_record.get("phases", [])
                ]
                summary_encoded = json.dumps(
                    summary_record, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                previous_record_bytes = path.stat().st_size if path.exists() else 0
                previous_summary_bytes = summary_path.stat().st_size if summary_path.exists() else 0
                current_accounted_bytes = record_bytes + summary_bytes + temporary_bytes
                projected_accounted_bytes = (
                    current_accounted_bytes
                    - previous_record_bytes
                    - previous_summary_bytes
                    + len(encoded)
                    + len(summary_encoded)
                )
                reservation_entries = self._reservation_entries()
                own_reservation = sum(
                    entry["reserved_bytes"]
                    for entry in reservation_entries
                    if not entry["invalid"] and entry["execution_id"] == record["execution_id"]
                )
                if own_reservation and not self.is_locked(record["execution_id"]):
                    own_reservation = 0
                other_reserved = self._reserved_checkpoint_bytes_locked(
                    exclude_execution_id=record["execution_id"],
                    clean_stale=True,
                )
                committed_growth = max(0, projected_accounted_bytes - current_accounted_bytes)
                if own_reservation and committed_growth > own_reservation:
                    raise ValueError(
                        "execution checkpoint exceeded its reserved growth; "
                        "the command result was not committed"
                    )
                if (
                    projected_accounted_bytes
                    + other_reserved
                    + (0 if consume_checkpoint_reservation else own_reservation)
                    > self.quota_bytes - self.checkpoint_reserve_bytes
                ):
                    raise ValueError(
                        "execution state aggregate quota would be exceeded; "
                        "retire durable records or increase EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES"
                    )
                for destination, payload in (
                    (path, encoded),
                    (summary_path, summary_encoded),
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
                    except OSError:
                        if destination == summary_path:
                            # The record is already current. Remove the previous
                            # summary so listing and retirement cannot mistake it
                            # for a matching pair after a failed replacement.
                            self._reject_symlink(summary_path, "summary file")
                            try:
                                summary_path.unlink()
                            except FileNotFoundError:
                                pass
                            else:
                                self._fsync_directory(self.state_dir)
                        raise
                    finally:
                        if temporary is not None and temporary.exists():
                            temporary.unlink()
                if consume_checkpoint_reservation and own_reservation:
                    reservation_path = self._reservation_path(record["execution_id"])
                    self._reject_symlink(reservation_path, "checkpoint reservation")
                    try:
                        reservation_path.unlink()
                    except FileNotFoundError:
                        pass
                directory_fd = os.open(
                    self.state_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

    def capacity(self) -> Dict[str, Any]:
        """Return read-only capacity and recovery-state diagnostics."""
        with self._lock:
            exists = self._validate_state_dir(require_exists=False)
            record_paths: Dict[str, Path] = {}
            summary_paths: Dict[str, Path] = {}
            record_bytes = 0
            summary_bytes = 0
            temporary_file_bytes = 0
            other_file_bytes = 0
            other_file_count = 0
            symlink_count = 0
            invalid_record_count = 0
            active_record_count = 0
            fence_pending_record_count = 0
            uncertain_lock_count = 0
            if exists:
                for path in self.state_dir.iterdir():
                    try:
                        info = path.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        symlink_count += 1
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    if path.name.startswith("."):
                        if path.name in {".records.lock"} or path.name.endswith(".lock"):
                            continue
                        if path.name.startswith(".checkpoint-"):
                            continue
                        other_file_count += 1
                        other_file_bytes += info.st_size
                        continue
                    if path.name.startswith("tmp"):
                        temporary_file_bytes += info.st_size
                        continue
                    if path.name.endswith(".summary.json"):
                        summary_paths[path.name.removesuffix(".summary.json") + ".json"] = path
                        summary_bytes += info.st_size
                    elif path.name.endswith(".json"):
                        record_paths[path.name] = path
                        record_bytes += info.st_size
                    else:
                        other_file_count += 1
                        other_file_bytes += info.st_size

            for name, path in record_paths.items():
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        record = json.load(stream)
                    execution_id = record.get("execution_id") if isinstance(record, dict) else None
                    if (
                        not isinstance(execution_id, str)
                        or self._filename(execution_id) != name
                        or not isinstance(record.get("phases"), list)
                        or not all(isinstance(phase, dict) for phase in record["phases"])
                    ):
                        invalid_record_count += 1
                        continue
                    try:
                        if self.is_locked(execution_id):
                            active_record_count += 1
                    except (OSError, ValueError):
                        uncertain_lock_count += 1
                    phases = record["phases"]
                    if record.get("execution_status") == "running" or any(
                        isinstance(phase, dict)
                        and (
                            phase.get("status") == "started"
                            or phase.get("process_fence_pending") is True
                        )
                        for phase in phases
                    ):
                        fence_pending_record_count += 1
                except (OSError, json.JSONDecodeError, UnicodeError, AttributeError):
                    invalid_record_count += 1

            reservations = self._reservation_entries() if exists else []
            active_reserved_bytes = 0
            stale_reserved_bytes = 0
            invalid_reservation_count = 0
            for entry in reservations:
                if entry["invalid"]:
                    invalid_reservation_count += 1
                    continue
                try:
                    active = self.is_locked(entry["execution_id"])
                except (OSError, ValueError):
                    active = None
                    uncertain_lock_count += 1
                if active is True:
                    active_reserved_bytes += entry["reserved_bytes"]
                elif active is False:
                    stale_reserved_bytes += entry["reserved_bytes"]

            main_names = set(record_paths)
            summary_names = set(summary_paths)
            paired_count = len(main_names & summary_names)
            try:
                filesystem_path = self.state_dir
                while not filesystem_path.exists() and filesystem_path != filesystem_path.parent:
                    filesystem_path = filesystem_path.parent
                disk = shutil.disk_usage(filesystem_path)
                filesystem = {
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                }
            except OSError:
                filesystem = {"total_bytes": None, "free_bytes": None}

            record_storage_limit = self.quota_bytes - self.checkpoint_reserve_bytes
            capacity_confident = invalid_reservation_count == 0 and uncertain_lock_count == 0
            available = (
                max(
                    0,
                    record_storage_limit
                    - record_bytes
                    - summary_bytes
                    - temporary_file_bytes
                    - active_reserved_bytes,
                )
                if capacity_confident
                else 0
            )
            return {
                "status": "ok",
                "state_directory": str(self.state_dir),
                "state_directory_exists": exists,
                "quota_bytes": self.quota_bytes,
                "checkpoint_reserve_bytes": self.checkpoint_reserve_bytes,
                "record_storage_limit_bytes": record_storage_limit,
                "record_count": len(record_paths),
                "summary_count": len(summary_paths),
                "paired_record_count": paired_count,
                "unpaired_record_count": len(main_names - summary_names),
                "unpaired_summary_count": len(summary_names - main_names),
                "record_bytes": record_bytes,
                "summary_bytes": summary_bytes,
                "committed_bytes": record_bytes + summary_bytes,
                "temporary_file_bytes": temporary_file_bytes,
                "accounted_bytes": record_bytes + summary_bytes + temporary_file_bytes,
                "active_reserved_bytes": active_reserved_bytes,
                "stale_reserved_bytes": stale_reserved_bytes,
                "available_bytes": available,
                "capacity_confident": capacity_confident,
                "active_record_count": active_record_count,
                "fence_pending_record_count": fence_pending_record_count,
                "invalid_record_count": invalid_record_count,
                "invalid_reservation_count": invalid_reservation_count,
                "uncertain_lock_count": uncertain_lock_count,
                "other_file_count": other_file_count,
                "other_file_bytes": other_file_bytes,
                "symlink_count": symlink_count,
                "filesystem": filesystem,
            }

    def _retirement_snapshot(
        self,
        execution_id: str,
        *,
        ignore_active_lease: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
        record_path = self._path(execution_id)
        summary_path = self._summary_path(execution_id)
        item: Dict[str, Any] = {
            "execution_id": execution_id,
            "eligible": False,
            "record_bytes": 0,
            "summary_bytes": 0,
            "projected_reclaimed_bytes": 0,
        }
        try:
            self._reject_symlink(record_path, "record file")
            self._reject_symlink(summary_path, "summary file")
            record = self._read_record(execution_id)
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            if (
                not isinstance(summary, dict)
                or summary.get("execution_id") != execution_id
                or not isinstance(summary.get("phases"), list)
                or not all(isinstance(phase, dict) for phase in summary["phases"])
            ):
                raise ValueError(f"Execution '{execution_id}' has an invalid summary")
            record_size = record_path.stat(follow_symlinks=False).st_size
            summary_size = summary_path.stat(follow_symlinks=False).st_size
        except (FileNotFoundError, KeyError):
            item["reason"] = "record or matching summary is missing"
            return item, None, None, None
        except (KeyError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            error_type = "ValueError" if isinstance(exc, ValueError) else type(exc).__name__
            item["reason"] = f"record pair is unreadable: {error_type}"
            return item, None, None, None

        status = record["execution_status"]
        label = record["label"]
        updated_at = record["updated_at"]
        record_metadata = {
            "execution_id": execution_id,
            "label": label,
            "execution_status": status,
            "updated_at": updated_at,
        }
        item.update({
            "label": label,
            "execution_status": status,
            "updated_at": updated_at,
            "record_bytes": record_size,
            "summary_bytes": summary_size,
            "projected_reclaimed_bytes": record_size + summary_size,
        })
        if not ignore_active_lease:
            try:
                if self.is_locked(execution_id):
                    item["reason"] = "execution has an active lease"
                    return item, record_metadata, record_path, summary_path
            except (OSError, ValueError) as exc:
                item["reason"] = f"execution lease state is uncertain: {type(exc).__name__}"
                return item, record_metadata, record_path, summary_path
        if record.get("execution_status") == "running" or any(
            phase.get("status") == "started" or phase.get("process_fence_pending") is True
            for phase in record["phases"]
        ):
            item["reason"] = "execution is started or fence-pending"
            return item, record_metadata, record_path, summary_path
        item["eligible"] = True
        return item, record_metadata, record_path, summary_path

    def _validate_archive_path(self, archive_path: str) -> Path:
        if not isinstance(archive_path, str) or not archive_path.strip():
            raise ValueError("archive_path must be a non-empty path")
        if len(archive_path.encode("utf-8")) > 4096:
            raise ValueError("archive_path is too long")
        target = Path(os.path.abspath(os.path.expanduser(archive_path)))
        parent = target.parent
        self._reject_symlink_components(parent)
        self._reject_symlink(parent, "archive directory")
        if not parent.is_dir():
            raise ValueError("archive_path parent directory must already exist")
        parent_stat = parent.stat()
        current_uid = getattr(os, "getuid", lambda: None)()
        if current_uid is not None and parent_stat.st_uid != current_uid:
            raise ValueError("archive_path parent directory must be owned by the current user")
        if stat.S_IMODE(parent_stat.st_mode) & 0o022:
            raise ValueError("archive_path parent directory must not be group/world writable")
        self._reject_symlink(target, "archive file")
        try:
            target.relative_to(self.state_dir)
        except ValueError:
            pass
        else:
            raise ValueError("archive_path must be outside the execution state directory")
        if target.exists():
            raise ValueError("archive_path already exists")
        if not os.access(parent, os.W_OK):
            raise ValueError("archive_path parent directory is not writable")
        return target

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_archive_temp(
        self,
        archive_path: Path,
        snapshots: List[Tuple[Dict[str, Any], Dict[str, Any], Path, Path]],
        *,
        source_digests: Optional[Dict[str, str]] = None,
    ) -> Path:
        source_digests = source_digests if source_digests is not None else {}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=archive_path.parent,
                prefix=".ephemeral-executions-archive-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                manifest = []
                with tarfile.open(fileobj=stream, mode="w") as archive:
                    for item, record_metadata, record_path, summary_path in snapshots:
                        digest = self._filename(record_metadata["execution_id"])
                        for member_name, source_path, size_key in (
                            (f"records/{digest}", record_path, "record_bytes"),
                            (f"records/{digest[:-5]}.summary.json", summary_path, "summary_bytes"),
                        ):
                            size = item[size_key]
                            info = tarfile.TarInfo(member_name)
                            info.size = size
                            info.mode = 0o600
                            info.mtime = 0
                            with source_path.open("rb") as source:
                                digesting_source = _DigestingReader(source)
                                archive.addfile(info, digesting_source)
                                source_digests[member_name] = digesting_source.hexdigest()
                        manifest.append({
                            "execution_id": record_metadata["execution_id"],
                            "label": record_metadata.get("label", ""),
                            "execution_status": record_metadata.get(
                                "execution_status", "unknown"
                            ),
                            "record": f"records/{digest}",
                            "summary": f"records/{digest[:-5]}.summary.json",
                            "source_bytes": item["projected_reclaimed_bytes"],
                        })
                    manifest_bytes = json.dumps(
                        {"schema_version": 1, "executions": manifest},
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    info = tarfile.TarInfo("manifest.json")
                    info.size = len(manifest_bytes)
                    info.mode = 0o600
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(manifest_bytes))
                stream.flush()
                os.fsync(stream.fileno())
            return temporary
        except BaseException:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            raise

    def retire(
        self,
        execution_ids: List[str],
        *,
        archive_path: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Preview or archive and retire explicit, inactive execution records."""
        if not isinstance(execution_ids, list) or not execution_ids:
            raise ValueError("execution_ids must be a non-empty list")
        if len(execution_ids) > MAX_EXECUTION_RETIRE_BATCH:
            raise ValueError(f"execution_ids may contain at most {MAX_EXECUTION_RETIRE_BATCH} items")
        normalized_ids = []
        for execution_id in execution_ids:
            normalized_ids.append(
                _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
            )
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("execution_ids must not contain duplicates")
        if not isinstance(dry_run, bool):
            raise ValueError("dry_run must be a boolean")

        initial = [self._retirement_snapshot(execution_id) for execution_id in normalized_ids]
        initial_items = [entry[0] for entry in initial]
        projected = sum(
            item["projected_reclaimed_bytes"] for item in initial_items if item["eligible"]
        )
        if dry_run:
            return {
                "status": "ok",
                "dry_run": True,
                "archive_required_for_retirement": True,
                "items": initial_items,
                "eligible_count": sum(item["eligible"] for item in initial_items),
                "blocked_count": sum(not item["eligible"] for item in initial_items),
                "projected_reclaimed_bytes": projected,
            }
        if archive_path is None:
            raise ValueError("archive_path is required when dry_run is false")
        archive_target = self._validate_archive_path(archive_path)
        if any(not item["eligible"] for item in initial_items):
            return {
                "status": "refused",
                "dry_run": False,
                "reason": "one or more selected executions are not eligible",
                "items": initial_items,
                "projected_reclaimed_bytes": projected,
            }

        snapshots: List[Tuple[Dict[str, Any], Dict[str, Any], Path, Path]] = []
        temporary_archive = None
        with ExitStack() as leases:
            try:
                for execution_id in sorted(normalized_ids):
                    leases.enter_context(self.lease(execution_id))
            except ExecutionBusyError:
                leases.close()
                refreshed = [self._retirement_snapshot(item) for item in normalized_ids]
                return {
                    "status": "refused",
                    "dry_run": False,
                    "reason": "one or more selected executions became active",
                    "items": [entry[0] for entry in refreshed],
                    "projected_reclaimed_bytes": sum(
                        entry[0]["projected_reclaimed_bytes"]
                        for entry in refreshed
                        if entry[0]["eligible"]
                    ),
                }
            try:
                refreshed = [
                    self._retirement_snapshot(execution_id, ignore_active_lease=True)
                    for execution_id in normalized_ids
                ]
                blocked_items = [entry[0] for entry in refreshed if not entry[0]["eligible"]]
                if blocked_items:
                    return {
                        "status": "refused",
                        "dry_run": False,
                        "reason": "one or more selected executions are no longer eligible",
                        "items": [entry[0] for entry in refreshed],
                        "projected_reclaimed_bytes": sum(
                            entry[0]["projected_reclaimed_bytes"]
                            for entry in refreshed
                            if entry[0]["eligible"]
                        ),
                    }
                for item, record, record_path, summary_path in refreshed:
                    snapshots.append((item, record, record_path, summary_path))

                archive_digests: Dict[str, str] = {}
                temporary_archive = self._write_archive_temp(
                    archive_target,
                    snapshots,
                    source_digests=archive_digests,
                )
                with self._record_lease():
                    # The per-record leases keep these pairs stable. Confirm
                    # their contents under the writer lock without reacquiring the
                    # store's thread lock in the opposite order from save().
                    for item, _, record_path, summary_path in snapshots:
                        self._reject_symlink(record_path, "record file")
                        self._reject_symlink(summary_path, "summary file")
                        record_member = f"records/{record_path.name}"
                        summary_member = f"records/{summary_path.name}"
                        if (
                            record_path.stat(follow_symlinks=False).st_size != item["record_bytes"]
                            or summary_path.stat(follow_symlinks=False).st_size != item["summary_bytes"]
                            or archive_digests.get(record_member) != self._file_digest(record_path)
                            or archive_digests.get(summary_member) != self._file_digest(summary_path)
                        ):
                            raise ValueError("selected execution state changed during retirement")
                        reservation = self._reservation_path(item["execution_id"])
                        self._reject_symlink(reservation, "checkpoint reservation")
                        try:
                            reservation_info = reservation.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            pass
                        else:
                            if not stat.S_ISREG(reservation_info.st_mode):
                                raise ValueError(
                                    "checkpoint reservation is not a regular file"
                                )
                    os.link(temporary_archive, archive_target)
                    temporary_archive.unlink()
                    temporary_archive = None
                    self._fsync_directory(archive_target.parent)
                    retired = []
                    removal_errors = []
                    for item, _, record_path, summary_path in snapshots:
                        try:
                            summary_path.unlink()
                            record_path.unlink()
                        except (OSError, ValueError) as exc:
                            removal_errors.append({
                                "execution_id": item["execution_id"],
                                "error": type(exc).__name__,
                            })
                            continue
                        retired.append(item)
                        reservation = self._reservation_path(item["execution_id"])
                        try:
                            self._reject_symlink(reservation, "checkpoint reservation")
                            try:
                                reservation.unlink()
                            except FileNotFoundError:
                                pass
                        except (OSError, ValueError) as exc:
                            removal_errors.append({
                                "execution_id": item["execution_id"],
                                "error": type(exc).__name__,
                            })
                    try:
                        self._fsync_directory(self.state_dir)
                    except OSError as exc:
                        removal_errors.append({"execution_id": None, "error": type(exc).__name__})
                return {
                    "status": "partial" if removal_errors else "ok",
                    "dry_run": False,
                    "archive_path": str(archive_target),
                    "archive_bytes": archive_target.stat().st_size,
                    "retired_count": len(retired),
                    "retired": retired,
                    "removal_errors": removal_errors,
                    "archive_is_durable": True,
                    "projected_reclaimed_bytes": sum(
                        item["projected_reclaimed_bytes"] for item in retired
                    ),
                }
            finally:
                if temporary_archive is not None:
                    try:
                        temporary_archive.unlink()
                    except FileNotFoundError:
                        pass

    def _read_record(self, execution_id: str) -> Dict[str, Any]:
        with self._lock:
            self._validate_state_dir(require_exists=False)
            path = self._path(execution_id)
            self._reject_symlink(path, "record file")
            try:
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ExecutionRecordError(
                        f"Execution '{execution_id}' has invalid state: record is not a regular file"
                    )
                if info.st_size > MAX_EXECUTION_STATE_BYTES:
                    raise ExecutionRecordError(
                        f"Execution '{execution_id}' has invalid state: record exceeds the size limit"
                    )
                with path.open("rb") as stream:
                    raw = stream.read(MAX_EXECUTION_STATE_BYTES + 1)
                if len(raw) > MAX_EXECUTION_STATE_BYTES:
                    raise ExecutionRecordError(
                        f"Execution '{execution_id}' has invalid state: record exceeds the size limit"
                    )
                record = json.loads(raw.decode("utf-8"))
            except FileNotFoundError as exc:
                raise KeyError(f"Execution '{execution_id}' was not found") from exc
            except ExecutionRecordError:
                raise
            except (OSError, UnicodeError, ValueError, RecursionError) as exc:
                raise ExecutionRecordError(
                    f"Execution '{execution_id}' has unreadable state",
                    reason_code="MALFORMED_RECORD",
                ) from exc
            return _validate_execution_record(record, execution_id)

    def inspect_record(
        self,
        execution_id: str,
        error: ExecutionRecordError,
    ) -> Dict[str, Any]:
        """Return a bounded sanitized preview without recovery or writes."""
        with self._lock:
            self._validate_state_dir(require_exists=False)
            path = self._path(execution_id)
            self._reject_symlink(path, "record file")
            try:
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("record is not a regular file")
                with path.open("rb") as stream:
                    raw_preview = stream.read(MAX_EXECUTION_RECORD_PREVIEW_BYTES + 1)
            except FileNotFoundError as exc:
                raise KeyError(f"Execution '{execution_id}' was not found") from exc
            except OSError as exc:
                raise ValueError(f"Execution '{execution_id}' has unreadable state") from exc
            truncated = len(raw_preview) > MAX_EXECUTION_RECORD_PREVIEW_BYTES
            raw_preview = raw_preview[:MAX_EXECUTION_RECORD_PREVIEW_BYTES]
            preview_text = "[preview omitted because it could not be safely sanitized]"
            preview_redacted = False
            preview_omitted = True
            if not truncated and info.st_size == len(raw_preview):
                try:
                    parsed_preview = json.loads(raw_preview.decode("utf-8"))

                    def allowlisted_phase(value: Any) -> Optional[Dict[str, Any]]:
                        nonlocal preview_redacted
                        if not isinstance(value, dict):
                            preview_redacted = True
                            return None
                        safe_phase: Dict[str, Any] = {}
                        status = value.get("status")
                        if status in PHASE_STATUSES:
                            safe_phase["status"] = status
                        elif "status" in value:
                            preview_redacted = True
                        attempts = value.get("attempts")
                        if (
                            not isinstance(attempts, bool)
                            and isinstance(attempts, int)
                            and 0 <= attempts <= MAX_PHASE_ATTEMPTS
                        ):
                            safe_phase["attempts"] = attempts
                        elif "attempts" in value:
                            preview_redacted = True
                        side_effects = value.get("side_effects")
                        if side_effects in ("none", "unsafe"):
                            safe_phase["side_effects"] = side_effects
                        elif "side_effects" in value:
                            preview_redacted = True
                        for key in (
                            "unsafe_side_effects", "process_fence_pending",
                        ):
                            if isinstance(value.get(key), bool):
                                safe_phase[key] = value[key]
                            elif key in value:
                                preview_redacted = True
                        if set(value) - {
                            "status", "attempts", "side_effects",
                            "unsafe_side_effects", "process_fence_pending",
                        }:
                            preview_redacted = True
                        return safe_phase

                    safe_preview: Dict[str, Any] = {}
                    if isinstance(parsed_preview, dict):
                        version = parsed_preview.get("schema_version")
                        if type(version) is int:
                            safe_preview["schema_version"] = version
                        elif "schema_version" in parsed_preview:
                            preview_redacted = True
                        status = parsed_preview.get("execution_status")
                        if status in (
                            "pending", "running", "completed", "partial", "interrupted"
                        ):
                            safe_preview["execution_status"] = status
                        elif "execution_status" in parsed_preview:
                            preview_redacted = True
                        partial = parsed_preview.get("partial")
                        if isinstance(partial, bool):
                            safe_preview["partial"] = partial
                        elif "partial" in parsed_preview:
                            preview_redacted = True
                        phases = parsed_preview.get("phases")
                        if isinstance(phases, list):
                            safe_preview["phases"] = [
                                phase_preview
                                for phase in phases
                                if (phase_preview := allowlisted_phase(phase)) is not None
                            ]
                        elif "phases" in parsed_preview:
                            preview_redacted = True
                        if set(parsed_preview) - {
                            "schema_version", "execution_status", "partial", "phases",
                        }:
                            preview_redacted = True
                    else:
                        preview_redacted = True

                    if safe_preview:
                        sanitized = json.dumps(
                            safe_preview,
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                        if len(sanitized) <= MAX_EXECUTION_RECORD_PREVIEW_BYTES:
                            preview_text = sanitized.decode("utf-8")
                            preview_omitted = False
                except (UnicodeError, ValueError, RecursionError):
                    pass
            schema_version = error.schema_version
            if (
                not isinstance(schema_version, (str, int, float, bool))
                or isinstance(schema_version, float) and not math.isfinite(schema_version)
            ):
                schema_version = None
            if isinstance(schema_version, str):
                schema_version = schema_version.encode("utf-8", errors="replace")[:64].decode(
                    "utf-8", errors="ignore"
                )
            return {
                "status": (
                    "unsupported_record"
                    if error.reason_code == "UNSUPPORTED_SCHEMA_VERSION"
                    else "invalid_record"
                ),
                "execution_id": execution_id,
                "schema_version": schema_version,
                "read_only": True,
                "reason_code": error.reason_code,
                "message": _bounded_error(error),
                "record_bytes": info.st_size,
                "record_preview": preview_text,
                "record_preview_sanitized": True,
                "record_preview_redacted": preview_redacted,
                "record_preview_truncated": (
                    truncated
                    or info.st_size > len(raw_preview)
                    or preview_omitted
                ),
            }

    def load(self, execution_id: str, recover: bool = True) -> Dict[str, Any]:
        """Load and validate a record, optionally recovering a started phase."""
        record = self._read_record(execution_id)
        if not recover or not any(
            phase.get("status") == "started"
            or phase.get("process_fence_pending") is True
            for phase in record.get("phases", [])
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
        recovery_candidate = False
        for phase in record.get("phases", []):
            status = phase.get("status")
            if status != "started" and not phase.get("process_fence_pending"):
                continue
            recovery_candidate = True
            prior_state = (
                phase.get("status"),
                phase.get("error"),
                phase.get("process_fence_pending"),
                phase.get("blocked_reason_code"),
            )
            already_pending = (
                status == "interrupted"
                and phase.get("process_fence_pending") is True
            )
            previous_reason_code = phase.get("blocked_reason_code")
            outcome = _terminate_stale_process_outcome(phase)
            blocked_reason_code = outcome.blocked_reason_code
            if (
                not outcome.confirmed
                and previous_reason_code in {
                    "CLEANUP_HOOK_UNAVAILABLE",
                    "CLEANUP_HOOK_FAILED",
                }
            ):
                # The runner's cleanup hook is the actionable blocker for an
                # injected runner; generic process inspection cannot replace it.
                blocked_reason_code = previous_reason_code
            phase["process_fence_pending"] = not outcome.confirmed
            phase["blocked_reason_code"] = blocked_reason_code
            if status == "started" or status == "interrupted":
                phase["status"] = "interrupted"
                phase["error"] = (
                    "process termination is pending before the phase can resume"
                    if not outcome.confirmed
                    else "process terminated before the phase completed"
                )
                if not already_pending or previous_reason_code != blocked_reason_code:
                    _phase_event(
                        phase,
                        "interrupted",
                        reason="process restart recovery",
                        blocked_reason_code=blocked_reason_code,
                    )
                    changed = True
            elif previous_reason_code != blocked_reason_code:
                if not outcome.confirmed and not phase.get("error"):
                    phase["error"] = "process termination is pending before the phase can resume"
                elif outcome.confirmed and phase.get("error") == (
                    "process termination is pending before the phase can resume"
                ):
                    phase["error"] = None
                _phase_event(
                    phase,
                    "blocked" if not outcome.confirmed else "recovered",
                    reason="process restart recovery",
                    blocked_reason_code=blocked_reason_code,
                )
                changed = True
            current_state = (
                phase.get("status"),
                phase.get("error"),
                phase.get("process_fence_pending"),
                phase.get("blocked_reason_code"),
            )
            changed = changed or current_state != prior_state
        if recovery_candidate:
            previous_overall_state = (
                record.get("execution_status"),
                record.get("partial"),
            )
            PhaseExecutionManager._refresh_overall_status(record)
            current_overall_state = (
                record.get("execution_status"),
                record.get("partial"),
            )
            changed = changed or current_overall_state != previous_overall_state
        return changed

    def list(
        self,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Any]:
        """List compact records, tolerating files removed during discovery."""
        with self._lock:
            if not self._validate_state_dir(require_exists=False):
                return []
            sort_by_name_only = False

            def is_not_symlink(path: Path) -> bool:
                nonlocal sort_by_name_only
                try:
                    return not path.is_symlink()
                except OSError:
                    sort_by_name_only = True
                    return False

            main_paths = {
                path.name: path
                for path in self.state_dir.glob("*.json")
                if (
                    not path.name.endswith(".summary.json")
                    and self._is_record_digest(path.name.removesuffix(".json"))
                    and is_not_symlink(path)
                )
            }
            summary_paths = {
                path.name.removesuffix(".summary.json") + ".json": path
                for path in self.state_dir.glob("*.summary.json")
                if (
                    self._is_record_digest(
                        path.name.removesuffix(".summary.json")
                    )
                    and is_not_symlink(path)
                )
            }
            entries: List[Tuple[str, str, Any]] = []
            for name in sorted(set(main_paths) | set(summary_paths)):
                main_path = main_paths.get(name)
                summary_path = summary_paths.get(name)
                if summary_path is None:
                    assert main_path is not None
                    candidates = [main_path]
                elif main_path is None:
                    candidates = [summary_path]
                else:
                    try:
                        summary_is_newer = (
                            summary_path.stat().st_mtime_ns
                            >= main_path.stat().st_mtime_ns
                        )
                    except OSError:
                        # The full record is authoritative when file metadata
                        # disappears during discovery. Name order is stable if
                        # timestamps cannot reliably order the discovered set.
                        sort_by_name_only = True
                        summary_is_newer = False
                    candidates = (
                        [summary_path, main_path]
                        if summary_is_newer
                        else [main_path, summary_path]
                    )

                discovered = None
                for candidate in candidates:
                    try:
                        with candidate.open("r", encoding="utf-8") as stream:
                            parsed = json.load(stream)
                    except (OSError, UnicodeError, ValueError, RecursionError):
                        continue
                    if (
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("execution_id"), str)
                        and self._matches_record_filename(
                            parsed["execution_id"], name
                        )
                    ):
                        discovered = parsed
                        break

                if discovered is not None:
                    execution_id = discovered["execution_id"]
                    updated_at = discovered.get("updated_at")
                    entries.append((
                        updated_at
                        if isinstance(updated_at, str)
                        else self._listing_file_sort_time(
                            main_path or summary_path
                        ),
                        name,
                        {"execution_id": execution_id},
                    ))
                    continue

                # The full record is the authoritative source. If it cannot be
                # parsed, use a matching summary only to recover its ID for the
                # sanitized read-only diagnostic.
                if main_path is None:
                    continue
                execution_id = None
                updated_at = self._listing_file_sort_time(main_path)
                if summary_path is not None:
                    try:
                        with summary_path.open("r", encoding="utf-8") as stream:
                            summary = json.load(stream)
                        candidate_id = (
                            summary.get("execution_id")
                            if isinstance(summary, dict)
                            else None
                        )
                        if (
                            isinstance(candidate_id, str)
                            and self._matches_record_filename(candidate_id, name)
                        ):
                            execution_id = candidate_id
                            candidate_updated_at = summary.get("updated_at")
                            if isinstance(candidate_updated_at, str):
                                updated_at = candidate_updated_at
                    except (OSError, UnicodeError, ValueError, RecursionError):
                        pass
                if execution_id is not None:
                    error = ExecutionRecordError(
                        f"Execution '{execution_id}' has unreadable state"
                    )
                    try:
                        diagnostic = self.inspect_record(execution_id, error)
                    except (KeyError, ValueError, OSError, json.JSONDecodeError):
                        diagnostic = self._listing_file_diagnostic(
                            main_path,
                            execution_id=execution_id,
                        )
                else:
                    diagnostic = self._listing_file_diagnostic(main_path)
                entries.append((updated_at, name, ExecutionListDiagnostic(diagnostic)))

            if sort_by_name_only:
                entries.sort(key=lambda item: item[1])
            else:
                entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if limit is not None:
                entries = entries[offset:offset + limit]
            elif offset:
                entries = entries[offset:]
            selected = []
            for _updated_at, _name, summary in entries:
                if isinstance(summary, ExecutionListDiagnostic):
                    selected.append(summary)
                    continue
                try:
                    execution_id = summary["execution_id"]
                    record = self._read_record(execution_id)
                    if any(
                        phase.get("status") == "started"
                        or phase.get("process_fence_pending") is True
                        for phase in record.get("phases", [])
                    ):
                        record = self.load(execution_id)
                    selected.append(self._compact_listing_record(record))
                except ExecutionRecordError as exc:
                    try:
                        selected.append(
                            ExecutionListDiagnostic(
                                self.inspect_record(execution_id, exc)
                            )
                        )
                    except (KeyError, ValueError, OSError, json.JSONDecodeError):
                        continue
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
        quota_bytes: Optional[int] = None,
        checkpoint_reserve_bytes: Optional[int] = None,
        command_runner: Optional[CommandRunner] = None,
        process_cleanup: Optional[ProcessCleanup] = None,
    ):
        if max_output_bytes < 512:
            raise ValueError("max_output_bytes must be at least 512")
        self.store = ExecutionStore(
            state_dir,
            quota_bytes=quota_bytes,
            checkpoint_reserve_bytes=checkpoint_reserve_bytes,
        )
        self.max_output_bytes = max_output_bytes
        self.command_runner = command_runner or run_command_bounded
        self.process_cleanup = process_cleanup
        self._lock = threading.RLock()
        self._active_lock = threading.RLock()
        self._active_cancellations: Dict[str, threading.Event] = {}
        self._background_futures = {}
        self._background_failures: Dict[str, str] = {}
        self._background_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="durable-execution",
        )
        self._closing = False

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
            "process_launch_token": None,
            "process_containment": None,
            "process_fence_pending": False,
            "blocked_reason_code": None,
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
        phase_output_limit = min(
            default_output,
            _max_phase_output_bytes(len(phases)),
        )
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
                (
                    "name", "status", "side_effects", "unsafe_side_effects",
                    "attempts", "error", "blocked_reason_code",
                )
                if compact
                else (
                    "name", "status", "cwd", "timeout_seconds", "max_output_bytes",
                    "side_effects", "unsafe_side_effects", "idempotency_key",
                    "structured_metrics", "attempts", "events", "attempt_results",
                    "result", "error", "blocked_reason_code",
                )
            )
            item = {
                key: phase.get(key)
                for key in keys
            }
            blocked_code = (
                phase.get("blocked_reason_code")
                if phase.get("process_fence_pending") is True
                else None
            )
            item["blocked_reason_code"] = blocked_code
            item["blocked_reason"] = _blocked_reason(blocked_code)
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
            "execution_in_progress": execution_in_progress,
            "partial": record["partial"],
            "summary": summary,
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            **(
                {"background_error": record["background_error"]}
                if record.get("background_error") is not None
                else {}
            ),
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
                "blocked_reason": (
                    _blocked_reason(next_record.get("blocked_reason_code"))
                    if next_record and next_record.get("process_fence_pending") is True
                    else None
                ),
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
        """Validate and persist a new execution without running its phases."""
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
        *,
        reserve_first_phase: bool = False,
    ):
        """Create a record and retain its lease through the caller's work."""
        self._ensure_runner_capabilities()
        record = self._new_record(
            phases, execution_id, label, resume_policy, cwd,
            timeout_seconds, max_output_bytes,
        )
        with self.store.lease(record["execution_id"]):
            try:
                self.store.load(record["execution_id"], recover=False)
            except KeyError:
                if reserve_first_phase:
                    self.store.reserve_checkpoint(
                        record["execution_id"],
                        self._checkpoint_reservation_size(record["phases"][0]),
                    )
                try:
                    self.store.save(record)
                except Exception:
                    if reserve_first_phase:
                        self.store.release_checkpoint(record["execution_id"])
                    raise
                yield record
                return
            raise ValueError(f"Execution '{record['execution_id']}' already exists")

    def _ensure_runner_capabilities(self) -> None:
        if self.command_runner is run_command_bounded and not _process_recovery_supported():
            raise RuntimeError(
                "built-in durable execution requires Linux /proc, pidfd, and selector support"
            )

    @staticmethod
    def _first_incomplete(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return next((phase for phase in record["phases"] if phase["status"] != "completed"), None)

    @staticmethod
    def _checkpoint_reservation_size(phase: Dict[str, Any]) -> int:
        return min(
            MAX_EXECUTION_STATE_BYTES,
            phase["max_output_bytes"] * JSON_OUTPUT_EXPANSION_BOUND
            + EXECUTION_METADATA_RESERVE_BYTES,
        )

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
        phase["process_fence_pending"] = True
        record["updated_at"] = _now()
        self.store.save(record)
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
        phase["process_launch_token"] = None
        phase["process_containment"] = None
        phase["process_fence_pending"] = False
        phase["blocked_reason_code"] = None

    def _cleanup_injected_runner(self) -> CleanupOutcome:
        """Fence an injected runner, whose child processes are opaque here."""
        if self.process_cleanup is None:
            # An injected runner may have spawned descendants that the manager
            # cannot discover.  Block resume rather than repeating side
            # effects until the runner supplies a successful cleanup hook.
            return CleanupOutcome(False, "CLEANUP_HOOK_UNAVAILABLE")
        try:
            self.process_cleanup()
        except BaseException:
            return CleanupOutcome(False, "CLEANUP_HOOK_FAILED")
        return CleanupOutcome(True)

    def _fence_interrupted_runner(
        self,
        cleanup_confirmed: Optional[bool] = None,
    ) -> CleanupOutcome:
        """Fence an interrupted runner before allowing a retry."""
        if self.command_runner is run_command_bounded:
            # The built-in runner owns its process group and cleans it up
            # before propagating an interruption.
            return (
                CleanupOutcome(True)
                if cleanup_confirmed is True
                else CleanupOutcome(False, "PROCESS_CLEANUP_UNCONFIRMED")
            )
        return self._cleanup_injected_runner()

    def _run(
        self,
        record: Dict[str, Any],
        *,
        retry_failed: bool,
        confirm_unsafe: bool,
        output_handler: Optional[OutputHandler],
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        while True:
            phase = self._first_incomplete(record)
            if phase is None:
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                self.store.save(record)
                return record
            self._ensure_runner_capabilities()
            if phase.get("process_fence_pending"):
                if self.command_runner is not run_command_bounded:
                    cleanup_outcome = self._cleanup_injected_runner()
                else:
                    cleanup_outcome = CleanupOutcome(
                        False,
                        phase.get("blocked_reason_code") or "PROCESS_STATE_UNVERIFIABLE",
                    )
                if cleanup_outcome.confirmed:
                    self._clear_process_identity(phase)
                    record["updated_at"] = _now()
                else:
                    previous_reason_code = phase.get("blocked_reason_code")
                    phase["blocked_reason_code"] = (
                        cleanup_outcome.blocked_reason_code
                        or "PROCESS_STATE_UNVERIFIABLE"
                    )
                    phase["error"] = "process termination is pending before the phase can resume"
                    if previous_reason_code != phase["blocked_reason_code"]:
                        _phase_event(
                            phase,
                            "blocked",
                            reason="runner cleanup",
                            blocked_reason_code=phase["blocked_reason_code"],
                        )
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

            if cancellation_event is not None and cancellation_event.is_set():
                if phase["status"] not in {"failed", "timed_out"}:
                    phase["status"] = "interrupted"
                    phase["error"] = "execution cancelled before the phase started"
                _phase_event(phase, "interrupted", reason="requested cancellation")
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                try:
                    self.store.save(record)
                finally:
                    self.store.release_checkpoint(record["execution_id"])
                return record

            checkpoint_reservation = self._checkpoint_reservation_size(phase)
            self.store.reserve_checkpoint(record["execution_id"], checkpoint_reservation)
            phase["attempts"] += 1
            phase["status"] = "started"
            phase["error"] = None
            phase["blocked_reason_code"] = None
            phase["output"] = ""
            phase["result"] = None
            phase["process_id"] = None
            phase["process_group_id"] = None
            phase["process_start_time"] = None
            phase["process_boot_id"] = None
            phase["process_launch_token"] = (
                uuid.uuid4().hex if self.command_runner is run_command_bounded else None
            )
            phase["process_containment"] = (
                PROCESS_CONTAINMENT_SUBREAPER if self.command_runner is run_command_bounded else None
            )
            phase["process_fence_pending"] = True
            _phase_event(phase, "started")
            self._refresh_overall_status(record)
            record["updated_at"] = _now()
            try:
                self.store.save(record)
            except Exception:
                self.store.release_checkpoint(record["execution_id"])
                raise
            started = time.perf_counter()
            cleanup_outcome = CleanupOutcome(True)
            try:
                if self.command_runner is run_command_bounded:
                    cancellation_options = (
                        {"cancellation_event": cancellation_event}
                        if cancellation_event is not None
                        else {}
                    )
                    command_result = run_command_bounded(
                        phase["command"],
                        phase["cwd"],
                        phase["max_output_bytes"],
                        phase["timeout_seconds"],
                        process_started=lambda process_id, process_group_id: self._record_process_identity(
                            record, phase, process_id, process_group_id
                        ),
                        process_marker=phase["process_launch_token"],
                        **cancellation_options,
                    )
                else:
                    command_result = self.command_runner(
                        phase["command"],
                        phase["cwd"],
                        phase["max_output_bytes"],
                        phase["timeout_seconds"],
                    )
                output, exit_code, truncated, original_byte_size, timed_out = command_result
                command_cancelled = bool(getattr(command_result, "cancelled", False))
                runner_cleanup_confirmed = getattr(command_result, "cleanup_confirmed", True)
                if timed_out and self.command_runner is not run_command_bounded:
                    cleanup_outcome = self._cleanup_injected_runner()
                else:
                    cleanup_outcome = (
                        CleanupOutcome(True)
                        if runner_cleanup_confirmed is True
                        else CleanupOutcome(False, "PROCESS_CLEANUP_UNCONFIRMED")
                    )
            except (KeyboardInterrupt, SystemExit) as exc:
                runner_cleanup_confirmed = getattr(exc, "cleanup_confirmed", None)
                cleanup_outcome = self._fence_interrupted_runner(
                    runner_cleanup_confirmed
                )
                if cleanup_outcome.confirmed:
                    self._clear_process_identity(phase)
                else:
                    phase["process_fence_pending"] = True
                    phase["blocked_reason_code"] = cleanup_outcome.blocked_reason_code
                phase["status"] = "interrupted"
                phase["error"] = "phase interrupted before a result was available"
                _phase_event(
                    phase,
                    "interrupted",
                    reason="runner interruption",
                    blocked_reason_code=phase.get("blocked_reason_code"),
                )
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                try:
                    self.store.save(record, consume_checkpoint_reservation=True)
                finally:
                    self.store.release_checkpoint(record["execution_id"])
                raise
            except Exception as exc:
                if self.command_runner is run_command_bounded:
                    runner_cleanup_confirmed = getattr(exc, "cleanup_confirmed", True)
                    cleanup_outcome = (
                        CleanupOutcome(True)
                        if runner_cleanup_confirmed is True
                        else CleanupOutcome(False, "PROCESS_CLEANUP_UNCONFIRMED")
                    )
                else:
                    cleanup_outcome = self._cleanup_injected_runner()
                if cleanup_outcome.confirmed:
                    self._clear_process_identity(phase)
                else:
                    phase["process_fence_pending"] = True
                    phase["blocked_reason_code"] = cleanup_outcome.blocked_reason_code
                phase["status"] = "failed"
                phase["error"] = _bounded_error(exc)
                phase["result"] = {
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                }
                phase["attempt_results"].append(self._attempt_snapshot(phase["result"]))
                _phase_event(
                    phase,
                    "failed",
                    error_type=type(exc).__name__,
                    blocked_reason_code=phase.get("blocked_reason_code"),
                )
                self._refresh_overall_status(record)
                record["updated_at"] = _now()
                try:
                    self.store.save(record, consume_checkpoint_reservation=True)
                finally:
                    self.store.release_checkpoint(record["execution_id"])
                return record

            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if cleanup_outcome.confirmed:
                self._clear_process_identity(phase)
            else:
                phase["process_fence_pending"] = True
                phase["blocked_reason_code"] = cleanup_outcome.blocked_reason_code
            phase["output"] = output
            phase["result"] = {
                "duration_ms": duration_ms,
                "exit_code": exit_code,
                "timed_out": bool(timed_out),
                "cancelled": command_cancelled,
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
            if command_cancelled:
                phase["status"] = "interrupted"
                phase["error"] = "phase cancelled by request"
                event_status = "interrupted"
            elif timed_out:
                phase["status"] = "timed_out"
                event_status = "timed_out"
            elif exit_code == 0:
                phase["status"] = "completed"
                event_status = "completed"
            else:
                phase["status"] = "failed"
                event_status = "failed"
            phase["attempt_results"].append(self._attempt_snapshot(phase["result"]))
            _phase_event(
                phase,
                event_status,
                exit_code=exit_code,
                **({"reason": "requested cancellation"} if command_cancelled else {}),
                blocked_reason_code=phase.get("blocked_reason_code"),
            )
            self._refresh_overall_status(record)
            record["updated_at"] = _now()
            try:
                self.store.save(record, consume_checkpoint_reservation=True)
            finally:
                self.store.release_checkpoint(record["execution_id"])
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
        """Create an execution and run phases until they complete or block."""
        with self._create_and_lease(
            phases, execution_id, label, resume_policy, cwd,
            timeout_seconds, max_output_bytes, reserve_first_phase=True,
        ) as record:
            record = self.store.load(record["execution_id"], recover=False)
            completed = self._run(
                record,
                retry_failed=False,
                confirm_unsafe=False,
                output_handler=output_handler,
            )
            return self._public(completed)

    def _submit_background(
        self,
        execution_id: str,
        *,
        resume: bool,
        options: Dict[str, Any],
    ) -> str:
        """Schedule one durable execution and return its ID promptly."""
        execution_id = _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
        with self._active_lock:
            if self._closing:
                raise RuntimeError("durable execution manager is shutting down")
            if execution_id in self._active_cancellations:
                raise ExecutionBusyError(f"Execution '{execution_id}' is already active")
            if len(self._background_futures) >= MAX_BACKGROUND_EXECUTIONS:
                raise ExecutionBusyError(
                    f"durable execution capacity is full ({MAX_BACKGROUND_EXECUTIONS} active or queued)"
                )
            self._background_failures.pop(execution_id, None)
            cancellation_event = threading.Event()
            self._active_cancellations[execution_id] = cancellation_event

            def run_background():
                try:
                    with self.store.lease(execution_id):
                        record = self.store.load(execution_id, recover=False)
                        if record.pop("background_error", None) is not None:
                            record["updated_at"] = _now()
                            self.store.save(record)
                        if resume and self.store._recover_started(record):
                            record["updated_at"] = _now()
                            self.store.save(record)
                        completed = self._run(
                            record,
                            retry_failed=options.get("retry_failed", False),
                            confirm_unsafe=options.get("confirm_unsafe", False),
                            output_handler=options.get("output_handler"),
                            cancellation_event=cancellation_event,
                        )
                        return self._public(completed)
                except BaseException as exc:
                    self._record_background_failure(execution_id, exc)
                    raise

            try:
                future = self._background_executor.submit(run_background)
            except BaseException:
                if self._active_cancellations.get(execution_id) is cancellation_event:
                    self._active_cancellations.pop(execution_id, None)
                raise
            self._background_futures[execution_id] = future
            future.add_done_callback(
                lambda done, active_id=execution_id, active_future=future,
                active_cancellation=cancellation_event: self._finish_background(
                    active_id, active_future, active_cancellation
                )
            )
        return execution_id

    def _record_background_failure(self, execution_id: str, exc: BaseException) -> None:
        message = _bounded_error(f"{type(exc).__name__}: {exc}")
        with self._active_lock:
            self._background_failures[execution_id] = message
        try:
            with self.store.lease(execution_id):
                record = self.store.load(execution_id, recover=False)
                record["background_error"] = message
                if record.get("execution_status") in {"pending", "running"}:
                    record["execution_status"] = "partial"
                    record["partial"] = True
                record["updated_at"] = _now()
                self.store.save(record)
        except BaseException:
            # Preserve the in-memory diagnostic when the state store itself is
            # unavailable; the done callback still consumes the future error.
            return

    def _finish_background(
        self,
        execution_id: str,
        future,
        cancellation_event: threading.Event,
    ) -> None:
        if future.cancelled():
            try:
                self._record_cancelled_before_start(execution_id)
            except BaseException as exc:
                with self._active_lock:
                    self._background_failures[execution_id] = _bounded_error(
                        "CancelledError: execution was cancelled before its worker started; "
                        f"the interrupted state could not be saved: {type(exc).__name__}: {exc}"
                    )
        else:
            error = future.exception()
            if error is not None:
                with self._active_lock:
                    recorded = execution_id in self._background_failures
                if not recorded:
                    self._record_background_failure(execution_id, error)
        with self._active_lock:
            if self._background_futures.get(execution_id) is future:
                self._background_futures.pop(execution_id, None)
            if self._active_cancellations.get(execution_id) is cancellation_event:
                self._active_cancellations.pop(execution_id, None)

    def _record_cancelled_before_start(self, execution_id: str) -> None:
        """Persist shutdown cancellation for work whose queued future never ran."""
        with self.store.lease(execution_id):
            try:
                record = self.store.load(execution_id, recover=False)
                phase = self._first_incomplete(record)
                if phase is not None and phase["status"] == "pending":
                    phase["status"] = "interrupted"
                    phase["error"] = "execution cancelled before the phase started"
                    _phase_event(phase, "interrupted", reason="requested cancellation")
                    self._refresh_overall_status(record)
                    record["updated_at"] = _now()
                    self.store.save(record)
            finally:
                self.store.release_checkpoint(execution_id)

    def start_background(
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
        """Persist a new execution, schedule it, and return its ID before phases finish."""
        with self._active_lock:
            if self._closing:
                raise RuntimeError("durable execution manager is shutting down")
            if len(self._background_futures) >= MAX_BACKGROUND_EXECUTIONS:
                raise ExecutionBusyError(
                    f"durable execution capacity is full ({MAX_BACKGROUND_EXECUTIONS} active or queued)"
                )
            with self._create_and_lease(
                phases, execution_id, label, resume_policy, cwd,
                timeout_seconds, max_output_bytes, reserve_first_phase=True,
            ) as record:
                normalized_id = record["execution_id"]
            normalized_id = self._submit_background(
                normalized_id,
                resume=False,
                options={"output_handler": output_handler},
            )
        return self.public(normalized_id)

    def resume_background(
        self,
        execution_id: str,
        *,
        retry_failed: bool = False,
        confirm_unsafe: bool = False,
        output_handler: Optional[OutputHandler] = None,
    ) -> Dict[str, Any]:
        """Schedule resume work and return its current persisted state promptly."""
        execution_id = _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
        self.store.load(execution_id, recover=False)
        with self._active_lock:
            if self._closing:
                raise RuntimeError("durable execution manager is shutting down")
            execution_id = self._submit_background(
                execution_id,
                resume=True,
                options={
                    "retry_failed": retry_failed,
                    "confirm_unsafe": confirm_unsafe,
                    "output_handler": output_handler,
                },
            )
        return self.public(execution_id)

    def request_cancel(self, execution_id: str) -> Dict[str, Any]:
        """Request cancellation of one active execution without blocking on its lease."""
        execution_id = _validate_text(execution_id, "execution_id", MAX_EXECUTION_ID_BYTES)
        with self._active_lock:
            cancellation_event = self._active_cancellations.get(execution_id)
            active = cancellation_event is not None
            if cancellation_event is not None:
                cancellation_event.set()
        if active:
            return {
                "status": "ok",
                "execution_id": execution_id,
                "execution_in_progress": True,
                "cancellation_requested": True,
                "message": "Cancellation requested; inspect this execution for its final state.",
            }
        payload = self.public(execution_id)
        payload["cancellation_requested"] = False
        return payload

    def shutdown(self, grace_seconds: float) -> Dict[str, Any]:
        """Cancel active durable phases and wait no longer than the grace period."""
        with self._active_lock:
            self._closing = True
            cancellations = list(self._active_cancellations.values())
            background_work = list(self._background_futures.items())
            futures = [future for _execution_id, future in background_work]
            for cancellation_event in cancellations:
                cancellation_event.set()
        _done, not_done = wait_futures(futures, timeout=max(0.0, grace_seconds))
        self._background_executor.shutdown(wait=not not_done, cancel_futures=True)
        with self._active_lock:
            unfinished_ids = list(self._background_futures)
        return {
            "unfinished_execution_ids": unfinished_ids,
            "unfinished_execution_count": len(unfinished_ids),
        }

    def resume(
        self,
        execution_id: str,
        *,
        retry_failed: bool = False,
        confirm_unsafe: bool = False,
        output_handler: Optional[OutputHandler] = None,
    ) -> Dict[str, Any]:
        """Continue an execution while honoring retry and unsafe-side-effect gates."""
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
        """Return the internal validated execution record."""
        with self._lock:
            return self._load(execution_id)

    def public(self, execution_id: str, *, include_output: bool = False) -> Dict[str, Any]:
        """Return a sanitized public view with current activity diagnostics."""
        with self._lock:
            execution_id = _validate_text(
                execution_id,
                "execution_id",
                MAX_EXECUTION_ID_BYTES,
            )
            try:
                record = self._load(execution_id)
            except ExecutionRecordError as exc:
                return self.store.inspect_record(execution_id, exc)
            with self._active_lock:
                cancellation_requested = (
                    execution_id in self._active_cancellations
                    and self._active_cancellations[execution_id].is_set()
                )
                tracked_active = execution_id in self._active_cancellations
                background_error = self._background_failures.get(execution_id)
            payload = self._public(
                record,
                include_output=include_output,
                execution_in_progress=(
                    tracked_active or self.store.is_locked(record["execution_id"])
                ),
            )
            payload["cancellation_requested"] = cancellation_requested
            if background_error is not None:
                payload["background_error"] = background_error
            return payload

    def output(
        self,
        execution_id: str,
        phase_name: Optional[str] = None,
        *,
        offset: int = 0,
        max_bytes: int = MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
    ) -> Dict[str, Any]:
        """Return bounded output; page through one phase by specifying phase_name."""
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
        """Return a paginated public view of durable execution records."""
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
            public_records = []
            for record in records:
                if isinstance(record, ExecutionListDiagnostic):
                    public_records.append(record.response)
                    continue
                execution_id = record["execution_id"]
                with self._active_lock:
                    cancellation_event = self._active_cancellations.get(execution_id)
                    tracked_active = cancellation_event is not None
                    cancellation_requested = (
                        cancellation_event is not None and cancellation_event.is_set()
                    )
                    background_error = self._background_failures.get(execution_id)
                public_record = self._public(
                    record,
                    compact=True,
                    execution_in_progress=(
                        tracked_active or self.store.is_locked(execution_id)
                    ),
                )
                public_record["cancellation_requested"] = cancellation_requested
                if background_error is not None:
                    public_record["background_error"] = background_error
                public_records.append(public_record)
            return public_records

    def capacity(self) -> Dict[str, Any]:
        """Return storage diagnostics and bounded in-process execution admission."""
        capacity = self.store.capacity()
        with self._active_lock:
            futures = list(self._background_futures.values())
        capacity.update(
            {
                "background_execution_limit": MAX_BACKGROUND_EXECUTIONS,
                "background_execution_active": sum(future.running() for future in futures),
                "background_execution_queued": sum(
                    not future.running() and not future.done() for future in futures
                ),
            }
        )
        return capacity

    def retire(
        self,
        execution_ids: List[str],
        *,
        archive_path: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Preview or archive and retire explicit durable execution records."""
        return self.store.retire(
            execution_ids,
            archive_path=archive_path,
            dry_run=dry_run,
        )
