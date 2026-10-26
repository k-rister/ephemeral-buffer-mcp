"""Private process identity and recovery helpers for durable executions."""

from __future__ import annotations

import errno
import os
import select
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .capture_utils import PROCESS_MARKER_ENV

try:
    import fcntl
except ImportError:  # pragma: no cover - durable process recovery requires Linux.
    fcntl = None

PROCESS_CONTAINMENT_SUBREAPER = "linux-subreaper"


@dataclass(frozen=True)
class CleanupOutcome:
    """Whether process cleanup was confirmed and why a fence remains."""

    confirmed: bool
    blocked_reason_code: Optional[str] = None



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

