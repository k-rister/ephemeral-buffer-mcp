"""Utilities for bounded command and stream output capture."""

import ctypes
import errno
import os
import logging
import select
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple
from config import DEFAULT_MAX_OUTPUT_BYTES
from logging_utils import get_logger, log_event


LOGGER = get_logger("capture")
_SUPERVISOR_ARG = "--ephemeral-supervise"
_SUPERVISOR_CLEANUP_FAILURE = 125
_SUPERVISOR_CLEANUP_TIMEOUT = 3
_SUPERVISOR_MARKER_SETTLE_TIMEOUT = 1
_SUPERVISOR_SIGNAL_RETRY_DELAY = 0.01
_SUPERVISOR_IDENTITY_RETRIES = 3


class BoundedCommandResult(tuple):
    """Five-value command result with an internal cleanup confirmation flag."""

    def __new__(
        cls,
        output: str,
        exit_code: int,
        truncated: bool,
        original_byte_size: int,
        timed_out: bool,
        *,
        cleanup_confirmed: bool = True,
    ):
        result = super().__new__(
            cls,
            (output, exit_code, truncated, original_byte_size, timed_out),
        )
        result.cleanup_confirmed = cleanup_confirmed
        return result


def _attach_cleanup_status(error: BaseException, confirmed: bool) -> None:
    """Make cleanup confirmation available to the execution manager."""
    try:
        error.cleanup_confirmed = confirmed
    except (AttributeError, TypeError):
        pass


class _BoundedCapture:
    """Incrementally retain bounded head/tail bytes from a binary stream."""

    def __init__(self, max_output_bytes: int):
        if max_output_bytes < 512:
            raise ValueError("max_output_bytes must be at least 512")
        payload_limit = max_output_bytes - 256
        self.max_output_bytes = max_output_bytes
        self.head_limit = payload_limit // 2
        self.tail_limit = payload_limit - self.head_limit
        self.captured = bytearray()
        self.head = bytearray()
        self.tail = bytearray()
        self.total_bytes = 0
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        if not self.truncated and len(self.captured) + len(chunk) <= self.head_limit + self.tail_limit:
            self.captured.extend(chunk)
            return
        if not self.truncated:
            combined = bytes(self.captured) + chunk
            self.head.extend(combined[:self.head_limit])
            self.tail.extend(combined[-self.tail_limit:])
            self.truncated = True
            return
        self.tail.extend(chunk)
        if len(self.tail) > self.tail_limit:
            del self.tail[:-self.tail_limit]

    @staticmethod
    def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
        """Return text whose UTF-8 representation fits without splitting characters."""
        if max_bytes <= 0:
            return ""
        encoded_size = 0
        retained = []
        for character in text:
            character_size = len(character.encode("utf-8"))
            if encoded_size + character_size > max_bytes:
                break
            retained.append(character)
            encoded_size += character_size
        return "".join(retained)

    def _finish_truncated(self) -> str:
        head_text = bytes(self.head).decode("utf-8", errors="replace")
        tail_text = bytes(self.tail).decode("utf-8", errors="replace")
        marker = (
            f"\n\n[output truncated: retained first {len(self.head):,} and last {len(self.tail):,} bytes "
            f"of {self.total_bytes:,}]\n\n"
        )
        marker_bytes = len(marker.encode("utf-8"))
        if marker_bytes >= self.max_output_bytes:
            return self._truncate_text_to_bytes(marker, self.max_output_bytes)

        content_budget = self.max_output_bytes - marker_bytes
        head_budget = min(
            len(head_text.encode("utf-8")),
            content_budget // 2,
        )
        head_text = self._truncate_text_to_bytes(head_text, head_budget)
        tail_budget = content_budget - len(head_text.encode("utf-8"))
        tail_text = self._truncate_text_to_bytes(tail_text, tail_budget)
        return head_text + marker + tail_text

    def finish(self) -> Tuple[str, bool, int]:
        if not self.truncated:
            output = self.captured.decode("utf-8", errors="replace")
            if len(output.encode("utf-8")) <= self.max_output_bytes:
                return output, False, self.total_bytes
            self.head = self.captured
            self.tail = bytearray()
            self.truncated = True
        return self._finish_truncated(), True, self.total_bytes


def bound_chunks(chunks: Iterable[bytes], max_output_bytes: int) -> Tuple[str, bool, int]:
    """Retain bounded head/tail bytes from an arbitrary binary stream."""
    capture = _BoundedCapture(max_output_bytes)
    for chunk in chunks:
        capture.add(chunk)
    return capture.finish()


def run_command_bounded(
    command: str,
    cwd: Optional[str],
    max_output_bytes: int,
    timeout_seconds: Optional[float] = None,
    process_started: Optional[Callable[[int, Optional[int]], None]] = None,
    process_marker: Optional[str] = None,
) -> Tuple[str, int, bool, int, bool]:
    """Run a command while retaining bounded output and enforcing an optional timeout."""
    return _run_command_bounded(
        command, cwd, max_output_bytes, timeout_seconds, process_started, process_marker
    )


def _run_command_bounded(
    command: str,
    cwd: Optional[str],
    max_output_bytes: int,
    timeout_seconds: Optional[float],
    process_started: Optional[Callable[[int, Optional[int]], None]] = None,
    process_marker: Optional[str] = None,
) -> Tuple[str, int, bool, int, bool]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    capture = _BoundedCapture(max_output_bytes)
    environment = None
    if process_marker is not None:
        environment = os.environ.copy()
        environment["EPHEMERAL_EXECUTION_PROCESS_MARKER"] = process_marker
    if process_marker is not None:
        # Keep a dedicated Linux subreaper alive until the command exits.  A
        # process-group kill cannot reach a child that calls setsid(); the
        # supervisor adopts such orphans and removes them before returning.
        popen_command = [sys.executable, os.path.abspath(__file__), _SUPERVISOR_ARG, command]
        shell = False
    else:
        popen_command = command
        shell = True
    proc = subprocess.Popen(
        popen_command,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    process_group_id = getattr(proc, "pid", None)
    try:
        process_group_id = os.getpgid(proc.pid)
    except (AttributeError, OSError):
        pass
    cleanup_confirmed = True
    if process_started is not None:
        try:
            process_started(proc.pid, process_group_id)
        except BaseException as exc:
            try:
                cleanup_confirmed = (
                    _terminate_supervised_process(proc, process_group_id, process_marker)
                    if process_marker is not None
                    else _terminate_process_group(proc, process_group_id)
                )
            except BaseException:
                cleanup_confirmed = False
            _attach_cleanup_status(exc, cleanup_confirmed)
            if proc.stdout is not None:
                proc.stdout.close()
            raise
    selector = None
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    timed_out = False
    aborted = False
    pending_error = None
    try:
        try:
            # Keep selector construction and registration inside the cleanup
            # boundary.  A platform selector can fail before the read loop
            # starts, and the child group must still be reaped in that case.
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                events = selector.select(remaining)
                if not events:
                    timed_out = True
                    break
                for key, _ in events:
                    chunk = key.fileobj.read1(65536)
                    if chunk:
                        capture.add(chunk)
                    else:
                        selector.unregister(key.fileobj)
            if not timed_out:
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
        except BaseException as exc:
            pending_error = exc
            aborted = True
            raise
    finally:
        if timed_out or aborted:
            if timed_out:
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "command_timeout",
                    pid=getattr(proc, "pid", None),
                    timeout_seconds=timeout_seconds,
                    output_bytes=capture.total_bytes,
                )
            try:
                if process_group_id is None:
                    cleanup_confirmed = _terminate_process_group(proc)
                elif process_marker is not None:
                    cleanup_confirmed = _terminate_supervised_process(
                        proc, process_group_id, process_marker
                    )
                else:
                    cleanup_confirmed = _terminate_process_group(proc, process_group_id)
            except BaseException as cleanup_error:
                cleanup_confirmed = False
                _attach_cleanup_status(cleanup_error, False)
                if pending_error is None:
                    raise
            if pending_error is not None:
                _attach_cleanup_status(pending_error, cleanup_confirmed)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            log_event(LOGGER, logging.WARNING, "process_wait_timeout", pid=getattr(proc, "pid", None))
            proc.kill()
            proc.wait()
        if selector is not None:
            selector.close()
        if proc.stdout is not None:
            proc.stdout.close()
    output, truncated, total_bytes = capture.finish()
    if process_marker is not None:
        # The subreaper result is the cleanup handshake.  A marker scan can
        # disprove it, but an unrelated protected /proc entry cannot.
        supervisor_confirmed = (
            proc.returncode is not None
            and proc.returncode >= 0
            and proc.returncode != _SUPERVISOR_CLEANUP_FAILURE
        )
        marker_absent = _marker_processes_absent(process_marker)
        cleanup_confirmed = (
            cleanup_confirmed
            and supervisor_confirmed
            and marker_absent is not False
        )
    return BoundedCommandResult(
        output,
        (124 if timed_out else proc.returncode),
        truncated,
        total_bytes,
        timed_out,
        cleanup_confirmed=cleanup_confirmed,
    )


def _terminate_process_group(proc: subprocess.Popen, process_group_id: Optional[int] = None) -> bool:
    """Terminate a shell command and all children started in its process group."""
    process_id = getattr(proc, "pid", None)
    group_id = process_group_id if process_group_id is not None else process_id
    try:
        os.killpg(group_id, signal.SIGTERM)
        log_event(LOGGER, logging.INFO, "process_group_terminate", pid=process_id, signal="SIGTERM")
    except (ProcessLookupError, OSError):
        log_event(LOGGER, logging.INFO, "process_terminate_fallback", pid=process_id)
        proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(group_id, signal.SIGKILL)
        log_event(LOGGER, logging.WARNING, "process_group_kill", pid=process_id, signal="SIGKILL")
    except ProcessLookupError:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return True
    except OSError:
        log_event(LOGGER, logging.WARNING, "process_kill_fallback", pid=process_id)
        proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait()
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _terminate_supervised_process(
    proc: subprocess.Popen,
    process_group_id: Optional[int],
    marker: str,
) -> bool:
    """Request cleanup and require the supervisor to complete its handshake."""
    group_id = process_group_id if process_group_id is not None else getattr(proc, "pid", None)
    if group_id is None:
        return False
    try:
        os.killpg(group_id, signal.SIGTERM)
        log_event(LOGGER, logging.INFO, "process_group_terminate", pid=proc.pid, signal="SIGTERM")
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return False
    forced = False
    try:
        proc.wait(timeout=_SUPERVISOR_CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            os.killpg(group_id, signal.SIGKILL)
            log_event(LOGGER, logging.WARNING, "process_group_kill", pid=proc.pid, signal="SIGKILL")
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                return False
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return False
    marker_absent = _marker_processes_absent(marker)
    return (
        not forced
        and proc.returncode is not None
        and proc.returncode >= 0
        and proc.returncode != _SUPERVISOR_CLEANUP_FAILURE
        and marker_absent is not False
    )


def read_file_bounded(file_path: str, max_bytes: int) -> str:
    """Read a UTF-8 file only when it fits within the configured byte limit."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")
    with Path(file_path).open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds the {max_bytes:,}-byte capture limit")
    return content.decode("utf-8", errors="replace")


def _enable_subreaper() -> bool:
    """Make the supervisor adopt descendants orphaned by the command."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        return prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except (AttributeError, OSError):
        return False


def _marker_processes(marker: str) -> Optional[set[int]]:
    """Find marked processes, or return None for an incomplete clean scan."""
    if not sys.platform.startswith("linux"):
        return None
    marker_bytes = f"EPHEMERAL_EXECUTION_PROCESS_MARKER={marker}".encode("utf-8")
    try:
        process_names = os.listdir("/proc")
    except OSError:
        return None
    processes: set[int] = set()
    scan_complete = True
    current_uid = getattr(os, "getuid", lambda: None)()
    for process_name in process_names:
        if not process_name.isdigit():
            continue
        process_id = int(process_name)
        try:
            if current_uid is not None and os.stat(f"/proc/{process_id}").st_uid != current_uid:
                continue
        except FileNotFoundError:
            continue
        except OSError:
            scan_complete = False
            continue
        try:
            environment = Path(f"/proc/{process_id}/environ").read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            scan_complete = False
            continue
        if marker_bytes in environment.split(b"\0"):
            processes.add(process_id)
    if processes:
        return processes
    return processes if scan_complete else None


def _marker_processes_absent(marker: str) -> Optional[bool]:
    """Return True, False, or None when process absence cannot be observed."""
    deadline = time.monotonic() + _SUPERVISOR_MARKER_SETTLE_TIMEOUT
    while True:
        processes = _marker_processes(marker)
        if processes == set():
            return True
        if time.monotonic() >= deadline:
            return None if processes is None else False
        time.sleep(0.01)


def _supervisor_descendants(root_pid: int) -> Optional[set[int]]:
    """Return all descendants of a supervisor from Linux proc metadata."""
    try:
        process_names = os.listdir("/proc")
    except OSError:
        return None
    parents = {}
    for process_name in process_names:
        if not process_name.isdigit():
            continue
        process_id = int(process_name)
        try:
            stat_line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
            remainder = stat_line[stat_line.rfind(") ") + 2 :].split()
            if len(remainder) > 1:
                parents[process_id] = int(remainder[1])
        except (OSError, UnicodeError, ValueError):
            continue
    children = {}
    for process_id, parent_id in parents.items():
        children.setdefault(parent_id, set()).add(process_id)
    descendants = set()
    pending = list(children.get(root_pid, set()))
    while pending:
        process_id = pending.pop()
        if process_id in descendants:  # pragma: no cover - proc parent maps are unique.
            continue
        descendants.add(process_id)
        pending.extend(children.get(process_id, set()))
    return descendants


def _supervisor_reap() -> bool:
    """Reap exited children and report whether the subreaper has none left."""
    while True:
        try:
            process_id, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        except OSError:
            return False
        if process_id == 0:
            return False


def _supervisor_process_identity(process_id: int) -> Optional[Tuple[str, str]]:
    """Return a Linux process generation identity for a cleanup target."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        stat_line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        remainder = stat_line[stat_line.rfind(") ") + 2 :].split()
        if not boot_id or len(remainder) <= 19:
            return None
        return remainder[19], boot_id
    except (OSError, UnicodeError, ValueError):
        return None


def _supervisor_process_absent(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    try:
        stat_line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        remainder = stat_line[stat_line.rfind(") ") + 2 :].split()
    except (OSError, UnicodeError, ValueError):
        return False
    return bool(remainder and remainder[0] == "Z")


def _supervisor_pidfd_terminated(pidfd: int) -> bool:
    try:
        ready, _write, _error = select.select([pidfd], [], [], 0)
    except (OSError, ValueError):
        return False
    return bool(ready)


def _supervisor_identity_with_retry(process_id: int) -> Optional[Tuple[str, str]]:
    for attempt in range(_SUPERVISOR_IDENTITY_RETRIES):
        identity = _supervisor_process_identity(process_id)
        if identity is not None:
            return identity
        if attempt + 1 < _SUPERVISOR_IDENTITY_RETRIES:
            time.sleep(_SUPERVISOR_SIGNAL_RETRY_DELAY)
    return None


def _supervisor_signal_descendants(descendants: set[int], signum: int) -> bool:
    """Signal descendants only after pinning their process generations."""
    if not descendants:
        return True
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        return False
    pidfds = []
    signalable_pidfds = []
    try:
        for process_id in descendants:
            expected_identity = _supervisor_identity_with_retry(process_id)
            if expected_identity is None:
                if _supervisor_process_absent(process_id):
                    continue
                return False
            try:
                pidfd = pidfd_open(process_id)
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.ESRCH:
                    continue
                return False
            pidfds.append(pidfd)
            current_identity = _supervisor_identity_with_retry(process_id)
            if current_identity is None:
                if _supervisor_pidfd_terminated(pidfd):
                    continue
                return False
            if current_identity != expected_identity:
                return False
            signalable_pidfds.append(pidfd)
        for pidfd in signalable_pidfds:
            try:
                pidfd_send_signal(pidfd, signum)
            except ProcessLookupError:
                pass
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.ESRCH or _supervisor_pidfd_terminated(pidfd):
                    continue
                return False
        return True
    finally:
        for pidfd in pidfds:
            try:
                os.close(pidfd)
            except OSError:
                pass


def _supervisor_cleanup(root_pid: int) -> bool:
    """Terminate every descendant before the supervisor exits."""
    descendants = _supervisor_descendants(root_pid)
    if descendants is None:
        return False
    if not _supervisor_signal_descendants(descendants, signal.SIGTERM):
        descendants = _supervisor_descendants(root_pid)
        if descendants is None:
            return False
        if descendants:
            time.sleep(_SUPERVISOR_SIGNAL_RETRY_DELAY)
            if not _supervisor_signal_descendants(descendants, signal.SIGTERM):
                return False
    deadline = time.monotonic() + 1
    while descendants and time.monotonic() < deadline:
        _supervisor_reap()
        descendants = _supervisor_descendants(root_pid)
        if descendants is None:
            return False
        if descendants:
            time.sleep(0.01)
    if not _supervisor_signal_descendants(descendants, signal.SIGKILL):
        descendants = _supervisor_descendants(root_pid)
        if descendants is None:
            return False
        if descendants:
            time.sleep(_SUPERVISOR_SIGNAL_RETRY_DELAY)
            if not _supervisor_signal_descendants(descendants, signal.SIGKILL):
                return False
    deadline = time.monotonic() + 1
    while descendants and time.monotonic() < deadline:
        _supervisor_reap()
        descendants = _supervisor_descendants(root_pid)
        if descendants is None:
            return False
        if descendants:
            time.sleep(0.01)
    return not descendants and _supervisor_reap()


def _supervisor_main(command: str) -> int:
    if not _enable_subreaper():
        return _SUPERVISOR_CLEANUP_FAILURE

    def stop(_signum, _frame):
        cleanup_confirmed = _supervisor_cleanup(os.getpid())
        os._exit(
            128 + _signum if cleanup_confirmed else _SUPERVISOR_CLEANUP_FAILURE
        )

    try:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        child = subprocess.Popen(command, shell=True)
        return_code = child.wait()
        cleanup_confirmed = _supervisor_cleanup(os.getpid())
    except BaseException:
        # An unhandled supervisor failure must never resemble command success.
        try:
            _supervisor_cleanup(os.getpid())
        except BaseException:
            pass
        return _SUPERVISOR_CLEANUP_FAILURE
    return return_code if cleanup_confirmed else _SUPERVISOR_CLEANUP_FAILURE


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == _SUPERVISOR_ARG:  # pragma: no cover
    raise SystemExit(_supervisor_main(sys.argv[2]))  # pragma: no cover
