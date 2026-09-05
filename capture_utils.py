"""Utilities for bounded command and stream output capture."""

import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from config import DEFAULT_MAX_OUTPUT_BYTES


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

    def finish(self) -> Tuple[str, bool, int]:
        if not self.truncated:
            return self.captured.decode("utf-8", errors="replace"), False, self.total_bytes
        marker = (
            f"\n\n[output truncated: retained first {len(self.head):,} and last {len(self.tail):,} bytes "
            f"of {self.total_bytes:,}]\n\n"
        ).encode("utf-8")
        output = bytes(self.head) + marker + bytes(self.tail)
        if len(output) > self.max_output_bytes:
            output = output[:self.max_output_bytes]
        return output.decode("utf-8", errors="replace"), True, self.total_bytes


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
) -> Tuple[str, int, bool, int, bool]:
    """Run a command while retaining bounded output and enforcing an optional timeout."""
    return _run_command_bounded(command, cwd, max_output_bytes, timeout_seconds)


def _run_command_bounded(
    command: str,
    cwd: Optional[str],
    max_output_bytes: int,
    timeout_seconds: Optional[float],
) -> Tuple[str, int, bool, int, bool]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    capture = _BoundedCapture(max_output_bytes)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    timed_out = False
    try:
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
    finally:
        if timed_out:
            _terminate_process_group(proc)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        selector.close()
        if proc.stdout is not None:
            proc.stdout.close()
    output, truncated, total_bytes = capture.finish()
    return output, (124 if timed_out else proc.returncode), truncated, total_bytes, timed_out


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate a shell command and all children started in its process group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()


def read_file_bounded(file_path: str, max_bytes: int) -> str:
    """Read a UTF-8 file only when it fits within the configured byte limit."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")
    with Path(file_path).open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds the {max_bytes:,}-byte capture limit")
    return content.decode("utf-8", errors="replace")
