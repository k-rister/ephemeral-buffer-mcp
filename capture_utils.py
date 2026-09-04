"""Utilities for bounded command and stream output capture."""

import subprocess
from typing import Iterable, Optional, Tuple


DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024 * 1024


def bound_chunks(chunks: Iterable[bytes], max_output_bytes: int) -> Tuple[str, bool, int]:
    """Retain bounded head/tail bytes from an arbitrary binary stream."""
    if max_output_bytes < 512:
        raise ValueError("max_output_bytes must be at least 512")

    payload_limit = max_output_bytes - 256
    head_limit = payload_limit // 2
    tail_limit = payload_limit - head_limit
    captured = bytearray()
    head = bytearray()
    tail = bytearray()
    total_bytes = 0
    truncated = False

    for chunk in chunks:
        if not chunk:
            continue
        total_bytes += len(chunk)
        if not truncated and len(captured) + len(chunk) <= payload_limit:
            captured.extend(chunk)
            continue

        if not truncated:
            combined = bytes(captured) + chunk
            head.extend(combined[:head_limit])
            tail.extend(combined[-tail_limit:])
            truncated = True
        else:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[:-tail_limit]

    if not truncated:
        return captured.decode("utf-8", errors="replace"), False, total_bytes

    marker = (
        f"\n\n[output truncated: retained first {len(head):,} and last {len(tail):,} bytes "
        f"of {total_bytes:,}]\n\n"
    ).encode("utf-8")
    output = bytes(head) + marker + bytes(tail)
    if len(output) > max_output_bytes:
        output = output[:max_output_bytes]
    return output.decode("utf-8", errors="replace"), True, total_bytes


def run_command_bounded(
    command: str,
    cwd: Optional[str],
    max_output_bytes: int,
) -> Tuple[str, int, bool, int]:
    """Run a command while retaining only bounded head/tail output."""
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        output, truncated, total_bytes = bound_chunks(
            iter(lambda: proc.stdout.read(65536), b""),
            max_output_bytes,
        )
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        exit_code = proc.wait()
    return output, exit_code, truncated, total_bytes
