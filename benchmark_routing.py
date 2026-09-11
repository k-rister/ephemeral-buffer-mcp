#!/usr/bin/env python3
"""Measure direct versus captured execution for routing guidance."""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_latency import _nearest_rank
from capture_utils import run_command_bounded
from engine import EphemeralEngine

SCHEMA_VERSION = 1
DEFAULT_LINE_COUNTS = (16, 256, 2048)
PROFILE_NAMES = {16: "targeted", 256: "test", 2048: "build-or-log"}


def _command_for_lines(line_count: int) -> str:
    """Return a portable shell command with predictable line-oriented output."""
    return f"printf 'benchmark line\\n%.0s' $(seq 1 {line_count})"


def _measure_direct(command: str) -> dict[str, float | int]:
    """Measure a direct command that returns its complete output to the caller."""
    started = time.perf_counter()
    completed = subprocess.run(command, shell=True, capture_output=True, check=False, text=True)
    output = completed.stdout + completed.stderr
    return {
        "seconds": time.perf_counter() - started,
        "output_bytes": len(output.encode("utf-8")),
        "line_count": len(output.splitlines()),
        "exit_code": completed.returncode,
    }


def _measure_captured(command: str, engine: EphemeralEngine, label: str) -> dict[str, float | int]:
    """Measure bounded capture, ingestion, and summary generation."""
    started = time.perf_counter()
    output, exit_code, truncated, original_bytes, timed_out = run_command_bounded(
        command, cwd=None, max_output_bytes=engine.max_buffer_bytes
    )
    command_seconds = time.perf_counter() - started
    started = time.perf_counter()
    capture = engine.ingest(
        output,
        label=f"routing-{label}",
        truncated=truncated,
        original_byte_size=original_bytes if truncated else None,
        command_exit_code=exit_code,
        timed_out=timed_out,
    )
    ingest_seconds = time.perf_counter() - started
    started = time.perf_counter()
    engine.get_summary(capture.capture_id)
    summary_seconds = time.perf_counter() - started
    return {
        "seconds": command_seconds + ingest_seconds + summary_seconds,
        "command_seconds": command_seconds,
        "ingest_seconds": ingest_seconds,
        "summary_seconds": summary_seconds,
        "output_bytes": len(output.encode("utf-8")),
        "line_count": len(output.splitlines()),
        "exit_code": exit_code,
    }


def _summary(samples: list[dict[str, float | int]], line_count: int) -> dict[str, Any]:
    """Return stable median and p95 timings for one output profile."""
    if not samples:
        raise ValueError("samples must not be empty")
    direct = sorted(float(sample["direct_seconds"]) for sample in samples)
    captured = sorted(float(sample["captured_seconds"]) for sample in samples)
    direct_median = statistics.median(direct)
    captured_median = statistics.median(captured)
    return {
        "profile": PROFILE_NAMES.get(line_count, "custom"),
        "line_count": line_count,
        "output_bytes": samples[0]["output_bytes"],
        "samples": len(samples),
        "direct_seconds_median": direct_median,
        "direct_seconds_p95": _nearest_rank(direct),
        "captured_seconds_median": captured_median,
        "captured_seconds_p95": _nearest_rank(captured),
        "capture_overhead_seconds_median": captured_median - direct_median,
        "capture_overhead_ratio_median": captured_median / direct_median if direct_median else None,
    }


def run_benchmark(line_counts: tuple[int, ...], samples: int) -> dict[str, Any]:
    """Run direct/captured comparisons for each requested output size."""
    if not line_counts or any(line_count < 1 for line_count in line_counts):
        raise ValueError("line_counts must contain only positive values")
    if samples < 1:
        raise ValueError("samples must be positive")
    engine = EphemeralEngine(max_captures=max(25, samples * len(line_counts) + 1))
    measurements = []
    for line_count in line_counts:
        command = _command_for_lines(line_count)
        for _ in range(samples):
            direct = _measure_direct(command)
            captured = _measure_captured(command, engine, PROFILE_NAMES.get(line_count, "custom"))
            measurements.append({
                "line_count": line_count,
                "output_bytes": direct["output_bytes"],
                "direct_seconds": direct["seconds"],
                "captured_seconds": captured["seconds"],
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "profiles": [
            _summary([sample for sample in measurements if sample["line_count"] == line_count], line_count)
            for line_count in line_counts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-counts", nargs="+", type=int, default=DEFAULT_LINE_COUNTS)
    parser.add_argument("--samples", type=int, default=5, help="Samples per output size")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    try:
        results = run_benchmark(tuple(args.line_counts), args.samples)
    except ValueError as exc:
        parser.error(str(exc))
    for profile in results["profiles"]:
        print(
            f"profile={profile['profile']} lines={profile['line_count']} bytes={profile['output_bytes']} "
            f"direct_median={profile['direct_seconds_median']:.6f}s "
            f"captured_median={profile['captured_seconds_median']:.6f}s "
            f"overhead_ratio={profile['capture_overhead_ratio_median']:.2f}x"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
