#!/usr/bin/env python3
"""Measure warm and cold latency for the command capture pipeline."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from capture_utils import run_command_bounded
from engine import EphemeralEngine


SCHEMA_VERSION = 1
DEFAULT_LINE_COUNTS = (16, 256, 2048)


def _command_for_lines(line_count: int) -> str:
    """Return a portable shell command with predictable line-oriented output."""
    return f"printf 'benchmark line\\n%.0s' $(seq 1 {line_count})"


def _measure_once(line_count: int, engine: EphemeralEngine) -> dict[str, float | int]:
    """Measure each phase of one warm capture pipeline invocation."""
    command = _command_for_lines(line_count)
    started = time.perf_counter()
    output, exit_code, truncated, original_bytes, timed_out = run_command_bounded(
        command, cwd=None, max_output_bytes=engine.max_buffer_bytes
    )
    command_seconds = time.perf_counter() - started

    started = time.perf_counter()
    capture = engine.ingest(
        output,
        label=f"latency-{line_count}",
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
        "line_count": line_count,
        "output_bytes": len(output.encode("utf-8")),
        "command_seconds": command_seconds,
        "ingest_seconds": ingest_seconds,
        "summary_seconds": summary_seconds,
        "total_seconds": command_seconds + ingest_seconds + summary_seconds,
    }


def _summary(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    """Return median and p95 timings while retaining sample count and size."""
    phases = ("command_seconds", "ingest_seconds", "summary_seconds", "total_seconds")
    first = samples[0]
    result: dict[str, Any] = {
        "line_count": first["line_count"],
        "output_bytes": first["output_bytes"],
        "samples": len(samples),
    }
    for phase in phases:
        values = sorted(float(sample[phase]) for sample in samples)
        result[f"{phase}_median"] = statistics.median(values)
        result[f"{phase}_p95"] = values[max(0, int(len(values) * 0.95) - 1)]
    return result


def run_benchmark(line_counts: tuple[int, ...], samples: int) -> dict[str, Any]:
    """Run cold-start and warm phase measurements for each output size."""
    if not line_counts or any(line_count < 1 for line_count in line_counts):
        raise ValueError("line_counts must contain only positive values")
    if samples < 1:
        raise ValueError("samples must be positive")

    cold_engine = EphemeralEngine(max_captures=2)
    started = time.perf_counter()
    _measure_once(line_counts[0], cold_engine)
    cold_start_seconds = time.perf_counter() - started

    engine = EphemeralEngine(max_captures=max(25, samples * len(line_counts) + 1))
    # Keep the reported size measurements focused on capture work, not model setup.
    engine.ingest("warmup", label="latency-warmup")
    by_size = []
    for line_count in line_counts:
        measurements = [_measure_once(line_count, engine) for _ in range(samples)]
        by_size.append(_summary(measurements))

    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "embedding_model": engine.embedding_model_name,
        "cold_start_seconds": cold_start_seconds,
        "warm_measurements": by_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--line-counts",
        nargs="+",
        type=int,
        default=DEFAULT_LINE_COUNTS,
        help="Output sizes to measure (default: 16 256 2048)",
    )
    parser.add_argument("--samples", type=int, default=5, help="Warm samples per output size")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    try:
        results = run_benchmark(tuple(args.line_counts), args.samples)
    except ValueError as exc:
        parser.error(str(exc))

    for measurement in results["warm_measurements"]:
        print(
            f"lines={measurement['line_count']} bytes={measurement['output_bytes']} "
            f"total_median={measurement['total_seconds_median']:.6f}s "
            f"command_median={measurement['command_seconds_median']:.6f}s "
            f"ingest_median={measurement['ingest_seconds_median']:.6f}s "
            f"summary_median={measurement['summary_seconds_median']:.6f}s"
        )
    print(f"cold_start={results['cold_start_seconds']:.6f}s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
