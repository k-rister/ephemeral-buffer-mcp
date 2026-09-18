#!/usr/bin/env python3
"""Measure warm and cold latency for the command capture pipeline."""

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import workload_results as wr
from capture_utils import run_command_bounded
from engine import EphemeralEngine


SCHEMA_VERSION = 1
DEFAULT_LINE_COUNTS = (16, 256, 2048)
PHASE_NAMES = ("command", "ingest", "semantic_index", "summary")


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
    engine._ensure_embeddings(capture)
    semantic_index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    engine.get_summary(capture.capture_id)
    summary_seconds = time.perf_counter() - started

    return {
        "line_count": line_count,
        "output_bytes": len(output.encode("utf-8")),
        "command_seconds": command_seconds,
        "ingest_seconds": ingest_seconds,
        "semantic_index_seconds": semantic_index_seconds,
        "summary_seconds": summary_seconds,
        "total_seconds": command_seconds + ingest_seconds + semantic_index_seconds + summary_seconds,
    }


def _nearest_rank(values: list[float], percentile: float = 0.95) -> float:
    """Return a percentile using the nearest-rank convention."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than 0 and at most 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _summary(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    """Return median and p95 timings while retaining sample count and size."""
    if not samples:
        raise ValueError("samples must not be empty")
    phases = ("command_seconds", "ingest_seconds", "semantic_index_seconds", "summary_seconds", "total_seconds")
    first = samples[0]
    result: dict[str, Any] = {
        "line_count": first["line_count"],
        "output_bytes": first["output_bytes"],
        "samples": len(samples),
    }
    for phase in phases:
        values = sorted(float(sample[phase]) for sample in samples)
        result[f"{phase}_median"] = statistics.median(values)
        result[f"{phase}_p95"] = _nearest_rank(values)
    return result


def run_benchmark(line_counts: tuple[int, ...], samples: int) -> dict[str, Any]:
    """Run cold-start and warm phase measurements for each output size."""
    if not line_counts or any(line_count < 1 for line_count in line_counts):
        raise ValueError("line_counts must contain only positive values")
    if samples < 1:
        raise ValueError("samples must be positive")

    # These harnesses time the lazy indexing path explicitly, so background
    # prefetch (on by default) is disabled to keep the phases distinct.
    cold_engine = EphemeralEngine(max_captures=2, semantic_prefetch=False)
    started = time.perf_counter()
    _measure_once(line_counts[0], cold_engine)
    cold_start_seconds = time.perf_counter() - started

    engine = EphemeralEngine(max_captures=max(25, samples * len(line_counts) + 1), semantic_prefetch=False)
    # Keep the reported size measurements focused on capture work, not model setup.
    warmup = engine.ingest("warmup", label="latency-warmup")
    engine._ensure_embeddings(warmup)
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


def workload_result(results: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-agnostic workload result for a benchmark record."""
    warm = results["warm_measurements"]
    runs = [
        wr.run(
            "cold-start",
            labels={"cache_state": "cold", "line_count": warm[0]["line_count"] if warm else None},
            measurements={
                "wall_time_seconds": wr.measurement(
                    "seconds",
                    value=results["cold_start_seconds"],
                    samples=1,
                    note="fresh engine: model setup plus one complete capture pipeline",
                ),
            },
        )
    ]
    for measurement in warm:
        samples = measurement["samples"]
        runs.append(wr.run(
            f"lines-{measurement['line_count']}",
            labels={"cache_state": "warm", "line_count": measurement["line_count"]},
            measurements={
                "output_bytes": wr.measurement("bytes", value=measurement["output_bytes"]),
                "wall_time_seconds": wr.measurement(
                    "seconds",
                    median=measurement["total_seconds_median"],
                    p95=measurement["total_seconds_p95"],
                    samples=samples,
                ),
            },
            phases=[
                wr.phase(
                    name,
                    median=measurement[f"{name}_seconds_median"],
                    p95=measurement[f"{name}_seconds_p95"],
                    samples=samples,
                )
                for name in PHASE_NAMES
            ],
        ))
    return wr.build_result(
        workload="capture-latency",
        kind="benchmark",
        producer="benchmark_latency.py",
        producer_schema_version=results["schema_version"],
        parameters={
            "line_counts": [measurement["line_count"] for measurement in warm],
            "samples": warm[0]["samples"] if warm else 0,
        },
        environment=wr.environment(embedding_model=results["embedding_model"]),
        runs=runs,
        details=results,
    )


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
    wr.add_result_argument(parser)
    args = parser.parse_args()

    try:
        results = run_benchmark(tuple(args.line_counts), args.samples)
    except ValueError as exc:
        parser.error(str(exc))

    report = wr.report_stream(args.result)
    for measurement in results["warm_measurements"]:
        print(
            f"lines={measurement['line_count']} bytes={measurement['output_bytes']} "
            f"total_median={measurement['total_seconds_median']:.6f}s "
            f"command_median={measurement['command_seconds_median']:.6f}s "
            f"ingest_median={measurement['ingest_seconds_median']:.6f}s "
            f"semantic_index_median={measurement['semantic_index_seconds_median']:.6f}s "
            f"summary_median={measurement['summary_seconds_median']:.6f}s",
            file=report,
        )
    print(f"cold_start={results['cold_start_seconds']:.6f}s", file=report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.result:
        wr.write_result(workload_result(results), args.result)


if __name__ == "__main__":
    main()
