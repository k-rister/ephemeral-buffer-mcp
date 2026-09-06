#!/usr/bin/env python3
"""Measure concurrent ingest and read throughput for the shared engine."""

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import DEFAULT_MAX_CAPTURES
from engine import EphemeralEngine


def run_benchmark(captures: int, workers: int) -> dict:
    """Run the benchmark and return machine-readable measurements."""
    engine = EphemeralEngine(max_captures=max(captures, DEFAULT_MAX_CAPTURES))
    payload = "benchmark line with representative output\n" * 20

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        captured = list(executor.map(
            lambda index: engine.ingest(payload, label=f"benchmark-{index}"),
            range(captures),
        ))
    ingest_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        summaries = list(executor.map(
            lambda capture: engine.get_summary(capture.capture_id),
            captured,
        ))
    read_seconds = time.perf_counter() - started

    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "captures": len(captured),
        "workers": workers,
        "ingest_seconds": ingest_seconds,
        "ingest_per_second": len(captured) / ingest_seconds,
        "read_seconds": read_seconds,
        "reads_per_second": len(summaries) / read_seconds,
        "buffer_stats": engine.get_buffer_stats(),
    }


def load_baseline(path: Path) -> dict:
    """Load and validate the checked-in benchmark baseline."""
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        ingest = float(baseline["ingest_per_second"])
        reads = float(baseline["reads_per_second"])
        tolerance = float(baseline["minimum_ratio"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark baseline {path}: {exc}") from exc
    if ingest <= 0 or reads <= 0 or not 0 < tolerance <= 1:
        raise ValueError("baseline rates must be positive and minimum_ratio must be in (0, 1]")
    return {
        "ingest_per_second": ingest,
        "reads_per_second": reads,
        "minimum_ratio": tolerance,
    }


def check_regression(results: dict, baseline: dict) -> list[str]:
    """Return regression messages for measurements below the tolerated baseline."""
    minimum_ratio = baseline["minimum_ratio"]
    failures = []
    for metric in ("ingest_per_second", "reads_per_second"):
        minimum = baseline[metric] * minimum_ratio
        actual = results[metric]
        if actual < minimum:
            failures.append(
                f"{metric} {actual:.2f}/s is below the tolerated baseline "
                f"{minimum:.2f}/s ({minimum_ratio:.0%} of {baseline[metric]:.2f}/s)"
            )
    return failures


def write_results(path: Path, results: dict, baseline: dict | None, failures: list[str]) -> None:
    """Write a stable JSON record suitable for workflow artifacts."""
    record = dict(results)
    record["baseline"] = baseline
    record["regressions"] = failures
    record["passed"] = not failures
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=int, default=32, help="Number of captures to ingest")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker count")
    parser.add_argument(
        "--min-ingest-per-second",
        type=float,
        default=0.0,
        help="Fail if ingest throughput falls below this value",
    )
    parser.add_argument(
        "--min-reads-per-second",
        type=float,
        default=0.0,
        help="Fail if read throughput falls below this value",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON baseline used for a tolerated regression comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write machine-readable benchmark results to this JSON file",
    )
    args = parser.parse_args()
    if args.captures < 1 or args.workers < 1:
        parser.error("--captures and --workers must be positive")
    if args.min_ingest_per_second < 0 or args.min_reads_per_second < 0:
        parser.error("throughput thresholds must not be negative")

    try:
        baseline = load_baseline(args.baseline) if args.baseline else None
    except ValueError as exc:
        parser.error(str(exc))

    results = run_benchmark(args.captures, args.workers)
    print(f"captures={results['captures']} workers={results['workers']}")
    print(f"ingest_seconds={results['ingest_seconds']:.3f} ingest_per_second={results['ingest_per_second']:.2f}")
    print(f"read_seconds={results['read_seconds']:.3f} reads_per_second={results['reads_per_second']:.2f}")
    print(f"buffer_stats={results['buffer_stats']}")

    failures = check_regression(results, baseline) if baseline else []
    if args.min_ingest_per_second and results["ingest_per_second"] < args.min_ingest_per_second:
        failures.append(
            f"ingest throughput {results['ingest_per_second']:.2f}/s is below "
            f"the minimum {args.min_ingest_per_second:.2f}/s"
        )
    if args.min_reads_per_second and results["reads_per_second"] < args.min_reads_per_second:
        failures.append(
            f"read throughput {results['reads_per_second']:.2f}/s is below "
            f"the minimum {args.min_reads_per_second:.2f}/s"
        )
    if args.output:
        write_results(args.output, results, baseline, failures)
    for failure in failures:
        print(f"REGRESSION: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
