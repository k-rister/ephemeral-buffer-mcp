#!/usr/bin/env python3
"""Measure concurrent ingest and read throughput for the shared engine."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from config import DEFAULT_MAX_CAPTURES
from engine import EphemeralEngine


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
    args = parser.parse_args()
    if args.captures < 1 or args.workers < 1:
        parser.error("--captures and --workers must be positive")
    if args.min_ingest_per_second < 0 or args.min_reads_per_second < 0:
        parser.error("throughput thresholds must not be negative")

    engine = EphemeralEngine(max_captures=max(args.captures, DEFAULT_MAX_CAPTURES))
    payload = "benchmark line with representative output\n" * 20

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        captures = list(executor.map(
            lambda index: engine.ingest(payload, label=f"benchmark-{index}"),
            range(args.captures),
        ))
    ingest_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(
            lambda capture: engine.get_summary(capture.capture_id),
            captures,
        ))
    read_seconds = time.perf_counter() - started

    print(f"captures={len(captures)} workers={args.workers}")
    print(f"ingest_seconds={ingest_seconds:.3f} ingest_per_second={len(captures) / ingest_seconds:.2f}")
    print(f"read_seconds={read_seconds:.3f} reads_per_second={len(summaries) / read_seconds:.2f}")
    print(f"buffer_stats={engine.get_buffer_stats()}")

    ingest_per_second = len(captures) / ingest_seconds
    reads_per_second = len(summaries) / read_seconds
    if ingest_per_second < args.min_ingest_per_second:
        parser.error(
            f"ingest throughput {ingest_per_second:.2f}/s is below "
            f"the minimum {args.min_ingest_per_second:.2f}/s"
        )
    if reads_per_second < args.min_reads_per_second:
        parser.error(
            f"read throughput {reads_per_second:.2f}/s is below "
            f"the minimum {args.min_reads_per_second:.2f}/s"
        )


if __name__ == "__main__":
    main()
