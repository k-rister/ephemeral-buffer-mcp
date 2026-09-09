#!/usr/bin/env python3
"""Compare lazy semantic indexing with bounded asynchronous prefetch."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from engine import EphemeralEngine


SCHEMA_VERSION = 1


def _fixture(line_count: int) -> str:
    return "\n".join(f"semantic benchmark line {index}" for index in range(line_count))


def _measure(prefetch: bool, line_count: int) -> dict[str, float | int | bool]:
    engine = EphemeralEngine(
        max_captures=2,
        semantic_prefetch=prefetch,
        semantic_prefetch_workers=1,
    )
    try:
        started = time.perf_counter()
        capture = engine.ingest(_fixture(line_count), label=f"prefetch-{prefetch}")
        ingest_seconds = time.perf_counter() - started

        started = time.perf_counter()
        engine.search_semantic(capture, "semantic benchmark")
        first_search_seconds = time.perf_counter() - started

        started = time.perf_counter()
        engine.search_semantic(capture, "semantic benchmark")
        second_search_seconds = time.perf_counter() - started
        return {
            "prefetch": prefetch,
            "line_count": line_count,
            "ingest_seconds": ingest_seconds,
            "first_search_seconds": first_search_seconds,
            "second_search_seconds": second_search_seconds,
        }
    finally:
        engine.shutdown()


def run_benchmark(line_count: int, samples: int) -> dict[str, Any]:
    if line_count < 1:
        raise ValueError("line_count must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    records = [_measure(prefetch, line_count) for _ in range(samples) for prefetch in (False, True)]
    summaries = {}
    for prefetch in (False, True):
        selected = [record for record in records if record["prefetch"] == prefetch]
        summaries[str(prefetch).lower()] = {
            field: statistics.median(float(record[field]) for record in selected)
            for field in ("ingest_seconds", "first_search_seconds", "second_search_seconds")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "line_count": line_count,
        "samples": samples,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-count", type=int, default=256)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_benchmark(args.line_count, args.samples)
    except ValueError as exc:
        parser.error(str(exc))
    for mode, summary in result["summaries"].items():
        print(
            f"prefetch={mode} ingest_median={summary['ingest_seconds']:.6f}s "
            f"first_search_median={summary['first_search_seconds']:.6f}s "
            f"second_search_median={summary['second_search_seconds']:.6f}s"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
