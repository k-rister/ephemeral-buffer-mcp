#!/usr/bin/env python3
"""Compare lazy first use with background startup embedding warm-up."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from engine import EphemeralEngine, process_rss_bytes


SCHEMA_VERSION = 1


def _measure_in_process(warmup: bool) -> dict[str, Any]:
    rss_before = process_rss_bytes()
    started = time.perf_counter()
    engine = EphemeralEngine(max_captures=1, embedding_warmup=warmup)
    engine_init_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        if warmup:
            engine.start_embedding_warmup()
            startup_ready_seconds = engine_init_seconds + (time.perf_counter() - started)
            embedding_ready_started = time.perf_counter()
            warmup_state = engine.wait_for_embedding_warmup()
            embedding_ready_seconds = time.perf_counter() - embedding_ready_started
        else:
            warmup_state = engine.embedding_warmup_state
            startup_ready_seconds = engine_init_seconds + (time.perf_counter() - started)
            embedding_ready_seconds = None

        capture = engine.ingest("embedding warmup benchmark", label=f"warmup-{warmup}")
        started = time.perf_counter()
        engine.search_semantic(capture, "embedding benchmark")
        first_search_seconds = time.perf_counter() - started
        if not warmup:
            # Lazy mode becomes embedding-ready only when its first semantic
            # operation completes; do not report readiness as instantaneous.
            embedding_ready_seconds = first_search_seconds
        rss_after = process_rss_bytes()
        rss_delta_bytes = (
            None
            if rss_before is None or rss_after is None
            else max(0, rss_after - rss_before)
        )
        return {
            "warmup": warmup,
            "warmup_state": warmup_state,
            "engine_init_seconds": engine_init_seconds,
            "startup_ready_seconds": startup_ready_seconds,
            "embedding_ready_seconds": embedding_ready_seconds,
            "first_search_seconds": first_search_seconds,
            "rss_delta_bytes": rss_delta_bytes,
        }
    finally:
        engine.shutdown()


def _measure(warmup: bool) -> dict[str, Any]:
    """Measure one policy in a fresh process so RSS baselines are independent."""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--warmup",
            str(warmup).lower(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_benchmark(samples: int) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    records = [_measure(warmup) for _ in range(samples) for warmup in (False, True)]
    summaries = {}
    for warmup in (False, True):
        selected = [record for record in records if record["warmup"] is warmup]
        summaries[str(warmup).lower()] = {
            field: statistics.median(float(record[field]) for record in selected)
            for field in (
                "engine_init_seconds",
                "startup_ready_seconds",
                "embedding_ready_seconds",
                "first_search_seconds",
            )
        }
        rss_values = [
            record["rss_delta_bytes"]
            for record in selected
            if record["rss_delta_bytes"] is not None
        ]
        summaries[str(warmup).lower()]["rss_delta_bytes"] = (
            statistics.median(rss_values) if rss_values else None
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "samples": samples,
        "records": records,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup", choices=("true", "false"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_measure_in_process(args.warmup == "true"), sort_keys=True))
        return
    try:
        result = run_benchmark(args.samples)
    except ValueError as exc:
        parser.error(str(exc))
    for mode, summary in result["summaries"].items():
        print(
            f"warmup={mode} startup_ready_median={summary['startup_ready_seconds']:.6f}s "
            f"embedding_ready_median={summary['embedding_ready_seconds']:.6f}s "
            f"first_search_median={summary['first_search_seconds']:.6f}s "
            f"rss_delta_median={summary['rss_delta_bytes']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
