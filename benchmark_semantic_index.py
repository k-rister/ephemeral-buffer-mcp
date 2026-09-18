#!/usr/bin/env python3
"""Measure semantic indexing cost and first hybrid-search latency by capture size."""

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from engine import EphemeralEngine


SCHEMA_VERSION = 1
FIXTURE_VERSION = 1
DEFAULT_LINE_COUNTS = (16, 256, 2048, 8192)
SEARCH_MODES = ("semantic", "hybrid")
PHASES = (
    "ingest_seconds",
    "semantic_index_seconds",
    "first_search_seconds",
    "subsequent_search_seconds",
)

# The needle is phrased so lexical and semantic retrieval both have a fair
# chance: the query shares vocabulary with it but is not a verbatim copy.
NEEDLE_LINE = "ERROR worker-7 lost connection to postgres primary: connection reset by peer"
NEEDLE_QUERY = "database connection dropped"

_LINE_TEMPLATES = (
    "INFO  {ts} build step {n}/{total} compiling module {mod} ({ms} ms)",
    "INFO  {ts} test_{mod}::test_case_{n} PASSED in {ms}ms",
    "WARN  {ts} deprecated option '{mod}-mode' used in config/{mod}.toml",
    "DEBUG {ts} cache lookup key={mod}:{n} hit=false latency={ms}us",
    "INFO  {ts} downloaded artifact {mod}-{n}.tar.gz ({kb} KiB)",
    "INFO  {ts} scheduler assigned job {n} to worker-{w} queue depth={d}",
    "DEBUG {ts} GET /api/v1/{mod}/{n} 200 {ms}ms bytes={kb}",
    "INFO  {ts} linking target {mod} with {d} objects",
)
_MODULES = (
    "parser", "lexer", "router", "storage", "auth", "metrics", "socket",
    "scheduler", "renderer", "codegen", "planner", "cache",
)


def build_fixture(line_count: int, seed: int = 1) -> str:
    """Return deterministic log-like output with one needle line near the middle.

    Lines are varied in vocabulary and length so embedding cost resembles real
    build or test output rather than a repeated short sentence.
    """
    if line_count < 1:
        raise ValueError("line_count must be positive")
    rng = random.Random(seed)
    needle_at = line_count // 2
    lines = []
    for index in range(line_count):
        if index == needle_at:
            lines.append(NEEDLE_LINE)
            continue
        template = rng.choice(_LINE_TEMPLATES)
        lines.append(
            template.format(
                ts=f"12:{(index // 60) % 60:02d}:{index % 60:02d}.{rng.randrange(1000):03d}",
                n=index,
                total=line_count,
                mod=rng.choice(_MODULES),
                ms=rng.randrange(1, 2500),
                kb=rng.randrange(1, 4096),
                w=rng.randrange(1, 9),
                d=rng.randrange(0, 64),
            )
        )
    return "\n".join(lines)


def needle_line_number(line_count: int) -> int:
    """Return the 1-based line number where the fixture places the needle."""
    return line_count // 2 + 1


def _needle_rank(result: dict[str, Any], line_count: int) -> int | None:
    """Return the 1-based rank of the first match whose core chunk holds the needle."""
    needle = needle_line_number(line_count)
    for rank, match in enumerate(result.get("matches", ()), start=1):
        start_text, _, end_text = match["matched_range"].partition("-")
        if int(start_text[1:]) <= needle <= int(end_text[1:]):
            return rank
    return None


@contextmanager
def _timed_semantic_index(engine: EphemeralEngine, sink: dict[str, float]) -> Iterator[None]:
    """Record how long lazy embedding materialization takes inside a search."""
    original = engine._ensure_embeddings

    def timed(capture):
        started = time.perf_counter()
        try:
            return original(capture)
        finally:
            sink["semantic_index_seconds"] += time.perf_counter() - started

    engine._ensure_embeddings = timed
    try:
        yield
    finally:
        engine._ensure_embeddings = original


def measure_once(
    engine: EphemeralEngine,
    line_count: int,
    mode: str = "hybrid",
    top_k: int = 5,
    seed: int = 1,
) -> dict[str, Any]:
    """Ingest one fresh capture and time the first and subsequent searches."""
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}")
    text = build_fixture(line_count, seed=seed)

    started = time.perf_counter()
    capture = engine.ingest(text, label=f"semantic-index-{line_count}")
    ingest_seconds = time.perf_counter() - started

    timing = {"semantic_index_seconds": 0.0}
    with _timed_semantic_index(engine, timing):
        started = time.perf_counter()
        first = engine.search(NEEDLE_QUERY, mode=mode, capture_id=capture.capture_id, top_k=top_k)
        first_search_seconds = time.perf_counter() - started

    started = time.perf_counter()
    engine.search(NEEDLE_QUERY, mode=mode, capture_id=capture.capture_id, top_k=top_k)
    subsequent_search_seconds = time.perf_counter() - started

    return {
        "line_count": line_count,
        "output_bytes": len(text.encode("utf-8")),
        "chunk_count": len(capture.chunks),
        "ingest_seconds": ingest_seconds,
        "semantic_index_seconds": timing["semantic_index_seconds"],
        "first_search_seconds": first_search_seconds,
        "subsequent_search_seconds": subsequent_search_seconds,
        "needle_rank": _needle_rank(first, line_count),
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


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return median and p95 timings plus throughput for one capture size."""
    if not samples:
        raise ValueError("samples must not be empty")
    first = samples[0]
    summary: dict[str, Any] = {
        "line_count": first["line_count"],
        "output_bytes": first["output_bytes"],
        "chunk_count": first["chunk_count"],
        "samples": len(samples),
        "needle_ranks": [sample["needle_rank"] for sample in samples],
    }
    for phase in PHASES:
        values = [float(sample[phase]) for sample in samples]
        summary[f"{phase}_median"] = statistics.median(values)
        summary[f"{phase}_p95"] = _nearest_rank(values)
    index_median = summary["semantic_index_seconds_median"]
    summary["chunks_per_second_median"] = (
        first["chunk_count"] / index_median if index_median > 0 else None
    )
    return summary


def run_benchmark(
    line_counts: tuple[int, ...],
    samples: int,
    mode: str = "hybrid",
    engine_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure semantic index and search latency for each capture size."""
    if not line_counts or any(line_count < 1 for line_count in line_counts):
        raise ValueError("line_counts must contain only positive values")
    if samples < 1:
        raise ValueError("samples must be positive")
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}")
    options = dict(engine_options or {})
    # Prefetch would hide the lazy indexing cost this benchmark exists to measure.
    options.setdefault("semantic_prefetch", False)
    options.setdefault("embedding_warmup", False)
    options.setdefault("max_captures", 4)

    engine = EphemeralEngine(**options)
    try:
        started = time.perf_counter()
        warmup = engine.ingest("semantic index benchmark warmup", label="semantic-index-warmup")
        engine._ensure_embeddings(warmup)
        model_load_seconds = time.perf_counter() - started

        by_size = []
        for line_count in line_counts:
            measurements = [
                measure_once(engine, line_count, mode=mode, seed=sample + 1)
                for sample in range(samples)
            ]
            by_size.append(summarize(measurements))
    finally:
        engine.shutdown()

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "embedding_model": engine.embedding_model_name,
        "embedding_threads": engine.embedding_threads,
        "test_embeddings": os.environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1",
        "engine_options": {key: value for key, value in options.items() if key != "max_captures"},
        "mode": mode,
        "samples": samples,
        "model_load_seconds": model_load_seconds,
        "measurements": by_size,
    }


def format_measurement(measurement: dict[str, Any]) -> str:
    """Return one human-readable summary line for a capture size."""
    throughput = measurement["chunks_per_second_median"]
    throughput_text = f"{throughput:.1f}" if throughput is not None else "n/a"
    return (
        f"lines={measurement['line_count']} chunks={measurement['chunk_count']} "
        f"ingest_median={measurement['ingest_seconds_median']:.6f}s "
        f"semantic_index_median={measurement['semantic_index_seconds_median']:.6f}s "
        f"first_search_median={measurement['first_search_seconds_median']:.6f}s "
        f"first_search_p95={measurement['first_search_seconds_p95']:.6f}s "
        f"subsequent_search_median={measurement['subsequent_search_seconds_median']:.6f}s "
        f"chunks_per_second={throughput_text} "
        f"needle_ranks={measurement['needle_ranks']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--line-counts",
        nargs="+",
        type=int,
        default=DEFAULT_LINE_COUNTS,
        help="Capture sizes to measure (default: 16 256 2048 8192)",
    )
    parser.add_argument("--samples", type=int, default=3, help="Fresh captures per size")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid", help="Search mode to time")
    parser.add_argument(
        "--embedding-model",
        help="FastEmbed model name to measure (default: EPHEMERAL_EMBEDDING_MODEL or the engine default)",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        help="ONNX Runtime thread count (default: EPHEMERAL_EMBEDDING_THREADS or the runtime default)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    engine_options = {}
    if args.embedding_model:
        engine_options["embedding_model_name"] = args.embedding_model
    if args.embedding_threads is not None:
        engine_options["embedding_threads"] = args.embedding_threads
    try:
        result = run_benchmark(
            tuple(args.line_counts), args.samples, mode=args.mode, engine_options=engine_options
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"model={result['embedding_model']} threads={result['embedding_threads']} "
        f"test_embeddings={result['test_embeddings']} "
        f"mode={result['mode']} model_load={result['model_load_seconds']:.3f}s"
    )
    for measurement in result["measurements"]:
        print(format_measurement(measurement))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
