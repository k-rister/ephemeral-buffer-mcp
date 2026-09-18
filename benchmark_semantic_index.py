#!/usr/bin/env python3
"""Measure semantic indexing cost and first hybrid-search latency by capture size.

The first search on a fresh capture runs against the engine's semantic wait
budget, so the harness records whether it answered with complete or pending
semantic coverage and how the needle rank differs from a fully indexed search.
"""

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

import workload_results as wr
from engine import EphemeralEngine


SCHEMA_VERSION = 3
FIXTURE_VERSION = 2
DEFAULT_LINE_COUNTS = (16, 256, 2048, 8192)
SEARCH_MODES = ("semantic", "hybrid")
PHASES = (
    "ingest_seconds",
    "semantic_index_seconds",
    "first_search_seconds",
    "subsequent_search_seconds",
)

# Needles are phrased so lexical and semantic retrieval both have a fair
# chance: each query shares vocabulary with its needle without copying it.
NEEDLES = (
    {
        "id": "database-disconnect",
        "line": "ERROR worker-7 lost connection to postgres primary: connection reset by peer",
        "query": "database connection dropped",
    },
    {
        "id": "out-of-memory",
        "line": "FATAL allocator: cannot allocate 2147483648 bytes for tensor buffer, out of memory",
        "query": "allocation failed because RAM ran out",
    },
    {
        "id": "certificate-expiry",
        "line": "WARN  tls: certificate for api.internal expires in 3 days; renew before the rotation deadline",
        "query": "TLS cert about to expire",
    },
)

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


def needle_positions(line_count: int) -> list[tuple[int, int]]:
    """Return (needle_index, 1-based line number) pairs that fit in the fixture.

    Needles sit at one quarter, one half, and three quarters of the capture.
    Small captures keep only the needles whose positions are distinct.
    """
    if line_count < 1:
        raise ValueError("line_count must be positive")
    placed: list[tuple[int, int]] = []
    used: set[int] = set()
    for index in range(len(NEEDLES)):
        line_number = line_count * (index + 1) // 4 + 1
        if line_number <= line_count and line_number not in used:
            used.add(line_number)
            placed.append((index, line_number))
    return placed


def build_fixture(line_count: int, seed: int = 1) -> str:
    """Return deterministic log-like output with known needle lines.

    Lines are varied in vocabulary and length so embedding cost resembles real
    build or test output rather than a repeated short sentence.
    """
    positions = {line_number: index for index, line_number in needle_positions(line_count)}
    rng = random.Random(seed)
    lines = []
    for index in range(line_count):
        needle = positions.get(index + 1)
        if needle is not None:
            lines.append(NEEDLES[needle]["line"])
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


def _needle_rank(result: dict[str, Any], line_number: int) -> int | None:
    """Return the 1-based rank of the first match whose core range holds the line."""
    for rank, match in enumerate(result.get("matches", ()), start=1):
        start_text, _, end_text = match["matched_range"].partition("-")
        if int(start_text[1:]) <= line_number <= int(end_text[1:]):
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
    """Ingest one fresh capture, time the first and subsequent searches, and rank needles.

    The first search triggers lazy indexing and may return before the index is
    ready (``first_search_semantic_coverage`` is then ``"pending"``).  The
    harness then waits for the index and searches every needle, including the
    first one again, over the complete index.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}")
    text = build_fixture(line_count, seed=seed)
    positions = needle_positions(line_count)

    started = time.perf_counter()
    capture = engine.ingest(text, label=f"semantic-index-{line_count}")
    ingest_seconds = time.perf_counter() - started

    timing = {"semantic_index_seconds": 0.0}
    first_index, first_line = positions[0]
    with _timed_semantic_index(engine, timing):
        started = time.perf_counter()
        first = engine.search(
            NEEDLES[first_index]["query"], mode=mode, capture_id=capture.capture_id, top_k=top_k
        )
        first_search_seconds = time.perf_counter() - started
        index_state = engine.wait_for_semantic_index(capture)
    if index_state != "ready":
        raise RuntimeError(f"semantic index did not become ready: {index_state}")

    needle_ranks: dict[str, int | None] = {}
    subsequent_times = []
    for needle_index, line_number in positions:
        started = time.perf_counter()
        result = engine.search(
            NEEDLES[needle_index]["query"], mode=mode, capture_id=capture.capture_id, top_k=top_k
        )
        subsequent_times.append(time.perf_counter() - started)
        needle_ranks[NEEDLES[needle_index]["id"]] = _needle_rank(result, line_number)

    return {
        "line_count": line_count,
        "output_bytes": len(text.encode("utf-8")),
        "chunk_count": len(capture.chunks),
        "semantic_chunk_count": len(capture.semantic_chunks),
        "ingest_seconds": ingest_seconds,
        "semantic_index_seconds": timing["semantic_index_seconds"],
        "first_search_seconds": first_search_seconds,
        "first_search_semantic_coverage": first["semantic_coverage"],
        "first_search_needle_rank": _needle_rank(first, first_line),
        "subsequent_search_seconds": statistics.median(subsequent_times),
        "needle_ranks": needle_ranks,
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
    """Return median and p95 timings, throughput, and needle retrieval quality.

    ``needle_*`` quality covers searches over the complete index;
    ``first_search_*`` covers the first search, which may have answered with
    pending semantic coverage inside the wait budget.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    first = samples[0]
    ranks = [rank for sample in samples for rank in sample["needle_ranks"].values()]
    first_ranks = [sample["first_search_needle_rank"] for sample in samples]
    summary: dict[str, Any] = {
        "line_count": first["line_count"],
        "output_bytes": first["output_bytes"],
        "chunk_count": first["chunk_count"],
        "semantic_chunk_count": first["semantic_chunk_count"],
        "samples": len(samples),
        "needle_ranks": [sample["needle_ranks"] for sample in samples],
        "needle_hit_at_1": sum(1 for rank in ranks if rank == 1) / len(ranks),
        "needle_mrr": sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks),
        "first_search_semantic_coverage": [
            sample["first_search_semantic_coverage"] for sample in samples
        ],
        "first_search_pending_rate": sum(
            1 for sample in samples if sample["first_search_semantic_coverage"] == "pending"
        ) / len(samples),
        "first_search_needle_ranks": first_ranks,
        "first_search_needle_hit_at_1": sum(1 for rank in first_ranks if rank == 1) / len(first_ranks),
        "first_search_needle_mrr": sum(
            1.0 / rank for rank in first_ranks if rank is not None
        ) / len(first_ranks),
    }
    for phase in PHASES:
        values = [float(sample[phase]) for sample in samples]
        summary[f"{phase}_median"] = statistics.median(values)
        summary[f"{phase}_p95"] = _nearest_rank(values)
    index_median = summary["semantic_index_seconds_median"]
    summary["semantic_chunks_per_second_median"] = (
        first["semantic_chunk_count"] / index_median if index_median > 0 else None
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
        "semantic_chunking": {
            "lines": engine.semantic_chunk_lines,
            "bytes": engine.semantic_chunk_bytes,
            "overlap": engine.semantic_chunk_overlap,
        },
        "test_embeddings": os.environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1",
        "engine_options": {key: value for key, value in options.items() if key != "max_captures"},
        "mode": mode,
        "samples": samples,
        "semantic_wait_seconds": engine.semantic_wait_seconds,
        "model_load_seconds": model_load_seconds,
        "measurements": by_size,
    }


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (an unbounded wait budget) so strict JSON accepts them."""
    if isinstance(value, float) and not math.isfinite(value):
        return "unbounded"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def workload_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-agnostic workload result for a benchmark record."""
    runs = [
        wr.run(
            "model-load",
            labels={"cache_state": "cold"},
            measurements={
                "wall_time_seconds": wr.measurement(
                    "seconds",
                    value=result["model_load_seconds"],
                    samples=1,
                    note="embedding model load plus one tiny capture",
                ),
            },
        )
    ]
    for measurement in result["measurements"]:
        samples = measurement["samples"]
        runs.append(wr.run(
            f"lines-{measurement['line_count']}",
            labels={"line_count": measurement["line_count"], "mode": result["mode"], "cache_state": "warm"},
            measurements={
                "output_bytes": wr.measurement("bytes", value=measurement["output_bytes"]),
                "chunk_count": wr.measurement("count", value=measurement["chunk_count"]),
                "semantic_chunk_count": wr.measurement("count", value=measurement["semantic_chunk_count"]),
                "throughput_per_second": wr.measurement(
                    "per_second",
                    median=measurement["semantic_chunks_per_second_median"],
                    samples=samples,
                    note="semantic chunks embedded per second",
                ),
                "first_search_pending_rate": wr.measurement(
                    "ratio", value=measurement["first_search_pending_rate"], samples=samples
                ),
                "first_search_needle_hit_at_1": wr.measurement(
                    "score", value=measurement["first_search_needle_hit_at_1"], samples=samples
                ),
                "first_search_needle_mrr": wr.measurement(
                    "score", value=measurement["first_search_needle_mrr"], samples=samples
                ),
                "needle_hit_at_1": wr.measurement("score", value=measurement["needle_hit_at_1"], samples=samples),
                "needle_mrr": wr.measurement("score", value=measurement["needle_mrr"], samples=samples),
            },
            phases=[
                wr.phase(
                    field.removesuffix("_seconds"),
                    median=measurement[f"{field}_median"],
                    p95=measurement[f"{field}_p95"],
                    samples=samples,
                )
                for field in PHASES
            ],
        ))
    return wr.build_result(
        workload="semantic-index",
        kind="benchmark",
        producer="benchmark_semantic_index.py",
        producer_schema_version=result["schema_version"],
        fixture_version=result["fixture_version"],
        parameters=_json_safe({
            "line_counts": [measurement["line_count"] for measurement in result["measurements"]],
            "samples": result["samples"],
            "mode": result["mode"],
            "semantic_wait_seconds": result["semantic_wait_seconds"],
            "semantic_chunking": result["semantic_chunking"],
            "engine_options": result["engine_options"],
            "test_embeddings": result["test_embeddings"],
        }),
        environment=wr.environment(
            embedding_model=result["embedding_model"],
            embedding_threads=result["embedding_threads"],
        ),
        runs=runs,
        details=_json_safe(result),
    )


def format_measurement(measurement: dict[str, Any]) -> str:
    """Return one human-readable summary line for a capture size."""
    throughput = measurement["semantic_chunks_per_second_median"]
    throughput_text = f"{throughput:.1f}" if throughput is not None else "n/a"
    return (
        f"lines={measurement['line_count']} chunks={measurement['chunk_count']} "
        f"semantic_chunks={measurement['semantic_chunk_count']} "
        f"ingest_median={measurement['ingest_seconds_median']:.6f}s "
        f"semantic_index_median={measurement['semantic_index_seconds_median']:.6f}s "
        f"first_search_median={measurement['first_search_seconds_median']:.6f}s "
        f"first_search_p95={measurement['first_search_seconds_p95']:.6f}s "
        f"subsequent_search_median={measurement['subsequent_search_seconds_median']:.6f}s "
        f"semantic_chunks_per_second={throughput_text} "
        f"first_search_pending_rate={measurement['first_search_pending_rate']:.2f} "
        f"first_search_needle_mrr={measurement['first_search_needle_mrr']:.2f} "
        f"needle_hit_at_1={measurement['needle_hit_at_1']:.2f} "
        f"needle_mrr={measurement['needle_mrr']:.2f}"
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
    parser.add_argument("--semantic-chunk-lines", type=int, help="Maximum lines per semantic window")
    parser.add_argument("--semantic-chunk-bytes", type=int, help="UTF-8 byte cap per semantic window")
    parser.add_argument("--semantic-chunk-overlap", type=int, help="Lines shared by consecutive windows")
    parser.add_argument(
        "--semantic-wait-seconds",
        type=float,
        help=(
            "Hybrid wait budget for the semantic index before answering lexical-first "
            "(default: EPHEMERAL_SEMANTIC_WAIT_SECONDS or the engine default; 'inf' waits for the index)"
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    wr.add_result_argument(parser)
    args = parser.parse_args()

    engine_options = {}
    if args.embedding_model:
        engine_options["embedding_model_name"] = args.embedding_model
    for option in (
        "embedding_threads",
        "semantic_chunk_lines",
        "semantic_chunk_bytes",
        "semantic_chunk_overlap",
        "semantic_wait_seconds",
    ):
        value = getattr(args, option)
        if value is not None:
            engine_options[option] = value
    try:
        result = run_benchmark(
            tuple(args.line_counts), args.samples, mode=args.mode, engine_options=engine_options
        )
    except ValueError as exc:
        parser.error(str(exc))

    chunking = result["semantic_chunking"]
    report = wr.report_stream(args.result)
    print(
        f"model={result['embedding_model']} threads={result['embedding_threads']} "
        f"semantic_chunking=lines:{chunking['lines']}/bytes:{chunking['bytes']}/overlap:{chunking['overlap']} "
        f"test_embeddings={result['test_embeddings']} "
        f"mode={result['mode']} semantic_wait={result['semantic_wait_seconds']:g}s "
        f"model_load={result['model_load_seconds']:.3f}s",
        file=report,
    )
    for measurement in result["measurements"]:
        print(format_measurement(measurement), file=report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.result:
        wr.write_result(workload_result(result), args.result)


if __name__ == "__main__":
    main()
