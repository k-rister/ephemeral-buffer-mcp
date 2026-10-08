#!/usr/bin/env python3
"""Measure model-load and semantic-index RSS on a reproducible synthetic capture."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import threading
import time
from typing import Any, Callable

from ephemeral_buffer_mcp.config import DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES
from ephemeral_buffer_mcp.engine import EphemeralEngine, process_rss_bytes
import benchmarks.workload_results as wr


DEFAULT_LINE_COUNT = 4096
DEFAULT_LINE_BYTES = 63
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.005
BUDGET_SENTINEL_TEXT = "budget sentinel target"
BUDGET_SENTINEL_QUERY = BUDGET_SENTINEL_TEXT


def synthetic_capture(line_count: int, line_bytes: int) -> str:
    """Create fixed-width text with stable line and byte counts."""
    lines = []
    for index in range(line_count):
        body = f"line={index:05d} command=build status=success file=src/module_{index % 97:02d}.py"
        is_sentinel = index == line_count // 2 and line_bytes >= len(BUDGET_SENTINEL_TEXT)
        if is_sentinel:
            body = BUDGET_SENTINEL_TEXT
        lines.append((body[:line_bytes]).ljust(line_bytes, " " if is_sentinel else "x"))
    return "\n".join(lines) + "\n"


def matched_target_rank(search_result: dict[str, Any], target_line: int) -> int | None:
    """Return the one-based rank of the result range containing a target line."""
    for rank, match in enumerate(search_result.get("matches", []), start=1):
        start, end = match["matched_range"].split("-")
        if int(start[1:]) <= target_line <= int(end[1:]):
            return rank
    return None


def measure_rss_stage(
    action: Callable[[], Any], sample_interval: float
) -> tuple[Any, float, int | None]:
    """Run an action while sampling current RSS and return its sampled peak."""
    samples = []
    initial = process_rss_bytes()
    if initial is not None:
        samples.append(initial)
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(sample_interval):
            current = process_rss_bytes()
            if current is not None:
                samples.append(current)

    sampler = threading.Thread(target=sample, name="rss-sampler", daemon=True)
    sampler.start()
    started_at = time.perf_counter()
    try:
        result = action()
    finally:
        elapsed = time.perf_counter() - started_at
        stop.set()
        sampler.join()
    final = process_rss_bytes()
    if final is not None:
        samples.append(final)
    return result, elapsed, max(samples) if samples else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-count", type=int, default=DEFAULT_LINE_COUNT)
    parser.add_argument("--line-bytes", type=int, default=DEFAULT_LINE_BYTES)
    parser.add_argument("--semantic-chunk-lines", type=int, default=8)
    parser.add_argument("--semantic-chunk-bytes", type=int, default=1024)
    parser.add_argument("--model", help="FastEmbed model name; defaults to the EB model setting")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-index-input-bytes", type=int, default=DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES)
    parser.add_argument("--cpu-mem-arena", choices=("on", "off"), default="off")
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS)
    wr.add_result_argument(parser)
    return parser.parse_args()


def workload_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert the native memory record into the shared workload format."""
    rss = result["rss_bytes"]
    timing = result["timing_seconds"]
    input_data = result["input"]
    fallback = result["budget_fallback_validation"]
    index_status = result["semantic_index_status"]
    index_success = index_status in {"ready", "budget_exceeded"} and fallback["status"] != "failed"
    if index_success:
        errors = []
    elif fallback["status"] == "failed":
        errors = ["semantic-index fallback validation failed"]
    else:
        errors = [f"unexpected semantic-index status: {index_status}"]

    runs = [
        wr.run(
            "model-load",
            labels={"cache_state": "cold", "stage": "model_load"},
            measurements={
                "wall_time_seconds": wr.measurement(
                    "seconds", value=timing["model_load"], samples=1
                ),
                "peak_rss_bytes": wr.measurement(
                    "bytes", value=rss["model_load_sampled_peak"], samples=1
                ),
                "rss_delta_bytes": wr.measurement(
                    "bytes",
                    value=(
                        rss["after_model_load"] - rss["before_model_load"]
                        if rss["after_model_load"] is not None and rss["before_model_load"] is not None
                        else None
                    ),
                    samples=1,
                ),
            },
            phases=[wr.phase("model_load", median=timing["model_load"], samples=1)],
        ),
        wr.run(
            "semantic-index",
            labels={
                "line_count": input_data["line_count"],
                "stage": "semantic_index",
                "index_status": index_status,
                "fallback_validation": fallback["status"],
            },
            status="success" if index_success else "failure",
            measurements={
                "wall_time_seconds": wr.measurement(
                    "seconds", value=timing["semantic_index"], samples=1
                ),
                "input_bytes": wr.measurement("bytes", value=input_data["capture_bytes"]),
                "semantic_input_bytes": wr.measurement(
                    "bytes", value=input_data["semantic_input_bytes"]
                ),
                "semantic_chunk_count": wr.measurement(
                    "count", value=input_data["semantic_chunk_count"]
                ),
                "retained_embedding_bytes": wr.measurement(
                    "bytes", value=result["retained_embedding_bytes"]
                ),
                "peak_rss_bytes": wr.measurement(
                    "bytes", value=rss["semantic_index_sampled_peak"], samples=1
                ),
                "rss_delta_bytes": wr.measurement(
                    "bytes",
                    value=(
                        rss["after_semantic_index"] - rss["before_semantic_index"]
                        if rss["after_semantic_index"] is not None and rss["before_semantic_index"] is not None
                        else None
                    ),
                    samples=1,
                ),
                "rss_after_clear_bytes": wr.measurement(
                    "bytes", value=rss["after_capture_clear_and_gc"], samples=1
                ),
            },
            phases=[wr.phase("semantic_index", median=timing["semantic_index"], samples=1)],
            errors=errors,
        ),
    ]
    return wr.build_result(
        workload="semantic-memory",
        kind="benchmark",
        producer="benchmark_semantic_memory.py",
        producer_schema_version=result["schema_version"],
        parameters={"configuration": result["configuration"], "input": input_data},
        environment=wr.environment(
            embedding_model=result["configuration"]["embedding_model"],
            embedding_threads=result["configuration"]["embedding_threads"],
        ),
        runs=runs,
        status="success" if index_success else "failure",
        errors=errors,
        details=result,
    )


def format_report(result: dict[str, Any]) -> str:
    """Return a concise report for --result -, which reserves stdout for JSON."""
    rss = result["rss_bytes"]
    timing = result["timing_seconds"]
    configuration = result["configuration"]
    return (
        f"model={configuration['embedding_model']} lines={result['input']['line_count']} "
        f"semantic_index={result['semantic_index_status']} "
        f"fallback_validation={result['budget_fallback_validation']['status']} "
        f"model_load={timing['model_load']:.3f}s semantic_index_time={timing['semantic_index']:.3f}s "
        f"model_load_peak_rss_bytes={rss['model_load_sampled_peak']} "
        f"semantic_index_peak_rss_bytes={rss['semantic_index_sampled_peak']} "
        f"retained_embedding_bytes={result['retained_embedding_bytes']}"
    )


def main() -> int:
    args = parse_args()
    if min(args.line_count, args.line_bytes, args.semantic_chunk_lines,
           args.semantic_chunk_bytes, args.threads, args.batch_size,
           args.max_batch_tokens, args.max_index_input_bytes) < 1:
        raise SystemExit("counts, byte limits, thread count, and batch limits must be positive")
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive")

    text = synthetic_capture(args.line_count, args.line_bytes)
    engine_options = {
        "max_captures": 1,
        "semantic_prefetch": False,
        "embedding_warmup": False,
        "embedding_threads": args.threads,
        "embedding_batch_size": args.batch_size,
        "embedding_max_batch_tokens": args.max_batch_tokens,
        "embedding_cpu_mem_arena_enabled": args.cpu_mem_arena == "on",
        "semantic_max_index_input_bytes": args.max_index_input_bytes,
        "semantic_chunk_lines": args.semantic_chunk_lines,
        "semantic_chunk_bytes": args.semantic_chunk_bytes,
    }
    if args.model:
        engine_options["embedding_model_name"] = args.model

    engine = EphemeralEngine(**engine_options)
    try:
        capture = engine.ingest(text, label="semantic-memory-benchmark")
        bytes_before_model = process_rss_bytes()
        _, model_load_seconds, model_load_peak = measure_rss_stage(
            engine.load_embedding_model, args.sample_interval
        )
        bytes_after_model = process_rss_bytes()
        bytes_before_index = process_rss_bytes()
        index_status, index_seconds, index_peak = measure_rss_stage(
            lambda: engine.index_capture(capture.capture_id),
            args.sample_interval,
        )
        bytes_after_index = process_rss_bytes()
        if index_status not in {"ready", "budget_exceeded"}:
            raise RuntimeError(f"semantic indexing failed with status: {index_status}")
        capture_diagnostics = engine.get_capture_diagnostics(capture.capture_id)
        if capture_diagnostics is None:
            raise RuntimeError("capture diagnostics unavailable after semantic indexing")
        embedding_bytes = capture_diagnostics.retained_embedding_bytes
        semantic_chunk_count = capture_diagnostics.semantic_chunk_count
        semantic_input_bytes = capture_diagnostics.semantic_input_bytes
        if index_status == "budget_exceeded":
            target_line = args.line_count // 2 + 1
            if args.line_bytes < len(BUDGET_SENTINEL_TEXT):
                fallback_validation = {
                    "status": "skipped",
                    "reason": "line-bytes is too small for the fallback sentinel",
                }
            else:
                searches = {}
                for mode in ("hybrid", "semantic"):
                    started_at = time.perf_counter()
                    search_result = engine.search(
                        BUDGET_SENTINEL_QUERY,
                        mode=mode,
                        top_k=5,
                    )
                    searches[mode] = {
                        "seconds": time.perf_counter() - started_at,
                        "semantic_coverage": search_result.get("semantic_coverage"),
                        "semantic_fallback": search_result.get("semantic_fallback"),
                        "target_rank": matched_target_rank(search_result, target_line),
                    }
                checks_passed = all(
                    search["semantic_coverage"] == "unavailable"
                    and search["semantic_fallback"] == "SemanticIndexBudgetExceeded"
                    and search["target_rank"] is not None
                    for search in searches.values()
                )
                fallback_validation = {
                    "status": "passed" if checks_passed else "failed",
                    "query": BUDGET_SENTINEL_QUERY,
                    "target_line": target_line,
                    "searches": searches,
                }
        else:
            fallback_validation = {"status": "not_needed"}

        capture_id = capture.capture_id
        engine.clear(capture_id)
        del capture
        gc.collect()
        time.sleep(0.1)
        bytes_after_clear = process_rss_bytes()

        stats = engine.get_buffer_stats()
        result = {
            "schema_version": 1,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
            "configuration": {
                "embedding_model": stats["embedding_model"],
                "embedding_threads": stats["embedding_threads"],
                "embedding_batch_size": stats["embedding_batch_size"],
                "embedding_max_batch_tokens": stats["embedding_max_batch_tokens"],
                "embedding_cpu_mem_arena_enabled": stats["embedding_cpu_mem_arena_enabled"],
                "semantic_max_index_input_bytes": stats["semantic_max_index_input_bytes"],
                "semantic_chunk_lines": stats["semantic_chunk_lines"],
                "semantic_chunk_bytes": stats["semantic_chunk_bytes"],
                "rss_sample_interval_seconds": args.sample_interval,
            },
            "input": {
                "capture_bytes": len(text.encode("utf-8")),
                "line_count": capture_line_count(text),
                "semantic_chunk_count": semantic_chunk_count,
                "semantic_input_bytes": semantic_input_bytes,
            },
            "timing_seconds": {
                "model_load": model_load_seconds,
                "semantic_index": index_seconds,
            },
            "semantic_index_status": index_status,
            "budget_fallback_validation": fallback_validation,
            "rss_bytes": {
                "before_model_load": bytes_before_model,
                "model_load_sampled_peak": model_load_peak,
                "after_model_load": bytes_after_model,
                "before_semantic_index": bytes_before_index,
                "semantic_index_sampled_peak": index_peak,
                "after_semantic_index": bytes_after_index,
                "after_capture_clear_and_gc": bytes_after_clear,
            },
            "retained_embedding_bytes": embedding_bytes,
        }
        common_result = workload_result(result) if args.result is not None else None
        if wr.result_to_stdout(args.result):
            print(format_report(result), file=wr.report_stream(args.result))
        else:
            print(json.dumps(result, indent=2, sort_keys=True), file=wr.report_stream(args.result))
        if common_result is not None:
            wr.write_result(common_result, args.result, experiment=wr.experiment_from_args(args))
    finally:
        engine.shutdown()
    return 0


def capture_line_count(text: str) -> int:
    return text.count("\n")


if __name__ == "__main__":
    raise SystemExit(main())
