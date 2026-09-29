#!/usr/bin/env python3
"""Measure model-load and semantic-index RSS on a reproducible synthetic capture."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import threading
import time
from typing import Any, Callable

from config import DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES
from engine import EphemeralEngine, process_rss_bytes


DEFAULT_LINE_COUNT = 4096
DEFAULT_LINE_BYTES = 63
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.005


def synthetic_capture(line_count: int, line_bytes: int) -> str:
    """Create fixed-width text with stable line and byte counts."""
    lines = []
    for index in range(line_count):
        body = f"line={index:05d} command=build status=success file=src/module_{index % 97:02d}.py"
        lines.append((body[:line_bytes]).ljust(line_bytes, "x"))
    return "\n".join(lines) + "\n"


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.line_count, args.line_bytes, args.semantic_chunk_lines,
           args.semantic_chunk_bytes, args.threads, args.batch_size,
           args.max_batch_tokens, args.max_index_input_bytes) < 1:
        raise SystemExit("counts, byte limits, thread count, and batch limits must be positive")
    if args.sample_interval <= 0:
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
            engine._get_embedding_model, args.sample_interval
        )
        bytes_after_model = process_rss_bytes()
        bytes_before_index = process_rss_bytes()
        _, index_seconds, index_peak = measure_rss_stage(
            lambda: engine._ensure_embeddings(capture),
            args.sample_interval,
        )
        bytes_after_index = process_rss_bytes()
        embedding_bytes = int(capture.embeddings.nbytes) if capture.embeddings is not None else 0
        semantic_chunk_count = len(capture.semantic_chunks)
        semantic_input_bytes = sum(
            len(chunk.text.encode("utf-8", errors="replace"))
            for chunk in capture.semantic_chunks
        )
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
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        engine.shutdown()
    return 0


def capture_line_count(text: str) -> int:
    return text.count("\n")


if __name__ == "__main__":
    raise SystemExit(main())
