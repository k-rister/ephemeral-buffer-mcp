# Changelog

## Unreleased

- Add a versioned, tool-agnostic workload result format
  (`coding-agent-workload-result`, format version 1) with a reference
  validator and JSON Schema in `workload_results.py` and
  `workload_result.schema.json`. Every benchmark, evaluation, and the Codex
  A/B runner accept `--result PATH` (`-` for stdout) to emit it alongside
  their existing reports, so latency, phase timings, output volume, token
  estimates, resource use, and success status can be compared across
  producers without reading producer-specific records.
- Add a semantic-index latency benchmark that reports ingestion, lazy embedding
  materialization, first and subsequent hybrid or semantic search timings,
  indexing throughput, and needle rank by capture size with the real model.
- Use the upstream fp32 ONNX export of bge-small-en-v1.5, registered with
  FastEmbed as `BAAI/bge-small-en-v1.5-fp32`, as the default embedding model.
  It produces identical vectors to FastEmbed's reduced-precision catalogue file
  but its kernels parallelize, indexing 23x faster on a Linux x86_64 host and
  3x faster on Apple silicon; the catalogue file remains selectable as
  `BAAI/bge-small-en-v1.5`. First use downloads about 130 MB instead of 67 MB.
- Add `EPHEMERAL_EMBEDDING_THREADS` to bound ONNX Runtime threads for embedding
  inference; the runtime default is fastest but uses every core.
- Index semantic embeddings over windows packed separately from the BM25
  sliding grid, bounded by `EPHEMERAL_SEMANTIC_CHUNK_LINES` (default 8),
  `EPHEMERAL_SEMANTIC_CHUNK_BYTES` (default 1024), and
  `EPHEMERAL_SEMANTIC_CHUNK_OVERLAP` (default 0), fuse hybrid results in line
  space so exact BM25 ranges are preserved, and report `chunk_index` per match.
  This stops embedding the two-line overlap twice and roughly halves first
  semantic or hybrid search latency on large captures.
- Replace best-effort semantic prefetch admission with a queue drained
  newest-first by the bounded worker pool, so ingestion bursts never silently
  leave captures unprefetched; searches index a still-queued capture inline,
  and diagnostics report queued and running counts.
- Enable semantic prefetch by default so the first semantic or hybrid search
  after a capture usually finds the index ready; set
  `EPHEMERAL_SEMANTIC_PREFETCH=0` to restore lazy-only indexing.
- Bound first hybrid-search latency on very large captures with
  `EPHEMERAL_SEMANTIC_WAIT_SECONDS` (default 10): hybrid search waits at most
  the budget for a capture's semantic index, then returns BM25 results marked
  `semantic_coverage: pending` while a background job finishes indexing, so
  repeating the search is fully hybrid. Responses always report
  `semantic_coverage` (`complete`, `pending`, `unavailable`, or
  `not-requested`); semantic mode still waits for the index. Searches on a
  queued or unindexed capture now index it on a dedicated thread instead of
  inline, `get_buffer_stats` reports the budget and on-demand job count, and
  `benchmark_semantic_index.py` records the first search's coverage and needle
  rank separately from complete-index quality.
- Add durable phase-level executions with persisted outputs, metrics, status
  history, restart recovery, explicit retries, and unsafe-side-effect resume
  confirmation for long-running agent workflows, including recoverable compact
  responses when detailed execution metadata exceeds the tool response budget,
  bounded list-page fallbacks, owner checks, and crash-durable directory sync.

## 0.4.0 - 2026-09-15

- Warm the embedding model asynchronously after server readiness by default,
  expose warm-up readiness and failures, and preserve lexical hybrid results
  when semantic initialization is unavailable.
- Add a warm-up benchmark covering startup readiness, first-query latency, and
  process RSS impact, with an opt-out for lexical-only deployments.
- Add explicit versioned length-prefix framing for socket requests and
  responses, including bounded and malformed-frame handling.
- Add opt-in runtime semantic-index budget adjustment with deterministic LRU
  eviction, diagnostics, concurrency coverage, and documentation.
- Add opt-in, session-scoped data-path byte counters to local diagnostics for
  capture input/retention, tool responses, and framed socket traffic.
- Extend the Codex A/B records to collect per-run data-path byte counters from
  content-free MCP metrics snapshots.
- Persist benchmark metrics snapshots after each MCP tool call so subprocess
  shutdown still leaves counters available for comparison runs.

## 0.3.1 - 2026-09-13

- Package Codex and generic agent session launchers alongside `ephbuf`.
- Allow explicitly configured stdio-only operation when the host denies Unix
  socket creation while preserving dual-mode startup by default.

## 0.3.0 - 2026-09-11

- Harden capture, search, consolidation, socket, timeout, shutdown, and
  process-group handling with stricter bounds and cleanup guarantees.
- Improve diff parsing, fallback tokenization, hybrid search context handling,
  semantic prefetch behavior, and diagnostic signal accuracy.
- Add repository-shaped agent A/B evaluation fixtures, repeatable Codex
  experiment tooling, provider-usage diagnostics, and privacy-safe baselines.
- Expand release, benchmark, lifecycle, and operational coverage and refresh
  the documented effectiveness measurements.

## 0.2.0 - 2026-09-07

- Add bounded `consolidate_captures` support for building one searchable JSON
  view over multiple active captures while preserving source IDs and line
  numbers.
- Add opt-in, content-free local usage metrics and aggregate diagnostics for
  observing capture, search, retrieval, eviction, cleanup, and process events.
- Add paired and consolidated effectiveness benchmarks, including reproducible
  README results that quantify context-size reductions and document benchmark
  limitations.

## 0.1.2 - 2026-09-06

- Add opt-in runtime diagnostics for content-free version, platform, socket,
  buffer, embedding, and memory reporting.
- Add privacy-safe structured operational logging for timeouts, process
  termination, embedding readiness and failures, limits, eviction, cleanup,
  and socket failures.
- Document field-observation procedures and add a privacy-conscious GitHub
  issue form for reporting real-world behavior.
- Expand project guidance and defensive-path coverage for ongoing maintenance.

## 0.1.1 - 2026-09-05

- Improve socket startup safety and per-session socket assignment.
- Add LRU capture eviction with a default capacity of 25 captures.
- Add command timeouts, process-group cleanup, and more accurate test-run
  signal handling.
- Expose embedding readiness in buffer statistics and close resources during
  shutdown.
- Offload socket ingestion from the asyncio event loop.
- Prepare trusted PyPI publishing with build checks, checksums, and provenance
  attestations.
