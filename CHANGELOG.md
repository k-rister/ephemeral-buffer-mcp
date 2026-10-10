# Changelog

## Unreleased

- Raise the default retained-capture limit from 25 to 256.
- Upgrade the MCP Python SDK to v2.3.0 and advertise five-minute cache hints for
  the deterministic tool catalog to modern clients, with modern and legacy
  protocol caching coverage.

## 0.9.0 - 2026-10-09

- Install the generic and Codex shell launchers with the PyPI package, and
  document the Codex MCP config update required for the package server entry
  point.
- Compact MCP text results by default to avoid repeating capture payloads and
  search snippets already available in structured results. Set
  `EPHEMERAL_COMPACT_TOOL_RESULTS=0` to restore legacy text output.

## 0.8.0 - 2026-10-08

- Move the runtime into the `ephemeral_buffer_mcp` package and organize
  benchmarks, tests, and release tooling into dedicated directories.
- Establish `create_mcp_server` and `create_service_context` as the supported
  Python embedding API. Previous imports from root-level implementation
  modules are no longer available; installed commands and MCP behavior remain
  available.

## 0.7.0 - 2026-10-07

- Record per-criterion phrase exposure in successful MCP search and capture-slice
  responses without retaining response text, summarize retrieval-to-answer
  outcomes by phrase-hit, response-without-hit, and no-successful-response state,
  report per-task success rates and paired deltas, and let release benchmark
  collection run only the agent A/B suite with an independent repetition count.
- Add a controlled MCP and socket admission saturation benchmark with latency,
  throughput, queue occupancy, and busy-rejection measurements; run it in the
  scheduled and manually dispatched benchmark workflow.
- Report unavailable routing ratios safely, write strict semantic-index
  benchmark JSON for unbounded wait budgets, preserve benchmark server errors,
  and include worker output when warm-up subprocesses fail.
- Run summary signal scans outside the engine-wide lock, preserve the available
  output budget when invalid UTF-8 expands during decoding, report effective
  consolidation limits, and make metrics accounting and process cleanup more
  reliable.
- Harden durable execution records against listing races and non-finite JSON,
  preserve failed-phase retry gates when cancellation arrives before a retry,
  invalidate stale summaries after failed writes, and document per-phase output
  paging.
- Return a frozen metadata view from `ingest` and `get_capture`, keep the
  historical lookup as a compatibility wrapper, and expose supported
  semantic-index operations and diagnostics for Python integrations and
  benchmarks.
- Isolate FastMCP private API access behind a compatibility adapter, delegate
  argument parsing and validation to the SDK, report validation-metrics
  limitations without interrupting tool dispatch, and test the minimum and
  latest supported FastMCP 1.x versions.
- Bound MCP and socket foreground work with configurable active slots and
  queues, return explicit busy errors on saturation, let the CLI stop a large
  socket upload when rejected, report admission and reader-pinned storage
  counters, and apply search response budgets while constructing match context.
- Add byte-bounded, resumable capture-slice pages and a versioned structured
  MCP result envelope for capture, search, retrieval, and clear tools while
  retaining their readable text responses.
- Validate optional workload-result environment fields against the published
  schema so malformed tool, machine, CPU-count, and source-revision metadata is
  rejected before listing or comparison consumers inspect it.
- Reject unknown agent A/B baseline threshold directions and nonboolean gate
  values in generated, file-loaded, and in-memory baselines.
- Reject nonfinite concurrency baseline throughput rates before regression
  comparisons so NaN or infinity cannot disable the performance gate.
- Give search-relevance fixtures four competing semantic windows and score
  expected matched ranges, so query-insensitive ranking can fail the baseline.
- Keep BM25 effectiveness benchmarks isolated from semantic prefetch work and
  shut down benchmark engines after each run. Reject nonfinite agent A/B
  measurements, use unique default run directories, clean up interrupted Codex
  process groups, and count response signals from stdout only.

## 0.6.3 - 2026-09-30

- Require urllib3 2.8.0 or newer at runtime so package upgrades replace
  vulnerable versions and receive the upstream fixes for CVE-2026-97687 and
  CVE-2026-97689.

## 0.6.2 - 2026-09-29

- Bound semantic-index memory work with token-length bucketed embedding batches,
  incremental embedding-matrix assembly, a per-capture semantic-input budget,
  and configurable ONNX CPU memory-arena use. Captures above the default 4 MiB
  semantic-input budget retain BM25 and slice access while semantic and hybrid
  search report unavailable semantic coverage and return lexical results.
- Add `benchmark_semantic_memory.py` to report RSS across model loading,
  semantic indexing, and capture cleanup.

## 0.6.1 - 2026-09-23

- Add automatic private JSONL session logs to the agent launchers, with a
  configurable `--log-level` and the active log path and effective level in
  runtime diagnostics. Log files are created with owner-only permissions and
  symlink redirection is rejected.
- Preserve valid TOML configuration when Codex launcher paths contain
  supplementary Unicode characters.

## 0.6.0 - 2026-09-22

- Add privacy-safe MCP transport-session attribution to usage metrics. MCP
  snapshots now report an opaque session scope and isolate coverage, funnel,
  byte, and per-tool counters between clients sharing one process, while
  persisted snapshots retain the aggregate process view. Per-session state is
  bounded and evicted capture IDs are removed from every scope.
- Add bounded per-tool latency distributions and content-free failure
  categories for validation, timeout, socket, embedding, eviction, and other
  operational failures, including coherent task-window deltas.
- Add content-free derived workflow-effectiveness signals with documented
  denominators, zero-denominator handling, response-byte ratios, and
  search/retrieval reduction metrics.
- Add scope-local snapshot tokens and non-resetting task-window deltas to
  `get_usage_metrics()`, distinguishing valid zero-activity windows from
  unavailable tokens across bounded history, MCP sessions, and server
  restarts.
- Add descriptive interface coverage by primary capability category, while
  retaining the complete tool-level coverage and unused-tool list.
- Derive interface coverage from the live MCP registration inventory so the
  available and unused tool lists cannot silently drift from the exposed API.
- Add `get_usage_metrics()`, a versioned JSON MCP interface for content-free
  usage metrics, including scope-lifetime measurement timestamps.
- Add content-free `interface_coverage` metrics to report the number and
  percentage of exposed MCP tools used during the active metrics scope and the
  complete unused-tool list.
- Session-aware coding-agent launchers now default `EPHEMERAL_METRICS=1` for
  their private server sessions, while preserving `EPHEMERAL_METRICS=0` as an
  explicit opt-out. Direct and shared server launches remain opt-in.

## 0.5.0 - 2026-09-18

- Reject duplicate `--line-counts` values in `benchmark_latency.py`,
  `benchmark_semantic_index.py`, and `benchmark_routing.py` before any
  measurement runs, since each size becomes a workload run id and a repeated
  size made `--result` fail with a duplicate-id error after the benchmark had
  finished. The workload result JSON Schema now pins canonical measurement
  names to their unit and rejects identical runs, matching the reference
  validator; run-id uniqueness beyond that stays with
  `workload_results.validate_result`, and the schema and documentation say so.
- Run on-demand semantic indexing on a pool bounded by
  `EPHEMERAL_SEMANTIC_PREFETCH_WORKERS` instead of one thread per capture, and
  cancel queued on-demand jobs when their capture is evicted or cleared, so
  hybrid searches that exceed the wait budget on captures that are then
  evicted no longer accumulate indexing threads beyond the bound. A hybrid
  search whose capture is evicted while it waits answers lexical-first within
  its budget instead of indexing the evicted capture inline, and
  `get_buffer_stats` reports queued on-demand jobs separately as
  `semantic_index_on_demand_queued`. Benchmarks shut their engines down so
  background indexing never delays process exit. A capture
  pulled out of the prefetch queue whose indexing then fails is marked `failed`
  and retries through the lazy path. Empty captures now report
  `semantic_coverage` (`complete` for semantic and hybrid searches,
  `not-requested` for BM25) like every other search response.
- Add a versioned, tool-agnostic workload result format
  (`coding-agent-workload-result`, format version 1) with a reference
  validator and JSON Schema in `workload_results.py` and
  `workload_result.schema.json`. Every benchmark, evaluation, and the Codex
  A/B runner accept `--result PATH` (`-` for stdout) to emit it alongside
  their existing reports, so latency, phase timings, output volume, token
  estimates, resource use, and success status can be compared across
  producers without reading producer-specific records.
- Add `compare_workload_results.py`, which compares two or more workload
  result documents (or single runs selected with `PATH#RUN_ID`) without
  rerunning them: it reports absolute and percentage deltas per run,
  measurement, phase, and statistic, classifies each as improved, regressed,
  changed, unchanged, missing, or incompatible, shows the parameter and
  environment differences that explain a delta, filters runs by label and
  metrics by name, and offers JSON output plus a `--check` mode that fails on
  regressions or non-success results for automated regression checks.
  Non-finite `--tolerance` values (`nan`, `inf`) are rejected as invalid
  input instead of failing with a traceback while writing the comparison.
- Add experiment groups and metadata to workload results: every benchmark,
  evaluation, and the Codex A/B runner accept `--experiment GROUP`,
  `--metadata KEY=VALUE`, and `--redact KEY`, recorded in an optional
  `experiment` block that consumers read alongside run summaries. Metadata
  keys that name credentials are always stored as `[redacted]`. Add
  `list_workload_results.py` to find result documents under directories and
  list or filter them by group, metadata, workload, and status (per document
  or per run, keeping failed and invalid documents visible), and teach
  `compare_workload_results.py` `DIR@GROUP` and `DIR@GROUP,KEY=VALUE`
  references plus an `experiment differences` line. `run_agent_ab_experiment.sh`
  now writes workload result documents tagged from `AGENT_AB_EXPERIMENT` and
  `AGENT_AB_VARIANT`.
- Add a semantic-index latency benchmark that reports ingestion, lazy embedding
  materialization, first and subsequent hybrid or semantic search timings,
  indexing throughput, and needle rank by capture size with the real model.
- Use the upstream fp32 ONNX export of bge-small-en-v1.5, registered with
  FastEmbed as `BAAI/bge-small-en-v1.5-fp32`, as the default embedding model.
  It produces identical vectors to FastEmbed's reduced-precision catalogue file
  but its kernels parallelize. A release-preparation comparison on a 16-vCPU
  Linux x86_64 host measured 69.1 semantic chunks per second and 1.85 seconds
  median indexing for the fp32 export versus 3.1 chunks per second and 41.15
  seconds for the catalogue file at 1,024 lines (about 22x); these are
  host-specific diagnostic measurements, not universal guarantees. The
  catalogue file remains selectable as `BAAI/bge-small-en-v1.5`. First use
  downloads about 130 MB instead of 67 MB.
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
