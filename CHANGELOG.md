# Changelog

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
