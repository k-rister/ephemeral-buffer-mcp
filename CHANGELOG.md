# Changelog

## 0.1.1 - Unreleased

- Improve socket startup safety and per-session socket assignment.
- Add LRU capture eviction with a default capacity of 25 captures.
- Add command timeouts, process-group cleanup, and more accurate test-run
  signal handling.
- Expose embedding readiness in buffer statistics and close resources during
  shutdown.
- Offload socket ingestion from the asyncio event loop.
- Prepare trusted PyPI publishing with build checks, checksums, and provenance
  attestations.
