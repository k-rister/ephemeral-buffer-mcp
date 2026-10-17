#!/usr/bin/env python3
"""Verify the lexical-only distribution works without semantic dependencies."""

import importlib.util

from ephemeral_buffer_mcp.engine import EphemeralEngine
import ephemeral_buffer_mcp.server  # noqa: F401


def main() -> None:
    """Check imports, BM25, and semantic-mode errors in an isolated install."""
    for dependency in ("fastembed", "numpy"):
        if importlib.util.find_spec(dependency) is not None:
            raise AssertionError(f"lexical-only environment unexpectedly has {dependency}")

    engine = EphemeralEngine(embedding_warmup=True, semantic_prefetch=True)
    try:
        capture = engine.ingest("lexical needle remains searchable", label="lexical-smoke")
        result = engine.search("needle", mode="bm25", capture_id=capture.capture_id)
        if result.get("status") != "ok" or not result.get("matches"):
            raise AssertionError(f"BM25 search failed: {result}")

        for mode in ("hybrid", "semantic"):
            result = engine.search("needle", mode=mode, capture_id=capture.capture_id)
            if result.get("error_code") != "semantic_backend_unavailable":
                raise AssertionError(f"{mode} search did not report missing backend: {result}")

        stats = engine.get_buffer_stats()
        if stats["semantic_backend_available"]:
            raise AssertionError("semantic backend reported available without its dependencies")
        if stats["embedding_warmup_state"] != "unavailable":
            raise AssertionError(f"unexpected warm-up state: {stats['embedding_warmup_state']}")
    finally:
        engine.shutdown()

    print("Lexical-only package smoke passed.")


if __name__ == "__main__":
    main()
