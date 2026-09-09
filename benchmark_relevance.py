#!/usr/bin/env python3
"""Evaluate deterministic BM25, semantic, and hybrid search relevance."""

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from engine import EphemeralEngine


SCHEMA_VERSION = 1
MODES = ("bm25", "semantic", "hybrid")


def relevance_cases() -> list[dict[str, Any]]:
    """Return privacy-safe fixtures with one expected relevant marker each."""
    return [
        {
            "id": "exact-database-error",
            "query": "database connection refused",
            "marker": "ERROR database connection refused",
            "lines": [
                "INFO service started",
                "INFO accepting requests",
                "WARN retry budget is low",
                "ERROR database connection refused retryable=true",
                "INFO retry scheduled",
                "INFO health check complete",
            ],
        },
        {
            "id": "punctuation-error-token",
            "query": "ECONNREFUSED:",
            "marker": "ECONNREFUSED: endpoint unavailable",
            "lines": [
                "INFO checking cache endpoint",
                "WARN cache request delayed",
                "ERROR ECONNREFUSED: endpoint unavailable",
                "INFO fallback enabled",
                "INFO request completed",
            ],
        },
        {
            "id": "semantic-network-disconnect",
            "query": "where did the network disconnect?",
            "marker": "remote host closed TCP connection",
            "lines": [
                "INFO replication sync started",
                "INFO primary node healthy",
                "WARN IO stream unexpectedly terminated; remote host closed TCP connection",
                "INFO fallback replica selected",
                "INFO replication sync resumed",
            ],
        },
        {
            "id": "lexical-semantic-conflict",
            "query": "payment test failure card declined",
            "marker": "PaymentGateway: received 402 Payment Required",
            "lines": [
                "INFO payment test suite started",
                "INFO payment authorization request sent",
                "INFO payment test completed successfully for test card",
                "FAIL PaymentGateway: received 402 Payment Required",
                "DETAIL card declined by issuing bank",
                "INFO retry disabled",
            ],
        },
    ]


def _marker_rank(matches: list[dict[str, Any]], marker: str) -> int | None:
    """Return the one-based rank of the first result containing marker."""
    for rank, match in enumerate(matches, start=1):
        if marker in match.get("context", "") or marker in match.get("snippet", ""):
            return rank
    return None


def run_relevance_benchmark(top_k: int = 3) -> dict[str, Any]:
    """Evaluate all modes against the deterministic relevance corpus."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    cases = relevance_cases()
    engine = EphemeralEngine(max_captures=len(cases))
    captures = {
        case["id"]: engine.ingest("\n".join(case["lines"]), label=f"relevance-{case['id']}")
        for case in cases
    }
    records = []
    for mode in MODES:
        for case in cases:
            result = engine.search(
                case["query"], mode=mode, capture_id=captures[case["id"]].capture_id,
                top_k=top_k, context_lines=1,
            )
            rank = _marker_rank(result.get("matches", []), case["marker"])
            records.append({
                "case": case["id"],
                "mode": mode,
                "query": case["query"],
                "expected_marker": case["marker"],
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_k": rank is not None and rank <= top_k,
                "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
            })

    summaries = {}
    for mode in MODES:
        mode_records = [record for record in records if record["mode"] == mode]
        summaries[mode] = {
            "queries": len(mode_records),
            "hit_at_1": sum(record["hit_at_1"] for record in mode_records) / len(mode_records),
            "hit_at_k": sum(record["hit_at_k"] for record in mode_records) / len(mode_records),
            "mrr": sum(record["reciprocal_rank"] for record in mode_records) / len(mode_records),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "search-relevance",
        "evaluation": "deterministic-synthetic-corpus",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "top_k": top_k,
        "controls": {
            "fixtures": "deterministic synthetic output with explicit expected markers",
            "embedding_model": "EPHEMERAL_TEST_EMBEDDINGS when enabled by caller",
            "telemetry": "none; all measurements are local",
            "scope": "retrieval relevance only; no agent answer quality or token usage",
        },
        "records": records,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_relevance_benchmark(args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
