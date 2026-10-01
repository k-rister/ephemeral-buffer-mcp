#!/usr/bin/env python3
"""Evaluate deterministic BM25, semantic, and hybrid search relevance."""

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import workload_results as wr
from engine import EphemeralEngine


SCHEMA_VERSION = 1
FIXTURE_VERSION = 2
SEMANTIC_CANDIDATE_LINES = 8
MODES = ("bm25", "semantic", "hybrid")
METRICS = ("hit_at_1", "hit_at_k", "mrr")
DEFAULT_TOLERANCES = {metric: 0.05 for metric in METRICS}


def _validate_tolerances(tolerances: dict[str, Any]) -> None:
    """Reject unknown, negative, or non-finite relevance tolerances."""
    for metric, value in tolerances.items():
        if metric not in METRICS:
            raise ValueError(f"relevance baseline has unknown tolerance metric {metric!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"relevance baseline tolerance {metric} must be finite and nonnegative")
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            numeric = math.inf
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"relevance baseline tolerance {metric} must be finite and nonnegative")


def relevance_cases() -> list[dict[str, Any]]:
    """Return synthetic cases with competing windows and explicit expected ranges."""

    def case(
        case_id: str,
        query: str,
        marker: str,
        candidates: list[list[str]],
        relevant_candidate: int,
        expected_lines: tuple[int, int],
    ) -> dict[str, Any]:
        lines = [line for candidate in candidates for line in candidate]
        expected_range = {
            "start_line": (
                relevant_candidate * SEMANTIC_CANDIDATE_LINES + expected_lines[0]
            ),
            "end_line": (
                relevant_candidate * SEMANTIC_CANDIDATE_LINES + expected_lines[1]
            ),
        }
        return {
            "id": case_id,
            "query": query,
            "expected_marker": marker,
            "expected_range": expected_range,
            "lines": lines,
        }

    return [
        case(
            "exact-database-error",
            "database connection refused",
            "ERROR database connection refused",
            [
                [
                    "INFO telemetry exporter started", "INFO scrape cycle completed",
                    "WARN storage watermark near threshold", "INFO metrics archive rotated",
                    "DETAIL counter shard four active", "INFO exporter flush successful",
                    "INFO dashboard cache refreshed", "INFO telemetry cycle complete",
                ],
                [
                    "INFO database client initialized", "INFO connection pool warmed",
                    "ERROR database connection refused retryable=true", "WARN reconnect backoff selected",
                    "INFO service retained request", "DETAIL retry budget remains",
                    "INFO health probe passed", "INFO diagnostic bundle ready",
                ],
                [
                    "INFO climate controller booted", "INFO thermostat reports stable",
                    "WARN ventilation threshold adjusted", "INFO cooling cycle entered idle",
                    "DETAIL ambient reading twenty one", "INFO fan speed returned to nominal",
                    "INFO climate record archived", "INFO thermal loop ready",
                ],
                [
                    "INFO batch compactor started", "INFO segment manifest loaded",
                    "DETAIL shard count remains constant", "INFO checksum verification passed",
                    "WARN archival queue reached half capacity", "INFO object store sync completed",
                    "INFO compaction checkpoint written", "INFO maintenance cycle complete",
                ],
            ],
            relevant_candidate=1,
            expected_lines=(3, 4),
        ),
        case(
            "punctuation-error-token",
            "ECONNREFUSED:",
            "ECONNREFUSED: endpoint unavailable",
            [
                [
                    "INFO image cache mounted", "INFO thumbnail worker ready",
                    "WARN cache refresh delayed", "INFO local asset served",
                    "DETAIL image dimensions validated", "INFO cache entry retained",
                    "INFO thumbnail queue drained", "INFO image pipeline idle",
                ],
                [
                    "INFO credential verifier initialized", "INFO token signature accepted",
                    "DETAIL account policy loaded", "INFO access scope evaluated",
                    "INFO audit record appended", "WARN session expires soon",
                    "INFO authentication response sent", "INFO identity check complete",
                ],
                [
                    "INFO outbound endpoint probe started", "DETAIL socket descriptor opened",
                    "ERROR ECONNREFUSED: endpoint unavailable", "INFO retry delay selected",
                    "WARN alternate endpoint configured", "INFO fallback path enabled",
                    "DETAIL request remains queued", "INFO endpoint probe complete",
                ],
                [
                    "INFO scheduler heartbeat received", "INFO worker lease renewed",
                    "DETAIL queue depth measured", "INFO background task resumed",
                    "WARN worker utilization increased", "INFO task checkpoint persisted",
                    "INFO scheduler cycle complete", "INFO lease monitor idle",
                ],
            ],
            relevant_candidate=2,
            expected_lines=(3, 4),
        ),
        case(
            "semantic-network-disconnect",
            "where did the network disconnect?",
            "remote host closed TCP connection",
            [
                [
                    "INFO audio mixer initialized", "INFO channel levels normalized",
                    "DETAIL equalizer profile selected", "INFO playback buffer filled",
                    "WARN output volume adjusted", "INFO stereo stream active",
                    "INFO audio frame rendered", "INFO mixer cycle complete",
                ],
                [
                    "INFO thermal sensor sampled", "DETAIL cooling fan remains steady",
                    "INFO voltage regulator checked", "WARN battery threshold updated",
                    "INFO power profile applied", "DETAIL temperature within range",
                    "INFO energy report stored", "INFO sensor cycle complete",
                ],
                [
                    "INFO document renderer started", "INFO page layout selected",
                    "DETAIL font cache loaded", "INFO vector layer rasterized",
                    "WARN output scale rounded", "INFO preview image written",
                    "INFO render queue drained", "INFO document task complete",
                ],
                [
                    "INFO replica synchronization started", "INFO primary node remained healthy",
                    "WARN IO stream unexpectedly terminated", "DETAIL remote host closed TCP connection",
                    "INFO fallback replica selected", "INFO replication resumed from checkpoint",
                    "DETAIL pending segment count recovered", "INFO synchronization cycle complete",
                ],
            ],
            relevant_candidate=3,
            expected_lines=(3, 4),
        ),
        case(
            "lexical-semantic-conflict",
            "payment test failure card declined",
            "PaymentGateway: received 402 Payment Required",
            [
                [
                    "INFO renderer started", "INFO frame palette loaded",
                    "DETAIL canvas dimensions accepted", "INFO texture atlas prepared",
                    "WARN animation frame skipped", "INFO scene graph updated",
                    "INFO display refresh completed", "INFO renderer idle",
                ],
                [
                    "INFO catalog snapshot loaded", "DETAIL product index verified",
                    "INFO inventory cache refreshed", "WARN catalog page took longer",
                    "INFO search index checkpointed", "DETAIL stock record normalized",
                    "INFO catalog response assembled", "INFO inventory task complete",
                ],
                [
                    "INFO integration suite started", "INFO authorization request dispatched",
                    "DETAIL issuer response received", "FAIL PaymentGateway: received 402 Payment Required",
                    "DETAIL card declined by issuing bank", "INFO retry disabled by policy",
                    "INFO failure summary recorded", "INFO integration task complete",
                ],
                [
                    "INFO media transcoder started", "INFO source track inspected",
                    "DETAIL codec profile selected", "WARN bitrate target adjusted",
                    "INFO segment encoder warmed", "INFO output container opened",
                    "INFO media manifest written", "INFO transcoding task complete",
                ],
            ],
            relevant_candidate=2,
            expected_lines=(4, 5),
        ),
    ]


def _expected_range_rank(
    matches: list[dict[str, Any]], expected_range: dict[str, int]
) -> int | None:
    """Return the first rank whose matched lines overlap the expected evidence range."""
    expected_start = expected_range["start_line"]
    expected_end = expected_range["end_line"]
    for rank, match in enumerate(matches, start=1):
        value = match.get("matched_range", "")
        try:
            start_text, end_text = value.split("-L", 1)
            matched_start = int(start_text.removeprefix("L"))
            matched_end = int(end_text)
        except (AttributeError, TypeError, ValueError):
            continue
        if matched_start <= expected_end and expected_start <= matched_end:
            return rank
    return None


def run_relevance_benchmark(top_k: int = 3) -> dict[str, Any]:
    """Evaluate all modes against the deterministic relevance corpus."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    cases = relevance_cases()
    engine = EphemeralEngine(
        max_captures=len(cases),
        semantic_chunk_lines=SEMANTIC_CANDIDATE_LINES,
        semantic_chunk_bytes=4096,
        semantic_chunk_overlap=0,
    )
    records = []
    try:
        captures = {
            case["id"]: engine.ingest("\n".join(case["lines"]), label=f"relevance-{case['id']}")
            for case in cases
        }
        for mode in MODES:
            for case in cases:
                result = engine.search(
                    case["query"], mode=mode, capture_id=captures[case["id"]].capture_id,
                    top_k=top_k, context_lines=0,
                )
                rank = _expected_range_rank(
                    result.get("matches", []), case["expected_range"]
                )
                records.append({
                    "case": case["id"],
                    "mode": mode,
                    "query": case["query"],
                    "expected_marker": case["expected_marker"],
                    "expected_range": case["expected_range"],
                    "rank": rank,
                    "hit_at_1": rank == 1,
                    "hit_at_k": rank is not None and rank <= top_k,
                    "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
                })
    finally:
        # Background index work must not outlive the run and delay process exit.
        engine.shutdown()

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
        "fixture_version": FIXTURE_VERSION,
        "benchmark": "search-relevance",
        "evaluation": "deterministic-synthetic-corpus",
        "embedding_mode": "deterministic-test" if os.environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1" else "configured-model",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "top_k": top_k,
        "controls": {
            "fixtures": "deterministic synthetic output with competing windows and expected matched ranges",
            "embedding_model": "EPHEMERAL_TEST_EMBEDDINGS when enabled by caller",
            "telemetry": "none; all measurements are local",
            "scope": "retrieval relevance only; no agent answer quality or token usage",
        },
        "records": records,
        "summaries": summaries,
    }


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the compact checked-in relevance baseline."""
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid relevance baseline {path}: {exc}") from exc
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("relevance baseline schema_version does not match benchmark")
    if baseline.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("relevance baseline fixture_version does not match benchmark")
    if baseline.get("benchmark") != "search-relevance":
        raise ValueError("relevance baseline benchmark name is invalid")
    if not isinstance(baseline.get("tolerances"), dict):
        raise ValueError("relevance baseline tolerances are missing")
    _validate_tolerances(baseline["tolerances"])
    if not isinstance(baseline.get("summaries"), dict):
        raise ValueError("relevance baseline summaries are missing")
    for mode in MODES:
        summary = baseline["summaries"].get(mode)
        if not isinstance(summary, dict) or any(metric not in summary for metric in METRICS):
            raise ValueError(f"relevance baseline is missing metrics for {mode}")
        for metric in METRICS:
            if not isinstance(summary[metric], (int, float)):
                raise ValueError(f"relevance baseline metric {mode}.{metric} is not numeric")
    return baseline


def compare_relevance(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare deterministic relevance scores and report material regressions."""
    regressions = []
    comparisons: dict[str, dict[str, Any]] = {}
    if result.get("fixture_version") != baseline.get("fixture_version"):
        regressions.append("fixture_version changed")
    if result.get("top_k") != baseline.get("top_k"):
        regressions.append("top_k changed")
    if result.get("embedding_mode") != baseline.get("embedding_mode"):
        regressions.append(
            f"embedding_mode changed from {baseline.get('embedding_mode')} to {result.get('embedding_mode')}"
        )

    baseline_tolerances = baseline.get("tolerances", {})
    if not isinstance(baseline_tolerances, dict):
        raise ValueError("relevance baseline tolerances are missing")
    tolerances = {**DEFAULT_TOLERANCES, **baseline_tolerances}
    _validate_tolerances(tolerances)
    for mode in MODES:
        current_summary = result["summaries"][mode]
        baseline_summary = baseline["summaries"][mode]
        comparisons[mode] = {}
        for metric in METRICS:
            current = float(current_summary[metric])
            expected = float(baseline_summary[metric])
            delta = current - expected
            allowed_drop = float(tolerances[metric])
            passed = delta >= -allowed_drop
            comparisons[mode][metric] = {
                "baseline": expected,
                "current": current,
                "delta": delta,
                "allowed_drop": allowed_drop,
                "passed": passed,
            }
            if not passed:
                regressions.append(
                    f"{mode}.{metric} dropped from {expected:.4f} to {current:.4f} "
                    f"(allowed drop {allowed_drop:.4f})"
                )
    return {
        "baseline_fixture_version": baseline.get("fixture_version"),
        "tolerances": tolerances,
        "comparisons": comparisons,
        "regressions": regressions,
        "passed": not regressions,
    }


def workload_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-agnostic workload result for a relevance record."""
    comparison = result.get("baseline_comparison")
    regressions = list(comparison["regressions"]) if comparison else []
    mode_regressions = {
        mode: [message for message in regressions if message.startswith(f"{mode}.")]
        for mode in result["summaries"]
    }
    global_regressions = [
        message
        for message in regressions
        if not any(message.startswith(f"{mode}.") for mode in mode_regressions)
    ]
    comparison_failed = comparison is not None and (
        comparison.get("passed") is False or bool(regressions)
    )
    runs = []
    for mode, summary in result["summaries"].items():
        errors = mode_regressions[mode]
        runs.append(wr.run(
            mode,
            labels={"mode": mode, "top_k": result["top_k"]},
            status="failure" if errors else "success",
            measurements={
                "queries": wr.measurement("count", value=summary["queries"]),
                "hit_at_1": wr.measurement("score", value=summary["hit_at_1"], samples=summary["queries"]),
                "hit_at_k": wr.measurement("score", value=summary["hit_at_k"], samples=summary["queries"]),
                "mrr": wr.measurement("score", value=summary["mrr"], samples=summary["queries"]),
            },
            errors=errors,
        ))
    return wr.build_result(
        workload="search-relevance",
        kind="evaluation",
        producer="benchmark_relevance.py",
        producer_schema_version=result["schema_version"],
        fixture_version=result["fixture_version"],
        parameters={
            "top_k": result["top_k"],
            "evaluation": result["evaluation"],
            "embedding_mode": result["embedding_mode"],
            "baseline_compared": comparison is not None,
            "tolerances": comparison["tolerances"] if comparison else None,
        },
        status="failure" if comparison_failed else None,
        errors=global_regressions,
        runs=runs,
        details=result,
        privacy="synthetic fixtures with explicit expected ranges; no user queries or captures",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    wr.add_result_argument(parser)
    parser.add_argument("--baseline", type=Path, help="Optional checked-in relevance baseline JSON")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when the supplied baseline comparison has a material regression",
    )
    args = parser.parse_args()
    try:
        result = run_relevance_benchmark(args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    comparison = None
    if args.baseline:
        try:
            comparison = compare_relevance(result, load_baseline(args.baseline))
        except ValueError as exc:
            parser.error(str(exc))
        result["baseline_comparison"] = comparison
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), file=wr.report_stream(args.result))
    if args.result:
        wr.write_result(workload_result(result), args.result, experiment=wr.experiment_from_args(args))
    if args.fail_on_regression and comparison and not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
