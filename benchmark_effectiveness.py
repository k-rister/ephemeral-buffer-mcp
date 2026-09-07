#!/usr/bin/env python3
"""Measure deterministic command-output workflows with and without the buffer."""

import argparse
import json
import platform
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine import EphemeralEngine, process_rss_bytes


SCHEMA_VERSION = 1
CONTEXT_LINES = 2


def _noise(prefix: str, count: int) -> List[str]:
    return [f"{prefix} line {index:03d}: synthetic output for benchmark" for index in range(1, count + 1)]


def scenarios() -> List[Dict[str, Any]]:
    """Return reproducible, content-free scenarios used by the benchmark."""
    large = _noise("compiler", 220)
    large.insert(180, "BUILD_RESULT=SUCCESS artifact=synthetic-package")
    failure = _noise("service", 140)
    failure.insert(103, "ERROR database connection refused retryable=false")
    diff = _noise("review", 90)
    diff[0:0] = [
        "diff --git a/src/auth.py b/src/auth.py",
        "--- a/src/auth.py",
        "+++ b/src/auth.py",
        "@@ -10,3 +10,4 @@",
        " def authenticate(token):",
        "+    return validate_token(token)",
    ]
    diff.insert(66, "REVIEW_NEEDED auth validation requires test coverage")
    timeout = _noise("worker", 160)
    timeout.insert(127, "TIMEOUT queue drain exceeded 30 seconds action=inspect")
    return [
        {"id": "large-build", "content_type": "text", "lines": large, "query": "BUILD RESULT SUCCESS", "marker": "BUILD_RESULT=SUCCESS"},
        {"id": "failure-log", "content_type": "log", "lines": failure, "query": "database connection refused", "marker": "ERROR database connection refused"},
        {"id": "review-diff", "content_type": "diff", "lines": diff, "query": "REVIEW NEEDED auth validation", "marker": "REVIEW_NEEDED auth validation"},
        {"id": "timeout-log", "content_type": "log", "lines": timeout, "query": "TIMEOUT queue drain", "marker": "TIMEOUT queue drain"},
    ]


def _record_base(scenario: Dict[str, Any], mode: str, started: float) -> Dict[str, Any]:
    text = "\n".join(scenario["lines"])
    return {
        "scenario": scenario["id"],
        "mode": mode,
        "input_lines": len(scenario["lines"]),
        "input_bytes": len(text.encode("utf-8")),
        "command_runs": 1,
        "reruns": 0,
        "token_usage": None,
        "token_usage_available": False,
        "time_seconds": time.perf_counter() - started,
    }


def run_baseline(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Model an agent receiving and scanning the complete command output."""
    started = time.perf_counter()
    text = "\n".join(scenario["lines"])
    found = scenario["marker"] in text
    result = _record_base(scenario, "baseline", started)
    result.update({
        "success": found,
        "searches": 0,
        "retrievals": 0,
        "bytes_examined": len(text.encode("utf-8")),
        "bytes_retrieved": len(text.encode("utf-8")),
        "search_useful": None,
    })
    result["time_seconds"] = time.perf_counter() - started
    return result


def _matched_range(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"L(\d+)-L(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def run_mcp(scenario: Dict[str, Any], engine: EphemeralEngine) -> Dict[str, Any]:
    """Model capture, targeted search, and exact-slice retrieval through the engine."""
    started = time.perf_counter()
    text = "\n".join(scenario["lines"])
    capture = engine.ingest(text, label=f"benchmark-{scenario['id']}", content_type=scenario["content_type"])
    search = engine.search(scenario["query"], mode="bm25", capture_id=capture.capture_id, top_k=3, context_lines=CONTEXT_LINES)
    matches = search.get("matches", [])
    useful = next((match for match in matches if scenario["marker"] in match["snippet"]), None)
    retrieved_bytes = len(json.dumps(search, sort_keys=True).encode("utf-8"))
    slice_result: Dict[str, Any] = {}
    if useful:
        line_range = _matched_range(useful.get("matched_range", ""))
        if line_range:
            slice_result = engine.get_slice(*line_range, capture_id=capture.capture_id)
            retrieved_bytes += len(json.dumps(slice_result, sort_keys=True).encode("utf-8"))
    result = _record_base(scenario, "mcp", started)
    result.update({
        "success": bool(useful and scenario["marker"] in slice_result.get("content", "")),
        "searches": 1,
        "retrievals": 1 if slice_result else 0,
        "bytes_examined": retrieved_bytes,
        "bytes_retrieved": retrieved_bytes,
        "search_useful": bool(useful),
        "capture_id": capture.capture_id,
    })
    result["time_seconds"] = time.perf_counter() - started
    return result


def _aggregate(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(results)
    baseline = {record["scenario"]: record for record in records if record["mode"] == "baseline"}
    assisted = {record["scenario"]: record for record in records if record["mode"] == "mcp"}
    reductions = [1 - (record["bytes_examined"] / baseline[scenario_id]["bytes_examined"])
                  for scenario_id, record in assisted.items()]
    return {
        "scenario_count": len(baseline),
        "baseline_successes": sum(record["success"] for record in baseline.values()),
        "mcp_successes": sum(record["success"] for record in assisted.values()),
        "mcp_searches": sum(record["searches"] for record in assisted.values()),
        "mcp_useful_searches": sum(record["search_useful"] is True for record in assisted.values()),
        "mean_bytes_reduction": sum(reductions) / len(reductions) if reductions else 0.0,
    }


def _mode_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize repeated records while preserving run-to-run variation."""
    times = [record["time_seconds"] for record in records]
    rss_deltas = [record["rss_delta_bytes"] for record in records if record["rss_delta_bytes"] is not None]
    return {
        "runs": len(records),
        "successes": sum(record["success"] for record in records),
        "success_rate": sum(record["success"] for record in records) / len(records),
        "mean_time_seconds": statistics.mean(times),
        "stdev_time_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time_seconds": min(times),
        "max_time_seconds": max(times),
        "mean_bytes_examined": statistics.mean(record["bytes_examined"] for record in records),
        "mean_repeated_commands": statistics.mean(record["reruns"] for record in records),
        "mean_rss_delta_bytes": statistics.mean(rss_deltas) if rss_deltas else None,
    }


def _recommendations(comparisons: List[Dict[str, Any]]) -> List[str]:
    """Turn the paired measurements into concrete, conservative guidance."""
    recommendations: List[str] = []
    if all(comparison["baseline"]["success_rate"] == comparison["mcp"]["success_rate"] for comparison in comparisons):
        recommendations.append("MCP preserved the baseline completion rate for every scenario.")
    useful = [comparison for comparison in comparisons if comparison["mcp_useful_search_rate"] > 0]
    if useful:
        recommendations.append("Use MCP for noisy output when targeted search is expected to reduce context size.")
    if any(comparison["local_mcp_overhead_ratio"] is not None and comparison["local_mcp_overhead_ratio"] > 1.0 for comparison in comparisons):
        recommendations.append("Review local MCP processing overhead where capture, indexing, and search cost exceeds baseline processing.")
    else:
        recommendations.append("No measured MCP timing regression exceeded the baseline in this synthetic run.")
    recommendations.append("Repeat on representative real agent tasks before generalizing these synthetic-fixture results.")
    return recommendations


def run_ab_evaluation(repetitions: int = 5, seed: int = 20260907) -> Dict[str, Any]:
    """Run paired, deterministic A/B measurements with controlled task order.

    The harness does not invoke a model, so token usage remains unavailable.
    Each repetition uses the same fixtures and a seeded task/mode order, while
    baseline and MCP records remain paired by repetition and scenario.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    import random

    selected = scenarios()
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    schedules: List[Dict[str, Any]] = []
    engine = EphemeralEngine(max_captures=len(selected) * repetitions)
    for repetition in range(1, repetitions + 1):
        ordered = list(selected)
        rng.shuffle(ordered)
        task_order = [scenario["id"] for scenario in ordered]
        mode_orders: Dict[str, List[str]] = {}
        for scenario in ordered:
            modes = ["baseline", "mcp"]
            rng.shuffle(modes)
            mode_orders[scenario["id"]] = modes
            for mode in modes:
                rss_before = process_rss_bytes()
                result = (run_baseline(scenario) if mode == "baseline" else run_mcp(scenario, engine))
                rss_after = process_rss_bytes()
                result.update({
                    "repetition": repetition,
                    "task_order": len(task_order) - len(ordered) + ordered.index(scenario) + 1,
                    "paired_mode_order": modes,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": rss_after,
                    "rss_delta_bytes": (rss_after - rss_before) if rss_before is not None and rss_after is not None else None,
                })
                records.append(result)
        schedules.append({"repetition": repetition, "task_order": task_order, "mode_orders": mode_orders})

    comparisons: List[Dict[str, Any]] = []
    for scenario in selected:
        scenario_records = [record for record in records if record["scenario"] == scenario["id"]]
        baseline_records = [record for record in scenario_records if record["mode"] == "baseline"]
        mcp_records = [record for record in scenario_records if record["mode"] == "mcp"]
        baseline = _mode_summary(baseline_records)
        mcp = _mode_summary(mcp_records)
        comparisons.append({
            "scenario": scenario["id"],
            "baseline": baseline,
            "mcp": mcp,
            "mcp_useful_search_rate": sum(record["search_useful"] is True for record in mcp_records) / len(mcp_records),
            "mean_bytes_reduction": 1 - (mcp["mean_bytes_examined"] / baseline["mean_bytes_examined"]),
            "local_mcp_overhead_ratio": mcp["mean_time_seconds"] / baseline["mean_time_seconds"] if baseline["mean_time_seconds"] else None,
        })
    return {
        "schema_version": 1,
        "benchmark": "mcp-effectiveness",
        "evaluation": "paired-ab",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repetitions": repetitions,
        "seed": seed,
        "controls": {
            "fixtures": "deterministic synthetic scenarios",
            "task_order": "seeded shuffle per repetition",
            "mode_order": "seeded shuffle per scenario and repetition",
            "model": "none; the harness does not invoke a model",
            "telemetry": "none; all measurements are local",
            "resource_measurement": "process RSS before and after each paired run",
        },
        "records": records,
        "schedules": schedules,
        "comparisons": comparisons,
        "recommendations": _recommendations(comparisons),
    }


def run_benchmark(mode: str = "both") -> Dict[str, Any]:
    """Run the selected deterministic baseline and/or MCP scenarios."""
    selected = scenarios()
    results: List[Dict[str, Any]] = []
    if mode in ("baseline", "both"):
        results.extend(run_baseline(scenario) for scenario in selected)
    if mode in ("mcp", "both"):
        engine = EphemeralEngine(max_captures=len(selected))
        results.extend(run_mcp(scenario, engine) for scenario in selected)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "mcp-effectiveness",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mode": mode,
        "scoring": {
            "success": "target marker is found in the final retrieved result",
            "search_useful": "a search match contains the target marker",
            "bytes_reduction": "1 - MCP bytes examined / baseline bytes examined",
        },
        "results": results,
        "aggregate": _aggregate(results) if mode == "both" else None,
    }


def write_results(path: Path, record: Dict[str, Any]) -> None:
    """Write benchmark output as stable, machine-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "mcp", "both"), default="both")
    parser.add_argument("--ab-runs", type=int, help="Run paired A/B evaluation this many times")
    parser.add_argument("--seed", type=int, default=20260907, help="Seed for paired A/B task and mode order")
    parser.add_argument("--output", type=Path, help="Write machine-readable results to this JSON file")
    args = parser.parse_args()
    record = run_ab_evaluation(args.ab_runs, args.seed) if args.ab_runs is not None else run_benchmark(args.mode)
    if args.output:
        write_results(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
