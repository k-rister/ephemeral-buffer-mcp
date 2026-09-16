#!/usr/bin/env python3
"""Measure deterministic command-output workflows with and without the buffer."""

import argparse
from contextlib import ContextDecorator
import json
import math
import platform
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

from engine import EphemeralEngine, process_rss_bytes
import server


SCHEMA_VERSION = 1
CONTEXT_LINES = 2


class _IsolatedSummaryEngine(ContextDecorator):
    """Give the summary benchmark a private engine without mutating callers."""

    def _recreate_cm(self):
        return type(self)()

    def __enter__(self):
        self._benchmark_engine = EphemeralEngine(max_captures=len(summary_scenarios()))
        self._engine_token = server._ENGINE_OVERRIDE.set(self._benchmark_engine)
        return self._benchmark_engine

    def __exit__(self, exc_type, exc_value, traceback):
        server._ENGINE_OVERRIDE.reset(self._engine_token)
        self._benchmark_engine.shutdown()
        return False


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


def summary_scenarios() -> List[Dict[str, Any]]:
    """Return representative coding-agent capture outcomes for summary measurement."""
    return [
        {
            "id": "successful-test",
            "task_prompt": "Decide whether the test suite is safe to continue.",
            "command": "pytest tests/test_orders.py -q",
            "content_type": "log",
            "text": "\n".join(
                [
                    f"test case {index}: passed; " + ("assertion detail " * 8)
                    for index in range(1, 81)
                ]
                + ["80 tests passed", "OK"]
            ),
            "command_exit_code": 0,
            "truncated": False,
            "original_byte_size": None,
            "timed_out": False,
        },
        {
            "id": "failed-test",
            "task_prompt": "Identify the failing test signal and decide what to inspect next.",
            "command": "pytest tests/test_payments.py -q",
            "content_type": "log",
            "text": "\n".join(
                [
                    f"test case {index}: passed; " + ("fixture detail " * 8)
                    for index in range(1, 81)
                ]
                + ["FAILED: 2 tests failed", "ERROR: assertion mismatch"]
            ),
            "command_exit_code": 1,
            "truncated": False,
            "original_byte_size": None,
            "timed_out": False,
        },
        {
            "id": "noisy-build",
            "task_prompt": "Decide whether the build completed successfully despite noisy output.",
            "command": "make all",
            "content_type": "log",
            "text": "\n".join(
                f"build step {index}: compiler output " + ("diagnostic context " * 20)
                for index in range(1, 501)
            ),
            "command_exit_code": 0,
            "truncated": False,
            "original_byte_size": None,
            "timed_out": False,
        },
        {
            "id": "truncated-command",
            "task_prompt": "Decide whether the retained command output is sufficient or needs retrieval.",
            "command": "./scripts/run-integration-suite.sh",
            "content_type": "log",
            "text": ("retained command output; " + ("captured context " * 12) + "\n") * 60,
            "command_exit_code": 1,
            "truncated": True,
            "original_byte_size": 80_000,
            "timed_out": False,
        },
        {
            "id": "timed-out-command",
            "task_prompt": "Decide whether the timed-out command needs a retry or targeted inspection.",
            "command": "python tools/long_running_worker.py",
            "content_type": "log",
            "text": ("partial command output; " + ("timeout context " * 12) + "\n") * 40,
            "command_exit_code": 124,
            "truncated": False,
            "original_byte_size": None,
            "timed_out": True,
            "timeout_seconds": 0.5,
        },
    ]


def _legacy_execute_response(
    command: str,
    summary: Dict[str, Any],
    exit_code: int,
    timed_out: bool,
    timeout_seconds: float | None,
) -> str:
    """Reproduce the parent commit's formatted execute response contract."""
    if timed_out:
        status = f"TIMED OUT after {timeout_seconds:g}s"
    else:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (Exit Code {exit_code})"
    truncation = ""
    if summary.get("truncated"):
        truncation = f"\nOutput: truncated from {summary['original_byte_size']:,} bytes\n"
    signals = summary.get("signals_summary", "None detected")
    return (
        f"Command: `{command}`\n"
        f"Status: {status}\n"
        f"Captured ID: `{summary['capture_id']}` ({summary['total_lines']:,} lines, {summary['byte_size']:,} bytes)\n"
        f"{truncation}"
        f"Detected Signals: {signals}\n\n"
        f"--- Head (First 5 lines) ---\n{summary['head_preview']}\n\n"
        f"--- Tail (Last 5 lines) ---\n{summary['tail_preview']}\n\n"
        f"Query details using `search_capture(query='...', capture_id='{summary['capture_id']}')`."
    )


@_IsolatedSummaryEngine()
def run_summary_benchmark() -> Dict[str, Any]:
    """Measure public preview responses versus summary-first prompts.

    This is a deterministic prompt-size proxy. It does not invoke a model or
    claim provider-reported token usage. The baseline reconstructs the parent
    commit's formatted-text response, while the comparison path exercises the
    public compact response from ``execute_and_capture``. Full retained output
    is verified through the public slice-retrieval API.
    """
    selected = summary_scenarios()
    records = []
    for scenario in selected:
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(
                scenario["text"],
                scenario["command_exit_code"],
                scenario["truncated"],
                scenario["original_byte_size"],
                scenario["timed_out"],
            ),
        ):
            compact_summary = json.loads(server.execute_and_capture(
                scenario["command"],
                label=f"summary-benchmark-{scenario['id']}",
                content_type=scenario["content_type"],
                max_output_bytes=server._active_engine().max_buffer_bytes,
            ))
        capture_id = compact_summary["capture_id"]
        detailed_summary = json.loads(
            server.get_capture_summary(capture_id, include_previews=True)
        )
        raw_summary = server._active_engine().get_summary(capture_id, include_previews=True)
        legacy_response = _legacy_execute_response(
            scenario["command"],
            raw_summary,
            scenario["command_exit_code"],
            scenario["timed_out"],
            scenario.get("timeout_seconds"),
        )
        compact_core = {
            key: value
            for key, value in compact_summary.items()
            if key not in {"command", "command_truncated"}
        }
        detailed_core = {
            key: value for key, value in detailed_summary.items() if key != "previews"
        }
        payload_shapes_aligned = set(compact_core) == set(detailed_core)
        if not payload_shapes_aligned:
            raise RuntimeError("summary benchmark payload shapes are not aligned")
        compact = json.dumps(
            compact_summary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        detailed = json.dumps(
            detailed_summary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        legacy_prompt = json.dumps(
            {"task": scenario["task_prompt"], "tool_response": legacy_response},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        summary_prompt = json.dumps(
            {"task": scenario["task_prompt"], "tool_response": compact_summary},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        retrieval = server.get_capture_slice(
            1,
            compact_summary["total_lines"],
            capture_id=capture_id,
        )
        retrieval_body = retrieval.split("```text\n", 1)[1].rsplit("\n```", 1)[0]
        retrieved_lines = [
            line.split(" | ", 1)[1]
            for line in retrieval_body.splitlines()
            if " | " in line
        ]
        retrieval_verified = (
            retrieved_lines == scenario["text"].splitlines()
        )
        compact_tokens = math.ceil(len(compact) / 4)
        detailed_tokens = math.ceil(len(detailed) / 4)
        legacy_prompt_tokens = math.ceil(len(legacy_prompt) / 4)
        summary_prompt_tokens = math.ceil(len(summary_prompt) / 4)
        records.append({
            "task_id": scenario["id"],
            "status": compact_summary["status"],
            "compact_summary_bytes": len(compact),
            "preview_summary_bytes": len(detailed),
            "compact_token_proxy": compact_tokens,
            "preview_token_proxy": detailed_tokens,
            "byte_reduction": 1 - (len(compact) / len(detailed)),
            "token_proxy_reduction": 1 - (compact_tokens / detailed_tokens),
            "legacy_prompt_bytes": len(legacy_prompt),
            "summary_prompt_bytes": len(summary_prompt),
            "legacy_prompt_token_proxy": legacy_prompt_tokens,
            "summary_prompt_token_proxy": summary_prompt_tokens,
            "prompt_byte_reduction": 1 - (len(summary_prompt) / len(legacy_prompt)),
            "prompt_token_proxy_reduction": 1 - (summary_prompt_tokens / legacy_prompt_tokens),
            "retrieval_bytes": len(retrieval.encode("utf-8")),
            "full_output_available_for_retrieval": retrieval_verified,
            "retrieval_verified": retrieval_verified,
            "payload_shapes_aligned": payload_shapes_aligned,
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "capture-summary",
        "evaluation": "public-preview-vs-summary-first-agent-prompt",
        "controls": {
            "fixtures": "deterministic coding-agent capture outcomes",
            "model": "none; the harness does not invoke a model",
            "token_proxy": "ceil(initial agent-prompt UTF-8 bytes / 4)",
            "baseline": "parent execute_and_capture formatted-text response reconstructed from the parent contract",
            "summary_path": "public execute_and_capture compact response; full output remains retrievable",
            "retrieval": "public get_capture_slice response is compared with retained fixture lines",
            "provider_usage": "not measured",
        },
        "records": records,
        "aggregate": {
            "task_count": len(records),
            "mean_byte_reduction": statistics.mean(record["byte_reduction"] for record in records),
            "mean_token_proxy_reduction": statistics.mean(record["token_proxy_reduction"] for record in records),
            "mean_prompt_byte_reduction": statistics.mean(record["prompt_byte_reduction"] for record in records),
            "mean_prompt_token_proxy_reduction": statistics.mean(
                record["prompt_token_proxy_reduction"] for record in records
            ),
        },
    }
    return result
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


def _run_sequential_workflow(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure one capture/search/slice path per synthetic repository."""
    started = time.perf_counter()
    engine = EphemeralEngine(max_captures=len(selected))
    overview_bytes = 0
    retrieval_bytes = 0
    successes = 0
    searches = 0
    retrievals = 0
    for scenario in selected:
        capture = engine.ingest(
            "\n".join(scenario["lines"]),
            label=f"benchmark-{scenario['id']}",
            content_type=scenario["content_type"],
        )
        summary = engine.get_summary(capture.capture_id)
        overview_bytes += len(json.dumps(summary, sort_keys=True).encode("utf-8"))
        search = engine.search(
            scenario["query"], mode="bm25", capture_id=capture.capture_id,
            top_k=3, context_lines=CONTEXT_LINES,
        )
        searches += 1
        retrieval_bytes += len(json.dumps(search, sort_keys=True).encode("utf-8"))
        useful = next((match for match in search.get("matches", [])
                       if scenario["marker"] in match["snippet"]), None)
        slice_result: Dict[str, Any] = {}
        if useful:
            line_range = _matched_range(useful.get("matched_range", ""))
            if line_range:
                slice_result = engine.get_slice(*line_range, capture_id=capture.capture_id)
                retrievals += 1
                retrieval_bytes += len(json.dumps(slice_result, sort_keys=True).encode("utf-8"))
        successes += int(bool(useful and scenario["marker"] in slice_result.get("content", "")))
    return {
        "mode": "sequential",
        "successes": successes,
        "scenario_count": len(selected),
        "searches": searches,
        "retrievals": retrievals,
        "overview_bytes": overview_bytes,
        "retrieval_bytes": retrieval_bytes,
        "time_seconds": time.perf_counter() - started,
    }


def _run_consolidated_workflow(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure one consolidated overview followed by targeted retrievals."""
    started = time.perf_counter()
    engine = EphemeralEngine(max_captures=len(selected) + 1)
    capture_ids = [
        engine.ingest(
            "\n".join(scenario["lines"]),
            label=f"benchmark-{scenario['id']}",
            content_type=scenario["content_type"],
        ).capture_id
        for scenario in selected
    ]
    # Keep the benchmark's source fixture complete; the MCP tool still exposes
    # bounded omission behavior independently through its caller-provided limit.
    consolidated = engine.consolidate(capture_ids, max_captures=len(selected), max_bytes=100000)
    consolidated_capture = engine.ingest(
        consolidated["content"], label="benchmark-consolidated", content_type="text"
    )
    overview = {
        key: value for key, value in consolidated.items() if key != "content"
    }
    overview_bytes = len(json.dumps(overview, sort_keys=True).encode("utf-8"))
    retrieval_bytes = 0
    successes = 0
    searches = 0
    retrievals = 0
    consolidated_id = consolidated_capture.capture_id
    for scenario in selected:
        search = engine.search(
            scenario["query"], mode="bm25", capture_id=consolidated_id,
            top_k=3, context_lines=CONTEXT_LINES,
        )
        searches += 1
        retrieval_bytes += len(json.dumps(search, sort_keys=True).encode("utf-8"))
        useful = next((match for match in search.get("matches", [])
                       if scenario["marker"] in match["snippet"]), None)
        slice_result: Dict[str, Any] = {}
        if useful:
            line_range = _matched_range(useful.get("matched_range", ""))
            if line_range:
                slice_result = engine.get_slice(*line_range, capture_id=consolidated_id)
                retrievals += 1
                retrieval_bytes += len(json.dumps(slice_result, sort_keys=True).encode("utf-8"))
        successes += int(bool(useful and scenario["marker"] in slice_result.get("content", "")))
    return {
        "mode": "consolidated",
        "successes": successes,
        "scenario_count": len(selected),
        "searches": searches,
        "retrievals": retrievals,
        "overview_bytes": overview_bytes,
        "retrieval_bytes": retrieval_bytes,
        "consolidated_bytes": len(consolidated["content"].encode("utf-8")),
        "omitted_record_count": consolidated["omitted_record_count"],
        "time_seconds": time.perf_counter() - started,
    }


def run_consolidation_benchmark(repetitions: int = 5, seed: int = 20260907) -> Dict[str, Any]:
    """Compare sequential per-capture retrieval with consolidated retrieval."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    import random

    selected = scenarios()
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    schedules: List[Dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        ordered = list(selected)
        rng.shuffle(ordered)
        modes = ["sequential", "consolidated"]
        rng.shuffle(modes)
        schedules.append({
            "repetition": repetition,
            "task_order": [scenario["id"] for scenario in ordered],
            "mode_order": modes,
        })
        for mode in modes:
            result = (_run_sequential_workflow(ordered) if mode == "sequential"
                      else _run_consolidated_workflow(ordered))
            result["repetition"] = repetition
            records.append(result)

    summaries: Dict[str, Dict[str, Any]] = {}
    for mode in ("sequential", "consolidated"):
        mode_records = [record for record in records if record["mode"] == mode]
        summaries[mode] = {
            "runs": len(mode_records),
            "success_rate": sum(record["successes"] for record in mode_records)
            / (len(mode_records) * len(selected)),
            "mean_time_seconds": statistics.mean(record["time_seconds"] for record in mode_records),
            "mean_overview_bytes": statistics.mean(record["overview_bytes"] for record in mode_records),
            "mean_retrieval_bytes": statistics.mean(record["retrieval_bytes"] for record in mode_records),
            "mean_searches": statistics.mean(record["searches"] for record in mode_records),
            "mean_retrievals": statistics.mean(record["retrievals"] for record in mode_records),
        }
    return {
        "schema_version": 1,
        "benchmark": "mcp-effectiveness",
        "evaluation": "sequential-vs-consolidated",
        "repetitions": repetitions,
        "seed": seed,
        "controls": {
            "fixtures": "deterministic synthetic scenarios as repository results",
            "task_order": "seeded shuffle per repetition",
            "mode_order": "seeded shuffle per repetition",
            "model": "none; the harness does not invoke a model",
            "telemetry": "none; all measurements are local",
        },
        "records": records,
        "schedules": schedules,
        "summaries": summaries,
        "comparison": {
            "overview_bytes_reduction": 1 - (
                summaries["consolidated"]["mean_overview_bytes"]
                / summaries["sequential"]["mean_overview_bytes"]
            ),
            "retrieval_bytes_reduction": 1 - (
                summaries["consolidated"]["mean_retrieval_bytes"]
                / summaries["sequential"]["mean_retrieval_bytes"]
            ),
            "time_ratio": summaries["consolidated"]["mean_time_seconds"]
            / summaries["sequential"]["mean_time_seconds"],
        },
    }


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
    parser.add_argument("--summary", action="store_true", help="Measure compact capture summaries for representative agent outcomes")
    parser.add_argument(
        "--consolidation-runs", type=int,
        help="Run sequential-vs-consolidated evaluation this many times",
    )
    parser.add_argument("--seed", type=int, default=20260907, help="Seed for paired A/B task and mode order")
    parser.add_argument("--output", type=Path, help="Write machine-readable results to this JSON file")
    args = parser.parse_args()
    if args.summary:
        record = run_summary_benchmark()
    elif args.consolidation_runs is not None:
        record = run_consolidation_benchmark(args.consolidation_runs, args.seed)
    elif args.ab_runs is not None:
        record = run_ab_evaluation(args.ab_runs, args.seed)
    else:
        record = run_benchmark(args.mode)
    if args.output:
        write_results(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
