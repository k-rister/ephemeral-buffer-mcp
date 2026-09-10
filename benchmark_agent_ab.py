#!/usr/bin/env python3
"""Plan and summarize privacy-safe agent-level MCP A/B evaluations."""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
RECORDS_SCHEMA_VERSION = 3
TASK_FIXTURE_VERSION = 1
MODES = ("control", "mcp")
TASKS = (
    {"id": "targeted-inspection", "category": "small-targeted-output"},
    {"id": "noisy-test-failure", "category": "noisy-test-output"},
    {"id": "build-log-search", "category": "noisy-build-output"},
    {"id": "follow-up-context", "category": "search-and-retrieval"},
)
RUN_KEYS = {
    "task_id",
    "repetition",
    "mode",
    "completed",
    "signal_retrieved",
    "duration_seconds",
    "tool_calls",
    "repeated_commands",
    "context_bytes",
    "peak_rss_bytes",
}
RUN_KEYS_V2 = RUN_KEYS - {"context_bytes"} | {
    "context_bytes_proxy",
    "exit_code",
    "failure_reason",
    "mcp_tool_calls",
    "input_tokens",
    "output_tokens",
}
RUN_KEYS_V3 = RUN_KEYS_V2 | {"input_token_samples", "output_token_samples"}
METRICS = (
    "completed",
    "signal_retrieved",
    "duration_seconds",
    "tool_calls",
    "mcp_tool_calls",
    "repeated_commands",
    "context_bytes_proxy",
    "peak_rss_bytes",
)
PROTOCOL_FIELDS = (
    "model_config",
    "repository_fixture",
    "environment",
    "reset_policy",
    "agent_adapter",
)


def build_schedule(repetitions: int = 5, seed: int = 20260909) -> dict[str, Any]:
    """Build a deterministic counterbalanced task/mode schedule."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    schedule = []
    sequence = 1
    for repetition in range(1, repetitions + 1):
        task_order = [task["id"] for task in TASKS]
        rng.shuffle(task_order)
        for task_id in task_order:
            mode_order = list(MODES)
            rng.shuffle(mode_order)
            for mode in mode_order:
                schedule.append({
                    "sequence": sequence,
                    "repetition": repetition,
                    "task_id": task_id,
                    "mode": mode,
                })
                sequence += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "agent-ab",
        "task_fixture_version": TASK_FIXTURE_VERSION,
        "seed": seed,
        "repetitions": repetitions,
        "modes": list(MODES),
        "tasks": list(TASKS),
        "schedule": schedule,
        "privacy": "schedule contains task IDs and categories only; no prompts, captures, commands, or user content",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root in {path} must be an object")
    return value


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("records protocol metadata is required")
    for field in PROTOCOL_FIELDS:
        value = protocol.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"records protocol field {field} must be a non-empty identifier")


def _run_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (int(record["repetition"]), record["task_id"], record["mode"])


def validate_records(payload: dict[str, Any], schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate balanced metadata-only agent records against a schedule."""
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("benchmark") != "agent-ab":
        raise ValueError("records schema or benchmark name is invalid")
    if payload.get("task_fixture_version") != schedule.get("task_fixture_version"):
        raise ValueError("records task_fixture_version does not match schedule")
    _validate_protocol(payload.get("protocol"))
    records_schema_version = payload.get("records_schema_version", 1)
    if records_schema_version not in (1, 2, RECORDS_SCHEMA_VERSION):
        raise ValueError("records schema version is unsupported")
    run_keys = (
        RUN_KEYS
        if records_schema_version == 1
        else RUN_KEYS_V2
        if records_schema_version == 2
        else RUN_KEYS_V3
    )
    expected = {_run_key(item): item for item in schedule.get("schedule", [])}
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("records runs must be a list")
    actual = {}
    for record in runs:
        if not isinstance(record, dict) or set(record) != run_keys:
            raise ValueError("each run must contain exactly the documented metadata fields")
        key = _run_key(record)
        if key not in expected:
            raise ValueError(f"run is not present in schedule: {key}")
        if key in actual:
            raise ValueError(f"duplicate run: {key}")
        if not isinstance(record["task_id"], str) or record["mode"] not in MODES:
            raise ValueError(f"invalid task or mode in run: {key}")
        if not isinstance(record["repetition"], int) or record["repetition"] < 1:
            raise ValueError(f"invalid repetition in run: {key}")
        for field in ("completed", "signal_retrieved"):
            if not isinstance(record[field], bool):
                raise ValueError(f"{field} must be boolean in run: {key}")
        numeric_fields = ("duration_seconds", "tool_calls", "repeated_commands", "peak_rss_bytes")
        numeric_fields += ("context_bytes",) if records_schema_version == 1 else ("context_bytes_proxy", "mcp_tool_calls")
        for field in numeric_fields:
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{field} must be a non-negative number in run: {key}")
        if records_schema_version >= 2:
            if record["exit_code"] is not None and (isinstance(record["exit_code"], bool) or not isinstance(record["exit_code"], int)):
                raise ValueError(f"exit_code must be an integer or null in run: {key}")
            if record["failure_reason"] is not None and not isinstance(record["failure_reason"], str):
                raise ValueError(f"failure_reason must be a string or null in run: {key}")
            for field in ("input_tokens", "output_tokens"):
                value = record[field]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                    raise ValueError(f"{field} must be a non-negative number or null in run: {key}")
        if records_schema_version >= 3:
            for field in ("input_token_samples", "output_token_samples"):
                samples = record[field]
                if not isinstance(samples, list):
                    raise ValueError(f"{field} must be a list in run: {key}")
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in samples):
                    raise ValueError(f"{field} must contain non-negative numbers in run: {key}")
        actual[key] = record
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"records are missing scheduled runs: {missing[0]}")
    return list(actual.values())


def _metric_value(record: dict[str, Any], metric: str) -> float:
    """Read a normalized metric from either records schema version."""
    if metric == "context_bytes_proxy":
        return float(record.get("context_bytes_proxy", record.get("context_bytes", 0)))
    return float(record.get(metric, 0))


def _optional_stats(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record[field] for record in records if record.get(field) is not None]
    if not values:
        return {"available": False, "count": 0}
    result = _stats([float(value) for value in values])
    result["available"] = True
    return result


def _stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.mean(values) if values else 0.0
    deviation = statistics.stdev(values) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "stdev": deviation,
        "ci95_half_width": 1.96 * deviation / (count ** 0.5) if count > 1 else 0.0,
    }


def summarize_records(payload: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate and paired outcome summaries without raw agent data."""
    runs = validate_records(payload, schedule)
    by_mode = {}
    for mode in MODES:
        selected = [record for record in runs if record["mode"] == mode]
        by_mode[mode] = {
            "runs": len(selected),
            "completion_rate": sum(record["completed"] for record in selected) / len(selected),
            "signal_retrieval_rate": sum(record["signal_retrieved"] for record in selected) / len(selected),
            "duration_seconds": _stats([_metric_value(record, "duration_seconds") for record in selected]),
            "tool_calls": _stats([_metric_value(record, "tool_calls") for record in selected]),
            "mcp_tool_calls": _stats([_metric_value(record, "mcp_tool_calls") for record in selected]),
            "repeated_commands": _stats([_metric_value(record, "repeated_commands") for record in selected]),
            "context_bytes_proxy": _stats([_metric_value(record, "context_bytes_proxy") for record in selected]),
            "peak_rss_bytes": _stats([_metric_value(record, "peak_rss_bytes") for record in selected]),
            "input_tokens": _optional_stats(selected, "input_tokens"),
            "output_tokens": _optional_stats(selected, "output_tokens"),
            "failure_reasons": {
                reason: sum(record.get("failure_reason") == reason for record in selected)
                for reason in sorted({record.get("failure_reason") for record in selected if record.get("failure_reason")})
            },
        }

    paired = {}
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault((record["repetition"], record["task_id"]), {})[record["mode"]] = record
    for metric in METRICS:
        deltas = []
        for pair in grouped.values():
            control = _metric_value(pair["control"], metric)
            mcp = _metric_value(pair["mcp"], metric)
            deltas.append(mcp - control)
        paired[metric] = _stats(deltas)

    recommendations = [
        "Treat completion and signal retrieval as primary outcomes; interpret cost metrics as secondary.",
        "Repeat with representative privacy-reviewed agent tasks before changing defaults.",
    ]
    if paired["completed"]["mean"] > 0 or paired["signal_retrieved"]["mean"] > 0:
        recommendations.insert(1, "MCP improved at least one paired task outcome; inspect task-level variance before generalizing.")
    elif paired["completed"]["mean"] < 0 or paired["signal_retrieved"]["mean"] < 0:
        recommendations.insert(1, "MCP reduced at least one paired task outcome; inspect failures and routing choices before enabling broader use.")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "agent-ab",
        "records_schema_version": max(payload.get("records_schema_version", 1), 1),
        "task_fixture_version": schedule["task_fixture_version"],
        "seed": schedule["seed"],
        "repetitions": schedule["repetitions"],
        "protocol": payload["protocol"],
        "mode_summaries": by_mode,
        "paired_deltas_mcp_minus_control": paired,
        "recommendations": recommendations,
        "privacy": "summary contains aggregate metadata only; raw prompts, commands, captures, transcripts, and user content are excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-output", type=Path, help="Write a deterministic run schedule")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--records", type=Path, help="Analyze an external metadata-only records envelope")
    parser.add_argument("--schedule", type=Path, help="Schedule JSON when it is not embedded in records")
    parser.add_argument("--output", type=Path, help="Write an aggregate summary JSON")
    args = parser.parse_args()
    if bool(args.schedule_output) == bool(args.records):
        parser.error("provide exactly one of --schedule-output or --records")
    if args.schedule_output:
        schedule = build_schedule(args.repetitions, args.seed)
        args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
        args.schedule_output.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(schedule, indent=2, sort_keys=True))
        return

    payload = _read_json(args.records)
    schedule = _read_json(args.schedule) if args.schedule else payload.get("schedule")
    if not isinstance(schedule, dict):
        parser.error("records must embed a schedule or provide --schedule")
    summary = summarize_records(payload, schedule)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
