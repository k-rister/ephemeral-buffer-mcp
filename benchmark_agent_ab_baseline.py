#!/usr/bin/env python3
"""Create and compare privacy-safe agent A/B aggregate baselines."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
BENCHMARK = "agent-ab-baseline"
MODES = ("control", "mcp")
METRICS = (
    "completion_rate",
    "signal_retrieval_rate",
    "duration_seconds",
    "tool_calls",
    "mcp_tool_calls",
    "repeated_commands",
    "context_bytes_proxy",
    "input_tokens",
    "output_tokens",
)
DEFAULT_THRESHOLDS = {
    "completion_rate": {"gate": True, "direction": "higher", "absolute_tolerance": 0.05, "relative_tolerance": 0.0},
    "signal_retrieval_rate": {"gate": True, "direction": "higher", "absolute_tolerance": 0.05, "relative_tolerance": 0.0},
    "duration_seconds": {"gate": True, "direction": "lower", "absolute_tolerance": 0.0, "relative_tolerance": 0.25},
    "tool_calls": {"gate": False, "direction": "lower", "absolute_tolerance": 0.0, "relative_tolerance": 0.25},
    "mcp_tool_calls": {"gate": False, "direction": "informational", "absolute_tolerance": 0.0, "relative_tolerance": 0.0},
    "repeated_commands": {"gate": True, "direction": "lower", "absolute_tolerance": 0.5, "relative_tolerance": 0.25},
    "context_bytes_proxy": {"gate": True, "direction": "lower", "absolute_tolerance": 0.0, "relative_tolerance": 0.25},
    "input_tokens": {"gate": True, "direction": "lower", "absolute_tolerance": 0.0, "relative_tolerance": 0.25},
    "output_tokens": {"gate": True, "direction": "lower", "absolute_tolerance": 0.0, "relative_tolerance": 0.25},
}
FORBIDDEN_KEYS = {"prompt", "prompts", "transcript", "transcripts", "capture", "captures", "command", "commands", "raw_output"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root in {path} must be an object")
    return value


def _assert_private(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError("aggregate summary contains a forbidden raw-data field")
        for child in value.values():
            _assert_private(child)
    elif isinstance(value, list):
        for child in value:
            _assert_private(child)


def _summary_metric(summary: dict[str, Any], mode: str, metric: str) -> float | None:
    mode_summary = summary.get("mode_summaries", {}).get(mode, {})
    entry = mode_summary.get(metric)
    if entry is None and metric == "context_bytes_proxy":
        entry = mode_summary.get("context_bytes")
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        value = entry
    elif isinstance(entry, dict) and entry.get("available", True):
        value = entry.get("mean")
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"summary metric {mode}.{metric} is not numeric")
    return float(value)


def _validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("benchmark") != "agent-ab":
        raise ValueError("summary benchmark must be agent-ab")
    if not isinstance(summary.get("protocol"), dict):
        raise ValueError("summary protocol metadata is required")
    for field in ("model_config", "repository_fixture", "environment", "reset_policy", "agent_adapter"):
        if not isinstance(summary["protocol"].get(field), str) or not summary["protocol"][field].strip():
            raise ValueError(f"summary protocol field {field} is required")
    if not isinstance(summary.get("repetitions"), int) or summary["repetitions"] < 1:
        raise ValueError("summary repetitions must be positive")
    _assert_private(summary)


def build_baseline(summary: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a compact baseline from an aggregate agent A/B summary."""
    _validate_summary(summary)
    merged_thresholds = {metric: dict(DEFAULT_THRESHOLDS[metric]) for metric in METRICS}
    for metric, threshold in (thresholds or {}).items():
        if metric not in METRICS or not isinstance(threshold, dict):
            raise ValueError(f"unknown baseline threshold: {metric}")
        merged_thresholds[metric].update(threshold)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "task_fixture_version": summary["task_fixture_version"],
        "records_schema_version": summary.get("records_schema_version", 1),
        "seed": summary["seed"],
        "repetitions": summary["repetitions"],
        "protocol": summary["protocol"],
        "metrics": {
            mode: {metric: _summary_metric(summary, mode, metric) for metric in METRICS}
            for mode in MODES
        },
        "thresholds": merged_thresholds,
        "privacy": "aggregate metadata only; raw prompts, transcripts, commands, captures, and user content are excluded",
    }


def load_baseline(path: Path) -> dict[str, Any]:
    baseline = _read_json(path)
    if baseline.get("schema_version") != SCHEMA_VERSION or baseline.get("benchmark") != BENCHMARK:
        raise ValueError("baseline schema or benchmark name is invalid")
    if not isinstance(baseline.get("metrics"), dict) or not isinstance(baseline.get("thresholds"), dict):
        raise ValueError("baseline metrics and thresholds are required")
    for mode in MODES:
        if not isinstance(baseline["metrics"].get(mode), dict):
            raise ValueError(f"baseline metrics are missing {mode}")
        for metric in METRICS:
            value = baseline["metrics"][mode].get(metric)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"baseline metric {mode}.{metric} is not numeric or null")
    _assert_private(baseline)
    return baseline


def compare_summary(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare aggregate metrics and return gated regressions plus observations."""
    _validate_summary(summary)
    load_baseline_from_memory = load_baseline_dict(baseline)
    regressions = []
    comparisons = {}
    if summary["task_fixture_version"] != baseline.get("task_fixture_version"):
        regressions.append("task_fixture_version changed")
    for field in ("model_config", "repository_fixture", "agent_adapter"):
        current = summary["protocol"].get(field)
        expected = baseline.get("protocol", {}).get(field)
        if current != expected:
            regressions.append(f"protocol.{field} changed from {expected} to {current}")
    for mode in MODES:
        comparisons[mode] = {}
        for metric in METRICS:
            current = _summary_metric(summary, mode, metric)
            expected = baseline["metrics"][mode].get(metric)
            threshold = {**DEFAULT_THRESHOLDS[metric], **baseline["thresholds"].get(metric, {})}
            result = {"baseline": expected, "current": current, "available": current is not None and expected is not None, "passed": True}
            if result["available"] and threshold["direction"] != "informational":
                allowed = float(threshold["absolute_tolerance"]) + abs(float(expected)) * float(threshold["relative_tolerance"])
                delta = float(current) - float(expected)
                result.update({"delta": delta, "allowed_change": allowed})
                if threshold["direction"] == "higher":
                    result["passed"] = delta >= -allowed
                else:
                    result["passed"] = delta <= allowed
                if threshold["gate"] and not result["passed"]:
                    regressions.append(f"{mode}.{metric} exceeded its tolerated change")
            comparisons[mode][metric] = result
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "baseline_protocol": baseline.get("protocol"),
        "current_protocol": summary.get("protocol"),
        "comparisons": comparisons,
        "regressions": regressions,
        "passed": not regressions,
        "privacy": "aggregate comparison only; raw prompts, transcripts, commands, captures, and user content are excluded",
    }


def load_baseline_dict(baseline: dict[str, Any]) -> dict[str, Any]:
    """Validate an in-memory baseline using the same rules as file loading."""
    if baseline.get("schema_version") != SCHEMA_VERSION or baseline.get("benchmark") != BENCHMARK:
        raise ValueError("baseline schema or benchmark name is invalid")
    if not isinstance(baseline.get("metrics"), dict) or not isinstance(baseline.get("thresholds"), dict):
        raise ValueError("baseline metrics and thresholds are required")
    for mode in MODES:
        if not isinstance(baseline["metrics"].get(mode), dict):
            raise ValueError(f"baseline metrics are missing {mode}")
    _assert_private(baseline)
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Aggregate agent-ab summary JSON")
    parser.add_argument("--baseline", type=Path, help="Existing baseline to compare")
    parser.add_argument("--create-baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    if args.create_baseline and args.fail_on_regression:
        parser.error("--fail-on-regression requires --baseline comparison")
    summary = _read_json(args.summary)
    if args.create_baseline == bool(args.baseline):
        parser.error("provide exactly one of --create-baseline or --baseline")
    result = build_baseline(summary) if args.create_baseline else compare_summary(summary, load_baseline(args.baseline))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_regression and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
