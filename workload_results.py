#!/usr/bin/env python3
"""Versioned, tool-agnostic result format for coding-agent workload measurements.

A *workload result* is one JSON object that any benchmark, evaluation, or agent
runner can emit so downstream tooling (regression checks, comparison commands,
dashboards) reads a single shape instead of every producer's private record.
The format describes *what* was measured, not *how*: it has no notion of
embeddings, BM25, or any other producer-specific mechanism.

Top-level shape (``FORMAT_VERSION`` 1)::

    {
      "format": "coding-agent-workload-result",
      "format_version": 1,
      "workload": {"name", "kind", "producer", "parameters", ...},
      "environment": {"python_version", "platform", "recorded_at", ...},
      "status": "success" | "failure" | "timeout" | "error" | "partial",
      "errors": ["..."],
      "measurements": {name: measurement},
      "runs": [{"id", "labels", "status", "measurements", "phases", "errors"}],
      "details": {...producer-native record, optional...}
    }

A *measurement* carries a unit plus one or more statistics::

    {"unit": "seconds", "median": 0.12, "p95": 0.31, "samples": 5}
    {"unit": "bytes", "value": 4096}
    {"unit": "tokens", "value": null, "note": "no model was invoked"}

``null`` means the statistic is unavailable; consumers must treat it as missing
rather than as zero.  Phase timings are an ordered list of ``seconds``
measurements with a ``name``.  Producers should prefer the canonical
measurement names in ``CANONICAL_MEASUREMENTS`` whenever the meaning matches so
results from different tools line up; other names are allowed.
"""

import argparse
import datetime as _datetime
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tomllib
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any, Iterable, TextIO


FORMAT = "coding-agent-workload-result"
FORMAT_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("workload_result.schema.json")
PROJECT_NAME = "ephemeral-buffer-mcp"

UNITS = ("seconds", "bytes", "count", "tokens", "ratio", "per_second", "score")
STATISTICS = ("value", "sum", "mean", "median", "min", "max", "p95", "stdev")
STATUSES = ("success", "failure", "timeout", "error", "partial")
KINDS = ("benchmark", "evaluation", "agent-run")

# Names producers should reuse when the meaning matches, so consumers can
# compare across tools.  Values are the required unit.
CANONICAL_MEASUREMENTS = {
    "wall_time_seconds": "seconds",
    "queue_wait_seconds": "seconds",
    "tool_call_seconds": "seconds",
    "tool_calls": "count",
    "repeated_commands": "count",
    "input_bytes": "bytes",
    "output_bytes": "bytes",
    "context_bytes": "bytes",
    "retained_summary_bytes": "bytes",
    "estimated_tokens": "tokens",
    "retained_summary_tokens": "tokens",
    "input_tokens": "tokens",
    "output_tokens": "tokens",
    "peak_rss_bytes": "bytes",
    "rss_delta_bytes": "bytes",
    "success_rate": "ratio",
    "throughput_per_second": "per_second",
}
# Labels with conventional values.  ``cache_state`` distinguishes cold and warm
# measurements; the others identify what a run measured.
CANONICAL_LABELS = ("cache_state", "mode", "task_id", "repetition", "line_count", "profile")

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_TYPES = (str, int, float, bool, type(None))


class WorkloadResultError(ValueError):
    """Raised when a result does not satisfy the format."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nearest_rank(values: list[float], percentile: float = 0.95) -> float:
    """Return a percentile using the nearest-rank convention."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than 0 and at most 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def measurement(unit: str, *, samples: int | None = None, note: str | None = None, **stats: Any) -> dict[str, Any]:
    """Return one measurement with a unit and at least one statistic.

    Statistics are keyword arguments named after ``STATISTICS``; ``None`` marks
    an unavailable value.  ``samples`` records how many observations produced
    the statistics and ``note`` explains a proxy or caveat.
    """
    if unit not in UNITS:
        raise WorkloadResultError(f"unknown unit {unit!r}; expected one of {', '.join(UNITS)}")
    if not stats:
        raise WorkloadResultError("a measurement needs at least one statistic")
    result: dict[str, Any] = {"unit": unit}
    for name, value in stats.items():
        if name not in STATISTICS:
            raise WorkloadResultError(f"unknown statistic {name!r}; expected one of {', '.join(STATISTICS)}")
        if value is not None and not _is_number(value):
            raise WorkloadResultError(f"statistic {name!r} must be a finite number or null")
        result[name] = value
    if samples is not None:
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
            raise WorkloadResultError("samples must be a non-negative integer")
        result["samples"] = samples
    if note is not None:
        if not isinstance(note, str) or not note:
            raise WorkloadResultError("note must be a non-empty string")
        result["note"] = note
    return result


def unavailable(unit: str, note: str | None = None) -> dict[str, Any]:
    """Return a measurement whose value is explicitly missing."""
    return measurement(unit, value=None, samples=0, note=note)


def summarize(values: Iterable[float | None], unit: str, note: str | None = None) -> dict[str, Any]:
    """Return a full statistical summary of raw observations.

    ``None`` observations are dropped; when nothing remains the measurement is
    marked unavailable instead of reporting a misleading zero.
    """
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return unavailable(unit, note)
    return measurement(
        unit,
        mean=statistics.mean(observed),
        median=statistics.median(observed),
        min=min(observed),
        max=max(observed),
        p95=nearest_rank(observed),
        stdev=statistics.stdev(observed) if len(observed) > 1 else 0.0,
        sum=sum(observed),
        samples=len(observed),
        note=note,
    )


def phase(name: str, **stats: Any) -> dict[str, Any]:
    """Return one ordered phase timing (always in seconds)."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise WorkloadResultError(f"phase name {name!r} must match {_NAME.pattern}")
    return {"name": name, **measurement("seconds", **stats)}


def run(
    run_id: str,
    *,
    labels: dict[str, Any] | None = None,
    status: str = "success",
    measurements: dict[str, dict[str, Any]] | None = None,
    phases: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Return one run: a unit of work with its own labels, status, and measurements."""
    return {
        "id": run_id,
        "labels": dict(labels or {}),
        "status": status,
        "measurements": dict(measurements or {}),
        "phases": list(phases or []),
        "errors": list(errors or []),
    }


def _project_version() -> str | None:
    pyproject = Path(__file__).with_name("pyproject.toml")
    try:
        with pyproject.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    try:
        return _metadata.version(PROJECT_NAME)
    except _metadata.PackageNotFoundError:
        return None


def _source_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def environment(**extra: Any) -> dict[str, Any]:
    """Describe the host well enough to explain why two results differ.

    Producer-specific facts (model names, thread counts) go in ``extra``;
    they must be JSON-serializable and must not contain secrets or paths that
    identify a user.
    """
    described = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "recorded_at": _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "tool": {"name": PROJECT_NAME, "version": _project_version()},
        "source_revision": _source_revision(),
    }
    for key, value in extra.items():
        if key in described:
            raise WorkloadResultError(f"environment field {key!r} is reserved")
        described[key] = value
    return described


_describe_environment = environment


def derive_status(runs: list[dict[str, Any]], errors: list[str]) -> str:
    """Return the overall status implied by the runs and top-level errors."""
    if errors:
        return "error"
    if not runs or all(item["status"] == "success" for item in runs):
        return "success"
    if any(item["status"] == "success" for item in runs):
        return "partial"
    return "failure"


def build_result(
    *,
    workload: str,
    kind: str,
    producer: str,
    parameters: dict[str, Any] | None = None,
    runs: Iterable[dict[str, Any]] = (),
    measurements: dict[str, dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
    status: str | None = None,
    errors: Iterable[str] = (),
    details: dict[str, Any] | None = None,
    producer_schema_version: int | None = None,
    fixture_version: int | None = None,
    description: str | None = None,
    privacy: str | None = None,
) -> dict[str, Any]:
    """Assemble and validate a complete workload result."""
    run_list = list(runs)
    error_list = list(errors)
    workload_block: dict[str, Any] = {
        "name": workload,
        "kind": kind,
        "producer": producer,
        "parameters": dict(parameters or {}),
    }
    if producer_schema_version is not None:
        workload_block["producer_schema_version"] = producer_schema_version
    if fixture_version is not None:
        workload_block["fixture_version"] = fixture_version
    if description is not None:
        workload_block["description"] = description
    if privacy is not None:
        workload_block["privacy"] = privacy
    result: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "workload": workload_block,
        "environment": environment if environment is not None else _describe_environment(),
        "status": status if status is not None else derive_status(run_list, error_list),
        "errors": error_list,
        "measurements": dict(measurements or {}),
        "runs": run_list,
    }
    if details is not None:
        result["details"] = details
    return validate_result(result)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise WorkloadResultError(message)


def _validate_measurement(value: Any, where: str, unit: str | None = None) -> None:
    _expect(isinstance(value, dict), f"{where} must be an object")
    _expect(value.get("unit") in UNITS, f"{where}.unit must be one of {', '.join(UNITS)}")
    if unit is not None:
        _expect(value["unit"] == unit, f"{where}.unit must be {unit!r}")
    present = [key for key in value if key in STATISTICS]
    _expect(bool(present), f"{where} needs at least one statistic")
    for key, item in value.items():
        if key in STATISTICS:
            _expect(item is None or _is_number(item), f"{where}.{key} must be a finite number or null")
        elif key == "samples":
            _expect(isinstance(item, int) and not isinstance(item, bool) and item >= 0, f"{where}.samples must be a non-negative integer")
        elif key == "note":
            _expect(isinstance(item, str) and bool(item), f"{where}.note must be a non-empty string")
        elif key != "unit":
            raise WorkloadResultError(f"{where}.{key} is not a recognised measurement field")


def _validate_measurements(block: Any, where: str) -> None:
    _expect(isinstance(block, dict), f"{where} must be an object")
    for name, value in block.items():
        _expect(isinstance(name, str) and bool(_NAME.match(name)), f"{where} key {name!r} must match {_NAME.pattern}")
        _validate_measurement(value, f"{where}.{name}", CANONICAL_MEASUREMENTS.get(name))


def _validate_run(item: Any, where: str) -> None:
    _expect(isinstance(item, dict), f"{where} must be an object")
    _expect(set(item) == {"id", "labels", "status", "measurements", "phases", "errors"}, f"{where} must have exactly id, labels, status, measurements, phases, errors")
    _expect(isinstance(item["id"], str) and bool(item["id"]), f"{where}.id must be a non-empty string")
    _expect(isinstance(item["labels"], dict), f"{where}.labels must be an object")
    for key, value in item["labels"].items():
        _expect(isinstance(key, str) and bool(_NAME.match(key)), f"{where}.labels key {key!r} must match {_NAME.pattern}")
        _expect(isinstance(value, _LABEL_TYPES), f"{where}.labels.{key} must be a string, number, boolean, or null")
    _expect(item["status"] in STATUSES, f"{where}.status must be one of {', '.join(STATUSES)}")
    _validate_measurements(item["measurements"], f"{where}.measurements")
    _expect(isinstance(item["phases"], list), f"{where}.phases must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(item["phases"]):
        label = f"{where}.phases[{index}]"
        _expect(isinstance(entry, dict) and isinstance(entry.get("name"), str) and bool(_NAME.match(entry["name"])), f"{label}.name must match {_NAME.pattern}")
        _expect(entry["name"] not in seen, f"{label}.name {entry['name']!r} is duplicated")
        seen.add(entry["name"])
        _validate_measurement({key: value for key, value in entry.items() if key != "name"}, label, "seconds")
    _expect(isinstance(item["errors"], list) and all(isinstance(error, str) for error in item["errors"]), f"{where}.errors must be a list of strings")


def validate_result(result: Any) -> dict[str, Any]:
    """Raise ``WorkloadResultError`` unless ``result`` satisfies the format."""
    _expect(isinstance(result, dict), "result must be an object")
    _expect(result.get("format") == FORMAT, f"format must be {FORMAT!r}")
    _expect(result.get("format_version") == FORMAT_VERSION, f"format_version must be {FORMAT_VERSION}")
    required = {"format", "format_version", "workload", "environment", "status", "errors", "measurements", "runs"}
    missing = required - set(result)
    _expect(not missing, f"result is missing {', '.join(sorted(missing))}")
    unknown = set(result) - required - {"details"}
    _expect(not unknown, f"result has unexpected fields {', '.join(sorted(unknown))}")

    workload = result["workload"]
    _expect(isinstance(workload, dict), "workload must be an object")
    for key in ("name", "producer"):
        _expect(isinstance(workload.get(key), str) and bool(workload[key]), f"workload.{key} must be a non-empty string")
    _expect(workload.get("kind") in KINDS, f"workload.kind must be one of {', '.join(KINDS)}")
    _expect(isinstance(workload.get("parameters"), dict), "workload.parameters must be an object")
    for key in ("producer_schema_version", "fixture_version"):
        if key in workload:
            _expect(isinstance(workload[key], int) and not isinstance(workload[key], bool) and workload[key] >= 0, f"workload.{key} must be a non-negative integer")
    for key in ("description", "privacy"):
        if key in workload:
            _expect(isinstance(workload[key], str) and bool(workload[key]), f"workload.{key} must be a non-empty string")
    allowed = {"name", "kind", "producer", "parameters", "producer_schema_version", "fixture_version", "description", "privacy"}
    extra = set(workload) - allowed
    _expect(not extra, f"workload has unexpected fields {', '.join(sorted(extra))}")

    env = result["environment"]
    _expect(isinstance(env, dict), "environment must be an object")
    for key in ("python_version", "platform", "recorded_at"):
        _expect(isinstance(env.get(key), str) and bool(env[key]), f"environment.{key} must be a non-empty string")

    _expect(result["status"] in STATUSES, f"status must be one of {', '.join(STATUSES)}")
    _expect(isinstance(result["errors"], list) and all(isinstance(error, str) for error in result["errors"]), "errors must be a list of strings")
    _validate_measurements(result["measurements"], "measurements")
    _expect(isinstance(result["runs"], list), "runs must be a list")
    ids: set[str] = set()
    for index, item in enumerate(result["runs"]):
        _validate_run(item, f"runs[{index}]")
        _expect(item["id"] not in ids, f"runs[{index}].id {item['id']!r} is duplicated")
        ids.add(item["id"])
    if "details" in result:
        _expect(isinstance(result["details"], dict), "details must be an object")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkloadResultError(f"result is not JSON-serializable: {exc}") from exc
    return result


def load_result(path: Path) -> dict[str, Any]:
    """Read and validate a workload result file."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadResultError(f"cannot read workload result {path}: {exc}") from exc
    return validate_result(value)


def add_result_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--result`` option to a producer's argument parser."""
    parser.add_argument(
        "--result",
        metavar="PATH",
        help=(
            f"Write a versioned {FORMAT} JSON document to PATH; use '-' for stdout, "
            "in which case the human-readable report moves to stderr"
        ),
    )


def result_to_stdout(result_path: str | os.PathLike[str] | None) -> bool:
    """Return whether ``--result`` asked for the JSON document on stdout."""
    return result_path is not None and str(result_path) == "-"


def report_stream(result_path: str | os.PathLike[str] | None) -> TextIO:
    """Return where the human-readable report should go for this invocation."""
    return sys.stderr if result_to_stdout(result_path) else sys.stdout


def write_result(result: dict[str, Any], result_path: str | os.PathLike[str] | None) -> None:
    """Validate ``result`` and write it to the requested destination, if any."""
    if result_path is None:
        return
    text = json.dumps(validate_result(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if result_to_stdout(result_path):
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_summary(result: dict[str, Any]) -> str:
    """Return a one-line human-readable description of a result."""
    workload = result["workload"]
    return (
        f"workload={workload['name']} kind={workload['kind']} producer={workload['producer']} "
        f"status={result['status']} runs={len(result['runs'])} errors={len(result['errors'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate coding-agent workload result files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Workload result JSON files to validate")
    args = parser.parse_args(argv)
    failures = 0
    for path in args.paths:
        try:
            result = load_result(path)
        except WorkloadResultError as exc:
            failures += 1
            print(f"{path}: INVALID: {exc}", file=sys.stderr)
            continue
        print(f"{path}: {format_summary(result)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
