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
      "experiment": {"group": "...", "metadata": {key: scalar}},  # optional
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

The optional ``experiment`` block assigns a document to a named group of
related runs (a sweep, an A/B study, a regression series) and carries flat
metadata describing what varied: task type, repository revision, agent
configuration, model, prompt or policy variant, workload size, start time.
Metadata values are scalars, keys that look like credentials are stored as
``REDACTED``, and every producer accepts ``--experiment``, ``--metadata``, and
``--redact`` (see ``add_result_argument``).  Listing and comparison tools
filter documents by group and metadata field with ``experiment_matches``.
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
# Which direction is better for each canonical measurement, for consumers that
# judge a change.  ``None`` means a change cannot be judged: ``input_bytes`` is
# the size of the work, not a cost of doing it.
CANONICAL_DIRECTIONS: dict[str, str | None] = {
    "wall_time_seconds": "lower",
    "queue_wait_seconds": "lower",
    "tool_call_seconds": "lower",
    "tool_calls": "lower",
    "repeated_commands": "lower",
    "input_bytes": None,
    "output_bytes": "lower",
    "context_bytes": "lower",
    "retained_summary_bytes": "lower",
    "estimated_tokens": "lower",
    "retained_summary_tokens": "lower",
    "input_tokens": "lower",
    "output_tokens": "lower",
    "peak_rss_bytes": "lower",
    "rss_delta_bytes": "lower",
    "success_rate": "higher",
    "throughput_per_second": "higher",
}
# Labels with conventional values.  ``cache_state`` distinguishes cold and warm
# measurements; the others identify what a run measured.
CANONICAL_LABELS = ("cache_state", "mode", "task_id", "repetition", "line_count", "profile")
# Experiment metadata keys with a conventional meaning, so listings and
# filters line up across producers.  ``started_at`` (UTC ISO 8601) orders the
# documents of a group; without it ``environment.recorded_at`` is used.
CANONICAL_METADATA = (
    "task_type",
    "repository_revision",
    "agent_configuration",
    "model",
    "tool_version",
    "variant",
    "environment",
    "workload_size",
    "started_at",
)
# Metadata values are identifiers, not content: a bound keeps prompts,
# transcripts, and command output from being pasted into a shared document.
MAX_METADATA_LENGTH = 256
# Value stored for metadata whose key names a credential or that a producer
# was asked to redact; the key stays visible so readers know it was set.
REDACTED = "[redacted]"
_SENSITIVE_PARTS = frozenset({
    "secret", "secrets", "password", "passwd", "credential", "credentials",
    "apikey", "authorization", "bearer",
})
_SENSITIVE_KEYS = frozenset({"token", "key"})
_SENSITIVE_SUFFIXES = ("_token", "_key")

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# PEP 621 version line; a regex keeps this module free of tomllib, which
# Python 3.10 lacks.
_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
LABEL_TYPES = (str, int, float, bool, type(None))


class WorkloadResultError(ValueError):
    """Raised when a result does not satisfy the format."""


# --------------------------------------------------------------------------
# Experiment groups and metadata


def is_sensitive_metadata_key(key: str) -> bool:
    """Return whether a metadata key names something that must not be stored.

    Matches whole underscore-separated parts (``client_secret``,
    ``password``), the bare keys ``token`` and ``key``, and the suffixes
    ``_token`` and ``_key`` (``access_token``, ``api_key``), never substrings,
    so ``max_tokens`` and ``token_budget`` stay ordinary metadata.
    """
    return (
        key in _SENSITIVE_KEYS
        or key.endswith(_SENSITIVE_SUFFIXES)
        or any(part in _SENSITIVE_PARTS for part in key.split("_"))
    )


def _check_metadata_key(key: Any, where: str) -> None:
    _expect(isinstance(key, str) and bool(_NAME.match(key)), f"{where} key {key!r} must match {_NAME.pattern}")


def _check_metadata_value(key: str, value: Any, where: str) -> None:
    _expect(isinstance(value, LABEL_TYPES), f"{where}.{key} must be a string, number, boolean, or null")
    if isinstance(value, str):
        _expect(len(value) <= MAX_METADATA_LENGTH, f"{where}.{key} must be at most {MAX_METADATA_LENGTH} characters")
    elif isinstance(value, float):
        _expect(math.isfinite(value), f"{where}.{key} must be a finite number")
    _expect(
        not is_sensitive_metadata_key(key) or value == REDACTED,
        f"{where}.{key} looks like a credential and must be omitted or redacted",
    )
    if key == "started_at":
        _expect(
            isinstance(value, str) and parse_timestamp(value) is not None,
            f"{where}.started_at must be an ISO 8601 timestamp with a UTC offset, such as 2026-09-18T10:00:00+00:00",
        )


def parse_timestamp(text: str) -> _datetime.datetime | None:
    """Parse an ISO 8601 timestamp with a UTC offset (``Z`` accepted) as an aware datetime, or ``None``."""
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = _datetime.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def parse_key_value(text: str, where: str = "metadata") -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` into a scalar: JSON when it parses, else a string."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise WorkloadResultError(f"{where} {text!r} must be KEY=VALUE")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if not isinstance(value, LABEL_TYPES):
        raise WorkloadResultError(f"{where} {text!r} must be a string, number, boolean, or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkloadResultError(f"{where} {text!r} must be a finite number")
    return key, value


def parse_metadata(text: str) -> tuple[str, Any]:
    """Parse one ``--metadata KEY=VALUE`` argument and check it fits the format.

    Credential-looking keys are accepted here because ``experiment`` redacts
    them; every other rule is enforced now so a bad value fails before the
    workload runs rather than when its result is written.
    """
    key, value = parse_key_value(text, "metadata")
    _check_metadata_key(key, "metadata")
    if not is_sensitive_metadata_key(key):
        _check_metadata_value(key, value, "metadata")
    return key, value


def experiment_group(text: str) -> str:
    """Check an experiment group name given on a command line."""
    if not text:
        raise WorkloadResultError("experiment group must not be empty")
    return text


def metadata_key(text: str) -> str:
    """Check a bare metadata key given on a command line."""
    _check_metadata_key(text, "metadata")
    return text


def experiment(
    group: str | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    redact: Iterable[str] = (),
) -> dict[str, Any]:
    """Return an experiment block, redacting credentials and requested keys.

    ``group`` names the experiment or run group; ``None`` records metadata
    alone.  Keys in ``redact`` and keys that look like credentials keep their
    name but store ``REDACTED`` so a document never carries the value.
    """
    if group is not None:
        _expect(isinstance(group, str) and bool(group), "experiment group must be a non-empty string or None")
    hidden = set(redact)
    described: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        _check_metadata_key(key, "experiment.metadata")
        if key in hidden or is_sensitive_metadata_key(key):
            value = REDACTED
        _check_metadata_value(key, value, "experiment.metadata")
        described[key] = value
    return {"group": group, "metadata": described}


def experiment_block(result: dict[str, Any]) -> dict[str, Any]:
    """Return a result's experiment block, or the empty block when it has none."""
    return result.get("experiment") or {"group": None, "metadata": {}}


def merge_experiment(existing: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine a producer's experiment block with one given on the command line.

    The command line wins key by key: its group replaces the producer's only
    when set, and its metadata keys replace matching producer keys while the
    rest are kept.
    """
    if existing is None or override is None:
        return override if existing is None else existing
    group = override["group"] if override["group"] is not None else existing["group"]
    return {"group": group, "metadata": {**existing["metadata"], **override["metadata"]}}


def value_matches(value: Any, wanted: Any) -> bool:
    """Return whether a label or metadata value equals a filter value.

    Booleans only match booleans: Python's ``True == 1`` would otherwise let a
    numeric filter select boolean values and vice versa.
    """
    if isinstance(value, bool) or isinstance(wanted, bool):
        return isinstance(value, bool) and isinstance(wanted, bool) and value is wanted
    return value == wanted


def experiment_matches(
    result: dict[str, Any],
    groups: Iterable[str] = (),
    where: Iterable[tuple[str, Any]] = (),
) -> bool:
    """Return whether a result belongs to one of ``groups`` and has every ``where`` value.

    An empty ``groups`` accepts any document, including ungrouped ones;
    metadata that is absent never matches, even a filter for ``null``.
    """
    block = experiment_block(result)
    wanted = list(groups)
    if wanted and block["group"] not in wanted:
        return False
    metadata = block["metadata"]
    return all(key in metadata and value_matches(metadata[key], value) for key, value in where)


def experiment_timestamp(result: dict[str, Any]) -> str:
    """Return the time that orders a document within its group, as a UTC ISO 8601 string.

    ``started_at`` metadata (validated as an aware timestamp) wins; otherwise
    ``environment.recorded_at`` is used, read as UTC when it has no offset.
    A ``recorded_at`` that is not ISO 8601 is returned as written.
    """
    started = experiment_block(result)["metadata"].get("started_at")
    recorded = result["environment"]["recorded_at"]
    if isinstance(started, str) and (parsed := parse_timestamp(started)) is not None:
        return parsed.astimezone(_datetime.timezone.utc).isoformat()
    parsed = parse_timestamp(recorded) or parse_timestamp(recorded + "+00:00")
    return parsed.astimezone(_datetime.timezone.utc).isoformat() if parsed is not None else recorded


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the group, metadata, status, and per-run status of a result.

    This is the record listing tools return: enough to organise and filter
    documents without their measurements.
    """
    block = experiment_block(result)
    env = result["environment"]
    return {
        "workload": {key: result["workload"][key] for key in ("name", "kind", "producer")},
        "group": block["group"],
        "metadata": dict(block["metadata"]),
        "status": result["status"],
        "errors": list(result["errors"]),
        "recorded_at": env["recorded_at"],
        "ordered_at": experiment_timestamp(result),
        "source_revision": env.get("source_revision"),
        "tool_version": (env.get("tool") or {}).get("version"),
        "runs": [
            {"id": item["id"], "labels": dict(item["labels"]), "status": item["status"], "errors": list(item["errors"])}
            for item in result["runs"]
        ],
    }


def redact_summary(summary: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return a copy of a summary with the named metadata values replaced by ``REDACTED``."""
    hidden = set(keys)
    metadata = {key: REDACTED if key in hidden else value for key, value in summary["metadata"].items()}
    return {**summary, "metadata": metadata}


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


def statistic(measurement: dict[str, Any], name: str) -> float | None:
    """Return one statistic of a measurement, or ``None`` when it is unavailable.

    A statistic is unavailable when it is absent, ``null``, or the measurement
    records ``samples: 0``; consumers must treat all three the same way.
    """
    value = measurement.get(name)
    if value is None or measurement.get("samples") == 0:
        return None
    return value


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
        match = _PYPROJECT_VERSION.search(pyproject.read_text(encoding="utf-8"))
    except OSError:
        match = None
    if match:
        return match.group(1)
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
    experiment: dict[str, Any] | None = None,
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
    if experiment is not None:
        result["experiment"] = experiment
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
        _expect(isinstance(value, LABEL_TYPES), f"{where}.labels.{key} must be a string, number, boolean, or null")
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


def _validate_experiment(block: Any) -> None:
    _expect(isinstance(block, dict), "experiment must be an object")
    _expect(set(block) == {"group", "metadata"}, "experiment must have exactly group and metadata")
    group = block["group"]
    _expect(group is None or (isinstance(group, str) and bool(group)), "experiment.group must be a non-empty string or null")
    _expect(isinstance(block["metadata"], dict), "experiment.metadata must be an object")
    for key, value in block["metadata"].items():
        _check_metadata_key(key, "experiment.metadata")
        _check_metadata_value(key, value, "experiment.metadata")


def validate_result(result: Any) -> dict[str, Any]:
    """Raise ``WorkloadResultError`` unless ``result`` satisfies the format."""
    _expect(isinstance(result, dict), "result must be an object")
    _expect(result.get("format") == FORMAT, f"format must be {FORMAT!r}")
    _expect(result.get("format_version") == FORMAT_VERSION, f"format_version must be {FORMAT_VERSION}")
    required = {"format", "format_version", "workload", "environment", "status", "errors", "measurements", "runs"}
    missing = required - set(result)
    _expect(not missing, f"result is missing {', '.join(sorted(missing))}")
    unknown = set(result) - required - {"details", "experiment"}
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
    if "experiment" in result:
        _validate_experiment(result["experiment"])
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
    """Add the shared ``--result``, ``--experiment``, ``--metadata``, and ``--redact`` options."""
    parser.add_argument(
        "--result",
        metavar="PATH",
        help=(
            f"Write a versioned {FORMAT} JSON document to PATH; use '-' for stdout, "
            "in which case the human-readable report moves to stderr"
        ),
    )
    parser.add_argument(
        "--experiment",
        metavar="GROUP",
        type=argument_type(experiment_group),
        help="Assign the result document to this experiment or run group",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        type=argument_type(parse_metadata),
        default=[],
        help=(
            "Record experiment metadata in the result document (repeatable; values parse as JSON when "
            f"possible; conventional keys: {', '.join(CANONICAL_METADATA)})"
        ),
    )
    parser.add_argument(
        "--redact",
        action="append",
        metavar="KEY",
        type=argument_type(metadata_key),
        default=[],
        help=f"Store {REDACTED} instead of the value of this metadata key (repeatable)",
    )


def argument_type(parse: Any) -> Any:
    """Wrap a parser for argparse ``type=`` so ``WorkloadResultError`` becomes a usage error."""
    def convert(text: str) -> Any:
        try:
            return parse(text)
        except WorkloadResultError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    convert.__name__ = getattr(parse, "__name__", "value")
    return convert


def experiment_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the experiment block requested by ``add_result_argument`` options, if any."""
    metadata = dict(getattr(args, "metadata", None) or [])
    group = getattr(args, "experiment", None)
    if group is None and not metadata:
        return None
    return experiment(group, metadata, redact=getattr(args, "redact", None) or ())


def result_to_stdout(result_path: str | os.PathLike[str] | None) -> bool:
    """Return whether ``--result`` asked for the JSON document on stdout."""
    return result_path is not None and str(result_path) == "-"


def report_stream(result_path: str | os.PathLike[str] | None) -> TextIO:
    """Return where the human-readable report should go for this invocation."""
    return sys.stderr if result_to_stdout(result_path) else sys.stdout


def write_result(
    result: dict[str, Any],
    result_path: str | os.PathLike[str] | None,
    *,
    experiment: dict[str, Any] | None = None,
) -> None:
    """Validate ``result`` and write it to the requested destination, if any.

    ``experiment`` (usually ``experiment_from_args(args)``) is merged into the
    document before validation (see ``merge_experiment``) so every producer
    assigns groups the same way and a producer's own block is kept.
    """
    if result_path is None:
        return
    if experiment is not None:
        result = {**result, "experiment": merge_experiment(result.get("experiment"), experiment)}
    text = json.dumps(validate_result(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if result_to_stdout(result_path):
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_metadata_value(value: Any) -> str:
    """Return a label or metadata value for display: strings bare, other values JSON."""
    return value if isinstance(value, str) else json.dumps(value)


def format_metadata(metadata: dict[str, Any]) -> str:
    """Return metadata as ``key=value`` words."""
    return " ".join(f"{key}={format_metadata_value(value)}" for key, value in metadata.items())


def format_summary(result: dict[str, Any]) -> str:
    """Return a one-line human-readable description of a result."""
    workload = result["workload"]
    block = experiment_block(result)
    described = (
        f"workload={workload['name']} kind={workload['kind']} producer={workload['producer']} "
        f"status={result['status']} runs={len(result['runs'])} errors={len(result['errors'])}"
    )
    if block["group"] is not None:
        described += f" group={block['group']}"
    if block["metadata"]:
        described += " " + format_metadata(block["metadata"])
    return described


def format_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    """Return aligned text rows under a header line."""
    widths = [max(len(text) for text in column) for column in zip(headers, *rows)]
    lines = ["  ".join(text.ljust(width) for text, width in zip(headers, widths)).rstrip()]
    for row in rows:
        lines.append("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip())
    return lines


def iter_result_files(paths: Iterable[str | os.PathLike[str]]) -> Iterable[tuple[Path, dict[str, Any] | None, str | None]]:
    """Yield ``(path, result, error)`` for every result document under ``paths``.

    A file is loaded as written.  A directory is searched recursively for
    ``*.json`` files, and only those that are ``coding-agent-workload-result``
    documents are yielded; other JSON files (producer records, schedules,
    manifests) are ignored.  A document that fails validation is yielded with
    ``result`` set to ``None`` and the error text, never silently dropped.
    """
    for item in paths:
        path = Path(item)
        if path.is_dir():
            candidates = sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
            explicit = False
        else:
            candidates = [path]
            explicit = True
        for candidate in candidates:
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if explicit:
                    yield candidate, None, f"cannot read workload result {candidate}: {exc}"
                continue
            if not explicit and not (isinstance(value, dict) and value.get("format") == FORMAT):
                continue
            try:
                yield candidate, validate_result(value), None
            except WorkloadResultError as exc:
                yield candidate, None, str(exc)


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
