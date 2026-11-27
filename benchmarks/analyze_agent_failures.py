#!/usr/bin/env python3
"""Classify agent A/B timeout and validation failures without retaining content.

The analyzer reads the privacy-safe records envelope, content-free MCP lifecycle
logs, and local failure traces.  Raw trace content is used only to derive
structural counts; prompts, commands, argument values, tool results, and
signature digests are never copied to the output document.
"""

import argparse
import datetime as datetime_module
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import benchmarks.workload_results as wr


FORMAT = "agent-failure-analysis"
FORMAT_VERSION = 1
DEFAULT_INACTIVITY_THRESHOLD_SECONDS = 300.0
NEAR_DEADLINE_SECONDS = 60.0
TOOL_EVENT_NAMES = {
    "mcp_tool_started",
    "mcp_tool_completed",
    "mcp_tool_result_failed",
    "mcp_tool_failed",
}
TERMINAL_TIMEOUT_EVENT = "mcp_invocation_timeout"
PRIVACY = (
    "contains only task identity, classification category, timing and structural "
    "counts, last content-free tool state, provider usage sample counts, and "
    "validation groupings; prompts, commands, argument values, tool results, "
    "transcripts, and signature digests are excluded"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _run_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["task_id"]),
        int(record["repetition"]),
        str(record["mode"]),
    )


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        raise ValueError("records input must contain a runs array of objects")
    return runs


def _schedule_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        return []
    items = schedule.get("schedule")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return []
    return items


def _sequence_by_key(items: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], int]:
    result = {}
    for item in items:
        try:
            result[_run_key(item)] = int(item["sequence"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _lifecycle_path(root: Path | None, sequence: int | None, mode: str) -> Path | None:
    if root is None or sequence is None:
        return None
    return Path(root) / f"{sequence:03d}-{mode}.jsonl"


def _failure_trace_path(root: Path | None, sequence: int | None, mode: str) -> Path | None:
    if root is None or sequence is None:
        return None
    return Path(root) / f"{sequence:04d}-{mode}" / "stdout.jsonl"


def _parse_timestamp(value: Any) -> datetime_module.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime_module.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(datetime_module.timezone.utc)


def _timestamp_text(value: datetime_module.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(datetime_module.timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_call_id(value: Any) -> str | None:
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value)
    return None


def _jsonl_objects(path: Path | None) -> tuple[list[dict[str, Any]], int]:
    if path is None or not path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    malformed = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return [], 1
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            malformed += 1
    return events, malformed


def _lifecycle_summary(path: Path | None) -> dict[str, Any]:
    events, malformed = _jsonl_objects(path)
    started: set[str] = set()
    finished: set[str] = set()
    timestamps: list[datetime_module.datetime] = []
    last_tool_event = None
    last_tool_name = None
    last_tool_call_id = None
    has_tool_event = False
    has_lifecycle_event = False
    for event in events:
        has_lifecycle_event = True
        name = event.get("event")
        if name not in TOOL_EVENT_NAMES:
            continue
        has_tool_event = True
        parsed = _parse_timestamp(event.get("timestamp"))
        if parsed is not None:
            timestamps.append(parsed)
        last_tool_event = name
        tool = event.get("tool")
        if isinstance(tool, str) and tool:
            last_tool_name = tool
        call_id = _safe_call_id(event.get("call_id"))
        if call_id is not None:
            last_tool_call_id = call_id
            if name == "mcp_tool_started":
                started.add(call_id)
            else:
                finished.add(call_id)
    return {
        "last_tool_event": last_tool_event,
        "last_tool_name": last_tool_name,
        "last_tool_call_id": last_tool_call_id,
        "pending_tool_calls": len(started - finished),
        "no_tool_event": has_lifecycle_event and not has_tool_event,
        "activity_timestamps": timestamps,
        "malformed_lines": malformed,
    }


def _metadata_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": None, "malformed": True}
    if not isinstance(value, dict):
        return {"schema_version": None, "malformed": True}
    schema_version = value.get("schema_version")
    if schema_version not in (1, 2):
        return {"schema_version": None, "malformed": True}
    timing = value.get("timing") if isinstance(value.get("timing"), dict) else {}
    return {
        "schema_version": int(schema_version),
        "malformed": False,
        "started_at": _parse_timestamp(timing.get("started_at")),
        "ended_at": _parse_timestamp(timing.get("ended_at")),
        "duration_seconds": _finite_number(timing.get("duration_seconds")),
        "timeout_seconds": _finite_number(timing.get("timeout_seconds")),
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _item_signature(item: dict[str, Any]) -> tuple[Any, ...] | None:
    item_type = item.get("type")
    if item_type == "command_execution":
        command = item.get("command")
        if isinstance(command, str) and command:
            return ("command", command)
        return None
    if item_type == "mcp_tool_call":
        tool = item.get("tool")
        if not isinstance(tool, str) or not tool:
            return None
        arguments = item.get("arguments")
        argument_names = tuple(sorted(arguments)) if isinstance(arguments, dict) else ()
        server = item.get("server")
        return ("tool", server if isinstance(server, str) else "", tool, argument_names)
    return None


def _trace_summary(path: Path | None) -> dict[str, Any]:
    events, malformed = _jsonl_objects(path)
    started_by_id: dict[Any, tuple[Any, ...]] = {}
    completed_ids: set[Any] = set()
    repetitions: Counter[tuple[Any, ...]] = Counter()
    distinct_actions: set[tuple[Any, ...]] = set()
    for event in events:
        if event.get("type") not in {"item.started", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        signature = _item_signature(item)
        if signature is None:
            continue
        distinct_actions.add(signature)
        item_id = item.get("id")
        if event["type"] == "item.started":
            if item_id is not None:
                started_by_id[item_id] = signature
            repetitions[signature] += 1
        elif item_id is not None:
            completed_ids.add(item_id)
    incomplete_repetitions = Counter(
        signature
        for item_id, signature in started_by_id.items()
        if item_id not in completed_ids
    )
    repeated_signatures = sum(
        1 for signature, count in incomplete_repetitions.items() if count >= 3
    )
    return {
        "repeated_signature_count": repeated_signatures,
        "distinct_action_count": len(distinct_actions),
        "malformed_lines": malformed,
    }


def _last_tool_state(
    record: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    summary = record.get("mcp_invocation_timeout")
    if not isinstance(summary, dict):
        summary = {}
    return {
        "last_tool_event": summary.get("last_tool_event", lifecycle["last_tool_event"]),
        "last_tool_name": summary.get("last_tool_name", lifecycle["last_tool_name"]),
        "last_tool_call_id": summary.get("last_tool_call_id", lifecycle["last_tool_call_id"]),
    }


def _timing(
    record: dict[str, Any],
    metadata: dict[str, Any] | None,
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    metadata = metadata or {}
    started_at = metadata.get("started_at")
    ended_at = metadata.get("ended_at")
    duration = metadata.get("duration_seconds")
    if duration is None:
        duration = _finite_number(record.get("duration_seconds"))
    if started_at is None and lifecycle["activity_timestamps"]:
        started_at = min(lifecycle["activity_timestamps"])
    if ended_at is None and started_at is not None and duration is not None:
        ended_at = started_at + datetime_module.timedelta(seconds=duration)
    return {
        "started_at": _timestamp_text(started_at),
        "ended_at": _timestamp_text(ended_at),
        "duration_seconds": duration,
        "timeout_budget_seconds": metadata.get("timeout_seconds"),
    }


def _inactivity_seconds(
    timing: dict[str, Any],
    lifecycle: dict[str, Any],
) -> float | None:
    timestamps = lifecycle["activity_timestamps"]
    if not timestamps:
        return None
    started = _parse_timestamp(timing["started_at"])
    ended = _parse_timestamp(timing["ended_at"])
    if started is None or ended is None or ended < started:
        return None
    ordered = sorted(set(timestamps))
    gaps = [
        (right - left).total_seconds()
        for left, right in zip(ordered, ordered[1:])
    ]
    gaps.append((ended - ordered[-1]).total_seconds())
    return max(gaps)


def _near_deadline(
    timing: dict[str, Any],
    lifecycle: dict[str, Any],
) -> bool:
    ended = _parse_timestamp(timing["ended_at"])
    if ended is None or not lifecycle["activity_timestamps"]:
        return False
    last_activity = max(lifecycle["activity_timestamps"])
    return 0 <= (ended - last_activity).total_seconds() <= NEAR_DEADLINE_SECONDS


def _classify_timeout(
    *,
    record: dict[str, Any],
    source_timeout: bool,
    replay: dict[str, Any] | None,
    lifecycle: dict[str, Any],
    trace: dict[str, Any],
    metadata: dict[str, Any] | None,
    inactivity_threshold_seconds: float,
) -> dict[str, Any]:
    timing = _timing(record, metadata, lifecycle)
    pending_calls = int(record.get("mcp_invocation_timeout", {}).get("pending_tool_calls", lifecycle["pending_tool_calls"])) if isinstance(record.get("mcp_invocation_timeout"), dict) else lifecycle["pending_tool_calls"]
    no_tool_event = bool(record.get("mcp_invocation_timeout", {}).get("no_tool_event", lifecycle["no_tool_event"])) if isinstance(record.get("mcp_invocation_timeout"), dict) else lifecycle["no_tool_event"]
    if source_timeout and replay is not None and replay.get("completed") is True:
        category = "not-reproduced"
    elif pending_calls > 0 or no_tool_event:
        category = "blocked"
    elif trace["repeated_signature_count"] > 0:
        category = "loop"
    elif (
        _inactivity_seconds(timing, lifecycle) is not None
        and _inactivity_seconds(timing, lifecycle) > inactivity_threshold_seconds
    ):
        category = "external-inactivity-suspected"
    elif _near_deadline(timing, lifecycle) and trace["distinct_action_count"] >= 3:
        category = "budget-limited"
    else:
        category = "insufficient-evidence"
    return {
        "task": record["task_id"],
        "repetition": record["repetition"],
        "mode": record["mode"],
        "category": category,
        **timing,
        "last_tool_state": _last_tool_state(record, lifecycle),
        "pending_call_count": pending_calls,
        "repeated_signature_count": trace["repeated_signature_count"],
        "distinct_action_count": trace["distinct_action_count"],
        "input_token_sample_count": len(record.get("input_token_samples") or []),
        "output_token_sample_count": len(record.get("output_token_samples") or []),
    }


def _record_validation_failures(
    records: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for record in records:
        failures = record.get("mcp_tool_failures")
        if not isinstance(failures, list):
            continue
        for failure in failures:
            if isinstance(failure, dict):
                yield failure


def _lifecycle_validation_failures(
    lifecycle_dir: Path | None,
) -> Iterable[dict[str, Any]]:
    if lifecycle_dir is None:
        return
    root = Path(lifecycle_dir)
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.jsonl")):
        events, _ = _jsonl_objects(path)
        for event in events:
            if event.get("event") not in {"mcp_tool_result_failed", "mcp_tool_failed"}:
                continue
            if event.get("failure_category") != "validation":
                continue
            request_shape = event.get("request_shape")
            if not isinstance(request_shape, dict):
                request_shape = {}
            yield {
                "tool": event.get("tool"),
                "error_code": event.get("error_code"),
                "request_shape": request_shape,
                "result_status": event.get("result_status"),
                "capture_id_supplied": event.get("capture_id_supplied"),
                "cursor_supplied": event.get("cursor_supplied"),
                "duration_ms": event.get("duration_ms"),
            }


def _validation_groups(
    records: Iterable[dict[str, Any]],
    *,
    lifecycle_dir: Path | None = None,
) -> list[dict[str, Any]]:
    failures = list(_lifecycle_validation_failures(lifecycle_dir))
    if not failures:
        failures = list(_record_validation_failures(records))
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for failure in failures:
        request_shape = failure.get("request_shape")
        if not isinstance(request_shape, dict):
            request_shape = {}
        key = (
            failure.get("tool"),
            failure.get("error_code"),
            tuple(sorted(request_shape.items())),
            failure.get("result_status"),
            failure.get("capture_id_supplied"),
            failure.get("cursor_supplied"),
        )
        duration = _finite_number(failure.get("duration_ms"))
        if duration is not None:
            grouped[key].append(duration)
    result = []
    for key, durations in sorted(grouped.items(), key=lambda item: repr(item[0])):
        tool, error_code, shape, result_status, capture_id_supplied, cursor_supplied = key
        result.append({
            "tool": tool,
            "error_code": error_code,
            "request_shape": dict(shape),
            "result_status": result_status,
            "capture_id_supplied": capture_id_supplied,
            "cursor_supplied": cursor_supplied,
            "count": len(durations),
            "mean_duration_ms": sum(durations) / len(durations),
        })
    return result


def build_analysis(
    *,
    records_payload: dict[str, Any],
    lifecycle_dir: Path | None,
    failure_traces_dir: Path | None,
    timeout_source_payload: dict[str, Any] | None,
    inactivity_threshold_seconds: float = DEFAULT_INACTIVITY_THRESHOLD_SECONDS,
) -> dict[str, Any]:
    """Build the privacy-safe analysis document."""
    if inactivity_threshold_seconds <= 0:
        raise ValueError("inactivity threshold must be positive")
    records = _records(records_payload)
    schedule = _schedule_items(records_payload)
    sequences = _sequence_by_key(schedule)
    records_by_key = {
        _run_key(record): record
        for record in records
        if all(field in record for field in ("task_id", "repetition", "mode"))
    }
    source_timeout_keys = set()
    if timeout_source_payload is not None:
        source_timeout_keys = {
            _run_key(record)
            for record in _records(timeout_source_payload)
            if record.get("failure_reason") == "timeout"
        }
    timeout_keys = source_timeout_keys or {
        _run_key(record)
        for record in records
        if record.get("failure_reason") == "timeout"
    }
    classifications = []
    lifecycle_files = 0
    trace_files = 0
    malformed_lifecycle_lines = 0
    malformed_trace_lines = 0
    malformed_metadata = 0
    for key in sorted(timeout_keys):
        replay = records_by_key.get(key)
        if replay is None:
            continue
        sequence = sequences.get(key)
        lifecycle_path = _lifecycle_path(lifecycle_dir, sequence, key[2])
        trace_path = _failure_trace_path(failure_traces_dir, sequence, key[2])
        metadata_path = trace_path.parent / "metadata.json" if trace_path is not None else None
        lifecycle = _lifecycle_summary(lifecycle_path)
        trace = _trace_summary(trace_path)
        metadata = _metadata_summary(metadata_path)
        if lifecycle_path is not None and lifecycle_path.is_file():
            lifecycle_files += 1
        if trace_path is not None and trace_path.is_file():
            trace_files += 1
        malformed_lifecycle_lines += lifecycle["malformed_lines"]
        malformed_trace_lines += trace["malformed_lines"]
        if metadata is not None and metadata.get("malformed"):
            malformed_metadata += 1
        classifications.append(
            _classify_timeout(
                record=replay,
                source_timeout=key in source_timeout_keys,
                replay=replay,
                lifecycle=lifecycle,
                trace=trace,
                metadata=metadata,
                inactivity_threshold_seconds=inactivity_threshold_seconds,
            )
        )
    validation = _validation_groups(records, lifecycle_dir=lifecycle_dir)
    category_counts = Counter(item["category"] for item in classifications)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "parameters": {
            "inactivity_threshold_seconds": inactivity_threshold_seconds,
            "near_deadline_seconds": NEAR_DEADLINE_SECONDS,
        },
        "inputs": {
            "lifecycle_files_read": lifecycle_files,
            "failure_trace_files_read": trace_files,
            "malformed_lifecycle_lines": malformed_lifecycle_lines,
            "malformed_failure_trace_lines": malformed_trace_lines,
            "malformed_failure_metadata_files": malformed_metadata,
        },
        "summary": {
            "timeout_classifications": len(classifications),
            "timeout_categories": dict(sorted(category_counts.items())),
            "validation_failure_events": sum(group["count"] for group in validation),
            "validation_failure_groups": len(validation),
        },
        "timeout_classifications": classifications,
        "validation_failure_groups": validation,
        "privacy": PRIVACY,
    }


def analysis_workload_result(analysis: dict[str, Any]) -> dict[str, Any]:
    """Return a tool-agnostic result document for the analysis."""
    return wr.build_result(
        workload=FORMAT,
        kind="evaluation",
        producer="analyze_agent_failures.py",
        producer_schema_version=FORMAT_VERSION,
        parameters=dict(analysis["parameters"]),
        measurements={
            "timeout_classifications": wr.measurement(
                "count", value=analysis["summary"]["timeout_classifications"]
            ),
            "validation_failure_events": wr.measurement(
                "count", value=analysis["summary"]["validation_failure_events"]
            ),
        },
        runs=[
            wr.run(
                "failure-analysis",
                labels={"result": "privacy-safe"},
                measurements={
                    "timeout_classifications": wr.measurement(
                        "count", value=analysis["summary"]["timeout_classifications"]
                    ),
                    "validation_failure_events": wr.measurement(
                        "count", value=analysis["summary"]["validation_failure_events"]
                    ),
                },
            )
        ],
        details=analysis,
        privacy=analysis["privacy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--lifecycle-dir", type=Path)
    parser.add_argument("--failure-traces-dir", type=Path)
    parser.add_argument("--timeout-source-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inactivity-threshold",
        type=float,
        default=DEFAULT_INACTIVITY_THRESHOLD_SECONDS,
    )
    wr.add_result_argument(parser)
    args = parser.parse_args()
    if args.inactivity_threshold <= 0:
        parser.error("--inactivity-threshold must be positive")
    try:
        records_payload = _read_json(args.records)
        source_payload = (
            _read_json(args.timeout_source_records)
            if args.timeout_source_records is not None
            else None
        )
        analysis = build_analysis(
            records_payload=records_payload,
            lifecycle_dir=args.lifecycle_dir,
            failure_traces_dir=args.failure_traces_dir,
            timeout_source_payload=source_payload,
            inactivity_threshold_seconds=args.inactivity_threshold,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({
            "timeout_classifications": analysis["summary"]["timeout_classifications"],
            "validation_failure_events": analysis["summary"]["validation_failure_events"],
            "output": str(args.output),
        }),
        file=wr.report_stream(args.result),
    )
    if args.result:
        wr.write_result(
            analysis_workload_result(analysis),
            args.result,
            experiment=wr.experiment_from_args(args),
        )


if __name__ == "__main__":
    main()
