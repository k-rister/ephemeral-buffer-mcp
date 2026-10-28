"""Execution response helpers used by the execution tool handlers."""

from __future__ import annotations

import errno
import json
import math
from contextvars import copy_context
from typing import Any, Dict, Optional

from ..execution import (
    ExecutionBusyError,
    ExecutionDiskSpaceError,
    ExecutionPhaseNotFoundError,
    ExecutionQuotaError,
    ExecutionRecordError,
)
from ..mcp_response import (
    ToolErrorEnvelope,
    ToolResponseEnvelope,
    call_tool_result_bytes,
)
from ..response_limits import (
    EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
    MCP_TOOL_RESPONSE_MAX_BYTES,
)
from ._context import active_engine as get_active_engine
from ._context import active_execution_manager as get_active_execution_manager


def _capture_execution_phase(
    phase: Dict[str, Any],
    output: str,
    result: Dict[str, Any],
) -> Optional[str]:
    """Retain a phase result in the searchable ring as a convenience snapshot."""
    active_engine = get_active_engine()
    capture = active_engine.ingest(
        output,
        label=f"execution phase: {phase['name']}",
        content_type="auto",
        truncated=bool(result.get("truncated", False)),
        original_byte_size=(
            result.get("original_byte_size") if result.get("truncated", False) else None
        ),
        command_exit_code=result.get("exit_code"),
        timed_out=bool(result.get("timed_out", False)),
        source="execution",
        duration_ms=result.get("duration_ms"),
        structured_metrics=phase.get("structured_metrics", {}),
    )
    result["capture_session_id"] = active_engine.session_id
    return capture.capture_id


def _background_execution_output_handler():
    """Preserve the calling service context for phase captures on worker threads."""
    context = copy_context()

    def handle(phase, output, result):
        return context.run(_capture_execution_phase, phase, output, result)

    return handle


def _execution_error_json(
    code: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    error: Dict[str, Any] = {"code": code, "message": message or "Execution operation failed."}
    if details:
        error["details"] = details
    return json.dumps(
        {"status": "error", "error": error},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _execution_exception_json(exc: Exception) -> str:
    """Map execution exceptions to stable codes without inspecting messages."""
    details = None
    if isinstance(exc, ExecutionBusyError):
        code = "execution_busy"
    elif isinstance(exc, ExecutionRecordError):
        code = (
            "unsupported_schema_version"
            if exc.reason_code == "UNSUPPORTED_SCHEMA_VERSION"
            else "malformed_record"
        )
        details = _execution_record_error_details(
            exc.reason_code,
            exc.schema_version,
        )
    elif isinstance(exc, ExecutionQuotaError):
        code = "quota_exceeded"
    elif isinstance(exc, ExecutionDiskSpaceError):
        code = "disk_full"
    elif isinstance(exc, ExecutionPhaseNotFoundError):
        code = "phase_not_found"
    elif isinstance(exc, KeyError):
        code = "execution_not_found"
    elif isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        code = "disk_full"
    elif isinstance(exc, OSError):
        code = "execution_storage_error"
    else:
        code = "execution_failed"
    return _execution_error_json(code, str(exc), details=details)


def _execution_record_error_details(
    reason_code: Any,
    schema_version: Any,
) -> Dict[str, Any]:
    """Return bounded, JSON-safe diagnostic fields for an invalid record."""
    details: Dict[str, Any] = {}
    if isinstance(reason_code, str):
        details["reason_code"] = reason_code
    if isinstance(schema_version, (str, int, float, bool)) and (
        not isinstance(schema_version, float) or math.isfinite(schema_version)
    ):
        if isinstance(schema_version, str):
            schema_version = schema_version.encode("utf-8", errors="replace")[:64].decode(
                "utf-8", errors="ignore"
            )
        details["schema_version"] = schema_version
    return details


def _execution_payload_with_error_codes(payload: Any) -> Any:
    """Add stable error codes to read-only malformed-record diagnostics."""
    if not isinstance(payload, dict):
        return payload

    def annotate(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
        status = diagnostic.get("status")
        if status not in {"invalid_record", "unsupported_record"}:
            return diagnostic
        code = (
            "unsupported_schema_version"
            if diagnostic.get("reason_code") == "UNSUPPORTED_SCHEMA_VERSION"
            or status == "unsupported_record"
            else "malformed_record"
        )
        annotated = dict(diagnostic)
        error = {
            "code": code,
            "message": str(diagnostic.get("message") or "Execution record is invalid."),
        }
        details = _execution_record_error_details(
            diagnostic.get("reason_code"),
            diagnostic.get("schema_version"),
        )
        if details:
            error["details"] = details
        annotated["error"] = error
        return annotated

    annotated = annotate(payload)
    executions = payload.get("executions")
    if isinstance(executions, list):
        annotated = dict(annotated)
        annotated["executions"] = [
            annotate(item) if isinstance(item, dict) else item for item in executions
        ]
    return annotated


def _execution_json(operation, *, max_response_bytes: Optional[int] = None) -> str:
    """Run an execution operation and return a compact machine-readable result."""
    try:
        payload = _execution_payload_with_error_codes(operation())
        result = _json_dumps_with_limit(payload, max_response_bytes)
        if result is None:
            if isinstance(payload, dict) and isinstance(payload.get("executions"), list):
                compact = {
                    key: payload[key]
                    for key in ("status", "limit", "offset")
                    if key in payload
                }
                compact["response_truncated"] = True
                compact["executions"] = []
                for execution in payload["executions"]:
                    if not isinstance(execution, dict):
                        continue
                    summary = {
                        key: execution[key]
                        for key in (
                            "status", "execution_id", "label", "execution_status", "partial",
                            "execution_in_progress", "cancellation_requested",
                            "background_error", "updated_at",
                            "completed_phase_count", "phase_count",
                            "reason_code", "schema_version", "read_only", "error",
                            "record_file_id", "message", "record_bytes",
                        )
                        if key in execution
                    }
                    candidate = dict(compact)
                    candidate["executions"] = [*compact["executions"], summary]
                    candidate["returned_count"] = len(candidate["executions"])
                    candidate["omitted_count"] = len(payload["executions"]) - candidate["returned_count"]
                    encoded_candidate = _json_dumps_with_limit(candidate, max_response_bytes)
                    if encoded_candidate is None:
                        break
                    compact["executions"].append(summary)
                compact["returned_count"] = len(compact["executions"])
                compact["omitted_count"] = len(payload["executions"]) - len(compact["executions"])
                compact_result = _json_dumps_with_limit(compact, max_response_bytes)
                if compact_result is not None:
                    return compact_result
            if isinstance(payload, dict) and payload.get("execution_id"):
                compact = {
                    key: payload[key]
                    for key in (
                        "status", "schema_version", "execution_id", "label",
                        "execution_status", "execution_in_progress",
                        "cancellation_requested", "partial", "summary", "created_at",
                        "background_error", "updated_at", "resume",
                        "completed_phase_count", "phase_count", "reason_code",
                        "read_only", "error", "record_bytes", "record_preview",
                        "record_preview_sanitized", "record_preview_redacted",
                        "record_preview_truncated",
                    )
                    if key in payload
                }
                compact["response_truncated"] = True
                compact["phases"] = [
                    {
                        key: phase.get(key)
                        for key in ("name", "status", "attempts", "error")
                    }
                    for phase in payload.get("phases", [])
                    if isinstance(phase, dict)
                ]
                compact_result = _json_dumps_with_limit(compact, max_response_bytes)
                if compact_result is not None:
                    return compact_result
            return _execution_error_json(
                "response_too_large",
                "Execution response exceeds the configured byte limit; request a smaller output chunk.",
            )
        return result
    except Exception as exc:
        return _execution_exception_json(exc)


def _json_dumps_with_limit(payload: Any, max_bytes: Optional[int]) -> Optional[str]:
    """Encode incrementally so an oversized response never builds in full."""
    if max_bytes is None:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    chunks = []
    total_bytes = 0
    for chunk in encoder.iterencode(payload):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > max_bytes:
            return None
        chunks.append(chunk)
    return "".join(chunks)


def _execution_public_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate session-local capture references before returning durable state."""
    active_engine = get_active_engine()
    for phase in payload.get("phases", []):
        result = phase.get("result")
        if not isinstance(result, dict) or "capture_id" not in result:
            continue
        result["capture_available"] = (
            result.get("capture_session_id") == active_engine.session_id
            and active_engine.get_capture(result["capture_id"]) is not None
        )
    return payload


def _compact_execution_public_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop verbose phase history and output while retaining execution identity."""
    compact = {
        key: payload[key]
        for key in (
            "status", "schema_version", "execution_id", "label",
            "execution_status", "execution_in_progress",
            "cancellation_requested", "partial", "summary", "created_at",
            "background_error", "updated_at", "resume",
            "completed_phase_count", "phase_count",
        )
        if key in payload
    }
    compact["response_truncated"] = True
    compact["phases"] = [
        {
            key: phase.get(key)
            for key in ("name", "status", "attempts", "error")
        }
        for phase in payload.get("phases", [])
        if isinstance(phase, dict)
    ]
    return compact


def _compact_execution_list_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce list entries while retaining each recoverable durable identity."""
    executions = payload.get("executions")
    if not isinstance(executions, list):
        return {}

    summaries = []
    for item in executions:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        execution_id = item.get("execution_id")
        record_file_id = item.get("record_file_id")
        if status == "invalid_record" and isinstance(record_file_id, str):
            summary = {
                "status": "invalid_record",
                "execution_id": execution_id if isinstance(execution_id, str) else None,
                "record_file_id": record_file_id,
            }
            for key in ("schema_version", "reason_code"):
                value = item.get(key)
                if isinstance(value, (str, int, float, bool)):
                    summary[key] = value
            if item.get("read_only") is True:
                summary["read_only"] = True
            summaries.append(summary)
            continue
        if not isinstance(execution_id, str):
            continue
        summary = {"status": status, "execution_id": execution_id}
        for key in (
            "execution_status", "partial", "execution_in_progress",
            "cancellation_requested", "completed_phase_count", "phase_count",
            "reason_code", "schema_version", "read_only",
        ):
            if key in item:
                summary[key] = item[key]
        summaries.append(summary)

    total = len(executions)
    return {
        "status": payload.get("status", "ok"),
        "limit": payload.get("limit", 1),
        "offset": payload.get("offset", 0),
        "response_truncated": True,
        "executions": summaries,
        "returned_count": len(summaries),
        "omitted_count": max(0, total - len(summaries)),
    }


def _compact_execution_output_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove output bytes while keeping the execution and phase identities."""
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return {}
    return {
        "status": payload.get("status", "ok"),
        "execution_id": payload.get("execution_id"),
        "execution_status": payload.get("execution_status"),
        "partial": payload.get("partial", False),
        "response_truncated": True,
        "phases": [
            {
                key: phase[key]
                for key in ("name", "status", "attempts", "error")
                if key in phase
            }
            for phase in phases
            if isinstance(phase, dict)
        ],
    }


def _compact_execution_retirement_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop descriptive retirement fields while retaining execution IDs."""
    compact_item_keys = (
        "execution_id", "eligible", "record_bytes", "summary_bytes",
        "projected_reclaimed_bytes",
    )

    def compact_items(items):
        if not isinstance(items, list):
            return None
        return [
            {key: item[key] for key in compact_item_keys if key in item}
            for item in items
            if isinstance(item, dict)
        ]

    items = compact_items(payload.get("items", []))
    if items is None:
        return {}
    compact = {
        key: payload[key]
        for key in (
            "status", "dry_run", "projected_reclaimed_bytes", "eligible_count",
            "blocked_count", "retired_count", "archive_bytes", "archive_is_durable",
        )
        if key in payload
    }
    compact["items"] = items
    compact["response_truncated"] = True
    retired = compact_items(payload.get("retired"))
    if retired is not None:
        compact["retired"] = retired
    removal_errors = payload.get("removal_errors")
    if isinstance(removal_errors, list):
        compact["removal_errors"] = [
            {
                key: item[key]
                for key in ("execution_id", "error")
                if key in item
            }
            for item in removal_errors
            if isinstance(item, dict)
        ]
    return compact


def _execution_compact_candidates(tool_name: str, payload: Dict[str, Any]):
    """Return typed summary shapes ordered from richest to smallest."""
    if tool_name in {
        "start_execution", "resume_execution", "cancel_execution", "get_execution",
    } and isinstance(payload.get("execution_id"), str) and isinstance(
        payload.get("phases"), list
    ):
        compact = _compact_execution_public_payload(payload)
        return [compact, {**compact, "phases": []}]
    if tool_name == "get_execution_output":
        compact = _compact_execution_output_payload(payload)
        if compact:
            return [compact, {**compact, "phases": []}]
    if tool_name == "list_executions":
        compact = _compact_execution_list_payload(payload)
        if compact:
            return [compact]
    if tool_name == "retire_executions":
        compact = _compact_execution_retirement_payload(payload)
        if compact:
            return [compact]
    return []


def _execution_identity_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep execution identifiers in the final bounded-size error when possible."""
    details: Dict[str, Any] = {}
    execution_id = payload.get("execution_id")
    if isinstance(execution_id, str):
        details["execution_id"] = execution_id
    executions = payload.get("executions")
    if isinstance(executions, list):
        identities = []
        for item in executions:
            if not isinstance(item, dict):
                continue
            identity = {}
            if isinstance(item.get("execution_id"), str):
                identity["execution_id"] = item["execution_id"]
            if isinstance(item.get("record_file_id"), str):
                identity["record_file_id"] = item["record_file_id"]
            if identity:
                identities.append(identity)
        if identities:
            details["executions"] = identities
    for key in ("items", "retired"):
        items = payload.get(key)
        if isinstance(items, list):
            ids = [
                item["execution_id"]
                for item in items
                if isinstance(item, dict) and isinstance(item.get("execution_id"), str)
            ]
            if ids:
                details["execution_ids"] = ids
                break
    return details


def _compact_execution_error_response(
    response: ToolResponseEnvelope,
) -> ToolResponseEnvelope:
    """Replace an oversized execution result with a small parseable JSON error."""
    data = response.data if isinstance(response.data, dict) else {}
    details = dict(response.error.details or {}) if response.error is not None else {}
    details.update(_execution_identity_details(data))
    code = response.error.code if response.error is not None else "response_too_large"
    message = (
        response.error.message
        if response.error is not None
        else "Execution response exceeds the MCP response byte budget."
    )
    message = message[:512]
    error_payload = {"code": code, "message": message}
    if details:
        error_payload["details"] = details
    payload = {"status": "error", "error": error_payload}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return response.model_copy(
        update={
            "status": "error",
            "data": payload,
            "error": ToolErrorEnvelope(
                code=code,
                message=message,
                details=details or None,
            ),
            "text": text,
            "truncated": True,
        }
    )


def fit_execution_mcp_response(
    tool_name: str,
    response: ToolResponseEnvelope,
) -> ToolResponseEnvelope:
    """Fit a typed execution result before generic MCP text truncation runs."""
    response_budget = MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
    if call_tool_result_bytes(response) <= response_budget:
        return response

    if response.status == "ok":
        for payload in _execution_compact_candidates(tool_name, response.data):
            text = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            candidate = response.model_copy(
                update={"data": payload, "text": text, "truncated": True}
            )
            if call_tool_result_bytes(candidate) <= response_budget:
                return candidate
    compact_error = _compact_execution_error_response(response)
    if call_tool_result_bytes(compact_error) <= response_budget:
        return compact_error
    fallback_payload = {
        "status": "error",
        "error": {
            "code": "response_too_large",
            "message": "Execution response exceeds the MCP response byte budget.",
        },
    }
    fallback_text = json.dumps(
        fallback_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return response.model_copy(
        update={
            "status": "error",
            "data": fallback_payload,
            "error": ToolErrorEnvelope(
                code="response_too_large",
                message=fallback_payload["error"]["message"],
            ),
            "text": fallback_text,
            "truncated": True,
        }
    )


def _execution_get_payload(execution_id: str, include_output: bool) -> Dict[str, Any]:
    """Avoid copying large durable output before the response budget is known."""
    metadata = get_active_execution_manager().public(execution_id, include_output=False)
    if not include_output:
        return metadata
    output_bytes = sum(
        phase.get("output_bytes", 0)
        for phase in metadata.get("phases", [])
        if isinstance(phase, dict) and isinstance(phase.get("output_bytes", 0), int)
    )
    if output_bytes > EXECUTION_OUTPUT_RESPONSE_MAX_BYTES:
        metadata["response_truncated"] = True
        return metadata
    return get_active_execution_manager().public(execution_id, include_output=True)
