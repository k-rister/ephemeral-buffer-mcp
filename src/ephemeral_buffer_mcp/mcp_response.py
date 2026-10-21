"""Transport-neutral MCP result models and response-size measurement."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from .response_formatting import bounded_summary_text
from .schema_versions import TOOL_RESPONSE_SCHEMA_VERSION


class ToolErrorEnvelope(BaseModel):
    """Stable machine-readable error information for public tool results."""

    code: str
    message: str


class ToolResponseEnvelope(BaseModel):
    """Versioned structured result paired with readable MCP text content."""

    schema_version: Literal[TOOL_RESPONSE_SCHEMA_VERSION] = TOOL_RESPONSE_SCHEMA_VERSION
    status: Literal["ok", "error"]
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ToolErrorEnvelope] = None
    text: Optional[str] = None
    truncated: bool = False
    next_cursor: Optional[str] = None


COMPACT_RESULT_SUMMARY_FIELDS = (
    "capture_id",
    "execution_id",
    "label",
    "query",
    "mode",
    "source",
    "total_lines",
    "byte_size",
    "original_byte_size",
    "content_bytes",
    "capture_count",
    "match_count",
    "available_match_count",
    "returned_count",
    "omitted_count",
    "requested_start_line",
    "requested_end_line",
    "start_line",
    "end_line",
    "partial",
    "exit_code",
)


def compact_tool_text(envelope: ToolResponseEnvelope) -> str:
    """Render concise MCP text without repeating structured payload fields."""
    parts = [f"status={envelope.status}"]
    if envelope.error is not None:
        parts.append(f"error_code={envelope.error.code}")

    data = envelope.data if isinstance(envelope.data, dict) else {}
    if data.get("status") not in (None, envelope.status):
        parts.append(f"result_status={data['status']}")
    for key in COMPACT_RESULT_SUMMARY_FIELDS:
        value = data.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, str):
            value, _ = bounded_summary_text(value, 160, "... [truncated]")
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={value}")
    if envelope.error is None and isinstance(data.get("message"), str):
        message, _ = bounded_summary_text(
            data["message"], 160, "... [truncated]"
        )
        parts.append(f"message={json.dumps(message, ensure_ascii=False)}")

    if envelope.truncated or data.get("truncated"):
        parts.append("truncated=true")
    next_cursor = envelope.next_cursor or data.get("next_cursor")
    if next_cursor:
        parts.append(f"next_cursor={json.dumps(next_cursor, ensure_ascii=False)}")

    matches = data.get("matches")
    if isinstance(matches, list) and matches:
        ranges = [
            str(match["matched_range"])
            for match in matches[:5]
            if isinstance(match, dict) and match.get("matched_range") is not None
        ]
        if ranges:
            parts.append(f"matched_ranges={json.dumps(ranges, ensure_ascii=False)}")
        parts.append("match_details=structuredContent.data.matches")
        parts.append("Use get_capture_slice for full content.")
    elif "content" in data:
        parts.append("content=structuredContent.data.content")
    elif isinstance(data.get("captures"), list):
        capture_ids = [
            capture["capture_id"]
            for capture in data["captures"][:5]
            if isinstance(capture, dict) and capture.get("capture_id") is not None
        ]
        if capture_ids:
            parts.append(f"capture_ids={json.dumps(capture_ids, ensure_ascii=False)}")
        parts.append("capture_details=structuredContent.data.captures")
    elif envelope.error is not None:
        parts.append("details=structuredContent.error.message")
    else:
        parts.append("details=structuredContent")

    parts.append("Set EPHEMERAL_COMPACT_TOOL_RESULTS=false for legacy text output.")
    return "Compact result: " + "; ".join(parts)


def call_tool_result_bytes(
    envelope: ToolResponseEnvelope,
    *,
    compact_text: bool = False,
) -> int:
    """Measure the serialized MCP result shape without importing the MCP SDK."""
    text = compact_tool_text(envelope) if compact_text else (envelope.text or "")
    result = {
        "_meta": None,
        "content": [
            {"type": "text", "text": text, "annotations": None, "_meta": None}
        ],
        "structuredContent": envelope.model_dump(
            mode="json", by_alias=True, exclude={"text"}
        ),
        "isError": False,
        "resultType": "complete",
    }
    return len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
