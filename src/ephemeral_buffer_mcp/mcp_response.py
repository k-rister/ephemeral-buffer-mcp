"""Transport-neutral MCP result models and response-size measurement."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from .response_formatting import bounded_summary_text
from .schema_versions import TOOL_RESPONSE_SCHEMA_VERSION


COMPACT_MATCH_LIMIT = 5
COMPACT_MATCH_SNIPPET_MAX_BYTES = 1024
COMPACT_PREVIEW_MAX_BYTES = 1024
COMPACT_SLICE_CONTENT_MAX_BYTES = 4096
_QUERY_MARKER_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9_])")


def _query_marker_line(query: Any, snippet: str) -> Optional[int]:
    """Find the numbered source line containing an identifier from the query."""
    if not isinstance(query, str) or not query:
        return None
    markers = [marker.casefold() for marker in _QUERY_MARKER_RE.findall(query)]
    if not markers:
        return None
    for line in snippet.splitlines():
        prefix, separator, content = line.partition("|")
        if not separator:
            continue
        line_number = next(
            (token for token in reversed(prefix.split()) if token.isdigit()),
            None,
        )
        if line_number is None:
            continue
        if any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])",
                content,
                flags=re.IGNORECASE,
            )
            for marker in markers
        ):
            return int(line_number)
    return None


class ToolErrorEnvelope(BaseModel):
    """Stable machine-readable error information for public tool results."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ToolResponseEnvelope(BaseModel):
    """Versioned structured result paired with readable MCP text content."""

    schema_version: Literal[TOOL_RESPONSE_SCHEMA_VERSION] = TOOL_RESPONSE_SCHEMA_VERSION
    payload_schema_version: Optional[int] = Field(default=None, exclude=True, ge=1)
    status: Literal["ok", "error"]
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ToolErrorEnvelope] = None
    text: Optional[str] = None
    truncated: bool = False
    next_cursor: Optional[str] = None


def structured_content(envelope: ToolResponseEnvelope) -> Dict[str, Any]:
    """Serialize an envelope while exposing a contract version only when set."""
    result = envelope.model_dump(
        mode="json",
        by_alias=True,
        exclude={"text", "payload_schema_version"},
    )
    if envelope.payload_schema_version is not None:
        result["payload_schema_version"] = envelope.payload_schema_version
    return result


def _strip_schema_titles(schema: Any) -> None:
    """Drop optional title metadata to keep the composed tool catalog bounded."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        for value in schema.values():
            _strip_schema_titles(value)
    elif isinstance(schema, list):
        for value in schema:
            _strip_schema_titles(value)


def structured_tool_output_schema(
    payload_schema: Dict[str, Any],
    payload_schema_version: int,
) -> Dict[str, Any]:
    """Describe the complete structured MCP result around one tool payload."""
    payload = deepcopy(payload_schema)
    definitions = payload.pop("$defs", {})
    error_schema = ToolErrorEnvelope.model_json_schema()

    schema: Dict[str, Any] = {
        "additionalProperties": False,
        "properties": {
            "schema_version": {
                "const": TOOL_RESPONSE_SCHEMA_VERSION,
                "type": "integer",
            },
            "payload_schema_version": {
                "const": payload_schema_version,
                "type": "integer",
            },
            "status": {
                "enum": ["ok", "error"],
                "type": "string",
            },
            "data": {
                "anyOf": [
                    payload,
                    {
                        "additionalProperties": False,
                        "maxProperties": 0,
                        "type": "object",
                    },
                ],
            },
            "error": {
                "anyOf": [error_schema, {"type": "null"}],
            },
            "truncated": {"type": "boolean"},
            "next_cursor": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        },
        "required": [
            "schema_version",
            "payload_schema_version",
            "status",
            "data",
            "error",
            "truncated",
            "next_cursor",
        ],
        "type": "object",
    }
    if definitions:
        schema["$defs"] = definitions
    _strip_schema_titles(schema)
    return schema


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
    """Render bounded metadata and evidence for text-only MCP clients."""
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
            for match in matches[:COMPACT_MATCH_LIMIT]
            if isinstance(match, dict) and match.get("matched_range") is not None
        ]
        if ranges:
            parts.append(f"matched_ranges={json.dumps(ranges, ensure_ascii=False)}")
        for index, match in enumerate(matches[:COMPACT_MATCH_LIMIT], start=1):
            if not isinstance(match, dict):
                continue
            matched_range = match.get("matched_range")
            location = ""
            if isinstance(matched_range, str):
                bounds = matched_range.removeprefix("L").split("-L", maxsplit=1)
                if len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
                    start_line, end_line = bounds
                    location = (
                        f"Matched line {start_line}"
                        if start_line == end_line
                        else f"Matched lines {start_line}-{end_line}"
                    )
            snippet = match.get("snippet")
            if isinstance(snippet, str) and snippet:
                snippet, _ = bounded_summary_text(
                    snippet,
                    COMPACT_MATCH_SNIPPET_MAX_BYTES,
                    "... [match excerpt truncated]",
                )
            marker_line = (
                _query_marker_line(data.get("query"), snippet)
                if isinstance(snippet, str)
                else None
            )
            if marker_line is not None:
                location = f"Matched line {marker_line}"
            if location or isinstance(snippet, str):
                heading = f"Match #{index}"
                if location:
                    heading += f" — {location}"
                if isinstance(matched_range, str):
                    heading += f" ({matched_range})"
                detail = heading
                if isinstance(snippet, str) and snippet:
                    detail += f"\nMatched text (verbatim):\n{snippet}"
                parts.append(detail)
        parts.append(
            "Use get_capture_slice only if the relevant line is truncated or missing from the excerpt."
        )
    elif "content" in data:
        content = data.get("content")
        if isinstance(content, str) and content:
            # The retrieval handler bounds pages before assigning their cursor.
            # Render that whole page so text-only clients never skip source bytes.
            start_line = data.get("start_line")
            end_line = data.get("end_line")
            if (
                isinstance(start_line, int)
                and not isinstance(start_line, bool)
                and isinstance(end_line, int)
                and not isinstance(end_line, bool)
            ):
                location = (
                    f"line {start_line}"
                    if start_line == end_line
                    else f"lines {start_line}-{end_line}"
                )
                heading = f"Capture slice, {location}:"
            else:
                heading = "Capture slice:"
            parts.append(f"{heading}\n{content}")
        else:
            parts.append("content=structuredContent.data.content")
    elif isinstance(data.get("previews"), dict):
        previews = data["previews"]
        for name in ("head", "tail"):
            preview = previews.get(name)
            if isinstance(preview, str) and preview:
                preview, _ = bounded_summary_text(
                    preview,
                    COMPACT_PREVIEW_MAX_BYTES,
                    "... [preview excerpt truncated] ...",
                )
                parts.append(f"{name.capitalize()} preview:\n{preview}")
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
        "structuredContent": structured_content(envelope),
        "isError": envelope.status == "error",
        "resultType": "complete",
    }
    return len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
