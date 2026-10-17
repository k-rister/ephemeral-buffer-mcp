"""Retrieval domain handlers."""

from ..server import *  # noqa: F403
from .capture import _bounded_json_text, _bounded_summary_text, _summary_json
from .registry import register_tool


@_instrument_tool("get_capture_slice")
def _capture_slice_response(
    start_line: Annotated[int, Field(ge=1)],
    end_line: Annotated[int, Field(ge=1)],
    capture_id: str = "latest",
    max_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
    cursor: Optional[str] = None,
    __invocation_mode: str = "direct",
) -> ToolResponseEnvelope:
    if any(
        isinstance(line, bool) or not isinstance(line, int) or line < 1
        for line in (start_line, end_line)
    ):
        message = "start_line and end_line must be positive integers."
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(code="invalid_range", message=message),
            text=f"Error: {message}",
        )

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 4096 <= max_bytes <= MCP_TOOL_RESPONSE_MAX_BYTES:
        message = f"max_bytes must be between 4,096 and {MCP_TOOL_RESPONSE_MAX_BYTES:,}."
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(code="invalid_byte_budget", message=message),
            text=f"Error: {message}",
        )

    response_budget = min(max_bytes, MCP_TOOL_RESPONSE_MAX_BYTES) - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
    active_engine = _active_engine()
    capture = active_engine.get_capture(capture_id)
    label_wire_budget = min(
        SUMMARY_LABEL_MAX_BYTES,
        max(128, response_budget // 12),
    )
    label_bytes = 0
    if capture is not None:
        safe_label, _ = _bounded_json_text(
            str(capture.label),
            label_wire_budget,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        label_bytes = len(json.dumps(safe_label, ensure_ascii=False).encode("utf-8")) - 2
    # The wire result repeats returned text in both channels. Limit individual
    # segments conservatively so even JSON-escaped control characters can fit
    # in the smallest accepted response budget.
    max_segment_bytes = max(
        4,
        min(
            CAPTURE_SLICE_SEGMENT_MAX_BYTES,
            max(4, response_budget - 1536 - (2 * label_bytes)) // 12,
        ),
    )
    page = active_engine.get_slice_page(
        start_line,
        end_line,
        capture_id=capture_id,
        max_content_bytes=min(MAX_CAPTURE_SLICE_CONTENT_BYTES, max_bytes // 2),
        cursor=cursor,
        max_segment_bytes=max_segment_bytes,
    )
    if page.get("status") != "ok":
        message = str(page.get("message", "capture slice unavailable"))
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(
                code=str(page.get("error_code") or "retrieval_failed"),
                message=message,
            ),
            text=f"Error: {message}",
        )

    selected: List[Dict[str, Any]] = []
    best: Optional[ToolResponseEnvelope] = None
    segments = page["segments"]
    for index, segment in enumerate(segments):
        candidate_segments = selected + [segment]
        candidate_cursor = segment["cursor_after"]
        content = "".join(item["content_piece"] for item in candidate_segments)
        rendered_segments = []
        current_line = None
        current_text: List[str] = []
        current_complete = False
        for item in candidate_segments:
            if current_line is not None and item["line"] != current_line:
                suffix = "" if current_complete else " [continued]"
                rendered_segments.append(f"  {current_line:5d} | {''.join(current_text)}{suffix}")
                current_text = []
            current_line = item["line"]
            current_text.append(item["text"])
            current_complete = item["line_complete"]
        if current_line is not None:
            suffix = "" if current_complete else " [continued]"
            rendered_segments.append(f"  {current_line:5d} | {''.join(current_text)}{suffix}")
        is_truncated = candidate_cursor is not None
        display_content = "\n".join(rendered_segments)
        if is_truncated:
            display_content += (
                "\n... [slice truncated; continue with next_cursor='"
                + str(candidate_cursor)
                + "']"
            )
        label, _ = _bounded_json_text(
            str(page["label"]),
            label_wire_budget,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        text = (
            f"Capture: `{page['capture_id']}` ({label}) | Lines {page['start_line']} to "
            f"{segment['line']} of {page['total_lines']}\n```text\n{display_content}\n```"
        )
        data = {
            "capture_id": page["capture_id"],
            "label": label,
            "requested_start_line": page["requested_start_line"],
            "requested_end_line": page["requested_end_line"],
            "start_line": candidate_segments[0]["line"],
            "end_line": segment["line"],
            "total_lines": page["total_lines"],
            "content": content,
            "content_bytes": len(content.encode("utf-8")),
            "segments": [
                {
                    key: item[key]
                    for key in ("line", "byte_offset", "byte_length", "line_complete", "separator_after")
                }
                for item in candidate_segments
            ],
        }
        candidate = ToolResponseEnvelope(
            status="ok",
            data=data,
            text=text,
            truncated=is_truncated,
            next_cursor=candidate_cursor,
        )
        if _call_tool_result_bytes(
            candidate,
            compact_text=_compact_tool_results_enabled_for_mcp(__invocation_mode),
        ) > response_budget:
            break
        selected.append(segment)
        best = candidate

    if best is None:
        message = "The response byte budget is too small to include capture content and its continuation metadata."
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(code="response_budget_too_small", message=message),
            text=f"Error: {message}",
        )
    return best


@register_tool("get_capture_slice", "retrieval", structured_result_factory=_capture_slice_response)
def get_capture_slice(
    start_line: Annotated[int, Field(ge=1)],
    end_line: Annotated[int, Field(ge=1)],
    capture_id: str = "latest",
    max_bytes: Annotated[int, Field(ge=4096, le=MCP_TOOL_RESPONSE_MAX_BYTES)] = MCP_TOOL_RESPONSE_MAX_BYTES,
    cursor: Optional[str] = None,
) -> str:
    """Fetch a byte-bounded page of 1-indexed capture lines.

    Repeat the range and capture ID from the first call, then pass the returned
    opaque ``next_cursor`` to continue. Pages never split a Unicode character;
    the cursor's line offset is a zero-based UTF-8 byte offset.
    """
    return get_capture_slice_result(start_line, end_line, capture_id, max_bytes, cursor).text or ""


def get_capture_slice_result(
    start_line: int,
    end_line: int,
    capture_id: str = "latest",
    max_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
    cursor: Optional[str] = None,
) -> ToolResponseEnvelope:
    """Return a typed paged retrieval result for Python callers."""
    return _capture_slice_response(start_line, end_line, capture_id, max_bytes, cursor)


@register_tool("get_capture_summary", "retrieval")
@_instrument_tool("get_capture_summary")
def get_capture_summary(capture_id: str = "latest", include_previews: bool = False) -> str:
    """
    Returns a compact versioned JSON summary. Set ``include_previews`` when
    head and tail samples are needed; full output remains available through
    ``get_capture_slice``.
    """
    return _summary_json(capture_id, include_previews=include_previews)


@_instrument_tool("list_captures")
def _list_captures_response() -> ToolResponseEnvelope:
    captures = []
    text_lines = ["Active Captures in Ephemeral Buffer:"]
    for capture in _active_engine().list_captures():
        label, label_truncated = _bounded_summary_text(
            capture["label"],
            SUMMARY_LABEL_MAX_BYTES,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        captures.append({
            "capture_id": capture["capture_id"],
            "label": label,
            "label_truncated": label_truncated,
            "total_lines": capture["total_lines"],
            "byte_size": capture["byte_size"],
            "timestamp": capture["timestamp"],
        })
        text_lines.append(
            f"- `{capture['capture_id']}`: \"{label}\" | {capture['total_lines']:,} lines | "
            f"{capture['byte_size']:,} bytes | {capture['timestamp']}"
        )
    if not captures:
        text = "Ephemeral buffer is empty. No captures currently stored."
    else:
        text = "\n".join(text_lines)
    return ToolResponseEnvelope(
        status="ok",
        data={"captures": captures, "capture_count": len(captures)},
        text=text,
    )


@register_tool("list_captures", "retrieval", structured_result_factory=_list_captures_response)
def list_captures() -> str:
    """List active captures with readable text and structured capture data.

    The structured result contains ``captures`` and ``capture_count`` alongside
    the text summary. Use ``get_capture_summary`` or ``get_capture_slice`` for
    details about an individual capture.
    """
    return list_captures_result().text or ""


def list_captures_result() -> ToolResponseEnvelope:
    """Return a typed list of active captures for Python callers."""
    return _list_captures_response()


@register_tool("clear_captures", "lifecycle")
@_instrument_tool("clear_captures")
def clear_captures(capture_id: str = "all") -> str:
    """
    Clears all or a specific capture from the ephemeral buffer to free memory.
    """
    return _active_engine().clear(capture_id)
