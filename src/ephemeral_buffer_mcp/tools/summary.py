"""Shared capture summary formatting for tool domains."""

from __future__ import annotations

import json
from typing import Any, Optional

from ..mcp_response import ToolErrorEnvelope, ToolResponseEnvelope
from ._context import active_engine
from .formatting import summary_payload
from ..response_formatting import bounded_summary_text
from ..response_limits import SUMMARY_LABEL_MAX_BYTES


def summary_json(
    capture_id: str,
    include_previews: bool = False,
    capture: Optional[Any] = None,
) -> str:
    """Return a compact JSON summary suitable for an agent response."""
    return summary_response(
        capture_id,
        include_previews=include_previews,
        capture=capture,
    ).text or ""


def summary_response(
    capture_id: str,
    include_previews: bool = False,
    capture: Optional[Any] = None,
) -> ToolResponseEnvelope:
    """Build a typed summary response while preserving engine error codes."""
    if capture is None:
        summary = active_engine().get_summary(capture_id, include_previews=include_previews)
    else:
        summary = active_engine().get_summary_for_capture(capture, include_previews=include_previews)
    if summary.get("status") == "error":
        message, _ = bounded_summary_text(
            str(summary.get("message", "capture summary unavailable")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        text = f"Error: {message}"
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(
                code=str(summary.get("error_code") or "retrieval_failed"),
                message=text,
            ),
            text=text,
        )
    payload = summary_payload(summary, include_previews=include_previews)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ToolResponseEnvelope(status="ok", data=payload, text=text)
