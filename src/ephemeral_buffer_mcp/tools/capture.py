"""Capture domain handlers."""

import inspect

from ..server import *  # noqa: F403
from .registry import register_tool
from .formatting import (
    bounded_summary_text,
    summary_payload,
)

CaptureContentType = Literal["auto", "diff", "log", "text"]
CAPTURE_CONTENT_TYPES = ("auto", "diff", "log", "text")


def _content_type_error(content_type: object) -> Optional[str]:
    """Return a stable error when a caller supplies an unsupported hint."""
    if isinstance(content_type, str) and content_type in CAPTURE_CONTENT_TYPES:
        return None
    return "Error: content_type must be one of: auto, diff, log, or text."


def _bounded_command(command: str) -> tuple[str, bool]:
    """Bound command metadata included alongside an execution summary."""
    return bounded_summary_text(
        command,
        SUMMARY_COMMAND_MAX_BYTES,
        SUMMARY_COMMAND_TRUNCATION_MARKER,
    )


def _summary_json(
    capture_id: str,
    include_previews: bool = False,
    capture: Optional[Any] = None,
) -> str:
    """Return a compact JSON summary suitable for an agent response."""
    if capture is None:
        summary = _active_engine().get_summary(capture_id, include_previews=include_previews)
    else:
        summary = _active_engine().get_summary_for_capture(capture, include_previews=include_previews)
    if summary.get("status") == "error":
        message, _ = bounded_summary_text(
            str(summary.get("message", "capture summary unavailable")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error: {message}"
    return json.dumps(
        summary_payload(summary, include_previews=include_previews),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _capture_text(
    content: str,
    label: str = "",
    content_type: CaptureContentType = "auto",
    source: str = "capture_text",
    duration_ms: Optional[float] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
    truncated: bool = False,
    original_byte_size: Optional[int] = None,
    command_exit_code: Optional[int] = None,
    timed_out: bool = False,
) -> str:
    """
    Ingests raw text output directly into the ephemeral search index.
    Automatically detects diffs, logs, and text structures.
    Returns a compact JSON summary while preserving the complete capture for later retrieval.
    """
    content_type_error = _content_type_error(content_type)
    if content_type_error:
        return content_type_error
    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"
    cap, summary = _active_engine().ingest_with_summary(
        content,
        label=label,
        content_type=content_type,
        truncated=truncated,
        original_byte_size=original_byte_size,
        command_exit_code=command_exit_code,
        timed_out=timed_out,
        source=source,
        duration_ms=duration_ms,
        structured_metrics=structured_metrics,
    )
    return json.dumps(
        summary_payload(summary),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _socket_capture_arguments(request):
    """Map the framed socket request onto the registered capture handler."""
    metrics = normalize_structured_metrics(request["structured_metrics"])
    return (request["text"],), {
        "label": request["label"],
        "content_type": request["content_type"],
        "structured_metrics": metrics,
        "__ingest_metadata": {
            "source": "socket",
            "duration_ms": request["duration_ms"],
            "truncated": request["truncated"],
            "original_byte_size": request["original_byte_size"],
            "command_exit_code": request["command_exit_code"],
            "timed_out": request["timed_out"],
        },
    }


def _socket_capture_response(result):
    """Project the registered capture result into the existing socket shape."""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("capture handler returned invalid summary JSON") from exc
    if not isinstance(payload, dict) or "capture_id" not in payload:
        message = (
            payload.get("message", "capture summary is unavailable")
            if isinstance(payload, dict)
            else "capture summary is unavailable"
        )
        raise RuntimeError(str(message))
    return {
        "status": "ok",
        "capture_id": payload["capture_id"],
        "label": payload["label"],
        "line_count": payload["total_lines"],
        "byte_size": payload["byte_size"],
        "summary": payload,
    }


@register_tool(
    "capture_text",
    "capture",
    socket_request_adapter=_socket_capture_arguments,
    socket_response_adapter=_socket_capture_response,
    legacy_error_code="capture_failed",
    legacy_error_mode="capture",
)
@_instrument_tool("capture_text")
def capture_text(
    content: str,
    label: str = "",
    content_type: CaptureContentType = "auto",
    structured_metrics: Optional[Dict[str, Any]] = None,
    __ingest_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Ingest already-collected text and return capture metadata.

    Use this when the caller already has output to index. For a noisy or
    potentially long command, use ``execute_and_capture`` so output remains
    bounded before it reaches the agent context. The return value is a compact
    versioned JSON summary; optional named metrics are retained in
    ``structured_metrics``.
    """
    metadata = __ingest_metadata or {}
    return _capture_text(
        content,
        label=label,
        content_type=content_type,
        structured_metrics=structured_metrics,
        source=metadata.get("source", "capture_text"),
        duration_ms=metadata.get("duration_ms"),
        truncated=metadata.get("truncated", False),
        original_byte_size=metadata.get("original_byte_size"),
        command_exit_code=metadata.get("command_exit_code"),
        timed_out=metadata.get("timed_out", False),
    )


_capture_text_signature = inspect.signature(capture_text)
capture_text.__signature__ = _capture_text_signature.replace(
    parameters=tuple(
        parameter
        for parameter in _capture_text_signature.parameters.values()
        if parameter.name != "__ingest_metadata"
    )
)


@register_tool(
    "capture_file",
    "capture",
    legacy_error_code="capture_failed",
    legacy_error_mode="capture",
)
@_instrument_tool("capture_file")
def capture_file(
    file_path: str,
    label: str = "",
    content_type: CaptureContentType = "auto",
    max_bytes: Optional[int] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Index a file and return a compact versioned JSON summary.

    Verify the path before reading. Symlinks are followed; pipes and devices are
    rejected. Resolve symlinks when path identity matters, confirm the expected
    workspace, and set max_bytes for large or untrusted files. Capture limits do
    not validate filesystem intent. Optional structured_metrics are retained.
    """
    content_type_error = _content_type_error(content_type)
    if content_type_error:
        return content_type_error
    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"
    try:
        file_path = os.fsdecode(os.fspath(file_path))
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        active_engine = _active_engine()
        if max_bytes is not None and max_bytes > active_engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "capture_file_limit_rejected",
                requested_bytes=max_bytes,
                max_buffer_bytes=active_engine.max_buffer_bytes,
            )
            return (
                f"Error: max_bytes ({max_bytes:,}) exceeds the configured "
                f"buffer limit ({active_engine.max_buffer_bytes:,})."
            )
        read_limit = active_engine.max_buffer_bytes if max_bytes is None else max_bytes
        if read_limit < 1:
            return "Error: max_bytes must be at least 1."
        if not stat.S_ISREG(os.stat(file_path).st_mode):
            return f"Error: File '{file_path}' is not a regular file."
        content = read_file_bounded(file_path, read_limit)
        if not label:
            label = os.path.basename(file_path)
        return _capture_text(
            content,
            label=label,
            content_type=content_type,
            source="file",
            structured_metrics=structured_metrics,
        )
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@register_tool(
    "execute_and_capture",
    "capture",
    legacy_error_code="capture_failed",
    legacy_error_mode="capture",
)
@_instrument_tool("execute_and_capture")
def execute_and_capture(
    command: str,
    cwd: Optional[str] = None,
    label: str = "",
    content_type: CaptureContentType = "auto",
    max_output_bytes: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Run a command, index bounded output, and return a compact JSON
    summary with status, duration, sizes, token estimates, truncation, signals,
    and optional metrics.

    Use for commands expected to finish within the MCP client's tool-call window
    when output benefits from bounded capture and later search. Use
    start_execution when duration may approach or exceed that window, even for a
    single phase; use direct execution for small targeted inspections. Choose by
    duration independently of output size; this routing guidance is advisory.

    Verify the command, intended repository, and working directory. Omitted cwd
    inherits the server process directory; symlinks and shell expansion may
    target another path. Output bounds do not validate command intent, path
    identity, or filesystem safety.

    An omitted label is derived from the command. content_type accepts auto
    (default), diff, log, or text. max_output_bytes defaults to the configured
    buffer limit and cannot exceed it. timeout_seconds bounds runtime; timeout
    retains collected output and returns exit code 124. structured_metrics accepts
    JSON-compatible named metrics.
    """
    content_type_error = _content_type_error(content_type)
    if content_type_error:
        return content_type_error
    if not label:
        label = command[:40] + ("..." if len(command) > 40 else "")

    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"

    try:
        active_engine = _active_engine()
        if max_output_bytes is not None and max_output_bytes > active_engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "command_output_limit_rejected",
                requested_bytes=max_output_bytes,
                max_buffer_bytes=active_engine.max_buffer_bytes,
            )
            return (
                f"Error: max_output_bytes ({max_output_bytes:,}) exceeds the configured "
                f"buffer limit ({active_engine.max_buffer_bytes:,})."
            )
        output_limit = active_engine.max_buffer_bytes if max_output_bytes is None else max_output_bytes
        command_started = time.perf_counter()
        command_result = run_command_bounded(
            command,
            cwd,
            output_limit,
            timeout_seconds,
            cancellation_event=_REQUEST_CANCEL_EVENT.get(),
        )
        output, exit_code, truncated, original_byte_size, timed_out = command_result
        duration_ms = round((time.perf_counter() - command_started) * 1000, 3)

        cap, summary = active_engine.ingest_with_summary(
            output,
            label=f"cmd: {label}",
            content_type=content_type,
            truncated=truncated,
            original_byte_size=original_byte_size if truncated else None,
            command_exit_code=exit_code,
            timed_out=timed_out,
            source="command",
            duration_ms=duration_ms,
            structured_metrics=structured_metrics,
        )
        payload = summary_payload(summary)
        bounded_command, command_truncated = _bounded_command(command)
        payload["command"] = bounded_command
        payload["command_truncated"] = command_truncated
        payload["command_cancelled"] = bool(getattr(command_result, "cancelled", False))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception as e:
        log_event(LOGGER, logging.ERROR, "command_execution_failed", error_type=type(e).__name__)
        return f"Error executing command: {str(e)}"


def _consolidated_jsonl(
    capture_ids: Optional[List[str]],
    max_captures: int,
    max_bytes: int,
) -> Dict[str, Any]:
    """Compatibility wrapper around the engine's consolidation implementation."""
    return _active_engine().consolidate(capture_ids, max_captures, max_bytes)


@register_tool("consolidate_captures", "capture")
@_instrument_tool("consolidate_captures")
def consolidate_captures(
    capture_ids: Optional[List[str]] = None,
    label: str = "consolidated captures",
    max_captures: int = 25,
    max_bytes: Optional[int] = None,
) -> str:
    """Create one bounded, searchable JSON capture from multiple captures.

    The consolidated capture keeps source capture IDs and source line numbers.
    If records do not fit, the complete source captures remain available through
    their original IDs until normal LRU eviction. The operation fails if the
    consolidated capture cannot be admitted while retaining those sources.
    """
    try:
        if max_captures > CONSOLIDATION_MAX_CAPTURES:
            return (
                f"Error consolidating captures: max_captures must be at most "
                f"{CONSOLIDATION_MAX_CAPTURES}"
            )
        if capture_ids is not None:
            if not isinstance(capture_ids, list):
                return "Error consolidating captures: capture_ids must be a list or null"
            if len(capture_ids) > CONSOLIDATION_MAX_CAPTURES:
                return (
                    f"Error consolidating captures: at most "
                    f"{CONSOLIDATION_MAX_CAPTURES} capture IDs may be requested"
                )
            for capture_id in capture_ids:
                if not isinstance(capture_id, str):
                    return "Error consolidating captures: capture_ids must contain strings"
                if len(capture_id.encode("utf-8")) > CONSOLIDATION_CAPTURE_ID_MAX_BYTES:
                    return "Error consolidating captures: capture ID exceeds the 256-byte limit"
        active_engine = _active_engine()
        result = active_engine.consolidate(capture_ids, max_captures, max_bytes)
        capture, summary = active_engine.ingest_with_summary(
            result["content"],
            label=label,
            content_type="text",
            source="consolidated",
            protected_capture_ids=result["source_capture_ids"],
        )
        payload = summary_payload(summary)
        payload.update({
            "source_capture_ids": result["source_capture_ids"],
            "requested_capture_count": result["requested_capture_count"],
            "selected_capture_count": result["selected_capture_count"],
            "source_count": result["source_count"],
            "record_count": result["record_count"],
            "omitted_record_count": result["omitted_record_count"],
            "missing_capture_ids": result["missing_capture_ids"],
            "next_steps": {
                "search": f"search_capture(query='...', capture_id='{capture.capture_id}')",
                "slice": f"get_capture_slice(start_line=..., end_line=..., capture_id='{capture.capture_id}')",
                "source_detail": "Use the original source capture IDs for complete omitted records.",
            },
        })
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        log_event(LOGGER, logging.ERROR, "capture_consolidation_failed", error_type=type(exc).__name__)
        message, _ = bounded_summary_text(
            str(exc),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error consolidating captures: {message}"
