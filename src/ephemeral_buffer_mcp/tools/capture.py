"""Capture domain handlers."""

from ..server import *  # noqa: F403
from .registry import register_tool

CaptureContentType = Literal["auto", "diff", "log", "text"]
CAPTURE_CONTENT_TYPES = ("auto", "diff", "log", "text")


def _content_type_error(content_type: object) -> Optional[str]:
    """Return a stable error when a caller supplies an unsupported hint."""
    if isinstance(content_type, str) and content_type in CAPTURE_CONTENT_TYPES:
        return None
    return "Error: content_type must be one of: auto, diff, log, or text."


def _bounded_diff_file_map(
    file_map: str,
    diff_meta: Optional[Dict[str, Any]] = None,
) -> tuple[str, int]:
    """Bound diff file-map text and return it with the omitted-entry count."""
    retained: List[str] = []
    files = diff_meta.get("files") if isinstance(diff_meta, dict) else None
    if isinstance(files, list):
        total_entries = len(files)
        # Format only the bounded prefix from structured metadata. The full
        # raw diff remains available through get_capture_slice.
        source_lines = []
        for file_info in files:
            status_tag = f" [{file_info['status'].upper()}]" if file_info["status"] != "modified" else ""
            source_lines.append(
                f"  - {file_info['path']}{status_tag} (+{file_info['additions']}, -{file_info['deletions']}) "
                f"| Buffer Lines: L{file_info['start_line']}-L{file_info['end_line']}"
            )
            if len(source_lines) >= SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES:
                break
    else:
        total_entries = file_map.count("\n") + (1 if file_map else 0)
        source_lines = []
        start = 0
        for _ in range(SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES):
            end = file_map.find("\n", start)
            if end < 0:
                if start < len(file_map):
                    source_lines.append(file_map[start:])
                break
            source_lines.append(file_map[start:end])
            start = end + 1

    for line in source_lines:
        candidate = "\n".join([*retained, line])
        if len(candidate.encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            break
        retained.append(line)

    omitted = total_entries - len(retained)
    if omitted:
        marker = (
            f"... [{omitted:,} diff file-map entries omitted; "
            "use get_capture_slice for complete diff details] ..."
        )
        while retained and len(("\n".join([*retained, marker])).encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            retained.pop()
            omitted += 1
        bounded = "\n".join([*retained, marker])
        if len(bounded.encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            bounded = marker[:SUMMARY_DIFF_FILE_MAP_MAX_BYTES]
        return bounded, omitted
    return "\n".join(retained), 0


def _bounded_summary_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    """Bound text metadata included alongside an execution summary."""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    marker_bytes = marker.encode("utf-8")
    if max_bytes <= 0:
        return "", True
    if len(marker_bytes) >= max_bytes:
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore"), True
    prefix_budget = max(0, max_bytes - len(marker_bytes))
    prefix = encoded[:prefix_budget].decode("utf-8", errors="ignore")
    return prefix + marker, True


def _bounded_json_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    """Bound the escaped JSON representation of metadata strings."""
    escaped_bytes = lambda text: len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2
    if escaped_bytes(value) <= max_bytes:
        return value, False

    marker_bytes = escaped_bytes(marker)
    if marker_bytes > max_bytes:
        return "", True
    prefix_budget = max_bytes - marker_bytes
    prefix = []
    prefix_bytes = 0
    for char in value:
        char_bytes = escaped_bytes(char)
        if prefix_bytes + char_bytes > prefix_budget:
            break
        prefix.append(char)
        prefix_bytes += char_bytes
    return "".join(prefix) + marker, True


def _truncate_utf8_with_marker(value: str, max_bytes: int, marker: str) -> Optional[str]:
    """Truncate UTF-8 text to a byte limit while retaining a visible marker."""
    marker_bytes = marker.encode("utf-8")
    if max_bytes < len(marker_bytes):
        return None
    prefix_budget = max_bytes - len(marker_bytes)
    prefix = []
    prefix_bytes = 0
    for char in value:
        char_bytes = len(char.encode("utf-8"))
        if prefix_bytes + char_bytes > prefix_budget:
            break
        prefix.append(char)
        prefix_bytes += char_bytes
    return "".join(prefix) + marker


def _bounded_command(command: str) -> tuple[str, bool]:
    """Bound command metadata included alongside an execution summary."""
    return _bounded_summary_text(
        command,
        SUMMARY_COMMAND_MAX_BYTES,
        SUMMARY_COMMAND_TRUNCATION_MARKER,
    )


def _summary_payload(summary: Dict[str, Any], include_previews: bool = False) -> Dict[str, Any]:
    """Project an engine summary into the compact, versioned MCP schema."""
    if summary.get("status") == "error":
        return dict(summary)

    execution_status = summary.get("execution_status")
    if execution_status is None:
        execution_status = "captured" if summary.get("command_exit_code") is None else (
            "success" if summary.get("command_exit_code") == 0 else "failed"
        )
    bounded_label, label_truncated = _bounded_summary_text(
        summary["label"],
        SUMMARY_LABEL_MAX_BYTES,
        SUMMARY_LABEL_TRUNCATION_MARKER,
    )
    evicted_captures = summary.get("evicted_captures", [])
    evicted_captures_omitted = max(
        0, summary.get("evicted_captures_omitted", 0)
    )
    if len(evicted_captures) > MAX_CAPTURE_EVICTION_DETAILS:
        evicted_captures_omitted += (
            len(evicted_captures) - MAX_CAPTURE_EVICTION_DETAILS
        )
        evicted_captures = evicted_captures[:MAX_CAPTURE_EVICTION_DETAILS]
    payload: Dict[str, Any] = {
        "schema_version": summary.get(
            "schema_version", CAPTURE_SUMMARY_SCHEMA_VERSION
        ),
        "capture_id": summary["capture_id"],
        "label": bounded_label,
        "label_truncated": label_truncated,
        "source": summary.get("source", "capture"),
        "status": execution_status,
        "partial": summary.get("partial", summary.get("timed_out", False)),
        "content_type": summary.get("content_type", "text"),
        "timestamp": summary.get("timestamp"),
        "duration_ms": summary.get("duration_ms"),
        "exit_code": summary.get("command_exit_code"),
        "timed_out": summary.get("timed_out", False),
        "total_lines": summary.get("total_lines", 0),
        "byte_size": summary.get("byte_size", 0),
        "original_byte_size": summary.get("original_byte_size"),
        "estimated_tokens": summary.get("estimated_tokens"),
        "original_estimated_tokens": summary.get("original_estimated_tokens"),
        "truncated": summary.get("truncated", False),
        "signals": summary.get("keyword_signals", {}),
        "errors": summary.get("errors", []),
        "warnings": summary.get("warnings", []),
        "structured_metrics": summary.get("structured_metrics", {}),
        "evicted_captures": list(evicted_captures),
        "evicted_captures_omitted": evicted_captures_omitted,
        "retrieval": {
            "search": f"search_capture(query='...', capture_id='{summary['capture_id']}')",
            "slice": f"get_capture_slice(start_line=..., end_line=..., capture_id='{summary['capture_id']}')",
        },
    }
    if summary.get("content_type") == "diff":
        bounded_file_map, omitted_files = _bounded_diff_file_map(
            summary.get("file_map", ""),
            summary.get("diff_meta"),
        )
        payload["diff"] = {
            "stats": summary.get("diff_stats", ""),
            "file_map": bounded_file_map,
            "file_map_truncated": omitted_files > 0,
            "omitted_file_count": omitted_files,
        }
    if include_previews:
        payload["previews"] = {
            "head": summary.get("head_preview", ""),
            "tail": summary.get("tail_preview", ""),
        }
    return payload


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
        message, _ = _bounded_summary_text(
            str(summary.get("message", "capture summary unavailable")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error: {message}"
    return json.dumps(
        _summary_payload(summary, include_previews=include_previews),
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
        source=source,
        duration_ms=duration_ms,
        structured_metrics=structured_metrics,
    )
    return json.dumps(
        _summary_payload(summary),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@register_tool("capture_text", "capture")
@_instrument_tool("capture_text")
def capture_text(
    content: str,
    label: str = "",
    content_type: CaptureContentType = "auto",
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Ingest already-collected text and return capture metadata.

    Use this when the caller already has output to index. For a noisy or
    potentially long command, use ``execute_and_capture`` so output remains
    bounded before it reaches the agent context. The return value is a compact
    versioned JSON summary; optional named metrics are retained in
    ``structured_metrics``.
    """
    return _capture_text(
        content,
        label=label,
        content_type=content_type,
        structured_metrics=structured_metrics,
    )


@register_tool("capture_file", "capture")
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
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
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


@register_tool("execute_and_capture", "capture")
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
        payload = _summary_payload(summary)
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
        payload = _summary_payload(summary)
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
        message, _ = _bounded_summary_text(
            str(exc),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error consolidating captures: {message}"
