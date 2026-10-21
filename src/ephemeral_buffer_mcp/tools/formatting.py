"""Shared bounded formatting for tool and transport responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..engine import MAX_CAPTURE_EVICTION_DETAILS
from ..response_formatting import (
    bounded_json_text,
    bounded_summary_text,
    truncate_utf8_with_marker,
)
from ..schema_versions import CAPTURE_SUMMARY_SCHEMA_VERSION

from ..response_limits import (
    SEARCH_QUERY_MAX_BYTES,
    SEARCH_QUERY_TRUNCATION_MARKER,
    SUMMARY_COMMAND_MAX_BYTES,
    SUMMARY_COMMAND_TRUNCATION_MARKER,
    SUMMARY_DIFF_FILE_MAP_MAX_BYTES,
    SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES,
    SUMMARY_LABEL_MAX_BYTES,
    SUMMARY_LABEL_TRUNCATION_MARKER,
)


def _bounded_diff_file_map(
    file_map: str,
    diff_meta: Optional[Dict[str, Any]] = None,
) -> tuple[str, int]:
    """Bound diff file-map text and return it with the omitted-entry count."""
    retained: List[str] = []
    files = diff_meta.get("files") if isinstance(diff_meta, dict) else None
    if isinstance(files, list):
        total_entries = len(files)
        source_lines = []
        for file_info in files:
            status_tag = (
                f" [{file_info['status'].upper()}]"
                if file_info["status"] != "modified"
                else ""
            )
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
        while retained and len(
            ("\n".join([*retained, marker])).encode("utf-8")
        ) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            retained.pop()
            omitted += 1
        bounded = "\n".join([*retained, marker])
        if len(bounded.encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            bounded = marker[:SUMMARY_DIFF_FILE_MAP_MAX_BYTES]
        return bounded, omitted
    return "\n".join(retained), 0


def summary_payload(
    summary: Dict[str, Any], include_previews: bool = False
) -> Dict[str, Any]:
    """Project an engine summary into the compact, versioned tool schema."""
    if summary.get("status") == "error":
        return dict(summary)

    execution_status = summary.get("execution_status")
    if execution_status is None:
        execution_status = "captured" if summary.get("command_exit_code") is None else (
            "success" if summary.get("command_exit_code") == 0 else "failed"
        )
    bounded_label, label_truncated = bounded_summary_text(
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
