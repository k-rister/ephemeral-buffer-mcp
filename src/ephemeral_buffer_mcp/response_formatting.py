"""Transport-neutral UTF-8 and JSON text formatting helpers."""

from __future__ import annotations

import json
from typing import Optional


def bounded_summary_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    """Bound UTF-8 text metadata included alongside an execution summary."""
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


def bounded_json_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
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


def truncate_utf8_with_marker(
    value: str, max_bytes: int, marker: str
) -> Optional[str]:
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
