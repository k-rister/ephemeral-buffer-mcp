"""
Core search and indexing engine for ephemeral command output buffer.
Provides hybrid search (BM25 lexical + dense semantic embeddings) with RRF ranking.
"""

import copy
import os
import sys
import time
import logging
import re
import math
import hashlib
import hmac
import sqlite3
import threading
import json
import uuid
import unicodedata
import base64
import binascii
import secrets
from types import MappingProxyType
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter, OrderedDict
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple, Mapping
from dataclasses import dataclass, field
from functools import wraps
from logging_utils import get_logger, log_event
from metrics import LocalMetrics
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from config import (
    BGE_SMALL_HF_REPO,
    DEFAULT_EMBEDDING_MODEL,
    FP32_EMBEDDING_MODEL,
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    SettingsSnapshot,
    load_settings,
)
from admission import admission_snapshot


LOGGER = get_logger("engine")
SEARCH_MODES = ("hybrid", "bm25", "semantic")
HYBRID_LEXICAL_WEIGHT = 2.0
PREVIEW_MAX_BYTES = 4 * 1024
PREVIEW_TRUNCATION_MARKER = "\n... [preview truncated; use get_capture_slice for full content] ..."
SEARCH_SNIPPET_MAX_BYTES = 8 * 1024
SEARCH_SNIPPET_TRUNCATION_MARKER = "... [search line truncated; use get_capture_slice for full content] ..."
SEARCH_MATCH_CONTEXT_MAX_BYTES = 4 * 1024
SEARCH_MATCH_SNIPPET_MAX_BYTES = 16 * 1024
MAX_SEARCH_TOP_K = 20
MAX_SEARCH_CONTEXT_LINES = 100
MAX_CAPTURE_SLICE_CONTENT_BYTES = 64 * 1024
CAPTURE_SLICE_SEGMENT_MAX_BYTES = 4 * 1024
SUMMARY_SCHEMA_VERSION = 1
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 4


class SemanticIndexBudgetExceeded(RuntimeError):
    """Raised when semantic indexing would exceed its configured work envelope."""

MAX_STRUCTURED_METRICS_BYTES = 16 * 1024


def sqlite_fts5_available() -> bool:
    """Return whether this Python build's SQLite library supports FTS5."""
    connection = None
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE fts5_capability_probe USING fts5(content)")
        return True
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def _fts5_tokens(text: str) -> List[str]:
    """Tokenize like FTS5 unicode61 with case- and diacritic-insensitive terms."""
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    # unicode61 treats underscores and punctuation as token separators while
    # retaining Unicode letters and numbers.
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return [token.casefold() for token in tokens]


def _fallback_tokens(text: str) -> List[str]:
    """Tokenize fallback documents with the same rules as FTS5 unicode61."""
    return _fts5_tokens(text)


def _bounded_preview(
    content: str,
    max_bytes: int = PREVIEW_MAX_BYTES,
    marker: str = PREVIEW_TRUNCATION_MARKER,
) -> str:
    """Return a UTF-8 bounded preview without encoding a potentially huge line."""
    if max_bytes <= 0:
        return ""

    # Short ASCII lines fit whenever their character count fits. Checking this
    # in C avoids the Python-level per-character loop for normal log snippets.
    if len(content) <= max_bytes and content.isascii():
        return content

    total_bytes = 0
    is_truncated = False
    for char in content:
        total_bytes += len(char.encode("utf-8"))
        if total_bytes > max_bytes:
            is_truncated = True
            break
    if not is_truncated:
        return content

    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= max_bytes:
        kept = []
        kept_bytes = 0
        for char in marker:
            char_bytes = len(char.encode("utf-8"))
            if kept_bytes + char_bytes > max_bytes:
                break
            kept.append(char)
            kept_bytes += char_bytes
        return "".join(kept)

    prefix_budget = max_bytes - len(marker_bytes)
    prefix = []
    prefix_bytes = 0
    for char in content:
        char_bytes = len(char.encode("utf-8"))
        if prefix_bytes + char_bytes > prefix_budget:
            break
        prefix.append(char)
        prefix_bytes += char_bytes
    return "".join(prefix) + marker


def _bounded_join_lines(
    lines: List[str],
    max_bytes: int,
    marker: str,
) -> Tuple[str, bool]:
    """Join lines with newlines while bounding work and retained UTF-8 bytes."""
    marker_bytes = marker.encode("utf-8")
    if max_bytes <= 0:
        return "", bool(lines)
    if len(marker_bytes) >= max_bytes:
        bounded_marker = _bounded_preview(marker, max_bytes=max_bytes, marker="")
        return bounded_marker, bool(lines)

    content_budget = max_bytes - len(marker_bytes)
    character_count = sum(len(line) for line in lines) + max(0, len(lines) - 1)
    if character_count <= content_budget and all(line.isascii() for line in lines):
        joined = "\n".join(lines)
        return joined, False

    pieces: List[str] = []
    used_bytes = 0
    for line_index, line in enumerate(lines):
        if line_index:
            if used_bytes + 1 > content_budget:
                return "".join(pieces) + marker, True
            pieces.append("\n")
            used_bytes += 1
        line_prefix: List[str] = []
        for char in line:
            char_bytes = len(char.encode("utf-8"))
            if used_bytes + char_bytes > content_budget:
                pieces.append("".join(line_prefix))
                return "".join(pieces) + marker, True
            line_prefix.append(char)
            used_bytes += char_bytes
        pieces.append("".join(line_prefix))
    return "".join(pieces), False
def estimate_tokens(text: str) -> int:
    """Return a deterministic approximate token count for retained text.

    This intentionally avoids a provider-specific tokenizer. Four UTF-8 bytes
    per token is a conservative planning estimate, not a billable token count.
    """
    if not text:
        return 0
    return estimate_tokens_from_bytes(len(text.encode("utf-8")))


def estimate_tokens_from_bytes(byte_count: Optional[int]) -> Optional[int]:
    """Return the approximate token count for a byte count, preserving null."""
    if byte_count is None:
        return None
    if byte_count <= 0:
        return 0
    return math.ceil(byte_count / TOKEN_ESTIMATE_BYTES_PER_TOKEN)


def normalize_structured_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and detach bounded JSON-compatible tool metrics."""
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        raise ValueError("structured_metrics must be a JSON object or null")
    try:
        encoded = json.dumps(
            metrics,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("structured_metrics must contain JSON-compatible values") from exc
    if len(encoded) > MAX_STRUCTURED_METRICS_BYTES:
        raise ValueError(
            f"structured_metrics exceeds the {MAX_STRUCTURED_METRICS_BYTES:,}-byte limit"
        )
    return json.loads(encoded.decode("utf-8"))


def _capture_execution_status(capture: "_CaptureState") -> str:
    """Return the stable execution status exposed by the summary schema."""
    if capture.timed_out:
        return "timed_out"
    if capture.command_exit_code is None:
        return "captured"
    return "success" if capture.command_exit_code == 0 else "failed"


def _signal_details(signals: Dict[str, int], names: set[str]) -> List[Dict[str, int | str]]:
    """Return stable typed signal details for the public summary."""
    return [
        {"type": name, "count": signals[name]}
        for name in sorted(names)
        if name in signals
    ]


def process_rss_bytes() -> Optional[int]:
    """Return current process RSS, when the host exposes it."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB; macOS reports bytes.
            return int(rss if sys.platform == "darwin" else rss * 1024)
        except (ImportError, OSError, ValueError):
            return None


def synchronized(method):
    """Serialize access to shared engine state, including nested calls."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            with self._lock:
                return method(self, *args, **kwargs)
        finally:
            self._flush_metrics_snapshot()
    return wrapper

DIFF_GIT_RE = re.compile(r"^diff --git (.+)$")
DIFF_PATH_TOKEN_RE = re.compile(r'"(?:\\.|[^"])*"|[^\s]+')
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
BENIGN_SIGNAL_RE = re.compile(
    r"(\b0\s*(errors?|failures?|failed|warnings?)\b|"
    r"\b(errors?|failures?|failed|warnings?)\s*[:=]\s*0\b|"
    r"\bno\s+(?:errors?|warnings?)\b)",
    re.IGNORECASE
)
LOG_SIGNAL_PATTERNS = {
    "error": re.compile(r"\b(ERROR|FATAL|PANIC|CRITICAL)\b", re.IGNORECASE),
    "exception": re.compile(r"\b(EXCEPTION|TRACEBACK)\b", re.IGNORECASE),
    "failure": re.compile(r"\b(FAILED|FAILURES?)\b", re.IGNORECASE),
    "timeout": re.compile(r"\b(TIMED\s*OUT|TIMEOUT)\b", re.IGNORECASE),
    "warning": re.compile(r"\b(WARN(?:ING)?S?)\b", re.IGNORECASE),
}
SUCCESS_TEST_RE = re.compile(
    r"(?:^\s*OK\s*$|\b\d+\s+(?:tests?|cases?)\s+.*\bOK\b|\b\d+\s+(?:tests?|cases?)\s+passed\b|\b\d+\s+passed\b)",
    re.IGNORECASE,
)
NONZERO_TEST_FAILURE_RE = re.compile(
    r"(?:\b[1-9]\d*\s+(?:failed|failures?|errors?)\b|\b(?:failed|failures?|errors?)\s*[:=]\s*[1-9]\d*)",
    re.IGNORECASE,
)
CONFLICT_START_RE = re.compile(r"^<<<<<<<(?:\s.*)?$")
CONFLICT_SEPARATOR_RE = re.compile(r"^=======$")
CONFLICT_END_RE = re.compile(r"^>>>>>>>(?:\s.*)?$")


def _diff_content_for_conflict_check(line: str) -> Optional[str]:
    """Return diff content while excluding removed lines and file headers."""
    if line.startswith(("diff --git", "@@ ", "--- ", "+++ ")):
        return None
    if line.startswith("-"):
        return None
    if line.startswith(("+", " ")):
        return line[1:]
    return line


def _contains_conflict_marker(line: str) -> bool:
    content = _diff_content_for_conflict_check(line)
    if content is None:
        return False
    content = content.rstrip()
    return bool(
        CONFLICT_START_RE.match(content)
        or CONFLICT_SEPARATOR_RE.match(content)
        or CONFLICT_END_RE.match(content)
    )


def _decode_git_path(token: str) -> str:
    """Decode a Git-quoted path, including octal C-style escapes."""
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1]

    decoded = bytearray()
    index = 0
    escape_bytes = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13}
    while index < len(token):
        if token[index] != "\\":
            decoded.extend(token[index].encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(token):
            decoded.extend(b"\\")
            break
        if token[index] in "01234567" and index + 2 < len(token):
            octal = token[index:index + 3]
            if all(char in "01234567" for char in octal):
                decoded.append(int(octal, 8))
                index += 3
                continue
        decoded.append(escape_bytes.get(token[index], ord(token[index])))
        index += 1
    return decoded.decode("utf-8", errors="replace")


def _normalize_diff_path_pair(
    old_path: str,
    new_path: str,
    *,
    allow_custom_prefixes: bool = False,
) -> Tuple[str, str]:
    """Remove paired Git diff prefixes without mistaking paths for renames."""
    if old_path == "/dev/null":
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        return old_path, new_path
    if new_path == "/dev/null":
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        return old_path, new_path

    if old_path.startswith("a/") and new_path.startswith("b/"):
        return old_path[2:], new_path[2:]

    old_prefix, old_separator, old_suffix = old_path.partition("/")
    new_prefix, new_separator, new_suffix = new_path.partition("/")
    if old_separator and new_separator:
        if old_path == new_path and old_prefix in {"a", "b"}:
            return old_suffix, new_suffix
        if (
            allow_custom_prefixes
            and old_prefix != new_prefix
            and old_suffix == new_suffix
        ):
            return old_suffix, new_suffix
    return old_path, new_path


def _parse_git_diff_paths(line: str) -> Optional[Tuple[str, str]]:
    """Return decoded old/new paths from a ``diff --git`` header."""
    match = DIFF_GIT_RE.match(line)
    if not match:
        return None
    header = match.group(1)
    tokens = DIFF_PATH_TOKEN_RE.findall(header)
    if len(tokens) != 2:
        # Git leaves ordinary spaces unquoted. A path-like token boundary
        # separates the two names in the usual case; paired ---/+++ headers
        # later resolve ambiguous names containing the same delimiter.
        candidates = []
        for delimiter in re.finditer(r" (?=[^/\s]+/)", header):
            old_token = header[:delimiter.start()]
            new_token = header[delimiter.start() + 1:]
            if old_token and new_token:
                old_path = _decode_git_path(old_token)
                new_path = _decode_git_path(new_token)
                candidates.append((old_path, new_path))
        matching_paths = [
            candidate for candidate in candidates
            if _normalize_diff_path_pair(
                *candidate,
                allow_custom_prefixes=True,
            )[0]
            == _normalize_diff_path_pair(
                *candidate,
                allow_custom_prefixes=True,
            )[1]
        ]
        if len(matching_paths) == 1:
            return matching_paths[0]
        if len(candidates) == 1:
            return candidates[0]
        return None
    return _decode_git_path(tokens[0]), _decode_git_path(tokens[1])


def _parse_unified_file_path(line: str, header: str) -> Optional[str]:
    """Parse one ---/+++ path, discarding an optional tab-separated timestamp."""
    if not line.startswith(header):
        return None
    value = line[len(header):].rstrip("\r\n")
    if value.startswith('"'):
        match = re.match(r'"(?:\\.|[^"])*"', value)
        if not match:
            return None
        value = match.group(0)
    else:
        value = value.partition("\t")[0]
    if not value:
        return None
    return _decode_git_path(value)


def parse_unified_diff(lines: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parses unified diff lines to extract structured file-level metadata and line boundaries.
    """
    if not lines:
        return None

    diff_markers = sum(
        1 for line in lines[:50]
        if line.startswith("diff --git") or line.startswith("@@ ") or line.startswith("--- ") or line.startswith("+++ ")
    )
    if diff_markers == 0:
        return None

    files: List[Dict[str, Any]] = []
    current_file: Optional[Dict[str, Any]] = None
    in_hunk = False
    hunk_old_remaining = 0
    hunk_new_remaining = 0
    pending_git_header = False
    pending_git_start_line = 0
    has_conflicts = False

    def new_file(
        old_path: str,
        new_path: str,
        start_line: int,
        *,
        allow_custom_prefixes: bool = False,
    ) -> Dict[str, Any]:
        old_path, new_path = _normalize_diff_path_pair(
            old_path,
            new_path,
            allow_custom_prefixes=allow_custom_prefixes,
        )
        path = new_path if new_path != "/dev/null" else old_path
        if old_path == "/dev/null":
            status = "added"
        elif new_path == "/dev/null":
            status = "deleted"
        elif old_path != new_path:
            status = "renamed"
        else:
            status = "modified"
        return {
            "path": path,
            "old_path": old_path,
            "new_path": new_path,
            "status": status,
            "start_line": start_line,
            "end_line": len(lines),
            "additions": 0,
            "deletions": 0,
            "hunks": 0,
        }

    def set_file_paths(file: Dict[str, Any], old_path: str, new_path: str) -> None:
        if old_path == "/dev/null" and file["new_path"] != "/dev/null":
            new_path = file["new_path"]
        elif new_path == "/dev/null" and file["old_path"] != "/dev/null":
            old_path = file["old_path"]
        else:
            old_path, new_path = _normalize_diff_path_pair(
                old_path,
                new_path,
                allow_custom_prefixes=True,
            )
        file["old_path"] = old_path
        file["new_path"] = new_path
        file["path"] = new_path if new_path != "/dev/null" else old_path
        if old_path == "/dev/null":
            file["status"] = "added"
        elif new_path == "/dev/null":
            file["status"] = "deleted"
        elif old_path != new_path and file["status"] == "modified":
            file["status"] = "renamed"

    for idx, line in enumerate(lines, start=1):
        if _contains_conflict_marker(line):
            has_conflicts = True

        if line.startswith("diff --git "):
            if current_file:
                current_file["end_line"] = idx - 1
                files.append(current_file)
            git_paths = _parse_git_diff_paths(line)
            current_file = (
                new_file(
                    git_paths[0],
                    git_paths[1],
                    idx,
                    allow_custom_prefixes=True,
                )
                if git_paths
                else None
            )
            pending_git_header = True
            pending_git_start_line = idx
            in_hunk = False
            hunk_old_remaining = 0
            hunk_new_remaining = 0
            continue

        if (
            not in_hunk
            and line.startswith("--- ")
            and idx < len(lines)
            and lines[idx].startswith("+++ ")
        ):
            old_path = _parse_unified_file_path(line, "--- ")
            new_path = _parse_unified_file_path(lines[idx], "+++ ")
            if old_path is not None and new_path is not None:
                if pending_git_header:
                    if current_file is None:
                        current_file = new_file(
                            old_path,
                            new_path,
                            pending_git_start_line or idx,
                            allow_custom_prefixes=True,
                        )
                    else:
                        set_file_paths(current_file, old_path, new_path)
                else:
                    if current_file:
                        current_file["end_line"] = idx - 1
                        files.append(current_file)
                    current_file = new_file(old_path, new_path, idx)
                pending_git_header = False
                in_hunk = False
                hunk_old_remaining = 0
                hunk_new_remaining = 0
                continue

        if current_file:
            if line.startswith("new file mode"):
                current_file["status"] = "added"
            elif line.startswith("deleted file mode"):
                current_file["status"] = "deleted"
            elif line.startswith("similarity index") or line.startswith("rename from"):
                current_file["status"] = "renamed"
            elif (hunk_match := HUNK_RE.match(line)):
                current_file["hunks"] += 1
                in_hunk = True
                hunk_old_remaining = int(hunk_match.group(1) or 1)
                hunk_new_remaining = int(hunk_match.group(2) or 1)
                if hunk_old_remaining == 0 and hunk_new_remaining == 0:
                    in_hunk = False
            elif in_hunk and line.startswith("+"):
                current_file["additions"] += 1
                hunk_new_remaining -= 1
            elif in_hunk and line.startswith("-"):
                current_file["deletions"] += 1
                hunk_old_remaining -= 1
            elif in_hunk and line.startswith(" "):
                hunk_old_remaining -= 1
                hunk_new_remaining -= 1
            if in_hunk and hunk_old_remaining <= 0 and hunk_new_remaining <= 0:
                in_hunk = False

    if current_file:
        current_file["end_line"] = len(lines)
        files.append(current_file)

    if not files:
        return None

    total_add = sum(f["additions"] for f in files)
    total_del = sum(f["deletions"] for f in files)

    return {
        "total_files": len(files),
        "total_additions": total_add,
        "total_deletions": total_del,
        "files": files,
        "has_conflicts": has_conflicts
    }


def detect_content_type(lines: List[str], label: str = "", content_type_hint: str = "auto") -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Classifies output content type: 'diff', 'log', or 'text'.
    Returns (content_type, diff_metadata).
    """
    if content_type_hint in ("diff", "log", "text"):
        if content_type_hint == "diff":
            diff_meta = parse_unified_diff(lines)
            return ("diff", diff_meta)
        return (content_type_hint, None)

    label_lower = label.lower()
    is_diff_command = any(k in label_lower for k in ["diff", "patch", "git show", "git log -p", "pr diff"])
    diff_meta = parse_unified_diff(lines)

    if diff_meta or is_diff_command:
        if diff_meta:
            return ("diff", diff_meta)
        return ("diff", None)

    is_log = any(k in label_lower for k in ["log", "test", "build", "make", "cargo", "pytest", "mvn", "gcc", "clang", "compile"])
    if is_log:
        return ("log", None)

    return ("text", None)


def detect_signals(
    lines: List[str],
    content_type: str,
    diff_meta: Optional[Dict[str, Any]] = None,
    command_exit_code: Optional[int] = None,
    timed_out: bool = False,
) -> Tuple[Dict[str, int], str]:
    """
    Extracts high-signal diagnostic indicators according to content type.
    Avoids false alarms on code/diff lines.
    """
    if content_type == "diff":
        if diff_meta and diff_meta.get("has_conflicts"):
            return ({"conflicts": 1}, "Conflict markers detected (<<<<<<< / >>>>>>>)!")
        return ({}, "None (Clean patch)")

    # Keyword signals are meaningful for command/test/build logs. Scanning
    # arbitrary text (for example source files or README content) produces
    # noisy matches for words such as "error" and "failure".
    if content_type != "log":
        return ({}, "None (non-log content)")

    if (
        not timed_out
        and command_exit_code in (None, 0)
        and any(SUCCESS_TEST_RE.search(line) for line in lines)
        and not any(NONZERO_TEST_FAILURE_RE.search(line) for line in lines)
    ):
        warning_hits = 0
        for line in lines:
            signal_line = BENIGN_SIGNAL_RE.sub("", line)
            if LOG_SIGNAL_PATTERNS["warning"].search(signal_line):
                warning_hits += 1
        if warning_hits:
            return ({"warning": warning_hits}, f"warning: {warning_hits}")
        return ({}, "None (successful test run)")

    detected = {}
    for name, pat in LOG_SIGNAL_PATTERNS.items():
        hits = 0
        for line in lines:
            # Remove only benign zero-valued phrases.  A summary line may
            # contain both a zero-valued category and a real failure, such as
            # ``FAILED: 1 failure, 0 errors``.
            signal_line = BENIGN_SIGNAL_RE.sub("", line)
            if not signal_line.strip():
                continue
            if pat.search(signal_line):
                hits += 1
        if hits > 0:
            detected[name] = hits

    if not detected:
        summary_str = "None detected"
    else:
        summary_str = ", ".join(f"{k}: {v}" for k, v in detected.items())

    return (detected, summary_str)


@dataclass
class Chunk:
    chunk_id: int
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed
    text: str


@dataclass
class _CaptureState:
    capture_id: str
    label: str
    timestamp: float
    raw_lines: List[str]
    input_byte_size: int
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    fts_conn: Optional[sqlite3.Connection] = None
    content_type: str = "text"
    diff_meta: Optional[Dict[str, Any]] = None
    truncated: bool = False
    original_byte_size: Optional[int] = None
    command_exit_code: Optional[int] = None
    timed_out: bool = False
    semantic_index_state: str = "not-requested"
    active_readers: int = 0
    storage_close_pending: bool = False
    # Append summary metadata after the legacy fields so positional state
    # construction remains compatible with the pre-summary data model.
    source: str = "capture"
    duration_ms: Optional[float] = None
    structured_metrics: Dict[str, Any] = field(default_factory=dict)
    # Semantic windows are packed separately from the lexical sliding windows
    # so embedding cost is bounded by line and byte caps rather than tied to
    # the overlap BM25 uses for exact line ranges.
    semantic_chunks: List[Chunk] = field(default_factory=list)
    # Internal reader-lease accounting is appended to preserve positional
    # construction compatibility for the state record.
    deferred_storage_tracked: bool = False
    deferred_storage_accounted_bytes: int = 0

    @property
    def line_count(self) -> int:
        return len(self.raw_lines)

    @property
    def byte_size(self) -> int:
        return self.input_byte_size

    @property
    def label_byte_size(self) -> int:
        """Return the UTF-8 bytes retained for the capture label."""
        return len(self.label.encode("utf-8"))

    @property
    def retained_byte_size(self) -> int:
        """Return content plus label bytes counted against the buffer limit."""
        return self.byte_size + self.label_byte_size


def _freeze_public_value(value: Any) -> Any:
    """Copy nested metadata into containers that cannot mutate engine state."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_public_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_public_value(item) for item in value)
    return value


@dataclass(frozen=True)
class CaptureView:
    """Read-only public metadata for an engine-owned capture.

    Capture text is retrieved through the bounded slice and search APIs. This
    view intentionally has no raw lines, chunks, embedding arrays, SQLite
    connections, or reader and deferred-close bookkeeping.
    """

    capture_id: str
    label: str
    timestamp: float
    line_count: int
    input_byte_size: int
    content_type: str
    diff_meta: Optional[Mapping[str, Any]]
    truncated: bool
    original_byte_size: Optional[int]
    command_exit_code: Optional[int]
    timed_out: bool
    semantic_index_state: str
    source: str
    duration_ms: Optional[float]
    structured_metrics: Mapping[str, Any]

    @property
    def byte_size(self) -> int:
        return self.input_byte_size

    @property
    def label_byte_size(self) -> int:
        return len(self.label.encode("utf-8"))

    @property
    def retained_byte_size(self) -> int:
        return self.byte_size + self.label_byte_size


@dataclass(frozen=True)
class CaptureDiagnostics:
    """Supported semantic-index statistics for diagnostics and benchmarks."""

    capture_id: str
    semantic_index_state: str
    lexical_chunk_count: int
    semantic_chunk_count: int
    semantic_input_bytes: int
    retained_embedding_bytes: int


# Keep the public type name pointed at the safe metadata view. Live storage is
# represented only by the private _CaptureState class.
Capture = CaptureView


_BUNDLED_MODELS_REGISTERED = False


def register_bundled_embedding_models() -> None:
    """Register the fp32 bge-small-en-v1.5 alias with FastEmbed once per process.

    FastEmbed's catalogue entry for bge-small-en-v1.5 downloads a reduced-precision
    ONNX export whose matrix kernels do not parallelize on common CPU hosts. The
    upstream fp32 export produces identical vectors, so it is registered under an
    alias that is the default and that ``EPHEMERAL_EMBEDDING_MODEL`` can select.
    """
    global _BUNDLED_MODELS_REGISTERED
    if _BUNDLED_MODELS_REGISTERED:
        return
    try:
        TextEmbedding.add_custom_model(
            model=FP32_EMBEDDING_MODEL,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=BGE_SMALL_HF_REPO),
            dim=384,
            model_file="onnx/model.onnx",
            description="fp32 ONNX export of BAAI/bge-small-en-v1.5",
            license="mit",
            size_in_gb=0.13,
        )
    except ValueError:
        # Another engine in this process already registered the alias.
        pass
    _BUNDLED_MODELS_REGISTERED = True


class _SemanticIndexJob:
    """Completion state for one on-demand embedding job.

    ``future`` is assigned by ``_start_semantic_index`` before the job is
    published to waiters, so it is always present once the job is visible.
    """

    __slots__ = (
        "done",
        "error",
        "future",
        "cancelled",
        "queued_at",
        "started_at",
        "measurement",
    )

    def __init__(
        self,
        queued_at: Optional[float] = None,
        measurement: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self.future: Future
        self.cancelled = False
        self.queued_at = time.perf_counter() if queued_at is None else queued_at
        self.started_at: Optional[float] = None
        self.measurement = measurement


class _DeterministicTestEmbedding:
    """Small deterministic substitute used only by the CI test environment."""

    _canonical_terms = {
        "network": "network-disconnect",
        "disconnect": "network-disconnect",
        "disconnected": "network-disconnect",
        "connection": "network-disconnect",
        "connected": "network-disconnect",
        "tcp": "network-disconnect",
        "remote": "network-disconnect",
        "closed": "network-disconnect",
        "terminated": "network-disconnect",
    }

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                token = self._canonical_terms.get(token, token)
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 384
                vector[index] += 1.0
            vectors.append(vector.tolist())
        return vectors


class EphemeralEngine:
    def __init__(
        self,
        max_captures: Optional[int] = None,
        max_buffer_bytes: Optional[int] = None,
        max_indexed_chunks: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
        embedding_warmup: Optional[bool] = None,
        embedding_threads: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_max_batch_tokens: Optional[int] = None,
        embedding_cpu_mem_arena_enabled: Optional[bool] = None,
        semantic_max_index_input_bytes: Optional[int] = None,
        metrics: Optional[LocalMetrics] = None,
        semantic_prefetch: Optional[bool] = None,
        semantic_prefetch_workers: Optional[int] = None,
        semantic_chunk_lines: Optional[int] = None,
        semantic_chunk_bytes: Optional[int] = None,
        semantic_chunk_overlap: Optional[int] = None,
        semantic_wait_seconds: Optional[float] = None,
        settings: Optional[SettingsSnapshot] = None,
    ):
        settings = settings or load_settings()
        max_captures = settings.max_captures.value if max_captures is None else max_captures
        max_buffer_bytes = settings.max_buffer_bytes.value if max_buffer_bytes is None else max_buffer_bytes
        self._lock = threading.RLock()
        self._slice_cursor_secret = secrets.token_bytes(32)
        if max_captures < 1:
            raise ValueError("max_captures must be at least 1")
        if max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be at least 1")
        self.max_indexed_chunks = (
            settings.max_indexed_chunks.value if max_indexed_chunks is None else max_indexed_chunks
        )
        if self.max_indexed_chunks < 1:
            raise ValueError("max_indexed_chunks must be at least 1")
        self.max_captures = max_captures
        self.max_buffer_bytes = max_buffer_bytes
        self._embedding_lock = threading.RLock()
        self._captures: Dict[str, _CaptureState] = {}
        self.capture_order: OrderedDict[str, None] = OrderedDict()
        self.session_id = uuid.uuid4().hex
        self._total_bytes = 0
        self._indexed_chunks = 0
        self._deferred_storage_capture_count = 0
        self._deferred_storage_readers = 0
        self._deferred_storage_bytes = 0
        self._last_index_budget_adjustment = {
            "status": "startup",
            "previous": self.max_indexed_chunks,
            "effective": self.max_indexed_chunks,
            "evicted_captures": 0,
        }
        self._next_id = 1
        self.lexical_backend = "fts5" if sqlite_fts5_available() else "python-fallback"
        
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name.value
        self.embedding_threads = (
            settings.embedding_threads.value if embedding_threads is None else embedding_threads
        )
        if self.embedding_threads is not None and self.embedding_threads < 1:
            raise ValueError("embedding_threads must be at least 1")
        self.embedding_batch_size = (
            settings.embedding_batch_size.value
            if embedding_batch_size is None
            else embedding_batch_size
        )
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        self.embedding_max_batch_tokens = (
            settings.embedding_max_batch_tokens.value
            if embedding_max_batch_tokens is None
            else embedding_max_batch_tokens
        )
        if self.embedding_max_batch_tokens < 1:
            raise ValueError("embedding_max_batch_tokens must be at least 1")
        self.embedding_cpu_mem_arena_enabled = (
            settings.embedding_cpu_mem_arena_enabled.value
            if embedding_cpu_mem_arena_enabled is None
            else embedding_cpu_mem_arena_enabled
        )
        self.semantic_max_index_input_bytes = (
            settings.semantic_max_index_input_bytes.value
            if semantic_max_index_input_bytes is None
            else semantic_max_index_input_bytes
        )
        if self.semantic_max_index_input_bytes < 1:
            raise ValueError("semantic_max_index_input_bytes must be at least 1")
        self.semantic_chunk_lines = (
            settings.semantic_chunk_lines.value if semantic_chunk_lines is None else semantic_chunk_lines
        )
        self.semantic_chunk_bytes = (
            settings.semantic_chunk_bytes.value if semantic_chunk_bytes is None else semantic_chunk_bytes
        )
        self.semantic_chunk_overlap = (
            settings.semantic_chunk_overlap.value
            if semantic_chunk_overlap is None
            else semantic_chunk_overlap
        )
        if self.semantic_chunk_lines < 1:
            raise ValueError("semantic_chunk_lines must be at least 1")
        if self.semantic_chunk_bytes < 1:
            raise ValueError("semantic_chunk_bytes must be at least 1")
        if not 0 <= self.semantic_chunk_overlap < self.semantic_chunk_lines:
            raise ValueError("semantic_chunk_overlap must be non-negative and smaller than semantic_chunk_lines")
        self.embedding_cache_path = embedding_cache_path or settings.embedding_cache_dir.value
        self.embedding_model = None
        self.embedding_warmup_enabled = (
            settings.embedding_warmup_enabled.value
            if embedding_warmup is None
            else embedding_warmup
        )
        self.embedding_warmup_state = (
            "not-started" if self.embedding_warmup_enabled else "disabled"
        )
        self.embedding_warmup_failure = None
        self._embedding_warmup_thread: Optional[threading.Thread] = None
        self.metrics = metrics or LocalMetrics(enabled=False)
        self._metrics_snapshot_callback: Optional[Callable[[], None]] = None
        self._metrics_snapshot_pending = False
        self.semantic_prefetch_enabled = (
            settings.semantic_prefetch_enabled.value if semantic_prefetch is None else semantic_prefetch
        )
        self.semantic_prefetch_workers = (
            settings.semantic_prefetch_workers.value
            if semantic_prefetch_workers is None
            else semantic_prefetch_workers
        )
        if self.semantic_prefetch_workers < 1:
            raise ValueError("semantic_prefetch_workers must be at least 1")
        self.semantic_wait_seconds = (
            settings.semantic_wait_seconds.value if semantic_wait_seconds is None else semantic_wait_seconds
        )
        if math.isnan(self.semantic_wait_seconds) or self.semantic_wait_seconds < 0:
            raise ValueError("semantic_wait_seconds must be non-negative")
        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=self.semantic_prefetch_workers,
                thread_name_prefix="semantic-prefetch",
            )
            if self.semantic_prefetch_enabled
            else None
        )
        # Eligible captures wait in an ordered queue that is drained newest-first,
        # so a burst of ingestion never silently skips a capture; the queue is
        # bounded by max_captures because eviction removes queued work.
        self._prefetch_queue: "OrderedDict[str, _CaptureState]" = OrderedDict()
        self._prefetch_running: Dict[str, threading.Event] = {}
        self._prefetch_job_meta: Dict[str, Dict[str, Any]] = {}
        self._semantic_job_dispositions: Dict[str, str] = {}
        self._prefetch_workers_active = 0
        # Searches that find no prefetch job running index the capture on a
        # bounded pool and wait for it only up to the configured budget, so a
        # very large capture yields lexical-first results instead of blocking.
        # The pool shares the prefetch worker count, so timed-out searches over
        # churning captures cannot accumulate threads beyond that bound; work
        # for evicted captures is cancelled while queued and skipped once run.
        self._on_demand_jobs: Dict[str, _SemanticIndexJob] = {}
        self._on_demand_executor = ThreadPoolExecutor(
            max_workers=self.semantic_prefetch_workers,
            thread_name_prefix="semantic-index",
        )
        self._shutdown = False

    def set_metrics_snapshot_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a best-effort callback for persisting asynchronous metric updates."""
        with self._lock:
            self._metrics_snapshot_callback = callback

    def _flush_metrics_snapshot(self) -> None:
        """Run a deferred metrics persistence callback after releasing the engine lock."""
        with self._lock:
            if not self._metrics_snapshot_pending:
                return
            self._metrics_snapshot_pending = False
            callback = self._metrics_snapshot_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:
            log_event(
                LOGGER,
                logging.WARNING,
                "metrics_snapshot_callback_failed",
            )

    def start_embedding_warmup(self) -> bool:
        """Start one non-blocking model warm-up and return whether work was started."""
        with self._lock:
            if self._shutdown or self.embedding_warmup_state != "not-started":
                return False
            self.embedding_warmup_state = "loading"
            try:
                thread = threading.Thread(
                    target=self._warm_embedding_model,
                    name="embedding-warmup",
                    daemon=True,
                )
                self._embedding_warmup_thread = thread
                thread.start()
            except Exception as exc:
                self._embedding_warmup_thread = None
                self.embedding_warmup_state = "failed"
                self.embedding_warmup_failure = type(exc).__name__
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "embedding_warmup_start_failed",
                    error_type=type(exc).__name__,
                )
                return False
            return True

    def _warm_embedding_model(self) -> None:
        """Load the model and run one deterministic inference off the serving thread."""
        try:
            with self._embedding_lock:
                model = self._get_embedding_model()
                list(model.embed(["ephemeral buffer embedding warmup"]))
        except Exception as exc:
            with self._lock:
                self.embedding_warmup_state = "failed"
                self.embedding_warmup_failure = type(exc).__name__
            log_event(
                LOGGER,
                logging.ERROR,
                "embedding_warmup_failed",
                error_type=type(exc).__name__,
            )
            return
        with self._lock:
            self.embedding_warmup_state = "ready"
            self.embedding_warmup_failure = None
        log_event(LOGGER, logging.INFO, "embedding_warmup_ready")

    def wait_for_embedding_warmup(self, timeout: Optional[float] = None) -> str:
        """Wait for an active warm-up, primarily for benchmarks and orderly tests."""
        with self._lock:
            thread = self._embedding_warmup_thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            return self.embedding_warmup_state

    def _get_embedding_model(self):
        """Load FastEmbed once, on first operation that needs embeddings."""
        with self._embedding_lock:
            if self.embedding_model is not None:
                return self.embedding_model
            if os.environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1":
                self.embedding_model = _DeterministicTestEmbedding()
                log_event(LOGGER, logging.INFO, "embedding_model_ready", model="deterministic-test")
                return self.embedding_model
            log_event(
                LOGGER,
                logging.INFO,
                "embedding_model_load_started",
                model=self.embedding_model_name,
                cache_dir=self.embedding_cache_path or "default",
            )
            sys.stderr.write(f"Loading embedding model: {self.embedding_model_name}...\n")
            sys.stderr.flush()
            kwargs = {"model_name": self.embedding_model_name}
            if self.embedding_cache_path:
                kwargs["cache_dir"] = self.embedding_cache_path
            if self.embedding_threads is not None:
                kwargs["threads"] = self.embedding_threads
            kwargs["enable_cpu_mem_arena"] = self.embedding_cpu_mem_arena_enabled
            if self.embedding_model_name == FP32_EMBEDDING_MODEL:
                register_bundled_embedding_models()
            try:
                self.embedding_model = TextEmbedding(**kwargs)
            except Exception:
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "embedding_model_load_failed",
                    model=self.embedding_model_name,
                    cache_dir=self.embedding_cache_path or "default",
                )
                LOGGER.exception("embedding_model_load_exception")
                raise
            sys.stderr.write("Embedding model ready.\n")
            sys.stderr.flush()
            log_event(LOGGER, logging.INFO, "embedding_model_ready", model=self.embedding_model_name)
            return self.embedding_model

    def load_embedding_model(self) -> None:
        """Ensure the configured embedding model is loaded without exposing it.

        This supported operation is useful for measuring model-load cost. The
        FastEmbed object remains owned by the engine.
        """
        self._get_embedding_model()

    def _chunk_lines(self, lines: List[str], window_size: int = 4, step_size: int = 2) -> List[Chunk]:
        """
        Creates sliding window chunks over lines with line numbers preserved.
        For short outputs (<= window_size), creates a single chunk.
        """
        if not lines:
            return []
            
        chunks: List[Chunk] = []
        n = len(lines)
        
        if n <= window_size:
            text = "\n".join(lines)
            return [Chunk(chunk_id=0, start_line=1, end_line=n, text=text)]
            
        chunk_idx = 0
        i = 0
        while i < n:
            end = min(i + window_size, n)
            chunk_text = "\n".join(lines[i:end])
            chunks.append(Chunk(
                chunk_id=chunk_idx,
                start_line=i + 1,
                end_line=end,
                text=chunk_text
            ))
            chunk_idx += 1
            if end == n:
                break
            i += step_size
            
        return chunks

    def _semantic_chunk_lines(self, lines: List[str]) -> List[Chunk]:
        """Pack consecutive lines into semantic windows bounded by line and byte caps.

        Each window takes at least one line, so a single oversized line becomes
        its own chunk and the tokenizer's truncation limit applies only to it.
        """
        if not lines:
            return []
        max_lines = self.semantic_chunk_lines
        max_bytes = self.semantic_chunk_bytes
        overlap = self.semantic_chunk_overlap
        chunks: List[Chunk] = []
        n = len(lines)
        start = 0
        while start < n:
            end = start + 1
            size = len(lines[start].encode("utf-8"))
            while end < n and end - start < max_lines:
                line_bytes = len(lines[end].encode("utf-8")) + 1
                if size + line_bytes > max_bytes:
                    break
                size += line_bytes
                end += 1
            chunks.append(Chunk(
                chunk_id=len(chunks),
                start_line=start + 1,
                end_line=end,
                text="\n".join(lines[start:end]),
            ))
            if end >= n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    @staticmethod
    def _chunk_count(line_count: int, window_size: int = 4, step_size: int = 2) -> int:
        """Return the number of sliding chunks without materializing them."""
        if line_count <= 0:
            return 0
        if line_count <= window_size:
            return 1
        return ((line_count - window_size + step_size - 1) // step_size) + 1

    def ingest(
        self,
        text: str,
        label: str = "",
        content_type: str = "auto",
        truncated: bool = False,
        original_byte_size: Optional[int] = None,
        command_exit_code: Optional[int] = None,
        timed_out: bool = False,
        protected_capture_ids: Optional[List[str]] = None,
        *,
        source: str = "capture",
        duration_ms: Optional[float] = None,
        structured_metrics: Optional[Dict[str, Any]] = None,
    ) -> CaptureView:
        """Ingest text and return a detached, read-only metadata view."""
        state = self._ingest_state(
            text,
            label=label,
            content_type=content_type,
            truncated=truncated,
            original_byte_size=original_byte_size,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
            protected_capture_ids=protected_capture_ids,
            source=source,
            duration_ms=duration_ms,
            structured_metrics=structured_metrics,
        )
        return self._capture_view(state)

    def ingest_with_summary(
        self,
        text: str,
        *,
        include_previews: bool = False,
        **ingest_options: Any,
    ) -> Tuple[CaptureView, Dict[str, Any]]:
        """Ingest text and atomically retain the summary needed by a caller.

        This avoids a lookup race if another ingestion evicts the capture
        before the caller serializes its response.
        """
        state = self._ingest_state(text, **ingest_options)
        view = self._capture_view(state)
        summary = self._build_summary(state, include_previews=include_previews)
        return view, summary

    def _ingest_state(
        self,
        text: str,
        label: str = "",
        content_type: str = "auto",
        truncated: bool = False,
        original_byte_size: Optional[int] = None,
        command_exit_code: Optional[int] = None,
        timed_out: bool = False,
        protected_capture_ids: Optional[List[str]] = None,
        *,
        source: str = "capture",
        duration_ms: Optional[float] = None,
        structured_metrics: Optional[Dict[str, Any]] = None,
    ) -> _CaptureState:
        """
        Ingests text, chunks it, and builds the SQLite FTS5 BM25 index.
        FastEmbed dense vector embeddings are materialized lazily when semantic
        or hybrid search first needs them. Automatically classifies content
        type (diff, log, text) and extracts structural metadata.
        """
        ingest_started = time.perf_counter()
        lines = text.splitlines()
        capture_bytes = len(text.encode("utf-8"))
        if original_byte_size is not None:
            if isinstance(original_byte_size, bool) or not isinstance(original_byte_size, int):
                raise ValueError("original_byte_size must be a non-negative integer or null")
            if original_byte_size < 0:
                raise ValueError("original_byte_size must be a non-negative integer or null")
        if duration_ms is not None:
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
                raise ValueError("duration_ms must be a non-negative finite number or null")
            if not math.isfinite(duration_ms) or duration_ms < 0:
                raise ValueError("duration_ms must be a non-negative finite number or null")
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        normalized_metrics = normalize_structured_metrics(structured_metrics)
        with self._lock:
            capture_number = self._next_id
            capture_id = f"cap_{capture_number}"
            self._next_id += 1

        if not label:
            label = f"Capture #{capture_number}"

        with self._lock:
            max_buffer_bytes = self.max_buffer_bytes
        label_bytes = len(label.encode("utf-8"))
        retained_bytes = capture_bytes + label_bytes
        if retained_bytes > max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "capture_rejected_limit",
                capture_id=capture_id,
                capture_bytes=capture_bytes,
                label_bytes=label_bytes,
                retained_bytes=retained_bytes,
                max_buffer_bytes=max_buffer_bytes,
            )
            raise ValueError(
                f"Capture content and label use {retained_bytes:,} bytes, exceeding the "
                f"{max_buffer_bytes:,}-byte buffer limit"
            )

        classified_type, diff_meta = detect_content_type(lines, label=label, content_type_hint=content_type)
        required_chunks = self._chunk_count(len(lines))
        if required_chunks > self.max_indexed_chunks:
            raise ValueError(
                f"Capture requires {required_chunks:,} indexed chunks, exceeding the "
                f"{self.max_indexed_chunks:,}-chunk index budget"
            )
        chunks = self._chunk_lines(lines)
        semantic_chunks = self._semantic_chunk_lines(lines)

        # Semantic embeddings are materialized lazily by the first semantic or
        # hybrid search. Ingestion remains useful for fast BM25 search without
        # paying the model/indexing cost when semantic ranking is unnecessary.
        embeddings = np.empty((0, 384), dtype=np.float32) if not semantic_chunks else None

        capture = _CaptureState(
            capture_id=capture_id,
            label=label,
            timestamp=time.time(),
            raw_lines=lines,
            input_byte_size=capture_bytes,
            chunks=chunks,
            semantic_chunks=semantic_chunks,
            embeddings=embeddings,
            fts_conn=None,
            content_type=classified_type,
            diff_meta=diff_meta,
            truncated=truncated,
            original_byte_size=original_byte_size,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
            source=source,
            duration_ms=duration_ms,
            structured_metrics=normalized_metrics,
        )

        with self._lock:
            protected_ids = set(protected_capture_ids or [])
            missing_protected_ids = sorted(protected_ids.difference(self._captures))
            if missing_protected_ids:
                raise ValueError(
                    "Cannot admit capture while retaining unavailable source captures: "
                    + ", ".join(missing_protected_ids)
                )

            # Plan LRU eviction before mutating state. Protected captures are
            # skipped so a consolidation can never evict its own sources.
            projected_count = len(self.capture_order) + 1
            projected_bytes = self._total_bytes + capture.retained_byte_size
            projected_chunks = self._indexed_chunks + len(capture.chunks)
            eviction_ids: List[str] = []
            for candidate_id in self.capture_order:
                if not (
                    projected_count > self.max_captures
                    or projected_bytes > self.max_buffer_bytes
                    or projected_chunks > self.max_indexed_chunks
                ):
                    break
                if candidate_id in protected_ids:
                    continue
                old_cap = self._captures[candidate_id]
                eviction_ids.append(candidate_id)
                projected_count -= 1
                projected_bytes -= old_cap.retained_byte_size
                projected_chunks -= len(old_cap.chunks)

            if (
                projected_count > self.max_captures
                or projected_bytes > self.max_buffer_bytes
                or projected_chunks > self.max_indexed_chunks
            ):
                raise ValueError(
                    "Cannot admit capture while retaining protected source captures "
                    f"(count={len(protected_ids)})"
                )

            for evicted_id in eviction_ids:
                self._evict_capture_locked(evicted_id)

            if self.lexical_backend == "fts5":
                # Captures may be ingested by the CLI socket listener thread
                # and queried by the MCP thread, so allow this connection to
                # cross threads. Build it after admission so rejected captures
                # never allocate index storage and LRU eviction can make room
                # first.
                fts_conn = sqlite3.connect(":memory:", check_same_thread=False)
                cur = fts_conn.cursor()
                cur.execute(
                    "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content, tokenize='unicode61')"
                )
                for chunk in capture.chunks:
                    cur.execute(
                        "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                        (chunk.chunk_id, chunk.text),
                    )
                fts_conn.commit()
                capture.fts_conn = fts_conn

            if capture.duration_ms is None:
                capture.duration_ms = round((time.perf_counter() - ingest_started) * 1000, 3)
            self._captures[capture_id] = capture
            self.capture_order[capture_id] = None
            self._total_bytes += capture.retained_byte_size
            self._indexed_chunks += len(capture.chunks)
            self.metrics.record_capture(capture_id)
            self.metrics.record_bytes("capture_input_bytes", capture.byte_size)
            self.metrics.record_bytes("capture_retained_bytes", capture.retained_byte_size)
            self.metrics.record_bytes(
                "capture_original_bytes",
                capture.original_byte_size if capture.original_byte_size is not None else capture.byte_size,
            )
        self._schedule_semantic_prefetch(capture)
        return capture

    def _evict_capture_locked(self, capture_id: str) -> bool:
        """Evict one capture; the caller must hold ``self._lock``."""
        self.capture_order.pop(capture_id, None)
        old_cap = self._captures.pop(capture_id, None)
        if old_cap is None:
            return False
        self._total_bytes -= old_cap.retained_byte_size
        self._indexed_chunks -= len(old_cap.chunks)
        log_event(
            LOGGER,
            logging.INFO,
            "capture_evicted",
            capture_id=capture_id,
            capture_bytes=old_cap.byte_size,
            label_bytes=old_cap.label_byte_size,
            retained_bytes=old_cap.retained_byte_size,
        )
        old_cap.semantic_index_state = "evicted"
        self._mark_semantic_job_disposition_locked(capture_id, "evicted")
        self._cancel_prefetch(capture_id, outcome="evicted")
        self._cancel_on_demand_job(capture_id, outcome="evicted")
        self._close_capture_storage(old_cap)
        self.metrics.record_event("evictions")
        self.metrics.forget_capture(capture_id)
        return True

    @synchronized
    def set_max_indexed_chunks(self, new_limit: int) -> Dict[str, Any]:
        """Adjust the session index budget and evict oldest captures if needed."""
        if isinstance(new_limit, bool) or not isinstance(new_limit, int) or new_limit < 1:
            raise ValueError("max_indexed_chunks must be a positive integer")
        previous = self.max_indexed_chunks
        self.max_indexed_chunks = new_limit
        evicted = 0
        while self._indexed_chunks > new_limit and self.capture_order:
            candidate_id = next(iter(self.capture_order))
            evicted += int(self._evict_capture_locked(candidate_id))
        result = {
            "status": "unchanged" if previous == new_limit else "updated",
            "previous": previous,
            "effective": new_limit,
            "indexed_chunks": self._indexed_chunks,
            "remaining_indexed_chunks": new_limit - self._indexed_chunks,
            "evicted_captures": evicted,
        }
        self._last_index_budget_adjustment = result
        return dict(result)

    def _record_semantic_job_metrics_locked(
        self,
        source: str,
        outcome: str,
        *,
        queued_at: Optional[float] = None,
        started_at: Optional[float] = None,
        indexed_chunks: int = 0,
        measurement: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.perf_counter()
        queue_wait_ms = (
            max(0.0, (started_at - queued_at) * 1000)
            if queued_at is not None and started_at is not None
            else None
        )
        indexing_duration_ms = (
            max(0.0, (now - started_at) * 1000)
            if started_at is not None
            else None
        )
        self.metrics.record_semantic_index(
            source,
            outcome,
            queue_wait_ms=queue_wait_ms,
            indexing_duration_ms=indexing_duration_ms,
            indexed_chunks=indexed_chunks if outcome == "completed" else 0,
            measurement=measurement,
        )
        if self._metrics_snapshot_callback is not None:
            self._metrics_snapshot_pending = True

    def _mark_semantic_job_disposition_locked(self, capture_id: str, outcome: str) -> None:
        """Remember why an in-flight semantic job will not publish its result."""
        if (
            capture_id in self._prefetch_queue
            or capture_id in self._prefetch_running
            or capture_id in self._on_demand_jobs
        ):
            self._semantic_job_dispositions[capture_id] = outcome

    def _finish_prefetch_job_locked(
        self,
        capture_id: str,
        capture: _CaptureState,
        outcome: str | None = None,
    ) -> None:
        metadata = self._prefetch_job_meta.pop(
            capture_id,
            {"queued_at": None, "started_at": None},
        )
        disposition = self._semantic_job_dispositions.pop(capture_id, None)
        if capture.embeddings is not None:
            effective_outcome = "completed"
        else:
            effective_outcome = disposition or outcome or "failed"
        self._record_semantic_job_metrics_locked(
            "prefetch",
            effective_outcome,
            queued_at=metadata.get("queued_at"),
            started_at=metadata.get("started_at"),
            indexed_chunks=len(capture.semantic_chunks),
            measurement=metadata.get("measurement"),
        )

    @staticmethod
    def _semantic_input_byte_count(capture: _CaptureState) -> int:
        """Return the UTF-8 bytes the semantic windows would send to inference."""
        return sum(
            len(chunk.text.encode("utf-8", errors="replace"))
            for chunk in capture.semantic_chunks
        )

    def _schedule_semantic_prefetch(self, capture: _CaptureState) -> None:
        """Queue post-ingestion indexing and make sure a bounded worker is draining."""
        if not capture.semantic_chunks:
            self._flush_metrics_snapshot()
            return
        if self._semantic_input_byte_count(capture) > self.semantic_max_index_input_bytes:
            with self._lock:
                if capture.embeddings is None:
                    capture.semantic_index_state = "budget-exceeded"
            log_event(
                LOGGER,
                logging.INFO,
                "semantic_index_budget_exceeded",
                capture_id=capture.capture_id,
                max_input_bytes=self.semantic_max_index_input_bytes,
                max_batch_tokens=self.embedding_max_batch_tokens,
            )
            self._flush_metrics_snapshot()
            return
        if not self.semantic_prefetch_enabled:
            self._flush_metrics_snapshot()
            return
        try:
            with self._lock:
                if self._shutdown or capture.capture_id not in self._captures:
                    return
                if (
                    capture.embeddings is not None
                    or capture.capture_id in self._prefetch_queue
                    or capture.capture_id in self._prefetch_running
                    or capture.capture_id in self._on_demand_jobs
                ):
                    return
                measurement = self.metrics.measurement_handle()
                self._prefetch_queue[capture.capture_id] = capture
                self._prefetch_job_meta[capture.capture_id] = {
                    "queued_at": time.perf_counter(),
                    "started_at": None,
                    "measurement": measurement,
                }
                self.metrics.record_semantic_index(
                    "prefetch",
                    "queued",
                    measurement=measurement,
                )
                capture.semantic_index_state = "pending"
                if self._prefetch_workers_active >= self.semantic_prefetch_workers:
                    return
                try:
                    self._prefetch_executor.submit(self._prefetch_worker)
                except Exception:
                    log_event(LOGGER, logging.ERROR, "semantic_prefetch_submit_failed", capture_id=capture.capture_id)
                    LOGGER.exception("semantic_prefetch_submit_exception")
                    if self._prefetch_workers_active == 0:
                        # Nothing will drain the queue, so leave the capture to the lazy path.
                        self._prefetch_queue.pop(capture.capture_id, None)
                        capture.semantic_index_state = "failed"
                        self._finish_prefetch_job_locked(
                            capture.capture_id,
                            capture,
                            outcome="failed",
                        )
                    return
                self._prefetch_workers_active += 1
        finally:
            self._flush_metrics_snapshot()

    def _prefetch_worker(self) -> None:
        """Drain queued captures newest-first until the queue is empty or shutdown."""
        while True:
            with self._lock:
                if self._shutdown or not self._prefetch_queue:
                    self._prefetch_workers_active -= 1
                    return
                capture_id, capture = self._prefetch_queue.popitem(last=True)
                done = threading.Event()
                self._prefetch_running[capture_id] = done
                metadata = self._prefetch_job_meta.setdefault(
                    capture_id,
                    {
                        "queued_at": time.perf_counter(),
                        "started_at": None,
                        "measurement": None,
                    },
                )
                metadata["started_at"] = time.perf_counter()
            try:
                self._ensure_embeddings(capture)
            except SemanticIndexBudgetExceeded:
                capture.semantic_index_state = "budget-exceeded"
                log_event(
                    LOGGER,
                    logging.INFO,
                    "semantic_prefetch_budget_exceeded",
                    capture_id=capture_id,
                    max_input_bytes=self.semantic_max_index_input_bytes,
                    max_batch_tokens=self.embedding_max_batch_tokens,
                )
            except Exception:
                capture.semantic_index_state = "failed"
                log_event(LOGGER, logging.ERROR, "semantic_prefetch_failed", capture_id=capture_id)
                LOGGER.exception("semantic_prefetch_exception")
            finally:
                with self._lock:
                    self._prefetch_running.pop(capture_id, None)
                    if capture.semantic_index_state == "pending":
                        # The capture was evicted before its embeddings were published.
                        capture.semantic_index_state = "not-requested"
                    self._finish_prefetch_job_locked(capture_id, capture)
                done.set()
                self._flush_metrics_snapshot()

    def _start_semantic_index(self, capture: _CaptureState) -> Optional[Tuple[threading.Event, Optional[_SemanticIndexJob]]]:
        """Return the completion event for the job indexing ``capture``, starting one if needed.

        A running prefetch job is reused.  Otherwise the capture is pulled out
        of the prefetch queue, because a search is a stronger signal than queue
        position, and indexed on the bounded on-demand pool, where the job
        outlives the caller's wait budget.  Returns ``None`` when the
        embeddings are already ready.
        """
        try:
            with self._lock:
                if getattr(capture, "semantic_index_state", None) == "budget-exceeded":
                    raise SemanticIndexBudgetExceeded(
                        "Semantic indexing previously exceeded its configured work budget."
                    )
                if capture.embeddings is not None:
                    return None
                running = self._prefetch_running.get(capture.capture_id)
                if running is not None:
                    return running, None
                job = self._on_demand_jobs.get(capture.capture_id)
                if job is not None:
                    return job.done, job
                if capture.capture_id in self._prefetch_queue:
                    self._cancel_prefetch(capture.capture_id)
                if self._shutdown:
                    # No background thread may start after shutdown; the caller
                    # indexes inline as the lazy path always could.
                    finished = threading.Event()
                    finished.set()
                    return finished, None
                job = _SemanticIndexJob(measurement=self.metrics.measurement_handle())
                try:
                    job.future = self._on_demand_executor.submit(
                        self._on_demand_index_worker,
                        capture,
                        job,
                    )
                except Exception:
                    capture.semantic_index_state = "failed"
                    self.metrics.record_semantic_index(
                        "on_demand",
                        "failed",
                        measurement=job.measurement,
                    )
                    raise
                self._on_demand_jobs[capture.capture_id] = job
                self.metrics.record_semantic_index(
                    "on_demand",
                    "queued",
                    measurement=job.measurement,
                )
                capture.semantic_index_state = "pending"
                return job.done, job
        finally:
            self._flush_metrics_snapshot()

    def _on_demand_index_worker(self, capture: _CaptureState, job: _SemanticIndexJob) -> None:
        """Materialize one capture's embeddings and publish the outcome to waiters."""
        job.started_at = time.perf_counter()
        try:
            self._ensure_embeddings(capture)
        except Exception as exc:
            job.error = exc
            if isinstance(exc, SemanticIndexBudgetExceeded):
                capture.semantic_index_state = "budget-exceeded"
                log_event(
                    LOGGER,
                    logging.INFO,
                    "semantic_index_budget_exceeded",
                    capture_id=capture.capture_id,
                    max_input_bytes=self.semantic_max_index_input_bytes,
                    max_batch_tokens=self.embedding_max_batch_tokens,
                )
            else:
                capture.semantic_index_state = "failed"
                log_event(LOGGER, logging.ERROR, "semantic_index_failed", capture_id=capture.capture_id)
                LOGGER.exception("semantic_index_exception")
        finally:
            with self._lock:
                disposition = self._semantic_job_dispositions.pop(capture.capture_id, None)
                outcome = (
                    "completed"
                    if capture.embeddings is not None
                    else disposition or "failed"
                )
                self._record_semantic_job_metrics_locked(
                    "on_demand",
                    outcome,
                    queued_at=job.queued_at,
                    started_at=job.started_at,
                    indexed_chunks=len(capture.semantic_chunks),
                    measurement=job.measurement,
                )
                self._on_demand_jobs.pop(capture.capture_id, None)
                if capture.semantic_index_state == "pending":
                    # The capture was evicted before its embeddings were published.
                    capture.semantic_index_state = "not-requested"
            job.done.set()
            self._flush_metrics_snapshot()

    def _await_semantic_index(self, capture: _CaptureState, timeout: Optional[float] = None) -> str:
        """Wait up to ``timeout`` seconds for the capture's semantic index.

        Returns ``"ready"`` or ``"pending"``.  ``None`` and ``inf`` wait until
        the index is ready or its job fails, in which case the job's exception
        is re-raised so callers keep the lazy-path error semantics. A bounded
        wait that expires, or observes a cancelled or evicted job, reports
        ``"pending"``. If a completed prefetch did not publish embeddings for
        a retained capture, indexing retries inline and may exceed the wait
        budget.
        """
        started = self._start_semantic_index(capture)
        if started is None:
            return "ready"
        done, job = started
        wait_seconds = None if timeout is None or math.isinf(timeout) else timeout
        if not done.wait(wait_seconds):
            return "pending"
        if job is not None and job.error is not None:
            raise job.error
        with self._lock:
            if capture.embeddings is not None:
                return "ready"
            retained = self._captures.get(capture.capture_id) is capture
        if wait_seconds is not None and (not retained or (job is not None and job.cancelled)):
            # The job was cancelled (eviction or shutdown) or finished without
            # publishing for an evicted capture.  Indexing inline would ignore
            # the budget and could block behind the model lock, so a bounded
            # waiter answers lexical-first; an unbounded one still indexes
            # under its reader lease below.
            return "pending"
        # The finished job could not publish (failed prefetch, or the capture
        # was evicted mid-flight); index inline so a retained capture still
        # gets a result and a failed one raises its error.
        self._ensure_embeddings(capture)
        return "ready"

    def _wait_for_state_index(self, capture: _CaptureState, timeout: Optional[float] = None) -> str:
        """Block until the capture's semantic index is ready, failed, or ``timeout`` elapses.

        Returns ``"ready"``, ``"pending"``, or ``"failed"``; it never raises for
        an indexing error, so callers can poll from tests and benchmarks.
        """
        try:
            return self._await_semantic_index(capture, timeout)
        except Exception:
            return "failed"

    def _close_capture_storage(self, capture: _CaptureState) -> None:
        """Close per-capture search storage and report cleanup failures."""
        if capture.active_readers:
            if not capture.deferred_storage_tracked:
                capture.deferred_storage_tracked = True
                self._deferred_storage_capture_count += 1
                self._deferred_storage_readers += capture.active_readers
                capture.deferred_storage_accounted_bytes = capture.retained_byte_size
                if capture.embeddings is not None:
                    capture.deferred_storage_accounted_bytes += int(capture.embeddings.nbytes)
                self._deferred_storage_bytes += capture.deferred_storage_accounted_bytes
            capture.storage_close_pending = True
            return
        if not capture.fts_conn:
            return
        try:
            capture.fts_conn.close()
            capture.fts_conn = None
        except Exception:
            log_event(
                LOGGER,
                logging.ERROR,
                "capture_storage_cleanup_failed",
                capture_id=capture.capture_id,
            )
            LOGGER.exception("capture_storage_cleanup_exception")

    def _acquire_capture_reader(self, capture_id: str) -> Optional[_CaptureState]:
        """Return a capture while retaining its storage for one search reader."""
        with self._lock:
            if not self._captures:
                return None
            if capture_id == "latest" or not capture_id:
                capture = self._captures[next(reversed(self.capture_order))]
            else:
                capture = self._captures.get(capture_id)
            if capture:
                self._touch_capture(capture.capture_id)
                capture.active_readers += 1
            return capture

    def _release_capture_reader(self, capture: _CaptureState) -> None:
        """Release a search reader and finish deferred storage cleanup."""
        with self._lock:
            capture.active_readers = max(0, capture.active_readers - 1)
            if capture.deferred_storage_tracked:
                self._deferred_storage_readers = max(0, self._deferred_storage_readers - 1)
                if capture.active_readers == 0:
                    self._deferred_storage_capture_count = max(
                        0, self._deferred_storage_capture_count - 1
                    )
                    self._deferred_storage_bytes = max(
                        0,
                        self._deferred_storage_bytes
                        - capture.deferred_storage_accounted_bytes,
                    )
                    capture.deferred_storage_tracked = False
                    capture.deferred_storage_accounted_bytes = 0
            if capture.active_readers == 0 and capture.storage_close_pending:
                capture.storage_close_pending = False
                self._close_capture_storage(capture)

    def _cancel_prefetch(self, capture_id: str, *, outcome: str = "cancelled") -> None:
        """Drop queued prefetch work for a capture; running work is allowed to finish."""
        capture = self._prefetch_queue.pop(capture_id, None)
        if capture is not None:
            capture.semantic_index_state = "not-requested"
            self._finish_prefetch_job_locked(capture_id, capture, outcome=outcome)

    def _cancel_on_demand_job(self, capture_id: str, *, outcome: str = "cancelled") -> None:
        """Cancel a capture's on-demand job unless it already holds a pool thread.

        The caller holds ``self._lock``.  A running job finishes on its own
        and skips publishing for an evicted capture.  A cancelled job runs no
        worker cleanup, so its waiters are released here: one past its budget
        has already answered, and ``_await_semantic_index`` keeps a bounded
        waiter from indexing inline.  Reached from eviction, ``clear``, and
        ``shutdown``; the capture is still buffered when clearing all captures
        (it is marked evicted right after) or at shutdown (it then stays
        lazily indexable).
        """
        job = self._on_demand_jobs.get(capture_id)
        if job is None or not job.future.cancel():
            return
        job.cancelled = True
        self._on_demand_jobs.pop(capture_id, None)
        disposition = self._semantic_job_dispositions.pop(capture_id, None)
        self._record_semantic_job_metrics_locked(
            "on_demand",
            disposition or outcome,
            queued_at=job.queued_at,
            started_at=job.started_at,
            measurement=job.measurement,
        )
        live = self._captures.get(capture_id)
        if live is not None and live.semantic_index_state == "pending":
            live.semantic_index_state = "not-requested"
        job.done.set()

    def shutdown(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        """Stop queued embedding work and bound waits for running native inference."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        with self._lock:
            self._shutdown = True
            executor = self._prefetch_executor
            warmup_thread = self._embedding_warmup_thread
            on_demand_executor = self._on_demand_executor
            for capture_id in list(self._prefetch_queue):
                self._cancel_prefetch(capture_id)
            for capture_id in list(self._on_demand_jobs):
                self._cancel_on_demand_job(capture_id)
        self._flush_metrics_snapshot()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        on_demand_executor.shutdown(wait=False, cancel_futures=True)
        if warmup_thread is not None and warmup_thread is not threading.current_thread():
            warmup_thread.join(max(0.0, deadline - time.monotonic()))
        self._wait_for_executor_workers(executor, deadline)
        self._wait_for_executor_workers(on_demand_executor, deadline)
        with self._lock:
            return {"unfinished_work": self._unfinished_shutdown_work_locked()}

    @staticmethod
    def _wait_for_executor_workers(executor, deadline: float) -> None:
        """Wait for a pool's threads only while the shared shutdown budget remains."""
        if executor is None:
            return
        workers = tuple(getattr(executor, "_threads", ()))
        current_thread = threading.current_thread()
        while True:
            live_workers = [
                worker
                for worker in workers
                if worker is not current_thread and worker.is_alive()
            ]
            if not live_workers:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            live_workers[0].join(min(remaining, 0.01))

    def _unfinished_shutdown_work_locked(self) -> List[str]:
        """Summarize running engine workers while the engine lock is held."""
        unfinished = []
        warmup_thread = self._embedding_warmup_thread
        if warmup_thread is not None and warmup_thread.is_alive():
            unfinished.append("embedding_warmup")
        if self._prefetch_workers_active:
            unfinished.append(f"semantic_prefetch:{self._prefetch_workers_active}")
        running_on_demand = sum(
            1 for job in self._on_demand_jobs.values() if job.future.running()
        )
        if running_on_demand:
            unfinished.append(f"semantic_index:{running_on_demand}")
        return unfinished

    @synchronized
    def _get_capture_state(self, capture_id: str = "latest") -> Optional[_CaptureState]:
        """Resolve engine-owned storage for internal engine operations only."""
        if not self._captures:
            return None
        if capture_id == "latest" or not capture_id:
            capture = self._captures[next(reversed(self.capture_order))]
        else:
            capture = self._captures.get(capture_id)
        if capture:
            self._touch_capture(capture.capture_id)
        return capture

    @staticmethod
    def _capture_view(capture: _CaptureState) -> CaptureView:
        """Create a detached, recursively read-only capture metadata view."""
        diff_meta = _freeze_public_value(capture.diff_meta) if capture.diff_meta is not None else None
        structured_metrics = _freeze_public_value(capture.structured_metrics)
        return CaptureView(
            capture_id=capture.capture_id,
            label=capture.label,
            timestamp=capture.timestamp,
            line_count=capture.line_count,
            input_byte_size=capture.input_byte_size,
            content_type=capture.content_type,
            diff_meta=diff_meta,
            truncated=capture.truncated,
            original_byte_size=capture.original_byte_size,
            command_exit_code=capture.command_exit_code,
            timed_out=capture.timed_out,
            semantic_index_state=capture.semantic_index_state,
            source=capture.source,
            duration_ms=capture.duration_ms,
            structured_metrics=structured_metrics,
        )

    @synchronized
    def get_capture_view(self, capture_id: str = "latest") -> Optional[CaptureView]:
        """Return immutable metadata without exposing capture storage or leases."""
        capture = self._get_capture_state(capture_id)
        return self._capture_view(capture) if capture is not None else None

    def get_capture(self, capture_id: str = "latest") -> Optional[CaptureView]:
        """Compatibility wrapper returning the supported read-only capture view.

        Callers that previously read ``raw_lines`` should use
        ``get_capture_slice``; indexing and storage fields are engine-owned.
        """
        return self.get_capture_view(capture_id)

    def index_capture(self, capture_id: str = "latest") -> str:
        """Ensure one capture's semantic index and return its supported status.

        Returns ``ready`` or ``budget_exceeded``. Other indexing failures are
        represented by the capture's ``failed`` diagnostic state.
        """
        capture = self._acquire_capture_reader(capture_id)
        if capture is None:
            return "not-found"
        try:
            if not capture.semantic_chunks:
                return "ready"
            try:
                self._ensure_embeddings(capture)
            except SemanticIndexBudgetExceeded:
                return "budget_exceeded"
            return "ready" if capture.semantic_index_state == "ready" else capture.semantic_index_state
        except Exception:
            with self._lock:
                if (
                    self._captures.get(capture.capture_id) is capture
                    and capture.embeddings is None
                ):
                    capture.semantic_index_state = "failed"
            return "failed"
        finally:
            self._release_capture_reader(capture)

    def wait_for_capture_index(
        self,
        capture_id: str = "latest",
        timeout: Optional[float] = None,
    ) -> str:
        """Wait for indexing by capture ID without exposing capture state."""
        capture = self._acquire_capture_reader(capture_id)
        if capture is None:
            return "not-found"
        try:
            return self._wait_for_state_index(capture, timeout)
        finally:
            self._release_capture_reader(capture)

    def wait_for_semantic_index(self, capture: Any, timeout: Optional[float] = None) -> str:
        """Compatibility wrapper accepting a view, ID, or legacy internal state."""
        if isinstance(capture, _CaptureState):
            return self._wait_for_state_index(capture, timeout)
        capture_id = capture.capture_id if isinstance(capture, CaptureView) else capture
        if not isinstance(capture_id, str):
            return "failed"
        return self.wait_for_capture_index(capture_id, timeout)

    @synchronized
    def get_capture_diagnostics(self, capture_id: str = "latest") -> Optional[CaptureDiagnostics]:
        """Return stable, supported semantic-index measurements for a capture."""
        capture = self._get_capture_state(capture_id)
        if capture is None:
            return None
        embedding_bytes = int(capture.embeddings.nbytes) if capture.embeddings is not None else 0
        return CaptureDiagnostics(
            capture_id=capture.capture_id,
            semantic_index_state=capture.semantic_index_state,
            lexical_chunk_count=len(capture.chunks),
            semantic_chunk_count=len(capture.semantic_chunks),
            semantic_input_bytes=self._semantic_input_byte_count(capture),
            retained_embedding_bytes=embedding_bytes,
        )

    def _touch_capture(self, capture_id: str) -> None:
        """Marks a capture as recently used for LRU eviction."""
        if capture_id in self.capture_order:
            self.capture_order.move_to_end(capture_id)

    @synchronized
    def search_bm25(self, capture: _CaptureState, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using SQLite FTS5 BM25. Query punctuation is treated as a
        separator; terms are combined with OR. Returns (chunk_id, score).
        """
        if not capture.chunks:
            return []

        # FTS5 syntax is intentionally not exposed: regex-like characters,
        # quotes, and operators are treated as punctuation rather than query
        # language. This keeps keyword search predictable and injection-safe.
        tokens = _fts5_tokens(query)
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        
        if not capture.fts_conn:
            return self._search_lexical_fallback(capture, tokens, top_k)

        cur = capture.fts_conn.cursor()
        try:
            cur.execute("""
                SELECT chunk_id, bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
            """, (fts_query, top_k))
            results = cur.fetchall()
            return [(int(row[0]), -float(row[1])) for row in results]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _search_lexical_fallback(
        capture: _CaptureState, tokens: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """Search chunks with complete token matching when SQLite lacks FTS5."""
        query_terms = set(tokens)
        ranked: List[Tuple[int, float]] = []
        for chunk in capture.chunks:
            chunk_terms = Counter(_fallback_tokens(chunk.text))
            matched_terms = query_terms.intersection(chunk_terms)
            if not matched_terms:
                continue
            score = sum(chunk_terms[term] for term in matched_terms) / max(sum(chunk_terms.values()), 1)
            ranked.append((chunk.chunk_id, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def search_semantic(self, capture: _CaptureState, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense vector cosine similarity search over the semantic windows.
        Returns list of (semantic_chunk_id, score).
        """
        if not capture.semantic_chunks:
            return []

        self._await_semantic_index(capture)
        with self._lock:
            embeddings = capture.embeddings
        if embeddings is None or len(embeddings) == 0:
            return []
            
        with self._embedding_lock:
            query_embed = list(self._get_embedding_model().embed([query]))[0]
        query_embed = np.array(query_embed, dtype=np.float32)
        norm = np.linalg.norm(query_embed)
        if norm > 0:
            query_embed = query_embed / norm

        similarities = np.dot(embeddings, query_embed)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.0]

    def _semantic_embedding_batches(self, model: Any, texts: List[str]) -> List[List[Tuple[int, str]]]:
        """Group similar token lengths into batches within both configured limits."""
        backend = getattr(model, "model", None)
        tokenizer = getattr(backend, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        indexed_texts = []
        for index, text in enumerate(texts):
            if callable(encode):
                encoded = encode(text)
                token_ids = getattr(encoded, "ids", None)
                token_length = len(token_ids) if token_ids is not None else 0
            else:
                # Test doubles and compatible non-FastEmbed adapters may not
                # expose a tokenizer. UTF-8 bytes are a conservative proxy.
                token_length = len(text.encode("utf-8", errors="replace"))
            token_length = max(1, token_length)
            if token_length > self.embedding_max_batch_tokens:
                raise SemanticIndexBudgetExceeded(
                    "A semantic chunk exceeds the configured padded-token batch limit."
                )
            indexed_texts.append((token_length, index, text))

        indexed_texts.sort(key=lambda entry: (entry[0], entry[1]))
        batches: List[List[Tuple[int, str]]] = []
        batch: List[Tuple[int, str]] = []
        batch_max_tokens = 0
        for token_length, index, text in indexed_texts:
            candidate_count = len(batch) + 1
            candidate_max_tokens = max(batch_max_tokens, token_length)
            if batch and (
                candidate_count > self.embedding_batch_size
                or candidate_count * candidate_max_tokens > self.embedding_max_batch_tokens
            ):
                batches.append(batch)
                batch = []
                batch_max_tokens = 0
                candidate_count = 1
                candidate_max_tokens = token_length
            batch.append((index, text))
            batch_max_tokens = candidate_max_tokens
        if batch:
            batches.append(batch)
        return batches

    def _ensure_embeddings(self, capture: _CaptureState) -> None:
        """Materialize and cache dense embeddings for a captured chunk set."""
        with self._lock:
            if capture.embeddings is not None:
                return
            if getattr(capture, "semantic_index_state", None) == "budget-exceeded":
                raise SemanticIndexBudgetExceeded(
                    "Semantic indexing previously exceeded its configured work budget."
                )
            # A search reader may outlive LRU admission.  Its lease keeps the
            # capture's chunks and storage alive long enough to finish lazy
            # indexing even after the capture is removed from ``self._captures``.
            if (
                self._captures.get(capture.capture_id) is not capture
                and not getattr(capture, "active_readers", 0)
            ):
                return
            chunk_texts = [chunk.text for chunk in capture.semantic_chunks]
        with self._embedding_lock:
            with self._lock:
                if capture.embeddings is not None:
                    return
                if (
                    self._captures.get(capture.capture_id) is not capture
                    and not getattr(capture, "active_readers", 0)
                ):
                    return
                if getattr(capture, "semantic_index_state", None) == "budget-exceeded":
                    raise SemanticIndexBudgetExceeded(
                        "Semantic indexing previously exceeded its configured work budget."
                    )
            input_bytes = self._semantic_input_byte_count(capture)
            if input_bytes > self.semantic_max_index_input_bytes:
                with self._lock:
                    capture.semantic_index_state = "budget-exceeded"
                raise SemanticIndexBudgetExceeded(
                    "Semantic indexing input exceeds the configured per-capture byte budget."
                )

            model = self._get_embedding_model()
            batches = self._semantic_embedding_batches(model, chunk_texts)
            embeddings: Optional[np.ndarray] = None
            for batch in batches:
                batch_embeddings = list(model.embed([text for _, text in batch]))
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError("Embedding model returned an unexpected batch size.")
                if embeddings is None:
                    first_embedding = np.asarray(batch_embeddings[0], dtype=np.float32)
                    if first_embedding.ndim != 1 or first_embedding.size == 0:
                        raise RuntimeError("Embedding model returned an invalid vector shape.")
                    embeddings = np.empty(
                        (len(chunk_texts), first_embedding.size),
                        dtype=np.float32,
                    )
                for (index, _), embedding in zip(batch, batch_embeddings):
                    vector = np.asarray(embedding, dtype=np.float32)
                    if vector.ndim != 1 or vector.size != embeddings.shape[1]:
                        raise RuntimeError("Embedding model returned inconsistent vector dimensions.")
                    embeddings[index] = vector
                del batch_embeddings
            if embeddings is None:
                return
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            np.divide(embeddings, norms, out=embeddings)
            with self._lock:
                if (
                    (
                        self._captures.get(capture.capture_id) is capture
                        or getattr(capture, "active_readers", 0)
                    )
                    and capture.embeddings is None
                ):
                    capture.embeddings = embeddings
                    capture.semantic_index_state = "ready"
                    if capture.deferred_storage_tracked:
                        embedding_bytes = int(embeddings.nbytes)
                        self._deferred_storage_bytes += embedding_bytes
                        capture.deferred_storage_accounted_bytes += embedding_bytes

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        capture_id: str = "latest",
        top_k: int = 5,
        context_lines: int = 3,
        response_budget_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Performs BM25, Semantic, or Hybrid (Reciprocal Rank Fusion) search across the capture.
        """
        if mode not in SEARCH_MODES:
            return {
                "status": "error",
                "error_code": "unsupported_search_mode",
                "message": f"Unsupported search mode '{mode}'. Choose one of: {', '.join(SEARCH_MODES)}.",
            }
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return {"status": "error", "error_code": "invalid_top_k", "message": "top_k must be at least 1."}
        if top_k > MAX_SEARCH_TOP_K:
            return {
                "status": "error",
                "error_code": "top_k_limit_exceeded",
                "message": f"top_k must not exceed {MAX_SEARCH_TOP_K}.",
            }
        if isinstance(context_lines, bool) or not isinstance(context_lines, int) or context_lines < 0:
            return {
                "status": "error",
                "error_code": "invalid_context_lines",
                "message": "context_lines must be non-negative.",
            }
        if context_lines > MAX_SEARCH_CONTEXT_LINES:
            return {
                "status": "error",
                "error_code": "context_limit_exceeded",
                "message": f"context_lines must not exceed {MAX_SEARCH_CONTEXT_LINES}.",
            }
        if response_budget_bytes is not None and (
            isinstance(response_budget_bytes, bool)
            or not isinstance(response_budget_bytes, int)
            or response_budget_bytes < 1
        ):
            return {
                "status": "error",
                "error_code": "invalid_response_budget",
                "message": "response_budget_bytes must be a positive integer or null.",
            }

        capture = self._acquire_capture_reader(capture_id)
        if not capture:
            return {
                "status": "error",
                "error_code": "capture_not_found",
                "message": f"No capture found for ID '{capture_id}'. Buffer is currently empty."
            }

        try:
            return self._search_capture(
                capture,
                query,
                mode,
                top_k,
                context_lines,
                response_budget_bytes,
            )
        finally:
            self._release_capture_reader(capture)

    def _search_capture(
        self,
        capture: _CaptureState,
        query: str,
        mode: str,
        top_k: int,
        context_lines: int,
        response_budget_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run search while the caller retains the capture storage lease."""

        if capture.line_count == 0:
            self.metrics.record_search(capture.capture_id, 0)
            # No lines means no semantic windows to index, so a semantic or
            # hybrid request is trivially complete rather than pending.
            return {
                "status": "ok",
                "capture_id": capture.capture_id,
                "label": capture.label,
                "matches": [],
                "semantic_coverage": "complete" if mode in ("semantic", "hybrid") else "not-requested",
                "message": "Capture is empty (0 lines).",
            }

        bm25_results = []
        semantic_results = []

        if mode in ("bm25", "hybrid"):
            bm25_results = self.search_bm25(capture, query, top_k=top_k * 3)
            
        semantic_fallback = None
        semantic_coverage = "not-requested"
        if mode in ("semantic", "hybrid"):
            semantic_coverage = "complete"
            try:
                # Hybrid has lexical results to fall back on, so it waits only
                # up to the budget; semantic mode has nothing else to return
                # and waits for the index.
                if mode == "hybrid":
                    index_state = self._await_semantic_index(capture, self.semantic_wait_seconds)
                else:
                    index_state = self._await_semantic_index(capture)
                if index_state == "ready":
                    semantic_results = self.search_semantic(capture, query, top_k=top_k * 3)
                else:
                    semantic_coverage = "pending"
                    self.metrics.record_semantic_search("pending_hybrid_responses")
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "hybrid_search_semantic_pending",
                        capture_id=capture.capture_id,
                        line_count=capture.line_count,
                        wait_seconds=self.semantic_wait_seconds,
                    )
            except SemanticIndexBudgetExceeded as exc:
                semantic_fallback = type(exc).__name__
                semantic_coverage = "unavailable"
                self.metrics.record_semantic_search("semantic_fallbacks")
                if mode == "semantic":
                    bm25_results = self.search_bm25(capture, query, top_k=top_k * 3)
                log_event(
                    LOGGER,
                    logging.INFO,
                    "semantic_search_budget_fallback",
                    capture_id=capture.capture_id,
                    error_type=semantic_fallback,
                )
            except Exception as exc:
                if mode == "semantic":
                    raise
                semantic_fallback = type(exc).__name__
                semantic_coverage = "unavailable"
                self.metrics.record_semantic_search("semantic_fallbacks")
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "hybrid_search_lexical_fallback",
                    error_type=semantic_fallback,
                )

        # Lexical and semantic hits come from different chunk grids, so fusion
        # happens in line space: a semantic hit boosts lexical windows whose
        # first line it contains. With overlapping semantic windows, the first
        # ranked matching window owns each lexical boost. A semantic hit stands
        # on its own only when no lexical windows belong to it.
        k_const = 60.0
        candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def add_candidate(index: str, chunk: Chunk, score: float) -> None:
            key = (index, chunk.chunk_id)
            entry = candidates.get(key)
            if entry is None:
                candidates[key] = {"index": index, "chunk": chunk, "score": score}
            else:
                entry["score"] += score

        if mode == "bm25":
            for cid, score in bm25_results:
                add_candidate("lexical", capture.chunks[cid], score)
        elif mode == "semantic":
            if semantic_fallback:
                for cid, score in bm25_results:
                    add_candidate("lexical", capture.chunks[cid], score)
            else:
                for sid, score in semantic_results:
                    add_candidate("semantic", capture.semantic_chunks[sid], score)
        else:  # hybrid
            for rank, (cid, _) in enumerate(bm25_results):
                add_candidate("lexical", capture.chunks[cid], HYBRID_LEXICAL_WEIGHT / (k_const + rank + 1))
            boosted_lexical_ids: set[int] = set()
            for rank, (sid, _) in enumerate(semantic_results):
                semantic_chunk = capture.semantic_chunks[sid]
                contribution = 1.0 / (k_const + rank + 1)
                members = [
                    entry for entry in candidates.values()
                    if entry["index"] == "lexical"
                    and semantic_chunk.start_line <= entry["chunk"].start_line <= semantic_chunk.end_line
                ]
                if members:
                    for entry in members:
                        chunk_id = entry["chunk"].chunk_id
                        if chunk_id not in boosted_lexical_ids:
                            entry["score"] += contribution
                            boosted_lexical_ids.add(chunk_id)
                else:
                    add_candidate("semantic", semantic_chunk, contribution)

        # Deduplicate before applying top_k: a candidate adds nothing when its
        # matched lines overlap an earlier match's matched lines or are already
        # fully visible inside an earlier match's context.  The search backends
        # intentionally over-fetch candidates so a sliding window cannot consume
        # the result quota with duplicate context, while adjacent windows that
        # would reveal new lines are still returned.
        ranked = sorted(candidates.values(), key=lambda entry: entry["score"], reverse=True)

        matches = []
        seen_line_ranges = []
        materialization_budget = (
            None if response_budget_bytes is None else response_budget_bytes // 2
        )
        response_matches_omitted = False

        for entry in ranked:
            chunk = entry["chunk"]
            score = entry["score"]
            ctx_start = max(1, chunk.start_line - context_lines)
            ctx_end = min(capture.line_count, chunk.end_line + context_lines)
            
            redundant = any(
                (chunk.start_line <= core_e and core_s <= chunk.end_line)
                or (prev_s <= chunk.start_line and chunk.end_line <= prev_e)
                for core_s, core_e, prev_s, prev_e in seen_line_ranges
            )
            if redundant:
                continue

            line_count = ctx_end - ctx_start + 1
            if materialization_budget is None:
                snippet_budget = SEARCH_MATCH_SNIPPET_MAX_BYTES
                context_budget = SEARCH_MATCH_CONTEXT_MAX_BYTES
                per_line_budget = SEARCH_SNIPPET_MAX_BYTES
            else:
                # The MCP response contains structured data and rendered text.
                # Reserve half of the wire budget for those two copies plus
                # metadata, and stop before constructing another raw context
                # when the remaining construction allowance is too small.
                if materialization_budget < 2 * line_count + 128:
                    response_matches_omitted = True
                    break
                snippet_budget = min(
                    SEARCH_MATCH_SNIPPET_MAX_BYTES,
                    materialization_budget // 2,
                )
                context_budget = min(
                    SEARCH_MATCH_CONTEXT_MAX_BYTES,
                    materialization_budget - snippet_budget,
                )
                if snippet_budget < 2 * line_count or context_budget < 64:
                    response_matches_omitted = True
                    break
                per_line_budget = max(1, (snippet_budget - line_count) // line_count)

            lines_with_numbers = []
            for line_no in range(ctx_start, ctx_end + 1):
                raw = capture.raw_lines[line_no - 1]
                is_match_core = chunk.start_line <= line_no <= chunk.end_line
                prefix = ">" if is_match_core else " "
                lines_with_numbers.append(_bounded_preview(
                    f"{prefix} {line_no:5d} | {raw}",
                    max_bytes=per_line_budget,
                    marker=SEARCH_SNIPPET_TRUNCATION_MARKER,
                ))
            snippet, snippet_truncated = _bounded_join_lines(
                lines_with_numbers,
                snippet_budget,
                "... [snippet truncated; use get_capture_slice for full content] ...",
            )
            raw_context, context_truncated = _bounded_join_lines(
                capture.raw_lines[ctx_start - 1:ctx_end],
                context_budget,
                "... [raw context truncated; use get_capture_slice for full content] ...",
            )
            seen_line_ranges.append((chunk.start_line, chunk.end_line, ctx_start, ctx_end))
            matches.append({
                "chunk_id": chunk.chunk_id,
                "chunk_index": entry["index"],
                "score": round(score, 4),
                "matched_range": f"L{chunk.start_line}-L{chunk.end_line}",
                "context_range": f"L{ctx_start}-L{ctx_end}",
                "context_start_line": ctx_start,
                "context_end_line": ctx_end,
                "context": raw_context,
                "context_truncated": context_truncated,
                "snippet": snippet,
                "snippet_truncated": snippet_truncated,
            })
            if materialization_budget is not None:
                # Debit the materialized strings rather than their maximum
                # allowances so short matches do not consume unused budget.
                materialization_budget -= (
                    len(snippet.encode("utf-8"))
                    + len(raw_context.encode("utf-8"))
                    + 256
                )
            if len(matches) >= top_k:
                break

        self.metrics.record_search(capture.capture_id, len(matches))
        result = {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "total_lines": capture.line_count,
            "mode": mode,
            "query": query,
            "match_count": len(matches),
            "matches": matches,
            "semantic_coverage": semantic_coverage,
            "response_matches_omitted": response_matches_omitted,
        }
        if semantic_fallback:
            result["semantic_fallback"] = semantic_fallback
            if mode == "semantic":
                result["message"] = "Semantic indexing exceeded its work budget; returning BM25 results."
        if semantic_coverage == "pending":
            result["semantic_index_state"] = capture.semantic_index_state
            result["semantic_wait_seconds"] = self.semantic_wait_seconds
            result["message"] = (
                "Semantic index still building for this capture; results are lexical "
                "(BM25) only. Repeat the search for hybrid ranking."
            )
        return result

    def _encode_slice_cursor(
        self,
        *,
        capture_id: str,
        start_line: int,
        end_line: int,
        line: int,
        character_offset: int,
        utf8_byte_offset: int,
    ) -> str:
        payload = json.dumps(
            {
                "v": 1,
                "capture_id": capture_id,
                "start_line": start_line,
                "end_line": end_line,
                "line": line,
                "character_offset": character_offset,
                "utf8_byte_offset": utf8_byte_offset,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(self._slice_cursor_secret, payload, hashlib.sha256).digest()[:16]
        token = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        token += "." + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        return token

    def _decode_slice_cursor(
        self,
        cursor: str,
        *,
        capture_id: str,
        start_line: int,
        end_line: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(cursor, str) or len(cursor) > 2048:
            return None
        try:
            payload_part, signature_part = cursor.split(".", 1)
            payload_bytes = base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
            signature = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
            expected = hmac.new(self._slice_cursor_secret, payload_bytes, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(signature, expected):
                return None
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
            return None

        if not isinstance(payload, dict):
            return None
        if (
            payload.get("v") != 1
            or payload.get("capture_id") != capture_id
            or payload.get("start_line") != start_line
            or payload.get("end_line") != end_line
        ):
            return None
        line = payload.get("line")
        character_offset = payload.get("character_offset")
        utf8_byte_offset = payload.get("utf8_byte_offset")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            line, character_offset, utf8_byte_offset
        )):
            return None
        return payload

    def get_slice_page(
        self,
        start_line: int,
        end_line: int,
        capture_id: str = "latest",
        max_content_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
        cursor: Optional[str] = None,
        max_segment_bytes: int = CAPTURE_SLICE_SEGMENT_MAX_BYTES,
    ) -> Dict[str, Any]:
        """Build one bounded, lossless page from the retained line representation.

        Cursor offsets are zero-based UTF-8 byte offsets within a line. The token
        also carries a character offset so continuation can seek without copying or
        re-encoding a very long retained line.
        """
        if isinstance(max_content_bytes, bool) or not isinstance(max_content_bytes, int):
            return {
                "status": "error",
                "error_code": "invalid_byte_budget",
                "message": "max_content_bytes must be an integer.",
            }
        if max_content_bytes < 4 or max_content_bytes > MAX_CAPTURE_SLICE_CONTENT_BYTES:
            return {
                "status": "error",
                "error_code": "invalid_byte_budget",
                "message": (
                    f"max_content_bytes must be between 4 and "
                    f"{MAX_CAPTURE_SLICE_CONTENT_BYTES:,}."
                ),
            }
        if (
            isinstance(max_segment_bytes, bool)
            or not isinstance(max_segment_bytes, int)
            or max_segment_bytes < 4
            or max_segment_bytes > CAPTURE_SLICE_SEGMENT_MAX_BYTES
        ):
            return {
                "status": "error",
                "error_code": "invalid_byte_budget",
                "message": (
                    f"max_segment_bytes must be between 4 and "
                    f"{CAPTURE_SLICE_SEGMENT_MAX_BYTES:,}."
                ),
            }

        # Resolve private storage while holding the engine lock. The retained
        # raw lines are immutable, so page construction can run outside it.
        capture = self._get_capture_state(capture_id)
        if not capture:
            return {
                "status": "error",
                "error_code": "capture_not_found",
                "message": f"Capture '{capture_id}' not found.",
            }

        start = max(1, start_line)
        end = min(capture.line_count, end_line)
        if start > end or start > capture.line_count:
            return {
                "status": "error",
                "error_code": "invalid_range",
                "message": f"Invalid range {start_line}-{end_line} for capture with {capture.line_count} lines.",
            }

        if cursor is None:
            line_number = start
            character_offset = 0
            utf8_byte_offset = 0
        else:
            cursor_data = self._decode_slice_cursor(
                cursor,
                capture_id=capture.capture_id,
                start_line=start,
                end_line=end,
            )
            if cursor_data is None:
                return {
                    "status": "error",
                    "error_code": "invalid_cursor",
                    "message": "The continuation cursor is invalid or belongs to another capture range.",
                }
            line_number = cursor_data["line"]
            character_offset = cursor_data["character_offset"]
            utf8_byte_offset = cursor_data["utf8_byte_offset"]
            if (
                line_number < start
                or line_number > end
                or character_offset < 0
                or utf8_byte_offset < 0
                or character_offset > len(capture.raw_lines[line_number - 1])
            ):
                return {
                    "status": "error",
                    "error_code": "invalid_cursor",
                    "message": "The continuation cursor points outside the requested capture range.",
                }

        segments: List[Dict[str, Any]] = []
        used_bytes = 0
        has_more = False
        while line_number <= end:
            raw = capture.raw_lines[line_number - 1]
            starts_at = character_offset
            remaining_bytes = max_content_bytes - used_bytes
            needs_separator = line_number < end
            # Reserve the line separator before consuming the last character of a
            # non-final line so concatenated page content exactly recreates the
            # retained newline-joined representation.
            text_budget = min(max_segment_bytes, remaining_bytes)
            if needs_separator:
                text_budget -= 1

            text_chars: List[str] = []
            text_bytes = 0
            next_character_offset = starts_at
            while next_character_offset < len(raw):
                char = raw[next_character_offset]
                char_bytes = len(char.encode("utf-8"))
                if text_bytes + char_bytes > text_budget:
                    break
                text_chars.append(char)
                text_bytes += char_bytes
                next_character_offset += 1

            line_complete = next_character_offset == len(raw)
            made_progress = next_character_offset > starts_at or line_complete
            if not made_progress:
                has_more = True
                if not segments:
                    return {
                        "status": "error",
                        "error_code": "response_budget_too_small",
                        "message": "The byte budget is too small to return the next Unicode character and its line metadata.",
                    }
                break

            separator_after = line_complete and needs_separator
            piece = "".join(text_chars) + ("\n" if separator_after else "")
            next_byte_offset = utf8_byte_offset + text_bytes
            if line_complete and needs_separator:
                next_line = line_number + 1
                next_character_offset = 0
                next_byte_offset = 0
            elif line_complete:
                next_line = end + 1
                next_character_offset = 0
                next_byte_offset = 0
            else:
                next_line = line_number

            cursor_after = None
            if next_line <= end:
                cursor_after = self._encode_slice_cursor(
                    capture_id=capture.capture_id,
                    start_line=start,
                    end_line=end,
                    line=next_line,
                    character_offset=next_character_offset,
                    utf8_byte_offset=next_byte_offset,
                )

            segments.append({
                "line": line_number,
                "text": "".join(text_chars),
                "byte_offset": utf8_byte_offset,
                "byte_length": text_bytes,
                "line_complete": line_complete,
                "separator_after": separator_after,
                "content_piece": piece,
                "cursor_after": cursor_after,
            })
            used_bytes += text_bytes + (1 if separator_after else 0)
            if not line_complete:
                has_more = True
                line_number = next_line
                character_offset = next_character_offset
                utf8_byte_offset = next_byte_offset
                if used_bytes >= max_content_bytes:
                    break
                continue
            if line_complete and needs_separator and used_bytes >= max_content_bytes:
                has_more = True
                break
            line_number = next_line
            character_offset = next_character_offset
            utf8_byte_offset = next_byte_offset

        self.metrics.record_retrieval(capture.capture_id)
        last_segment = segments[-1]
        next_cursor = last_segment["cursor_after"] if has_more else None
        content = "".join(segment["content_piece"] for segment in segments)
        return {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "requested_start_line": start,
            "requested_end_line": end,
            "start_line": segments[0]["line"],
            "end_line": segments[-1]["line"],
            "total_lines": capture.line_count,
            "content": content,
            "content_bytes": used_bytes,
            "segments": segments,
            "truncated": next_cursor is not None,
            "next_cursor": next_cursor,
        }

    def get_slice(
        self,
        start_line: int,
        end_line: int,
        capture_id: str = "latest",
        max_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a readable, byte-bounded page of numbered capture lines."""
        page = self.get_slice_page(
            start_line,
            end_line,
            capture_id=capture_id,
            max_content_bytes=max_bytes,
            cursor=cursor,
        )
        if page.get("status") != "ok":
            return page

        lines_with_numbers = []
        current_line = None
        current_text: List[str] = []
        current_complete = False
        for segment in page["segments"]:
            if current_line is not None and segment["line"] != current_line:
                suffix = "" if current_complete else " [continued]"
                lines_with_numbers.append(
                    f"  {current_line:5d} | {''.join(current_text)}{suffix}"
                )
                current_text = []
            current_line = segment["line"]
            current_text.append(segment["text"])
            current_complete = segment["line_complete"]
        if current_line is not None:
            suffix = "" if current_complete else " [continued]"
            lines_with_numbers.append(
                f"  {current_line:5d} | {''.join(current_text)}{suffix}"
            )
        content = "\n".join(lines_with_numbers)
        if page["truncated"]:
            content += "\n... [slice truncated; continue with next_cursor]"
        return {**page, "content": content, "raw_content": page["content"]}

    def _build_summary(self, capture: _CaptureState, include_previews: bool = True) -> Dict[str, Any]:
        """Build a summary from a captured object without looking it up by ID."""
        signals, signals_str = detect_signals(
            capture.raw_lines,
            capture.content_type,
            capture.diff_meta,
            capture.command_exit_code,
            capture.timed_out,
        )

        file_map_str = ""
        diff_stats_str = ""
        if capture.content_type == "diff" and capture.diff_meta:
            meta = capture.diff_meta
            diff_stats_str = f"{meta['total_files']} file(s) changed, +{meta['total_additions']}, -{meta['total_deletions']}"
            files_lines = []
            for f in meta["files"]:
                status_tag = f" [{f['status'].upper()}]" if f["status"] != "modified" else ""
                files_lines.append(
                    f"  - {f['path']}{status_tag} (+{f['additions']}, -{f['deletions']}) | Buffer Lines: L{f['start_line']}-L{f['end_line']}"
                )
            file_map_str = "\n".join(files_lines)

        summary = {
            "status": "ok",
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "capture_id": capture.capture_id,
            "label": capture.label,
            "source": capture.source,
            "content_type": capture.content_type,
            "diff_stats": diff_stats_str,
            "file_map": file_map_str,
            "diff_meta": capture.diff_meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(capture.timestamp)),
            "total_lines": capture.line_count,
            "byte_size": capture.byte_size,
            "truncated": capture.truncated,
            "original_byte_size": capture.original_byte_size,
            "command_exit_code": capture.command_exit_code,
            "timed_out": capture.timed_out,
            "execution_status": _capture_execution_status(capture),
            "partial": capture.timed_out,
            "duration_ms": capture.duration_ms,
            "estimated_tokens": estimate_tokens_from_bytes(capture.byte_size),
            "original_estimated_tokens": (
                estimate_tokens_from_bytes(capture.original_byte_size)
                if capture.original_byte_size is not None else None
            ),
            "keyword_signals": signals,
            "signals_summary": signals_str,
            "errors": _signal_details(signals, {"error", "exception", "failure", "timeout", "conflicts"}),
            "warnings": _signal_details(signals, {"warning"}),
            "structured_metrics": dict(capture.structured_metrics),
        }
        if include_previews:
            summary["head_preview"] = _bounded_preview(
                "\n".join(f"  {i+1:5d} | {line}" for i, line in enumerate(capture.raw_lines[:5]))
            )
            summary["tail_preview"] = _bounded_preview(
                "\n".join(
                    f"  {capture.line_count - len(capture.raw_lines[-5:]) + i + 1:5d} | {line}"
                    for i, line in enumerate(capture.raw_lines[-5:])
                )
            )
        return summary

    def get_summary(self, capture_id: str = "latest", include_previews: bool = True) -> Dict[str, Any]:
        """Generate a quick diagnostic summary for an active capture."""
        try:
            with self._lock:
                capture = self._get_capture_state(capture_id)
                # Summary inputs are immutable after ingestion. Keep the state
                # alive across concurrent eviction, then scan its raw lines
                # after releasing the engine-wide lock.
                capture_snapshot = copy.copy(capture) if capture is not None else None
            if capture_snapshot is None:
                return {
                    "status": "error",
                    "error_code": "capture_not_found",
                    "message": f"Capture '{capture_id}' not found.",
                }
            return self._build_summary(capture_snapshot, include_previews=include_previews)
        finally:
            self._flush_metrics_snapshot()

    def get_summary_for_capture(
        self,
        capture: Any,
        include_previews: bool = True,
    ) -> Dict[str, Any]:
        """Build a summary from a capture or its read-only metadata view.

        Internal callers may pass a private state snapshot. A ``CaptureView``
        is resolved through the engine while active and never exposes backing
        storage. Use ``ingest_with_summary`` when an eviction-safe summary is
        required as part of ingestion.
        """
        if isinstance(capture, CaptureView):
            state = self._get_capture_state(capture.capture_id)
            if state is None:
                return {
                    "status": "error",
                    "error_code": "capture_not_found",
                    "message": f"Capture '{capture.capture_id}' not found.",
                }
            capture = state
        return self._build_summary(capture, include_previews=include_previews)

    @synchronized
    def list_captures(self) -> List[Dict[str, Any]]:
        """
        Lists all active captures in the ring buffer.
        """
        result = []
        for cid in reversed(self.capture_order):
            cap = self._captures[cid]
            result.append({
                "capture_id": cap.capture_id,
                "label": cap.label,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cap.timestamp)),
                "total_lines": cap.line_count,
                "byte_size": cap.byte_size
            })
        return result

    def consolidate(
        self,
        capture_ids: Optional[List[str]] = None,
        max_captures: int = 25,
        max_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a bounded, searchable JSON view over active captures."""
        if max_captures < 1:
            raise ValueError("max_captures must be at least 1")
        output_limit = self.max_buffer_bytes if max_bytes is None else max_bytes
        if output_limit < 512:
            raise ValueError(
                f"effective output limit ({output_limit}) must be at least 512 bytes"
            )
        if output_limit > self.max_buffer_bytes:
            raise ValueError(
                f"max_bytes ({output_limit:,}) exceeds the configured buffer limit "
                f"({self.max_buffer_bytes:,})"
            )

        # Capture content and chunks are immutable after ingestion. Snapshot
        # their metadata and lines under the lock, then serialize outside it.
        with self._lock:
            requested_ids = list(capture_ids) if capture_ids else list(reversed(self.capture_order))
            selected_ids = list(dict.fromkeys(requested_ids))[:max_captures]
            sources: List[Dict[str, Any]] = []
            records: List[Dict[str, Any]] = []
            missing_ids: List[str] = []
            for capture_id in selected_ids:
                capture = self._captures.get(capture_id)
                if not capture:
                    missing_ids.append(capture_id)
                    continue
                sources.append({
                    "capture_id": capture.capture_id,
                    "label": capture.label,
                    "content_type": capture.content_type,
                    "total_lines": capture.line_count,
                    "byte_size": capture.byte_size,
                    "truncated": capture.truncated,
                    "original_byte_size": capture.original_byte_size,
                    "command_exit_code": capture.command_exit_code,
                    "timed_out": capture.timed_out,
                })
                records.extend(
                    {"capture_id": capture.capture_id, "source_line": line_number, "text": line}
                    for line_number, line in enumerate(capture.raw_lines, start=1)
                )

        metadata_before_records = json.dumps(
            {"schema_version": 1, "sources": sources},
            ensure_ascii=False,
            separators=(",", ":"),
        )[:-1] + ',"records":['

        def compact_suffix(omitted_count: int, record_count: int) -> str:
            metadata_after_records = {
                "requested_capture_count": len(requested_ids),
                "selected_capture_count": len(selected_ids),
                "missing_capture_ids": missing_ids,
                "omitted_record_count": omitted_count,
            }
            closing = "\n]," if record_count else "],"
            return closing + json.dumps(
                metadata_after_records,
                ensure_ascii=False,
                separators=(",", ":"),
            )[1:]

        encoded_records = [
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            for record in records
        ]
        metadata_bytes = len(metadata_before_records.encode("utf-8"))
        record_bytes = 0
        included_count = 0
        for index, encoded_record in enumerate(encoded_records):
            candidate_record_bytes = record_bytes + len(encoded_record.encode("utf-8"))
            if index:
                candidate_record_bytes += 2  # comma and newline between records
            candidate_bytes = (
                metadata_bytes
                + candidate_record_bytes
                + len(compact_suffix(len(records) - index - 1, index + 1).encode("utf-8"))
            )
            if candidate_bytes > output_limit:
                break
            record_bytes = candidate_record_bytes
            included_count = index + 1

        omitted_count = len(records) - included_count
        encoded = (
            metadata_before_records
            + ",\n".join(encoded_records[:included_count])
            + compact_suffix(omitted_count, included_count)
        )
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "sources": sources,
            "records": records[:included_count],
            "requested_capture_count": len(requested_ids),
            "selected_capture_count": len(selected_ids),
            "missing_capture_ids": missing_ids,
            "omitted_record_count": omitted_count,
        }
        pretty_encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(pretty_encoded.encode("utf-8")) <= output_limit:
            encoded = pretty_encoded
        if len(encoded.encode("utf-8")) > output_limit:
            # Source metadata can be arbitrarily large (for example, a user
            # supplied label). Keep the consolidated capture valid and bounded
            # rather than returning content that the caller cannot ingest.
            payload = {
                "schema_version": 1,
                "records": [],
                "omitted_record_count": len(records),
                "metadata_omitted": True,
            }
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return {
            "content": encoded,
            "source_capture_ids": [item["capture_id"] for item in sources],
            "source_count": len(sources),
            "record_count": len(payload["records"]),
            "omitted_record_count": payload["omitted_record_count"],
            "missing_capture_ids": missing_ids,
            "requested_capture_count": len(requested_ids),
            "selected_capture_count": len(selected_ids),
        }

    @synchronized
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Returns aggregate capture and memory-accounting metrics."""
        total_lines = sum(cap.line_count for cap in self._captures.values())
        total_chunks = sum(len(cap.chunks) for cap in self._captures.values())
        total_semantic_chunks = sum(len(cap.semantic_chunks) for cap in self._captures.values())
        embedding_bytes = sum(
            int(cap.embeddings.nbytes) for cap in self._captures.values()
            if cap.embeddings is not None
        )
        accounted_bytes = self._total_bytes + self._deferred_storage_bytes + embedding_bytes
        rss_bytes = process_rss_bytes()
        admission = admission_snapshot()
        return {
            "capture_count": len(self._captures),
            "max_captures": self.max_captures,
            "total_lines": total_lines,
            "total_chunks": total_chunks,
            "total_semantic_chunks": total_semantic_chunks,
            "semantic_chunk_lines": self.semantic_chunk_lines,
            "semantic_chunk_bytes": self.semantic_chunk_bytes,
            "semantic_chunk_overlap": self.semantic_chunk_overlap,
            "indexed_chunks": self._indexed_chunks,
            "max_indexed_chunks": self.max_indexed_chunks,
            "last_index_budget_adjustment": dict(self._last_index_budget_adjustment),
            "remaining_indexed_chunks": self.max_indexed_chunks - self._indexed_chunks,
            "total_bytes": self._total_bytes,
            "max_buffer_bytes": self.max_buffer_bytes,
            "deferred_storage_capture_count": self._deferred_storage_capture_count,
            "deferred_storage_readers": self._deferred_storage_readers,
            "deferred_storage_bytes": self._deferred_storage_bytes,
            "embedding_bytes": embedding_bytes,
            "embedding_model": self.embedding_model_name,
            "embedding_model_loaded": self.embedding_model is not None,
            "embedding_threads": self.embedding_threads,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_batch_tokens": self.embedding_max_batch_tokens,
            "embedding_cpu_mem_arena_enabled": self.embedding_cpu_mem_arena_enabled,
            "semantic_max_index_input_bytes": self.semantic_max_index_input_bytes,
            "embedding_warmup_enabled": self.embedding_warmup_enabled,
            "embedding_warmup_state": self.embedding_warmup_state,
            "embedding_warmup_failure": self.embedding_warmup_failure,
            "lexical_backend": self.lexical_backend,
            "embedding_cache_dir": self.embedding_cache_path,
            "semantic_prefetch_enabled": self.semantic_prefetch_enabled,
            "semantic_prefetch_workers": self.semantic_prefetch_workers,
            "semantic_prefetch_pending": sum(
                1 for cap in self._captures.values() if cap.semantic_index_state == "pending"
            ),
            "semantic_prefetch_queued": len(self._prefetch_queue),
            "semantic_prefetch_running": len(self._prefetch_running),
            "semantic_index_on_demand_running": sum(
                1 for job in self._on_demand_jobs.values() if job.future.running()
            ),
            "semantic_index_on_demand_queued": sum(
                1 for job in self._on_demand_jobs.values()
                if not job.future.running() and not job.future.done()
            ),
            "semantic_wait_seconds": self.semantic_wait_seconds,
            "semantic_prefetch_failed": sum(
                1 for cap in self._captures.values() if cap.semantic_index_state == "failed"
            ),
            "semantic_index_budget_exceeded": sum(
                1
                for cap in self._captures.values()
                if cap.semantic_index_state == "budget-exceeded"
            ),
            "accounted_bytes": accounted_bytes,
            **admission,
            "process_rss_bytes": rss_bytes,
            "unaccounted_rss_bytes": (
                max(0, rss_bytes - accounted_bytes) if rss_bytes is not None else None
            ),
        }

    @synchronized
    def clear(self, capture_id: str = "all") -> str:
        """
        Clears one or all captures from the buffer.
        """
        if capture_id == "all":
            capture_ids = list(self._captures)
            for cap in self._captures.values():
                self._mark_semantic_job_disposition_locked(cap.capture_id, "cleared")
                self._cancel_prefetch(cap.capture_id, outcome="cleared")
                self._cancel_on_demand_job(cap.capture_id, outcome="cleared")
                cap.semantic_index_state = "evicted"
                self._close_capture_storage(cap)
            self._captures.clear()
            self.capture_order.clear()
            self._total_bytes = 0
            self._indexed_chunks = 0
            self.metrics.record_event("cleanups")
            for current_id in capture_ids:
                self.metrics.forget_capture(current_id)
            return "Cleared all captures from ephemeral buffer."
        elif capture_id in self._captures:
            cap = self._captures.pop(capture_id)
            self._mark_semantic_job_disposition_locked(capture_id, "cleared")
            self._cancel_prefetch(capture_id, outcome="cleared")
            self._cancel_on_demand_job(capture_id, outcome="cleared")
            cap.semantic_index_state = "evicted"
            self._total_bytes -= cap.retained_byte_size
            self._indexed_chunks -= len(cap.chunks)
            self._close_capture_storage(cap)
            self.capture_order.pop(capture_id, None)
            self.metrics.record_event("cleanups")
            self.metrics.forget_capture(capture_id)
            return f"Cleared capture '{capture_id}'."
        else:
            return f"Capture '{capture_id}' not found."
