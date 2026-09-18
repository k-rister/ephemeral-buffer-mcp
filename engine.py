"""
Core search and indexing engine for ephemeral command output buffer.
Provides hybrid search (BM25 lexical + dense semantic embeddings) with RRF ranking.
"""

import os
import sys
import time
import logging
import re
import math
import hashlib
import sqlite3
import threading
import json
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter, OrderedDict
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
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
    embedding_cache_dir,
    embedding_model_name as configured_embedding_model_name,
    embedding_threads as configured_embedding_threads,
    embedding_warmup_enabled as configured_embedding_warmup_enabled,
    max_indexed_chunks as configured_max_indexed_chunks,
    semantic_prefetch_enabled as configured_semantic_prefetch_enabled,
    semantic_prefetch_workers as configured_semantic_prefetch_workers,
    semantic_wait_seconds as configured_semantic_wait_seconds,
    semantic_chunk_lines as configured_semantic_chunk_lines,
    semantic_chunk_bytes as configured_semantic_chunk_bytes,
    semantic_chunk_overlap as configured_semantic_chunk_overlap,
)


LOGGER = get_logger("engine")
SEARCH_MODES = ("hybrid", "bm25", "semantic")
HYBRID_LEXICAL_WEIGHT = 2.0
PREVIEW_MAX_BYTES = 4 * 1024
PREVIEW_TRUNCATION_MARKER = "\n... [preview truncated; use get_capture_slice for full content] ..."
SEARCH_SNIPPET_MAX_BYTES = 8 * 1024
SEARCH_SNIPPET_TRUNCATION_MARKER = "... [search line truncated; use get_capture_slice for full content] ..."
SUMMARY_SCHEMA_VERSION = 1
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 4
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
    """Return a UTF-8 bounded preview with an explicit truncation marker."""
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content

    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= max_bytes:
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore")

    prefix = encoded[: max_bytes - len(marker_bytes)].decode("utf-8", errors="ignore")
    return prefix + marker


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


def _capture_execution_status(capture: "Capture") -> str:
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
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

DIFF_GIT_RE = re.compile(r"^diff --git (.+)$")
DIFF_PATH_TOKEN_RE = re.compile(r'"(?:\\.|[^"])*"|[^\s]+')
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
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


def _parse_git_diff_paths(line: str) -> Optional[Tuple[str, str]]:
    """Return decoded old/new paths from a ``diff --git`` header."""
    match = DIFF_GIT_RE.match(line)
    if not match:
        return None
    tokens = DIFF_PATH_TOKEN_RE.findall(match.group(1))
    if len(tokens) != 2:
        return None
    return _decode_git_path(tokens[0]), _decode_git_path(tokens[1])


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
    has_conflicts = False

    for idx, line in enumerate(lines, start=1):
        if _contains_conflict_marker(line):
            has_conflicts = True

        git_paths = _parse_git_diff_paths(line)
        if git_paths:
            if current_file:
                current_file["end_line"] = idx - 1
                files.append(current_file)
            old_p, new_p = git_paths
            if old_p.startswith("a/"):
                old_p = old_p[2:]
            if new_p.startswith("b/"):
                new_p = new_p[2:]
            path = new_p if new_p != "/dev/null" else old_p
            current_file = {
                "path": path,
                "old_path": old_p,
                "new_path": new_p,
                "status": "modified",
                "start_line": idx,
                "end_line": len(lines),
                "additions": 0,
                "deletions": 0,
                "hunks": 0
            }
            in_hunk = False
            continue

        if current_file is None and (line.startswith("--- ") or line.startswith("+++ ")):
            path = _decode_git_path(line[4:].strip())
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if path:
                current_file = {
                    "path": path,
                    "old_path": path,
                    "new_path": path,
                    "status": "modified",
                    "start_line": idx,
                    "end_line": len(lines),
                    "additions": 0,
                    "deletions": 0,
                    "hunks": 0
                }
                in_hunk = False

        if current_file:
            if line.startswith("new file mode"):
                current_file["status"] = "added"
            elif line.startswith("deleted file mode"):
                current_file["status"] = "deleted"
            elif line.startswith("similarity index") or line.startswith("rename from"):
                current_file["status"] = "renamed"
            elif HUNK_RE.match(line):
                current_file["hunks"] += 1
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                current_file["additions"] += 1
            elif in_hunk and line.startswith("-"):
                current_file["deletions"] += 1

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
class Capture:
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
    # Append summary metadata after the legacy fields so positional Capture
    # construction remains compatible with the pre-summary data model.
    source: str = "capture"
    duration_ms: Optional[float] = None
    structured_metrics: Dict[str, Any] = field(default_factory=dict)
    # Semantic windows are packed separately from the lexical sliding windows
    # so embedding cost is bounded by line and byte caps rather than tied to
    # the overlap BM25 uses for exact line ranges.
    semantic_chunks: List[Chunk] = field(default_factory=list)

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

    __slots__ = ("done", "error", "future", "cancelled")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self.future: Future
        self.cancelled = False


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
        max_captures: int = DEFAULT_MAX_CAPTURES,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        max_indexed_chunks: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
        embedding_warmup: Optional[bool] = None,
        embedding_threads: Optional[int] = None,
        metrics: Optional[LocalMetrics] = None,
        semantic_prefetch: Optional[bool] = None,
        semantic_prefetch_workers: Optional[int] = None,
        semantic_chunk_lines: Optional[int] = None,
        semantic_chunk_bytes: Optional[int] = None,
        semantic_chunk_overlap: Optional[int] = None,
        semantic_wait_seconds: Optional[float] = None,
    ):
        self._lock = threading.RLock()
        if max_captures < 1:
            raise ValueError("max_captures must be at least 1")
        if max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be at least 1")
        self.max_indexed_chunks = (
            configured_max_indexed_chunks() if max_indexed_chunks is None else max_indexed_chunks
        )
        if self.max_indexed_chunks < 1:
            raise ValueError("max_indexed_chunks must be at least 1")
        self.max_captures = max_captures
        self.max_buffer_bytes = max_buffer_bytes
        self._embedding_lock = threading.RLock()
        self.captures: Dict[str, Capture] = {}
        self.capture_order: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._indexed_chunks = 0
        self._last_index_budget_adjustment = {
            "status": "startup",
            "previous": self.max_indexed_chunks,
            "effective": self.max_indexed_chunks,
            "evicted_captures": 0,
        }
        self._next_id = 1
        self.lexical_backend = "fts5" if sqlite_fts5_available() else "python-fallback"
        
        self.embedding_model_name = embedding_model_name or configured_embedding_model_name()
        self.embedding_threads = (
            configured_embedding_threads() if embedding_threads is None else embedding_threads
        )
        if self.embedding_threads is not None and self.embedding_threads < 1:
            raise ValueError("embedding_threads must be at least 1")
        self.semantic_chunk_lines = (
            configured_semantic_chunk_lines() if semantic_chunk_lines is None else semantic_chunk_lines
        )
        self.semantic_chunk_bytes = (
            configured_semantic_chunk_bytes() if semantic_chunk_bytes is None else semantic_chunk_bytes
        )
        self.semantic_chunk_overlap = (
            configured_semantic_chunk_overlap()
            if semantic_chunk_overlap is None
            else semantic_chunk_overlap
        )
        if self.semantic_chunk_lines < 1:
            raise ValueError("semantic_chunk_lines must be at least 1")
        if self.semantic_chunk_bytes < 1:
            raise ValueError("semantic_chunk_bytes must be at least 1")
        if not 0 <= self.semantic_chunk_overlap < self.semantic_chunk_lines:
            raise ValueError("semantic_chunk_overlap must be non-negative and smaller than semantic_chunk_lines")
        self.embedding_cache_path = embedding_cache_path or embedding_cache_dir()
        self.embedding_model = None
        self.embedding_warmup_enabled = (
            configured_embedding_warmup_enabled()
            if embedding_warmup is None
            else embedding_warmup
        )
        self.embedding_warmup_state = (
            "not-started" if self.embedding_warmup_enabled else "disabled"
        )
        self.embedding_warmup_failure = None
        self._embedding_warmup_thread: Optional[threading.Thread] = None
        self.metrics = metrics or LocalMetrics(enabled=False)
        self.semantic_prefetch_enabled = (
            configured_semantic_prefetch_enabled() if semantic_prefetch is None else semantic_prefetch
        )
        self.semantic_prefetch_workers = (
            configured_semantic_prefetch_workers()
            if semantic_prefetch_workers is None
            else semantic_prefetch_workers
        )
        if self.semantic_prefetch_workers < 1:
            raise ValueError("semantic_prefetch_workers must be at least 1")
        self.semantic_wait_seconds = (
            configured_semantic_wait_seconds() if semantic_wait_seconds is None else semantic_wait_seconds
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
        self._prefetch_queue: "OrderedDict[str, Capture]" = OrderedDict()
        self._prefetch_running: Dict[str, threading.Event] = {}
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
    ) -> Capture:
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

        capture = Capture(
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
            missing_protected_ids = sorted(protected_ids.difference(self.captures))
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
                old_cap = self.captures[candidate_id]
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
            self.captures[capture_id] = capture
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
        old_cap = self.captures.pop(capture_id, None)
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
        self._cancel_prefetch(capture_id)
        self._cancel_on_demand_job(capture_id)
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

    def _schedule_semantic_prefetch(self, capture: Capture) -> None:
        """Queue post-ingestion indexing and make sure a bounded worker is draining."""
        if not self.semantic_prefetch_enabled or not capture.semantic_chunks:
            return
        with self._lock:
            if self._shutdown or capture.capture_id not in self.captures:
                return
            if (
                capture.embeddings is not None
                or capture.capture_id in self._prefetch_queue
                or capture.capture_id in self._prefetch_running
                or capture.capture_id in self._on_demand_jobs
            ):
                return
            self._prefetch_queue[capture.capture_id] = capture
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
                return
            self._prefetch_workers_active += 1

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
            try:
                self._ensure_embeddings(capture)
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
                    done.set()

    def _start_semantic_index(self, capture: Capture) -> Optional[Tuple[threading.Event, Optional[_SemanticIndexJob]]]:
        """Return the completion event for the job indexing ``capture``, starting one if needed.

        A running prefetch job is reused.  Otherwise the capture is pulled out
        of the prefetch queue, because a search is a stronger signal than queue
        position, and indexed on the bounded on-demand pool, where the job
        outlives the caller's wait budget.  Returns ``None`` when the
        embeddings are already ready.
        """
        with self._lock:
            if capture.embeddings is not None:
                return None
            running = self._prefetch_running.get(capture.capture_id)
            if running is not None:
                return running, None
            job = self._on_demand_jobs.get(capture.capture_id)
            if job is not None:
                return job.done, job
            self._prefetch_queue.pop(capture.capture_id, None)
            if self._shutdown:
                # No background thread may start after shutdown; the caller
                # indexes inline as the lazy path always could.
                finished = threading.Event()
                finished.set()
                return finished, None
            job = _SemanticIndexJob()
            job.future = self._on_demand_executor.submit(self._on_demand_index_worker, capture, job)
            self._on_demand_jobs[capture.capture_id] = job
            capture.semantic_index_state = "pending"
            return job.done, job

    def _on_demand_index_worker(self, capture: Capture, job: _SemanticIndexJob) -> None:
        """Materialize one capture's embeddings and publish the outcome to waiters."""
        try:
            self._ensure_embeddings(capture)
        except Exception as exc:
            job.error = exc
            capture.semantic_index_state = "failed"
            log_event(LOGGER, logging.ERROR, "semantic_index_failed", capture_id=capture.capture_id)
            LOGGER.exception("semantic_index_exception")
        finally:
            with self._lock:
                self._on_demand_jobs.pop(capture.capture_id, None)
                if capture.semantic_index_state == "pending":
                    # The capture was evicted before its embeddings were published.
                    capture.semantic_index_state = "not-requested"
                job.done.set()

    def _await_semantic_index(self, capture: Capture, timeout: Optional[float] = None) -> str:
        """Wait up to ``timeout`` seconds for the capture's semantic index.

        Returns ``"ready"`` or ``"pending"``.  ``None`` and ``inf`` wait until
        the index is ready or its job fails, in which case the job's exception
        is re-raised so callers keep the lazy-path error semantics.  A bounded
        wait never indexes inline: if the job was cancelled or the capture was
        evicted, it reports ``"pending"`` instead.
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
            retained = self.captures.get(capture.capture_id) is capture
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

    def wait_for_semantic_index(self, capture: Capture, timeout: Optional[float] = None) -> str:
        """Block until the capture's semantic index is ready, failed, or ``timeout`` elapses.

        Returns ``"ready"``, ``"pending"``, or ``"failed"``; it never raises for
        an indexing error, so callers can poll from tests and benchmarks.
        """
        try:
            return self._await_semantic_index(capture, timeout)
        except Exception:
            return "failed"

    def _close_capture_storage(self, capture: Capture) -> None:
        """Close per-capture search storage and report cleanup failures."""
        if capture.active_readers:
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

    def _acquire_capture_reader(self, capture_id: str) -> Optional[Capture]:
        """Return a capture while retaining its storage for one search reader."""
        with self._lock:
            if not self.captures:
                return None
            if capture_id == "latest" or not capture_id:
                capture = self.captures[next(reversed(self.capture_order))]
            else:
                capture = self.captures.get(capture_id)
            if capture:
                self._touch_capture(capture.capture_id)
                capture.active_readers += 1
            return capture

    def _release_capture_reader(self, capture: Capture) -> None:
        """Release a search reader and finish deferred storage cleanup."""
        with self._lock:
            capture.active_readers = max(0, capture.active_readers - 1)
            if capture.active_readers == 0 and capture.storage_close_pending:
                capture.storage_close_pending = False
                self._close_capture_storage(capture)

    def _cancel_prefetch(self, capture_id: str) -> None:
        """Drop queued prefetch work for a capture; running work is allowed to finish."""
        capture = self._prefetch_queue.pop(capture_id, None)
        if capture is not None:
            capture.semantic_index_state = "not-requested"

    def _cancel_on_demand_job(self, capture_id: str) -> None:
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
        live = self.captures.get(capture_id)
        if live is not None and live.semantic_index_state == "pending":
            live.semantic_index_state = "not-requested"
        job.done.set()

    def shutdown(self) -> None:
        """Stop background embedding work without holding the engine lock while waiting."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            executor = self._prefetch_executor
            warmup_thread = self._embedding_warmup_thread
            on_demand_executor = self._on_demand_executor
            for capture_id in list(self._prefetch_queue):
                self._cancel_prefetch(capture_id)
            for capture_id in list(self._on_demand_jobs):
                self._cancel_on_demand_job(capture_id)
        if warmup_thread is not None and warmup_thread is not threading.current_thread():
            warmup_thread.join()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        on_demand_executor.shutdown(wait=True, cancel_futures=True)

    @synchronized
    def get_capture(self, capture_id: str = "latest") -> Optional[Capture]:
        if not self.captures:
            return None
        if capture_id == "latest" or not capture_id:
            capture = self.captures[next(reversed(self.capture_order))]
        else:
            capture = self.captures.get(capture_id)
        if capture:
            self._touch_capture(capture.capture_id)
        return capture

    def _touch_capture(self, capture_id: str) -> None:
        """Marks a capture as recently used for LRU eviction."""
        if capture_id in self.capture_order:
            self.capture_order.move_to_end(capture_id)

    @synchronized
    def search_bm25(self, capture: Capture, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
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
        capture: Capture, tokens: List[str], top_k: int
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

    def search_semantic(self, capture: Capture, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
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

    def _ensure_embeddings(self, capture: Capture) -> None:
        """Materialize and cache dense embeddings for a captured chunk set."""
        with self._lock:
            if capture.embeddings is not None:
                return
            # A search reader may outlive LRU admission.  Its lease keeps the
            # capture's chunks and storage alive long enough to finish lazy
            # indexing even after the capture is removed from ``self.captures``.
            if (
                self.captures.get(capture.capture_id) is not capture
                and not getattr(capture, "active_readers", 0)
            ):
                return
            chunk_texts = [chunk.text for chunk in capture.semantic_chunks]
        with self._embedding_lock:
            with self._lock:
                if capture.embeddings is not None:
                    return
                if (
                    self.captures.get(capture.capture_id) is not capture
                    and not getattr(capture, "active_readers", 0)
                ):
                    return
            embed_list = list(self._get_embedding_model().embed(chunk_texts))
            embeddings = np.array(embed_list, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = embeddings / norms
            with self._lock:
                if (
                    (
                        self.captures.get(capture.capture_id) is capture
                        or getattr(capture, "active_readers", 0)
                    )
                    and capture.embeddings is None
                ):
                    capture.embeddings = normalized
                    capture.semantic_index_state = "ready"

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        capture_id: str = "latest",
        top_k: int = 5,
        context_lines: int = 3
    ) -> Dict[str, Any]:
        """
        Performs BM25, Semantic, or Hybrid (Reciprocal Rank Fusion) search across the capture.
        """
        if mode not in SEARCH_MODES:
            return {
                "status": "error",
                "message": f"Unsupported search mode '{mode}'. Choose one of: {', '.join(SEARCH_MODES)}.",
            }
        if top_k < 1:
            return {"status": "error", "message": "top_k must be at least 1."}
        if context_lines < 0:
            return {"status": "error", "message": "context_lines must be non-negative."}

        capture = self._acquire_capture_reader(capture_id)
        if not capture:
            return {
                "status": "error",
                "message": f"No capture found for ID '{capture_id}'. Buffer is currently empty."
            }

        try:
            return self._search_capture(capture, query, mode, top_k, context_lines)
        finally:
            self._release_capture_reader(capture)

    def _search_capture(
        self,
        capture: Capture,
        query: str,
        mode: str,
        top_k: int,
        context_lines: int,
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
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "hybrid_search_semantic_pending",
                        capture_id=capture.capture_id,
                        line_count=capture.line_count,
                        wait_seconds=self.semantic_wait_seconds,
                    )
            except Exception as exc:
                if mode == "semantic":
                    raise
                semantic_fallback = type(exc).__name__
                semantic_coverage = "unavailable"
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "hybrid_search_lexical_fallback",
                    error_type=semantic_fallback,
                )

        # Lexical and semantic hits come from different chunk grids, so fusion
        # happens in line space: each lexical window belongs to the semantic
        # window that contains its first line, a semantic hit boosts the lexical
        # hits that belong to it, and it stands on its own only when none do.
        # Exact BM25 line ranges are therefore preserved, and a lexical window
        # straddling two semantic windows cannot collect both boosts.
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
            for sid, score in semantic_results:
                add_candidate("semantic", capture.semantic_chunks[sid], score)
        else:  # hybrid
            for rank, (cid, _) in enumerate(bm25_results):
                add_candidate("lexical", capture.chunks[cid], HYBRID_LEXICAL_WEIGHT / (k_const + rank + 1))
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
                        entry["score"] += contribution
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

            seen_line_ranges.append((chunk.start_line, chunk.end_line, ctx_start, ctx_end))

            lines_with_numbers = []
            for line_no in range(ctx_start, ctx_end + 1):
                raw = capture.raw_lines[line_no - 1]
                is_match_core = chunk.start_line <= line_no <= chunk.end_line
                prefix = ">" if is_match_core else " "
                lines_with_numbers.append(_bounded_preview(
                    f"{prefix} {line_no:5d} | {raw}",
                    max_bytes=SEARCH_SNIPPET_MAX_BYTES,
                    marker=SEARCH_SNIPPET_TRUNCATION_MARKER,
                ))

            snippet = "\n".join(lines_with_numbers)
            matches.append({
                "chunk_id": chunk.chunk_id,
                "chunk_index": entry["index"],
                "score": round(score, 4),
                "matched_range": f"L{chunk.start_line}-L{chunk.end_line}",
                "context_range": f"L{ctx_start}-L{ctx_end}",
                "context_start_line": ctx_start,
                "context_end_line": ctx_end,
                "context": "\n".join(capture.raw_lines[ctx_start - 1:ctx_end]),
                "snippet": snippet
            })
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
        }
        if semantic_fallback:
            result["semantic_fallback"] = semantic_fallback
        if semantic_coverage == "pending":
            result["semantic_index_state"] = capture.semantic_index_state
            result["semantic_wait_seconds"] = self.semantic_wait_seconds
            result["message"] = (
                "Semantic index still building for this capture; results are lexical "
                "(BM25) only. Repeat the search for hybrid ranking."
            )
        return result

    @synchronized
    def get_slice(self, start_line: int, end_line: int, capture_id: str = "latest") -> Dict[str, Any]:
        """
        Retrieves an exact slice of lines from a capture.
        """
        capture = self.get_capture(capture_id)
        if not capture:
            return {"status": "error", "message": f"Capture '{capture_id}' not found."}

        self.metrics.record_retrieval(capture.capture_id)

        start = max(1, start_line)
        end = min(capture.line_count, end_line)

        if start > end or start > capture.line_count:
            return {
                "status": "error",
                "message": f"Invalid range {start_line}-{end_line} for capture with {capture.line_count} lines."
            }

        lines_with_numbers = [
            f"  {line_no:5d} | {capture.raw_lines[line_no - 1]}"
            for line_no in range(start, end + 1)
        ]

        return {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "start_line": start,
            "end_line": end,
            "total_lines": capture.line_count,
            "content": "\n".join(lines_with_numbers)
        }

    def _build_summary(self, capture: Capture, include_previews: bool = True) -> Dict[str, Any]:
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

    @synchronized
    def get_summary(self, capture_id: str = "latest", include_previews: bool = True) -> Dict[str, Any]:
        """Generate a quick diagnostic summary for an active capture."""
        capture = self.get_capture(capture_id)
        if not capture:
            return {"status": "error", "message": f"Capture '{capture_id}' not found."}
        return self._build_summary(capture, include_previews=include_previews)

    def get_summary_for_capture(
        self,
        capture: Capture,
        include_previews: bool = True,
    ) -> Dict[str, Any]:
        """Build a summary from an ingestion result even after LRU eviction.

        Ingestion callers retain the returned capture object. Reading that
        snapshot avoids a lookup race where another capture evicts it before
        the response summary is assembled.
        """
        return self._build_summary(capture, include_previews=include_previews)

    @synchronized
    def list_captures(self) -> List[Dict[str, Any]]:
        """
        Lists all active captures in the ring buffer.
        """
        result = []
        for cid in reversed(self.capture_order):
            cap = self.captures[cid]
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
            raise ValueError("max_bytes must be at least 512")
        if output_limit > self.max_buffer_bytes:
            raise ValueError(
                f"max_bytes ({output_limit:,}) exceeds the configured buffer limit "
                f"({self.max_buffer_bytes:,})"
            )

        # Capture objects are immutable after ingestion. Snapshot their metadata
        # and lines while holding the lock, then release it before doing the
        # potentially expensive serialization work.
        with self._lock:
            requested_ids = list(capture_ids) if capture_ids else list(reversed(self.capture_order))
            selected_ids = list(dict.fromkeys(requested_ids))[:max_captures]
            sources: List[Dict[str, Any]] = []
            records: List[Dict[str, Any]] = []
            missing_ids: List[str] = []
            for capture_id in selected_ids:
                capture = self.captures.get(capture_id)
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
        total_lines = sum(cap.line_count for cap in self.captures.values())
        total_chunks = sum(len(cap.chunks) for cap in self.captures.values())
        total_semantic_chunks = sum(len(cap.semantic_chunks) for cap in self.captures.values())
        embedding_bytes = sum(
            int(cap.embeddings.nbytes) for cap in self.captures.values()
            if cap.embeddings is not None
        )
        accounted_bytes = self._total_bytes + embedding_bytes
        rss_bytes = process_rss_bytes()
        return {
            "capture_count": len(self.captures),
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
            "embedding_bytes": embedding_bytes,
            "embedding_model": self.embedding_model_name,
            "embedding_model_loaded": self.embedding_model is not None,
            "embedding_threads": self.embedding_threads,
            "embedding_warmup_enabled": self.embedding_warmup_enabled,
            "embedding_warmup_state": self.embedding_warmup_state,
            "embedding_warmup_failure": self.embedding_warmup_failure,
            "lexical_backend": self.lexical_backend,
            "embedding_cache_dir": self.embedding_cache_path,
            "semantic_prefetch_enabled": self.semantic_prefetch_enabled,
            "semantic_prefetch_workers": self.semantic_prefetch_workers,
            "semantic_prefetch_pending": sum(
                1 for cap in self.captures.values() if cap.semantic_index_state == "pending"
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
                1 for cap in self.captures.values() if cap.semantic_index_state == "failed"
            ),
            "accounted_bytes": accounted_bytes,
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
            capture_ids = list(self.captures)
            for cap in self.captures.values():
                self._cancel_prefetch(cap.capture_id)
                self._cancel_on_demand_job(cap.capture_id)
                cap.semantic_index_state = "evicted"
                self._close_capture_storage(cap)
            self.captures.clear()
            self.capture_order.clear()
            self._total_bytes = 0
            self._indexed_chunks = 0
            self.metrics.record_event("cleanups")
            for current_id in capture_ids:
                self.metrics.forget_capture(current_id)
            return "Cleared all captures from ephemeral buffer."
        elif capture_id in self.captures:
            cap = self.captures.pop(capture_id)
            self._cancel_prefetch(capture_id)
            self._cancel_on_demand_job(capture_id)
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
