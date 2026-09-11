"""
Core search and indexing engine for ephemeral command output buffer.
Provides hybrid search (BM25 lexical + dense semantic embeddings) with RRF ranking.
"""

import os
import sys
import time
import logging
import re
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
from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    embedding_cache_dir,
    embedding_model_name as configured_embedding_model_name,
    max_indexed_chunks as configured_max_indexed_chunks,
    semantic_prefetch_enabled as configured_semantic_prefetch_enabled,
    semantic_prefetch_workers as configured_semantic_prefetch_workers,
)


LOGGER = get_logger("engine")
SEARCH_MODES = ("hybrid", "bm25", "semantic")
HYBRID_LEXICAL_WEIGHT = 2.0
PREVIEW_MAX_BYTES = 4 * 1024
PREVIEW_TRUNCATION_MARKER = "\n... [preview truncated; use get_capture_slice for full content] ..."
SEARCH_SNIPPET_MAX_BYTES = 8 * 1024
SEARCH_SNIPPET_TRUNCATION_MARKER = "... [search line truncated; use get_capture_slice for full content] ..."


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

DIFF_GIT_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
BENIGN_SIGNAL_RE = re.compile(
    r"(\b0\s*(errors?|failures?|failed)\b|\b(errors?|failures?|failed)\s*[:=]\s*0\b|\bno\s+errors?\b)",
    re.IGNORECASE
)
LOG_SIGNAL_PATTERNS = {
    "error": re.compile(r"\b(ERROR|FATAL|PANIC|CRITICAL)\b", re.IGNORECASE),
    "exception": re.compile(r"\b(EXCEPTION|TRACEBACK)\b", re.IGNORECASE),
    "failure": re.compile(r"\b(FAILED|FAILURES?)\b", re.IGNORECASE),
    "timeout": re.compile(r"\b(TIMED\s*OUT|TIMEOUT)\b", re.IGNORECASE),
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

        m = DIFF_GIT_RE.match(line)
        if m:
            if current_file:
                current_file["end_line"] = idx - 1
                files.append(current_file)
            old_p, new_p = m.group(1), m.group(2)
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
            path = line[4:].strip()
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
        return ({}, "None (successful test run)")

    detected = {}
    for name, pat in LOG_SIGNAL_PATTERNS.items():
        hits = 0
        for line in lines:
            if BENIGN_SIGNAL_RE.search(line):
                continue
            if pat.search(line):
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

    @property
    def line_count(self) -> int:
        return len(self.raw_lines)

    @property
    def byte_size(self) -> int:
        return self.input_byte_size


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
        metrics: Optional[LocalMetrics] = None,
        semantic_prefetch: Optional[bool] = None,
        semantic_prefetch_workers: Optional[int] = None,
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
        self._next_id = 1
        self.lexical_backend = "fts5" if sqlite_fts5_available() else "python-fallback"
        
        self.embedding_model_name = embedding_model_name or configured_embedding_model_name()
        self.embedding_cache_path = embedding_cache_path or embedding_cache_dir()
        self.embedding_model = None
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
        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=self.semantic_prefetch_workers,
                thread_name_prefix="semantic-prefetch",
            )
            if self.semantic_prefetch_enabled
            else None
        )
        self._prefetch_futures: Dict[str, Future[None]] = {}
        self._prefetch_slots = threading.BoundedSemaphore(self.semantic_prefetch_workers * 2)
        self._shutdown = False

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
    ) -> Capture:
        """
        Ingests text, chunks it, and builds the SQLite FTS5 BM25 index.
        FastEmbed dense vector embeddings are materialized lazily when semantic
        or hybrid search first needs them. Automatically classifies content
        type (diff, log, text) and extracts structural metadata.
        """
        lines = text.splitlines()
        with self._lock:
            capture_number = self._next_id
            capture_id = f"cap_{capture_number}"
            self._next_id += 1

        if not label:
            label = f"Capture #{capture_number}"

        capture_bytes = len(text.encode("utf-8"))
        with self._lock:
            max_buffer_bytes = self.max_buffer_bytes
        if capture_bytes > max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "capture_rejected_limit",
                capture_id=capture_id,
                capture_bytes=capture_bytes,
                max_buffer_bytes=max_buffer_bytes,
            )
            raise ValueError(
                f"Capture is {capture_bytes:,} bytes, exceeding the {max_buffer_bytes:,}-byte buffer limit"
            )

        classified_type, diff_meta = detect_content_type(lines, label=label, content_type_hint=content_type)
        required_chunks = self._chunk_count(len(lines))
        if required_chunks > self.max_indexed_chunks:
            raise ValueError(
                f"Capture requires {required_chunks:,} indexed chunks, exceeding the "
                f"{self.max_indexed_chunks:,}-chunk index budget"
            )
        chunks = self._chunk_lines(lines)

        # Semantic embeddings are materialized lazily by the first semantic or
        # hybrid search. Ingestion remains useful for fast BM25 search without
        # paying the model/indexing cost when semantic ranking is unnecessary.
        embeddings = np.empty((0, 384), dtype=np.float32) if not chunks else None

        capture = Capture(
            capture_id=capture_id,
            label=label,
            timestamp=time.time(),
            raw_lines=lines,
            input_byte_size=capture_bytes,
            chunks=chunks,
            embeddings=embeddings,
            fts_conn=None,
            content_type=classified_type,
            diff_meta=diff_meta,
            truncated=truncated,
            original_byte_size=original_byte_size,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
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
            projected_bytes = self._total_bytes + capture.byte_size
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
                projected_bytes -= old_cap.byte_size
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
                self.capture_order.pop(evicted_id, None)
                if evicted_id in self.captures:
                    old_cap = self.captures.pop(evicted_id)
                    self._total_bytes -= old_cap.byte_size
                    self._indexed_chunks -= len(old_cap.chunks)
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "capture_evicted",
                        capture_id=evicted_id,
                        capture_bytes=old_cap.byte_size,
                    )
                    old_cap.semantic_index_state = "evicted"
                    self._cancel_prefetch(evicted_id)
                    self._close_capture_storage(old_cap)
                    self.metrics.record_event("evictions")
                    self.metrics.forget_capture(evicted_id)

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

            self.captures[capture_id] = capture
            self.capture_order[capture_id] = None
            self._total_bytes += capture.byte_size
            self._indexed_chunks += len(capture.chunks)
            self.metrics.record_capture(capture_id)
        self._schedule_semantic_prefetch(capture)
        return capture

    def _schedule_semantic_prefetch(self, capture: Capture) -> None:
        """Submit at most a bounded number of post-ingestion indexing jobs."""
        if not self.semantic_prefetch_enabled or not capture.chunks:
            return
        with self._lock:
            if self._shutdown or capture.capture_id not in self.captures:
                return
            if capture.capture_id in self._prefetch_futures or capture.embeddings is not None:
                return
            if not self._prefetch_slots.acquire(blocking=False):
                return
            capture.semantic_index_state = "pending"
            try:
                future = self._prefetch_executor.submit(self._prefetch_capture, capture)
            except Exception:
                self._prefetch_slots.release()
                capture.semantic_index_state = "failed"
                log_event(LOGGER, logging.ERROR, "semantic_prefetch_submit_failed", capture_id=capture.capture_id)
                LOGGER.exception("semantic_prefetch_submit_exception")
                return
            self._prefetch_futures[capture.capture_id] = future
            future.add_done_callback(
                lambda completed, capture_id=capture.capture_id: self._prefetch_finished(capture_id, completed)
            )

    def _prefetch_capture(self, capture: Capture) -> None:
        """Build one capture's semantic index in a background worker."""
        try:
            self._ensure_embeddings(capture)
            capture.semantic_index_state = "ready"
        except Exception:
            capture.semantic_index_state = "failed"
            log_event(LOGGER, logging.ERROR, "semantic_prefetch_failed", capture_id=capture.capture_id)
            LOGGER.exception("semantic_prefetch_exception")
            raise

    def _prefetch_finished(self, capture_id: str, future: Future[None]) -> None:
        """Release bounded worker capacity and retain a content-free state."""
        with self._lock:
            self._prefetch_futures.pop(capture_id, None)
            capture = self.captures.get(capture_id)
            if capture and future.cancelled():
                capture.semantic_index_state = "not-requested"
            elif capture and future.exception() is not None:
                capture.semantic_index_state = "failed"
            elif capture and capture.embeddings is not None:
                capture.semantic_index_state = "ready"
            self._prefetch_slots.release()

    def _wait_for_prefetch(self, capture: Capture) -> None:
        """Wait for a relevant prefetch, leaving lazy indexing as fallback."""
        with self._lock:
            future = self._prefetch_futures.get(capture.capture_id)
        if future is not None:
            try:
                future.result()
            except Exception:
                # The synchronous path below retries failed work so search remains
                # correct even when a background model operation fails.
                pass

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
        """Cancel queued work for a capture; running work is allowed to finish."""
        future = self._prefetch_futures.get(capture_id)
        if future is not None:
            future.cancel()

    def shutdown(self) -> None:
        """Stop background indexing without holding the engine lock while waiting."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            executor = self._prefetch_executor
            # Future cancellation can synchronously run _prefetch_finished,
            # which removes the future from this mapping.
            for future in list(self._prefetch_futures.values()):
                future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

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
        Dense vector cosine similarity search. Returns list of (chunk_id, score).
        """
        if not capture.chunks:
            return []

        self._wait_for_prefetch(capture)
        self._ensure_embeddings(capture)
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
            chunk_texts = [chunk.text for chunk in capture.chunks]
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
            return {
                "status": "ok",
                "capture_id": capture.capture_id,
                "label": capture.label,
                "matches": [],
                "message": "Capture is empty (0 lines)."
            }

        bm25_results = []
        semantic_results = []

        if mode in ("bm25", "hybrid"):
            bm25_results = self.search_bm25(capture, query, top_k=top_k * 3)
            
        if mode in ("semantic", "hybrid"):
            semantic_results = self.search_semantic(capture, query, top_k=top_k * 3)

        rrf_scores: Dict[int, float] = {}
        k_const = 60.0

        if mode == "bm25":
            for rank, (cid, score) in enumerate(bm25_results):
                rrf_scores[cid] = score
        elif mode == "semantic":
            for rank, (cid, score) in enumerate(semantic_results):
                rrf_scores[cid] = score
        else: # hybrid
            for rank, (cid, _) in enumerate(bm25_results):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + HYBRID_LEXICAL_WEIGHT / (k_const + rank + 1)
            for rank, (cid, _) in enumerate(semantic_results):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_const + rank + 1)

        # Deduplicate overlapping context windows before applying top_k.  The
        # search backends intentionally over-fetch candidates so a sliding
        # window cannot consume the result quota with duplicate context.
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        matches = []
        seen_line_ranges = []

        for cid, score in sorted_chunks:
            chunk = capture.chunks[cid]
            ctx_start = max(1, chunk.start_line - context_lines)
            ctx_end = min(capture.line_count, chunk.end_line + context_lines)
            
            overlap = False
            for prev_s, prev_e in seen_line_ranges:
                if not (ctx_end < prev_s or ctx_start > prev_e):
                    overlap = True
                    break
            if overlap:
                continue
                
            seen_line_ranges.append((ctx_start, ctx_end))

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
                "chunk_id": cid,
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
        return {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "total_lines": capture.line_count,
            "mode": mode,
            "query": query,
            "match_count": len(matches),
            "matches": matches
        }

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

    @synchronized
    def get_summary(self, capture_id: str = "latest") -> Dict[str, Any]:
        """
        Generates a quick diagnostic summary of the capture.
        """
        capture = self.get_capture(capture_id)
        if not capture:
            return {"status": "error", "message": f"Capture '{capture_id}' not found."}

        signals, signals_str = detect_signals(
            capture.raw_lines,
            capture.content_type,
            capture.diff_meta,
            capture.command_exit_code,
            capture.timed_out,
        )

        head_preview = _bounded_preview(
            "\n".join(f"  {i+1:5d} | {line}" for i, line in enumerate(capture.raw_lines[:5]))
        )
        tail_preview = _bounded_preview(
            "\n".join(
                f"  {capture.line_count - len(capture.raw_lines[-5:]) + i + 1:5d} | {line}"
                for i, line in enumerate(capture.raw_lines[-5:])
            )
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

        return {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
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
            "keyword_signals": signals,
            "signals_summary": signals_str,
            "head_preview": head_preview,
            "tail_preview": tail_preview
        }

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
            "indexed_chunks": self._indexed_chunks,
            "max_indexed_chunks": self.max_indexed_chunks,
            "remaining_indexed_chunks": self.max_indexed_chunks - self._indexed_chunks,
            "total_bytes": self._total_bytes,
            "max_buffer_bytes": self.max_buffer_bytes,
            "embedding_bytes": embedding_bytes,
            "embedding_model": self.embedding_model_name,
            "embedding_model_loaded": self.embedding_model is not None,
            "lexical_backend": self.lexical_backend,
            "embedding_cache_dir": self.embedding_cache_path,
            "semantic_prefetch_enabled": self.semantic_prefetch_enabled,
            "semantic_prefetch_workers": self.semantic_prefetch_workers,
            "semantic_prefetch_pending": sum(
                1 for cap in self.captures.values() if cap.semantic_index_state == "pending"
            ),
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
            cap.semantic_index_state = "evicted"
            self._total_bytes -= cap.byte_size
            self._indexed_chunks -= len(cap.chunks)
            self._close_capture_storage(cap)
            self.capture_order.pop(capture_id, None)
            self.metrics.record_event("cleanups")
            self.metrics.forget_capture(capture_id)
            return f"Cleared capture '{capture_id}'."
        else:
            return f"Capture '{capture_id}' not found."
