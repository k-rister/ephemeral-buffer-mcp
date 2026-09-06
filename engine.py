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
from collections import OrderedDict
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import wraps
from logging_utils import get_logger, log_event
from fastembed import TextEmbedding
from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    embedding_cache_dir,
    embedding_model_name as configured_embedding_model_name,
)


LOGGER = get_logger("engine")


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
    has_conflicts = False

    for idx, line in enumerate(lines, start=1):
        if line.startswith("<<<<<<<") or line.startswith("=======") or line.startswith(">>>>>>>"):
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

        if current_file:
            if line.startswith("new file mode"):
                current_file["status"] = "added"
            elif line.startswith("deleted file mode"):
                current_file["status"] = "deleted"
            elif line.startswith("similarity index") or line.startswith("rename from"):
                current_file["status"] = "renamed"
            elif HUNK_RE.match(line):
                current_file["hunks"] += 1
            elif line.startswith("+") and not line.startswith("+++"):
                current_file["additions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
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
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    fts_conn: Optional[sqlite3.Connection] = None
    content_type: str = "text"
    diff_meta: Optional[Dict[str, Any]] = None
    truncated: bool = False
    original_byte_size: Optional[int] = None
    command_exit_code: Optional[int] = None
    timed_out: bool = False

    @property
    def line_count(self) -> int:
        return len(self.raw_lines)

    @property
    def byte_size(self) -> int:
        return sum(len(line.encode("utf-8")) + 1 for line in self.raw_lines)


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
        embedding_model_name: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
    ):
        self._lock = threading.RLock()
        if max_captures < 1:
            raise ValueError("max_captures must be at least 1")
        if max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be at least 1")
        self.max_captures = max_captures
        self.max_buffer_bytes = max_buffer_bytes
        self._embedding_lock = threading.RLock()
        self.captures: Dict[str, Capture] = {}
        self.capture_order: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._next_id = 1
        
        self.embedding_model_name = embedding_model_name or configured_embedding_model_name()
        self.embedding_cache_path = embedding_cache_path or embedding_cache_dir()
        self.embedding_model = None

    def _get_embedding_model(self):
        """Load FastEmbed once, on first operation that needs embeddings."""
        if self.embedding_model is None:
            with self._embedding_lock:
                if self.embedding_model is None:
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

    def ingest(
        self,
        text: str,
        label: str = "",
        content_type: str = "auto",
        truncated: bool = False,
        original_byte_size: Optional[int] = None,
        command_exit_code: Optional[int] = None,
        timed_out: bool = False,
    ) -> Capture:
        """
        Ingests text, chunks it, builds SQLite FTS5 BM25 index and FastEmbed dense vector embeddings.
        Automatically classifies content type (diff, log, text) and extracts structural metadata.
        """
        lines = text.splitlines()
        with self._lock:
            capture_number = self._next_id
            capture_id = f"cap_{capture_number}"
            self._next_id += 1

        if not label:
            label = f"Capture #{capture_number}"

        capture_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines)
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
        chunks = self._chunk_lines(lines)
        
        # 1. Setup SQLite FTS5 in-memory DB for this capture
        # Captures may be ingested by the CLI socket listener thread and queried
        # by the MCP thread, so allow the per-capture connection to cross threads.
        fts_conn = sqlite3.connect(":memory:", check_same_thread=False)
        cur = fts_conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content, tokenize='unicode61')")
        
        for c in chunks:
            cur.execute("INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)", (c.chunk_id, c.text))
        fts_conn.commit()

        # 2. Compute dense embeddings without holding the engine state lock.
        with self._embedding_lock:
            if chunks:
                chunk_texts = [c.text for c in chunks]
                embed_list = list(self._get_embedding_model().embed(chunk_texts))
                embeddings = np.array(embed_list, dtype=np.float32)
                # Normalize embeddings for cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
            else:
                embeddings = np.empty((0, 384), dtype=np.float32)

        capture = Capture(
            capture_id=capture_id,
            label=label,
            timestamp=time.time(),
            raw_lines=lines,
            chunks=chunks,
            embeddings=embeddings,
            fts_conn=fts_conn,
            content_type=classified_type,
            diff_meta=diff_meta,
            truncated=truncated,
            original_byte_size=original_byte_size,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
        )

        with self._lock:
            # LRU eviction: capture_order is ordered from least to most recently used.
            while self.capture_order and (
                len(self.capture_order) >= self.max_captures
                or self._total_bytes + capture.byte_size > self.max_buffer_bytes
            ):
                evicted_id, _ = self.capture_order.popitem(last=False)
                if evicted_id in self.captures:
                    old_cap = self.captures.pop(evicted_id)
                    self._total_bytes -= old_cap.byte_size
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "capture_evicted",
                        capture_id=evicted_id,
                        capture_bytes=old_cap.byte_size,
                    )
                    self._close_capture_storage(old_cap)

            self.captures[capture_id] = capture
            self.capture_order[capture_id] = None
            self._total_bytes += capture.byte_size
            return capture

    def _close_capture_storage(self, capture: Capture) -> None:
        """Close per-capture search storage and report cleanup failures."""
        if not capture.fts_conn:
            return
        try:
            capture.fts_conn.close()
        except Exception:
            log_event(
                LOGGER,
                logging.ERROR,
                "capture_storage_cleanup_failed",
                capture_id=capture.capture_id,
            )
            LOGGER.exception("capture_storage_cleanup_exception")

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
        Search using SQLite FTS5 BM25. Returns list of (chunk_id, score).
        """
        if not capture.chunks or not capture.fts_conn:
            return []
            
        tokens = [t.replace('"', '""') for t in query.split() if t.isalnum() or '_' in t or '-' in t]
        if not tokens:
            return []
            
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        
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

    @synchronized
    def search_semantic(self, capture: Capture, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense vector cosine similarity search. Returns list of (chunk_id, score).
        """
        if not capture.chunks or capture.embeddings is None or len(capture.embeddings) == 0:
            return []
            
        query_embed = list(self._get_embedding_model().embed([query]))[0]
        query_embed = np.array(query_embed, dtype=np.float32)
        norm = np.linalg.norm(query_embed)
        if norm > 0:
            query_embed = query_embed / norm

        similarities = np.dot(capture.embeddings, query_embed)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.0]

    @synchronized
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
        capture = self.get_capture(capture_id)
        if not capture:
            return {
                "status": "error",
                "message": f"No capture found for ID '{capture_id}'. Buffer is currently empty."
            }

        if capture.line_count == 0:
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
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_const + rank + 1)
            for rank, (cid, _) in enumerate(semantic_results):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_const + rank + 1)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

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
                lines_with_numbers.append(f"{prefix} {line_no:5d} | {raw}")

            snippet = "\n".join(lines_with_numbers)
            matches.append({
                "chunk_id": cid,
                "score": round(score, 4),
                "matched_range": f"L{chunk.start_line}-L{chunk.end_line}",
                "context_range": f"L{ctx_start}-L{ctx_end}",
                "snippet": snippet
            })

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

        head_preview = [f"  {i+1:5d} | {line}" for i, line in enumerate(capture.raw_lines[:5])]
        tail_preview = [f"  {capture.line_count - len(capture.raw_lines[-5:]) + i + 1:5d} | {line}" for i, line in enumerate(capture.raw_lines[-5:])]

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
            "head_preview": "\n".join(head_preview),
            "tail_preview": "\n".join(tail_preview)
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
            "total_bytes": self._total_bytes,
            "max_buffer_bytes": self.max_buffer_bytes,
            "embedding_bytes": embedding_bytes,
            "embedding_model": self.embedding_model_name,
            "embedding_model_loaded": self.embedding_model is not None,
            "embedding_cache_dir": self.embedding_cache_path,
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
            for cap in self.captures.values():
                self._close_capture_storage(cap)
            self.captures.clear()
            self.capture_order.clear()
            self._total_bytes = 0
            return "Cleared all captures from ephemeral buffer."
        elif capture_id in self.captures:
            cap = self.captures.pop(capture_id)
            self._total_bytes -= cap.byte_size
            self._close_capture_storage(cap)
            self.capture_order.pop(capture_id, None)
            return f"Cleared capture '{capture_id}'."
        else:
            return f"Capture '{capture_id}' not found."
