"""
Core search and indexing engine for ephemeral command output buffer.
Provides hybrid search (BM25 lexical + dense semantic embeddings) with RRF ranking.
"""

import sys
import time
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from fastembed import TextEmbedding


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

    @property
    def line_count(self) -> int:
        return len(self.raw_lines)

    @property
    def byte_size(self) -> int:
        return sum(len(line.encode("utf-8")) + 1 for line in self.raw_lines)


class EphemeralEngine:
    def __init__(self, max_captures: int = 10, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.max_captures = max_captures
        self.captures: Dict[str, Capture] = {}
        self.capture_order: List[str] = []
        self._next_id = 1
        
        # Lazy or fast load embedding model
        sys.stderr.write(f"Loading embedding model: {embedding_model_name}...\n")
        sys.stderr.flush()
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
        sys.stderr.write("Embedding model ready.\n")
        sys.stderr.flush()

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

    def ingest(self, text: str, label: str = "") -> Capture:
        """
        Ingests text, chunks it, builds SQLite FTS5 BM25 index and FastEmbed dense vector embeddings.
        """
        lines = text.splitlines()
        capture_id = f"cap_{self._next_id}"
        self._next_id += 1
        
        if not label:
            label = f"Capture #{self._next_id - 1}"

        chunks = self._chunk_lines(lines)
        
        # 1. Setup SQLite FTS5 in-memory DB for this capture
        fts_conn = sqlite3.connect(":memory:")
        cur = fts_conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content, tokenize='unicode61')")
        
        for c in chunks:
            cur.execute("INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)", (c.chunk_id, c.text))
        fts_conn.commit()

        # 2. Compute dense embeddings for chunks
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embed_list = list(self.embedding_model.embed(chunk_texts))
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
            fts_conn=fts_conn
        )

        # Ring buffer eviction
        if len(self.capture_order) >= self.max_captures:
            evicted_id = self.capture_order.pop(0)
            if evicted_id in self.captures:
                old_cap = self.captures.pop(evicted_id)
                if old_cap.fts_conn:
                    old_cap.fts_conn.close()

        self.captures[capture_id] = capture
        self.capture_order.append(capture_id)
        return capture

    def get_capture(self, capture_id: str = "latest") -> Optional[Capture]:
        if not self.captures:
            return None
        if capture_id == "latest" or not capture_id:
            return self.captures[self.capture_order[-1]]
        return self.captures.get(capture_id)

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

    def search_semantic(self, capture: Capture, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense vector cosine similarity search. Returns list of (chunk_id, score).
        """
        if not capture.chunks or capture.embeddings is None or len(capture.embeddings) == 0:
            return []
            
        query_embed = list(self.embedding_model.embed([query]))[0]
        query_embed = np.array(query_embed, dtype=np.float32)
        norm = np.linalg.norm(query_embed)
        if norm > 0:
            query_embed = query_embed / norm

        similarities = np.dot(capture.embeddings, query_embed)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.0]

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

    def get_summary(self, capture_id: str = "latest") -> Dict[str, Any]:
        """
        Generates a quick diagnostic summary of the capture.
        """
        capture = self.get_capture(capture_id)
        if not capture:
            return {"status": "error", "message": f"Capture '{capture_id}' not found."}

        error_keywords = ["error", "exception", "failed", "fatal", "panic", "traceback", "critical", "timed out", "unhandled"]
        detected_counts = {}
        for kw in error_keywords:
            hits = sum(1 for line in capture.raw_lines if kw in line.lower())
            if hits > 0:
                detected_counts[kw] = hits

        head_preview = [f"  {i+1:5d} | {line}" for i, line in enumerate(capture.raw_lines[:5])]
        tail_preview = [f"  {capture.line_count - len(capture.raw_lines[-5:]) + i + 1:5d} | {line}" for i, line in enumerate(capture.raw_lines[-5:])]

        return {
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(capture.timestamp)),
            "total_lines": capture.line_count,
            "byte_size": capture.byte_size,
            "keyword_signals": detected_counts,
            "head_preview": "\n".join(head_preview),
            "tail_preview": "\n".join(tail_preview)
        }

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

    def clear(self, capture_id: str = "all") -> str:
        """
        Clears one or all captures from the buffer.
        """
        if capture_id == "all":
            for cap in self.captures.values():
                if cap.fts_conn:
                    cap.fts_conn.close()
            self.captures.clear()
            self.capture_order.clear()
            return "Cleared all captures from ephemeral buffer."
        elif capture_id in self.captures:
            cap = self.captures.pop(capture_id)
            if cap.fts_conn:
                cap.fts_conn.close()
            if capture_id in self.capture_order:
                self.capture_order.remove(capture_id)
            return f"Cleared capture '{capture_id}'."
        else:
            return f"Capture '{capture_id}' not found."
