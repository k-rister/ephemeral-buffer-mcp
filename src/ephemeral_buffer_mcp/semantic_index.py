"""Embedding model loading and asynchronous semantic-index coordination."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_utils import get_logger, log_event

LOGGER = get_logger("engine")
np = None
TextEmbedding = None
EMBEDDING_MODEL_LOAD_RETRY_SECONDS = 60.0


class SemanticIndexBudgetExceeded(RuntimeError):
    """Raised when semantic indexing would exceed its configured work envelope."""


class SemanticBackendUnavailable(RuntimeError):
    """Raised when optional semantic-search dependencies are not installed."""


_SEMANTIC_INSTALL_GUIDANCE = (
    "Semantic and hybrid search require the full distribution. Install it with "
    "`python -m pip install --upgrade ephemeral-buffer-mcp`."
)


def _semantic_dependencies_available() -> bool:
    """Return whether the optional semantic runtime packages can be located."""
    try:
        dependencies = ("numpy", "onnxruntime", "huggingface_hub", "tokenizers")
        return all(importlib.util.find_spec(name) is not None for name in dependencies)
    except (ImportError, ValueError):
        return False


def _load_semantic_dependencies() -> None:
    """Import the semantic stack only when a dense-search path needs it."""
    global np, TextEmbedding
    try:
        if np is None:
            import numpy as numpy

            np = numpy
        if TextEmbedding is None:
            from .embedding_backend import TextEmbedding as text_embedding

            TextEmbedding = text_embedding
    except ImportError as exc:
        raise SemanticBackendUnavailable(_SEMANTIC_INSTALL_GUIDANCE) from exc


def _embedding_nbytes(embeddings: Any) -> int:
    """Return retained dense-vector bytes, including for lexical empty markers."""
    return int(getattr(embeddings, "nbytes", 0)) if embeddings is not None else 0


@dataclass
class Chunk:
    chunk_id: int
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed
    text: str


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
        _load_semantic_dependencies()
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                token = self._canonical_terms.get(token, token)
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 384
                vector[index] += 1.0
            vectors.append(vector.tolist())
        return vectors


class SemanticIndexCoordinator:
    """Own embedding inference, semantic jobs, and their bounded lifecycle."""

    def __init__(
        self,
        *,
        lock: Any,
        metrics: Any,
        get_retained_capture: Callable[[str], Any],
        publish_embeddings: Callable[[Any, Any], bool],
        mark_metrics_snapshot_pending: Callable[[], None],
        flush_metrics_snapshot: Callable[[], None],
        embedding_model_name: str,
        embedding_cache_path: Optional[str],
        embedding_threads: Optional[int],
        embedding_batch_size: int,
        embedding_max_batch_tokens: int,
        embedding_cpu_mem_arena_enabled: bool,
        semantic_max_index_input_bytes: int,
        semantic_chunk_lines: int,
        semantic_chunk_bytes: int,
        semantic_chunk_overlap: int,
        semantic_backend_available: bool,
        embedding_warmup_enabled: bool,
        embedding_warmup_state: str,
        embedding_warmup_failure: Optional[str],
        semantic_prefetch_enabled: bool,
        semantic_prefetch_workers: int,
        semantic_wait_seconds: float,
    ) -> None:
        self._lock = lock
        self._embedding_lock = threading.RLock()
        self.metrics = metrics
        self._get_retained_capture = get_retained_capture
        self._publish_embeddings_callback = publish_embeddings
        self._mark_metrics_snapshot_pending_callback = mark_metrics_snapshot_pending
        self._flush_metrics_snapshot_callback = flush_metrics_snapshot

        self.embedding_model_name = embedding_model_name
        self.embedding_cache_path = embedding_cache_path
        self.embedding_threads = embedding_threads
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_batch_tokens = embedding_max_batch_tokens
        self.embedding_cpu_mem_arena_enabled = embedding_cpu_mem_arena_enabled
        self.semantic_max_index_input_bytes = semantic_max_index_input_bytes
        self.semantic_chunk_lines = semantic_chunk_lines
        self.semantic_chunk_bytes = semantic_chunk_bytes
        self.semantic_chunk_overlap = semantic_chunk_overlap
        self.semantic_backend_available = semantic_backend_available
        self.embedding_model = None
        self._embedding_model_load_failure: Optional[Tuple[type, float]] = None
        self.embedding_warmup_enabled = embedding_warmup_enabled
        self.embedding_warmup_state = embedding_warmup_state
        self.embedding_warmup_failure = embedding_warmup_failure
        self._embedding_warmup_thread: Optional[threading.Thread] = None
        self.semantic_prefetch_enabled = semantic_prefetch_enabled
        self.semantic_prefetch_workers = semantic_prefetch_workers
        self.semantic_wait_seconds = semantic_wait_seconds

        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=self.semantic_prefetch_workers,
                thread_name_prefix="semantic-prefetch",
            )
            if self.semantic_prefetch_enabled and self.semantic_backend_available
            else None
        )
        # Eligible captures wait in an ordered queue that is drained newest-first.
        self._prefetch_queue: "OrderedDict[str, Any]" = OrderedDict()
        self._prefetch_running: Dict[str, threading.Event] = {}
        self._prefetch_job_meta: Dict[str, Dict[str, Any]] = {}
        self._semantic_job_dispositions: Dict[str, str] = {}
        self._prefetch_workers_active = 0
        self._on_demand_jobs: Dict[str, _SemanticIndexJob] = {}
        self._on_demand_executor = ThreadPoolExecutor(
            max_workers=self.semantic_prefetch_workers,
            thread_name_prefix="semantic-index",
        )
        self._shutdown = False

    @property
    def numpy(self):
        """Return the optional NumPy module after lazy dependency loading."""
        return np

    def embed_query(self, query: str):
        """Embed one search query using the configured semantic model."""
        _load_semantic_dependencies()
        with self._embedding_lock:
            return list(self._get_embedding_model().embed([query]))[0]

    def capture_removed_locked(self, capture_id: str, outcome: str) -> None:
        """Cancel queued semantic work for a removed capture under ``self._lock``."""
        self._mark_semantic_job_disposition_locked(capture_id, outcome)
        self._cancel_prefetch(capture_id, outcome=outcome)
        self._cancel_on_demand_job(capture_id, outcome=outcome)

    def capture_diagnostics(self, capture: Any) -> Tuple[int, int]:
        """Return semantic input and embedding bytes for one engine-owned capture."""
        return (
            self._semantic_input_byte_count(capture),
            _embedding_nbytes(capture.embeddings),
        )

    def diagnostics(self, captures: Any) -> Dict[str, Any]:
        """Snapshot semantic-index accounting while the shared engine lock is held."""
        values = tuple(captures)
        embedding_load_failure = self._embedding_model_load_failure
        return {
            "embedding_bytes": sum(
                _embedding_nbytes(capture.embeddings)
                for capture in values
                if capture.embeddings is not None
            ),
            "embedding_model_load_failure": (
                embedding_load_failure[0].__name__
                if embedding_load_failure is not None
                else None
            ),
            "embedding_model_retry_after_seconds": (
                max(0.0, embedding_load_failure[1] - time.monotonic())
                if embedding_load_failure is not None
                else 0.0
            ),
            "semantic_prefetch_pending": sum(
                1 for capture in values if capture.semantic_index_state == "pending"
            ),
            "semantic_prefetch_queued": len(self._prefetch_queue),
            "semantic_prefetch_running": len(self._prefetch_running),
            "semantic_index_on_demand_running": sum(
                1 for job in self._on_demand_jobs.values() if job.future.running()
            ),
            "semantic_index_on_demand_queued": sum(
                1
                for job in self._on_demand_jobs.values()
                if not job.future.running() and not job.future.done()
            ),
            "semantic_prefetch_failed": sum(
                1 for capture in values if capture.semantic_index_state == "failed"
            ),
            "semantic_index_budget_exceeded": sum(
                1
                for capture in values
                if capture.semantic_index_state == "budget-exceeded"
            ),
        }

    def _flush_metrics_snapshot(self) -> None:
        """Flush a deferred metrics snapshot after releasing the shared lock."""
        callback = self._flush_metrics_snapshot_callback
        if callback is not None:
            callback()

    def start_embedding_warmup(self) -> bool:
        """Start one non-blocking model warm-up and return whether work was started."""
        with self._lock:
            if not self.semantic_backend_available:
                self.embedding_warmup_state = "unavailable"
                return False
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
        """Load the ONNX embedding model once, when semantic search first needs it."""
        with self._embedding_lock:
            if self.embedding_model is not None:
                return self.embedding_model
            if os.environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1":
                self.embedding_model = _DeterministicTestEmbedding()
                self._embedding_model_load_failure = None
                log_event(LOGGER, logging.INFO, "embedding_model_ready", model="deterministic-test")
                return self.embedding_model
            try:
                _load_semantic_dependencies()
            except SemanticBackendUnavailable:
                self.semantic_backend_available = False
                raise
            load_failure = self._embedding_model_load_failure
            if load_failure is not None:
                failure_type, retry_at = load_failure
                if time.monotonic() < retry_at:
                    try:
                        cached_failure = failure_type(
                            "Embedding model load is in retry backoff."
                        )
                    except Exception:
                        cached_failure = RuntimeError(
                            "Embedding model load is in retry backoff after "
                            f"{failure_type.__name__}."
                        )
                    raise cached_failure
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
            try:
                self.embedding_model = TextEmbedding(**kwargs)
            except Exception as exc:
                self._embedding_model_load_failure = (
                    type(exc),
                    time.monotonic() + EMBEDDING_MODEL_LOAD_RETRY_SECONDS,
                )
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "embedding_model_load_failed",
                    model=self.embedding_model_name,
                    cache_dir=self.embedding_cache_path or "default",
                )
                LOGGER.exception("embedding_model_load_exception")
                raise
            self._embedding_model_load_failure = None
            sys.stderr.write("Embedding model ready.\n")
            sys.stderr.flush()
            log_event(LOGGER, logging.INFO, "embedding_model_ready", model=self.embedding_model_name)
            return self.embedding_model

    def load_embedding_model(self) -> None:
        """Ensure the configured embedding model is loaded without exposing it.

        This supported operation is useful for measuring model-load cost. The
        ONNX embedding object remains owned by the engine.
        """
        self._get_embedding_model()

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
        self._mark_metrics_snapshot_pending_callback()

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
        capture: Any,
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
    def _semantic_input_byte_count(capture: Any) -> int:
        """Return the UTF-8 bytes the semantic windows would send to inference."""
        return sum(
            len(chunk.text.encode("utf-8", errors="replace"))
            for chunk in capture.semantic_chunks
        )

    def _schedule_semantic_prefetch(self, capture: Any) -> None:
        """Queue post-ingestion indexing and make sure a bounded worker is draining."""
        if not self.semantic_backend_available:
            self._flush_metrics_snapshot()
            return
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
                if self._shutdown or self._get_retained_capture(capture.capture_id) is not capture:
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

    def _start_semantic_index(self, capture: Any) -> Optional[Tuple[threading.Event, Optional[_SemanticIndexJob]]]:
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

    def _on_demand_index_worker(self, capture: Any, job: _SemanticIndexJob) -> None:
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

    def _await_semantic_index(self, capture: Any, timeout: Optional[float] = None) -> str:
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
            retained = self._get_retained_capture(capture.capture_id) is capture
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

    def _wait_for_state_index(self, capture: Any, timeout: Optional[float] = None) -> str:
        """Block until the capture's semantic index is ready, failed, or ``timeout`` elapses.

        Returns ``"ready"``, ``"pending"``, or ``"failed"``; it never raises for
        an indexing error, so callers can poll from tests and benchmarks.
        """
        try:
            return self._await_semantic_index(capture, timeout)
        except Exception:
            return "failed"

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
        live = self._get_retained_capture(capture_id)
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
                # Test doubles and compatible embedding adapters may not
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

    def _ensure_embeddings(self, capture: Any) -> None:
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
                self._get_retained_capture(capture.capture_id) is not capture
                and not getattr(capture, "active_readers", 0)
            ):
                return
            chunk_texts = [chunk.text for chunk in capture.semantic_chunks]
        with self._embedding_lock:
            with self._lock:
                if capture.embeddings is not None:
                    return
                if (
                    self._get_retained_capture(capture.capture_id) is not capture
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
            embeddings: Optional[Any] = None
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
                        self._get_retained_capture(capture.capture_id) is capture
                        or getattr(capture, "active_readers", 0)
                    )
                    and capture.embeddings is None
                ):
                    self._publish_embeddings_callback(capture, embeddings)

