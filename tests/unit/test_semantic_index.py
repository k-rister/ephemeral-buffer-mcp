"""Direct tests for semantic-index coordination and capture-store boundaries."""

from __future__ import annotations

import io
import threading
import unittest
from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, Optional
from unittest.mock import patch

import numpy as np

import ephemeral_buffer_mcp.semantic_index as semantic_index
from ephemeral_buffer_mcp.metrics import LocalMetrics
from ephemeral_buffer_mcp.semantic_index import (
    Chunk,
    SemanticCaptureLease,
    SemanticIndexConfig,
    SemanticIndexCoordinator,
)


@dataclass
class _Capture:
    capture_id: str
    semantic_chunks: list[Chunk]
    embeddings: Any = None
    semantic_index_state: str = "not-requested"
    active_readers: int = 0
    deferred_storage_tracked: bool = False
    deferred_storage_accounted_bytes: int = 0


class _CaptureStore:
    def __init__(self) -> None:
        self.captures: dict[str, _Capture] = {}
        self.published: list[str] = []

    def get_retained_capture(self, capture_id: str) -> Optional[SemanticCaptureLease]:
        return self.captures.get(capture_id)

    def publish_embeddings(self, capture: SemanticCaptureLease, embeddings: Any) -> bool:
        if self.captures.get(capture.capture_id) is not capture or capture.embeddings is not None:
            return False
        capture.embeddings = embeddings
        capture.semantic_index_state = "ready"
        self.published.append(capture.capture_id)
        return True


def _config(**overrides: Any) -> SemanticIndexConfig:
    values = {
        "embedding_model_name": "test-model",
        "embedding_cache_path": None,
        "embedding_threads": None,
        "embedding_batch_size": 8,
        "embedding_max_batch_tokens": 1024,
        "embedding_cpu_mem_arena_enabled": False,
        "semantic_max_index_input_bytes": 1024 * 1024,
        "semantic_chunk_lines": 8,
        "semantic_chunk_bytes": 1024,
        "semantic_chunk_overlap": 0,
        "embedding_warmup_enabled": False,
        "semantic_prefetch_enabled": False,
        "semantic_prefetch_workers": 1,
        "semantic_wait_seconds": 0.1,
    }
    values.update(overrides)
    return SemanticIndexConfig(**values)


class TestSemanticIndexCoordinator(unittest.TestCase):
    def _coordinator(
        self,
        *,
        config: Optional[SemanticIndexConfig] = None,
        warmup_state: str = "disabled",
        backend_available: bool = True,
    ) -> tuple[SemanticIndexCoordinator, _CaptureStore]:
        store = _CaptureStore()
        coordinator = SemanticIndexCoordinator(
            lock=threading.RLock(),
            metrics=LocalMetrics(enabled=False),
            capture_store=store,
            mark_metrics_snapshot_pending=lambda: None,
            flush_metrics_snapshot=lambda: None,
            config=config or _config(),
            semantic_backend_available=backend_available,
            embedding_warmup_state=warmup_state,
            embedding_warmup_failure=None,
        )
        self.addCleanup(coordinator.shutdown)
        return coordinator, store

    @staticmethod
    def _capture(capture_id: str) -> _Capture:
        return _Capture(capture_id, [Chunk(0, 1, 1, f"payload for {capture_id}")])

    def test_config_is_frozen_and_capture_publication_uses_the_store(self):
        config = _config()
        with self.assertRaises(FrozenInstanceError):
            config.semantic_chunk_lines = 2

        coordinator, store = self._coordinator(config=config)
        capture = self._capture("retained")
        store.captures[capture.capture_id] = capture

        class Embedding:
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        coordinator.embedding_model = Embedding()
        with patch.object(semantic_index, "np", np):
            coordinator.ensure_embeddings(capture)

        self.assertEqual(capture.semantic_index_state, "ready")
        self.assertEqual(store.published, [capture.capture_id])
        self.assertEqual(capture.embeddings.shape, (1, 2))

    def test_timing_observer_runs_before_job_completion_and_is_removed(self):
        coordinator, store = self._coordinator()
        capture = self._capture("timed")
        store.captures[capture.capture_id] = capture
        embedding_started = threading.Event()
        release_embedding = threading.Event()
        observations: list[tuple[float, bool]] = []
        job_holder = []

        class BlockingEmbedding:
            def embed(self, texts):
                embedding_started.set()
                if not release_embedding.wait(timeout=5):
                    raise AssertionError("test did not release the embedding worker")
                return [[1.0, 0.0] for _ in texts]

        coordinator.embedding_model = BlockingEmbedding()

        def observe(elapsed_seconds: float) -> None:
            observations.append((elapsed_seconds, job_holder[0].done.is_set()))

        try:
            with patch.object(semantic_index, "np", np):
                with coordinator.observe_indexing(observe):
                    done, job = coordinator._start_semantic_index(capture)
                    self.assertIsNotNone(job)
                    job_holder.append(job)
                    self.assertTrue(embedding_started.wait(timeout=2))
                    release_embedding.set()
                    self.assertTrue(done.wait(timeout=2))
            self.assertEqual(len(observations), 1)
            self.assertGreater(observations[0][0], 0)
            self.assertFalse(observations[0][1])
            self.assertEqual(capture.semantic_index_state, "ready")
            self.assertEqual(coordinator._indexing_observers, {})
        finally:
            release_embedding.set()

    def test_observer_errors_do_not_change_index_result_and_cleanup_on_failure(self):
        coordinator, store = self._coordinator()
        successful = self._capture("observer-error")
        failed = self._capture("index-error")
        store.captures[successful.capture_id] = successful
        store.captures[failed.capture_id] = failed

        class Embedding:
            def embed(self, texts):
                if texts[0] == failed.semantic_chunks[0].text:
                    raise RuntimeError("embedding failed")
                return [[1.0, 0.0] for _ in texts]

        def broken_observer(_elapsed_seconds: float) -> None:
            raise ValueError("observer failed")

        coordinator.embedding_model = Embedding()
        with patch.object(semantic_index, "np", np):
            with coordinator.observe_indexing(broken_observer):
                coordinator.ensure_embeddings(successful)
            self.assertEqual(successful.semantic_index_state, "ready")
            self.assertEqual(coordinator._indexing_observers, {})

            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                with coordinator.observe_indexing(broken_observer):
                    coordinator.ensure_embeddings(failed)

        self.assertEqual(coordinator._indexing_observers, {})

    def test_prefetch_queue_cancellation_diagnostics_and_shutdown(self):
        coordinator, store = self._coordinator(
            config=replace(_config(), semantic_prefetch_enabled=True)
        )
        first = self._capture("first")
        queued = self._capture("queued")
        store.captures[first.capture_id] = first
        store.captures[queued.capture_id] = queued
        embedding_started = threading.Event()
        release_embedding = threading.Event()

        class BlockingEmbedding:
            def embed(self, texts):
                embedding_started.set()
                if not release_embedding.wait(timeout=5):
                    raise AssertionError("test did not release the prefetch worker")
                return [[1.0, 0.0] for _ in texts]

        coordinator.embedding_model = BlockingEmbedding()
        try:
            with patch.object(semantic_index, "np", np):
                coordinator.schedule_semantic_prefetch(first)
                self.assertTrue(embedding_started.wait(timeout=2))
                coordinator.schedule_semantic_prefetch(queued)
                diagnostics = coordinator.diagnostics(store.captures.values())
                self.assertEqual(diagnostics["semantic_prefetch_queued"], 1)
                self.assertEqual(diagnostics["semantic_prefetch_running"], 1)

                store.captures.pop(queued.capture_id)
                coordinator.capture_removed_locked(queued.capture_id, "evicted")
                self.assertNotIn(queued.capture_id, coordinator._prefetch_queue)
                self.assertEqual(queued.semantic_index_state, "not-requested")
                self.assertIn("semantic_prefetch:1", coordinator.shutdown(0)["unfinished_work"])

                release_embedding.set()
                report = coordinator.shutdown(2)

            self.assertEqual(report["unfinished_work"], [])
            self.assertEqual(first.semantic_index_state, "ready")
            self.assertEqual(queued.embeddings, None)
        finally:
            release_embedding.set()

    def test_warmup_readiness_is_owned_by_the_coordinator(self):
        coordinator, _store = self._coordinator(
            config=replace(_config(), embedding_warmup_enabled=True),
            warmup_state="not-started",
        )

        class WarmEmbedding:
            def embed(self, texts):
                self.texts = list(texts)
                return [[1.0, 0.0] for _ in texts]

        coordinator.embedding_model = WarmEmbedding()
        self.assertTrue(coordinator.start_embedding_warmup())
        self.assertEqual(coordinator.wait_for_embedding_warmup(timeout=2), "ready")
        self.assertEqual(coordinator.embedding_warmup_failure, None)

    def test_model_load_backoff_and_retry_are_coordinator_state(self):
        coordinator, _store = self._coordinator()
        loaded_model = object()
        with patch.dict("os.environ", {"EPHEMERAL_TEST_EMBEDDINGS": "0"}, clear=False), \
                patch.object(semantic_index, "_load_semantic_dependencies"), \
                patch.object(
                    semantic_index,
                    "TextEmbedding",
                    side_effect=[RuntimeError("model load failed"), loaded_model],
                ) as embedding, \
                patch.object(semantic_index.time, "monotonic", side_effect=[100.0, 110.0, 161.0]), \
                patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                coordinator._get_embedding_model()
            with self.assertRaisesRegex(RuntimeError, "retry backoff"):
                coordinator._get_embedding_model()
            self.assertIs(coordinator._get_embedding_model(), loaded_model)

        self.assertEqual(embedding.call_count, 2)
        self.assertIsNone(coordinator._embedding_model_load_failure)


if __name__ == "__main__":
    unittest.main()
