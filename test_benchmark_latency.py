"""Tests for the command-capture latency benchmark."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_latency


class FakeEngine:
    instances = []

    def __init__(self, max_captures):
        self.max_buffer_bytes = 1024
        self.embedding_model_name = "test-model"
        self.events = []
        self._next_capture = 1
        self.__class__.instances.append(self)

    def ingest(self, content, *, label, **_kwargs):
        capture = SimpleNamespace(capture_id=f"cap-{self._next_capture}")
        self._next_capture += 1
        self.events.append(("ingest", label, content))
        return capture

    def _ensure_embeddings(self, capture):
        self.events.append(("ensure_embeddings", capture.capture_id))

    def get_summary(self, capture_id):
        self.events.append(("summary", capture_id))
        return {}


class TestBenchmarkLatency(unittest.TestCase):
    def test_warmup_materializes_embeddings_before_warm_samples(self):
        FakeEngine.instances = []
        with patch.object(benchmark_latency, "EphemeralEngine", FakeEngine), patch.object(
            benchmark_latency,
            "run_command_bounded",
            return_value=("benchmark line\n", 0, False, 15, False),
        ):
            benchmark_latency.run_benchmark((1,), samples=1)

        self.assertEqual(len(FakeEngine.instances), 2)
        warm_events = FakeEngine.instances[1].events
        self.assertEqual(warm_events[:2], [
            ("ingest", "latency-warmup", "warmup"),
            ("ensure_embeddings", "cap-1"),
        ])
        self.assertLess(
            warm_events.index(("ensure_embeddings", "cap-1")),
            warm_events.index(("ingest", "latency-1", "benchmark line\n")),
        )


if __name__ == "__main__":
    unittest.main()
