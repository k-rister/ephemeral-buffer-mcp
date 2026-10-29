"""Offline tests for the scheduled real-model compatibility smoke tooling."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

from ephemeral_buffer_mcp import config, embedding_backend
from scripts import real_model_compatibility_smoke as smoke


class _FakeEmbedding:
    model_name = smoke.MODEL_NAME

    def embed(self, _texts):
        vectors = np.zeros((len(smoke.EMBEDDING_INPUTS), smoke.EXPECTED_EMBEDDING_DIMENSIONS))
        vectors[:, 0] = 1.0
        return vectors


class _FakeEngine:
    def __init__(self, **options):
        self.options = options
        self._semantic_index = SimpleNamespace(embedding_model=_FakeEmbedding())
        self.shutdown_called = False

    def load_embedding_model(self):
        return None

    def ingest(self, _text, label):
        return SimpleNamespace(capture_id="smoke-capture", label=label)

    def search(self, query, *, mode, capture_id, top_k, context_lines):
        self.search_args = (query, mode, capture_id, top_k, context_lines)
        return {
            "status": "ok",
            "mode": mode,
            "semantic_coverage": "complete",
            "match_count": 1,
            "matches": [{"score": 1.0}],
        }

    def get_capture_diagnostics(self, _capture_id):
        return SimpleNamespace(semantic_index_state="ready")

    def shutdown(self):
        self.shutdown_called = True


class TestRealModelCompatibilitySmoke(unittest.TestCase):
    def test_report_model_metadata_matches_the_runtime_pin(self):
        self.assertEqual(smoke.MODEL_NAME, config.DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(smoke.MODEL_ID, config.BGE_SMALL_HF_REPO)
        self.assertEqual(smoke.MODEL_REVISION, embedding_backend._BGE_SMALL_REVISION)

    def test_report_version_uses_core_distribution_and_ignores_full_package(self):
        def version(distribution):
            versions = {
                "ephemeral-buffer-mcp-core": "0.11.0",
                "ephemeral-buffer-mcp": "0.10.0",
            }
            if distribution in versions:
                return versions[distribution]
            raise smoke.importlib.metadata.PackageNotFoundError(distribution)

        with patch.object(smoke.importlib.metadata, "version", side_effect=version) as lookup:
            packages = smoke._new_report("package-version-test")["packages"]

        self.assertEqual(packages["ephemeral-buffer-mcp-core"], "0.11.0")
        self.assertNotIn("ephemeral-buffer-mcp", packages)
        lookup.assert_any_call("ephemeral-buffer-mcp-core")
        self.assertNotIn(call("ephemeral-buffer-mcp"), lookup.call_args_list)

    def test_embedding_validation_checks_shape_finiteness_and_normalization(self):
        vectors = np.zeros((3, smoke.EXPECTED_EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[:, 0] = 1.0
        result = smoke.validate_embeddings(vectors, expected_count=3)
        self.assertEqual(result["shape"], [3, 384])
        self.assertTrue(result["finite"])
        self.assertTrue(result["l2_normalized"])
        self.assertEqual(result["norms"], [1.0, 1.0, 1.0])

        with self.assertRaisesRegex(AssertionError, "expected embedding shape"):
            smoke.validate_embeddings(vectors[:-1], expected_count=3)
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            invalid = vectors.copy()
            invalid[0, 0] = np.nan
            smoke.validate_embeddings(invalid, expected_count=3)
        with self.assertRaisesRegex(AssertionError, "not L2-normalized"):
            unnormalized = vectors.copy()
            unnormalized[0, 0] = 2.0
            smoke.validate_embeddings(unnormalized, expected_count=3)

    def test_run_smoke_reports_real_embedding_and_ready_semantic_search(self):
        engines = []

        def create_engine(**options):
            engine = _FakeEngine(**options)
            engines.append(engine)
            return engine

        report = smoke.run_smoke(
            "offline-test",
            environ={"EPHEMERAL_EMBEDDING_CACHE_DIR": "/tmp/model-cache"},
            engine_factory=create_engine,
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["lane"], "offline-test")
        self.assertEqual(report["model"]["id"], smoke.MODEL_ID)
        self.assertEqual(report["model"]["revision"], smoke.MODEL_REVISION)
        self.assertTrue(report["checks"]["deterministic_embedding_guard"]["passed"])
        self.assertEqual(report["checks"]["embeddings"]["shape"], [3, 384])
        self.assertEqual(report["checks"]["semantic_search"]["semantic_coverage"], "complete")
        self.assertEqual(report["checks"]["semantic_search"]["semantic_index_state"], "ready")
        self.assertEqual(report["packages"]["numpy"], smoke._package_versions()["numpy"])
        self.assertEqual(engines[0].options["embedding_cache_path"], "/tmp/model-cache")
        self.assertTrue(engines[0].shutdown_called)

    def test_run_smoke_rejects_deterministic_embeddings_before_engine_creation(self):
        factory_called = False

        def create_engine(**_options):
            nonlocal factory_called
            factory_called = True
            return _FakeEngine()

        report = smoke.run_smoke(
            "deterministic-guard",
            environ={"EPHEMERAL_TEST_EMBEDDINGS": "1"},
            engine_factory=create_engine,
        )

        self.assertEqual(report["status"], "failed")
        self.assertFalse(factory_called)
        self.assertEqual(report["failure"]["stage"], "deterministic_embedding_guard")
        self.assertIn("deterministic test vectors", report["failure"]["message"])

    def test_run_smoke_rejects_a_deterministic_model_instance(self):
        class _DeterministicTestEmbedding:
            pass

        class DeterministicEngine(_FakeEngine):
            def __init__(self, **options):
                super().__init__(**options)
                self._semantic_index.embedding_model = _DeterministicTestEmbedding()

        report = smoke.run_smoke(
            "deterministic-model-guard",
            environ={},
            engine_factory=DeterministicEngine,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["stage"], "model_load")
        self.assertIn("deterministic test embeddings", report["failure"]["message"])

    def test_run_smoke_records_model_load_failure_and_shuts_down_engine(self):
        class FailingEngine(_FakeEngine):
            def load_embedding_model(self):
                raise RuntimeError("offline fake model-load failure")

        engine = FailingEngine()
        report = smoke.run_smoke(
            "model-load-failure",
            environ={},
            engine_factory=lambda **_options: engine,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["stage"], "model_load")
        self.assertIn("offline fake model-load failure", report["failure"]["message"])
        self.assertIn("Traceback", report["failure"]["traceback"])
        self.assertTrue(engine.shutdown_called)

    def test_main_writes_failure_report_and_returns_nonzero(self):
        report = smoke._new_report("write-test")
        report["status"] = "failed"
        report["failure"] = {"stage": "offline-test", "message": "expected failure"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "report.json"
            with patch.object(smoke, "run_smoke", return_value=report):
                status = smoke.main(["--lane", "write-test", "--report", str(target)])

            written = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(status, 1)
            self.assertEqual(written["failure"]["message"], "expected failure")

    def test_setup_failure_report_is_written_and_existing_smoke_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            created = smoke.ensure_failure_report(
                target,
                lane="install-failure",
                details="dependency install= failure",
            )
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertTrue(created)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["model"]["revision"], smoke.MODEL_REVISION)
            self.assertEqual(report["failure"]["stage"], "workflow_setup")
            self.assertIn("dependency install", report["failure"]["message"])

            created_again = smoke.ensure_failure_report(
                target,
                lane="install-failure",
                details="replacement details",
            )
            self.assertFalse(created_again)
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8"))["failure"]["message"],
                "dependency install= failure",
            )


if __name__ == "__main__":
    unittest.main()
