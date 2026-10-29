#!/usr/bin/env python3
"""Exercise the pinned real embedding model and semantic search path.

This script's report helpers use only the Python standard library so a
diagnostic report can still be written when dependency installation fails.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping


MODEL_NAME = "BAAI/bge-small-en-v1.5-fp32"
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EXPECTED_EMBEDDING_DIMENSIONS = 384
EMBEDDING_INPUTS = (
    "A red bicycle leaned against the oak tree.",
    "The deployment completed without errors.",
    "A small sailboat crossed the blue lake.",
)
CAPTURE_TEXT = "\n".join(EMBEDDING_INPUTS)
PACKAGE_DISTRIBUTIONS = {
    "ephemeral-buffer-mcp": "ephemeral-buffer-mcp",
    "mcp": "mcp",
    "filelock": "filelock",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "tokenizers": "tokenizers",
    "huggingface-hub": "huggingface-hub",
    "urllib3": "urllib3",
}


def _package_versions() -> dict[str, str | None]:
    """Return installed runtime package versions, preserving missing packages."""
    versions: dict[str, str | None] = {}
    for display_name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[display_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[display_name] = None
    return versions


def _new_report(lane: str) -> dict[str, Any]:
    """Create a dependency-free report skeleton with the pinned model identity."""
    return {
        "schema_version": 1,
        "lane": lane,
        "status": "not_run",
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "packages": _package_versions(),
        "model": {
            "name": MODEL_NAME,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "checks": {
            "deterministic_embedding_guard": {"passed": False},
            "embeddings": {"passed": False},
            "semantic_search": {"passed": False},
        },
        "failure": None,
    }


def _runtime_model_metadata() -> dict[str, str]:
    """Read model identity from the runtime and catch report metadata drift."""
    from ephemeral_buffer_mcp.config import BGE_SMALL_HF_REPO, DEFAULT_EMBEDDING_MODEL
    from ephemeral_buffer_mcp.embedding_backend import _BGE_SMALL_REVISION

    metadata = {
        "name": DEFAULT_EMBEDDING_MODEL,
        "id": BGE_SMALL_HF_REPO,
        "revision": _BGE_SMALL_REVISION,
    }
    if metadata != {"name": MODEL_NAME, "id": MODEL_ID, "revision": MODEL_REVISION}:
        raise RuntimeError(
            "Smoke report model metadata does not match the runtime's configured model and pinned revision: "
            f"{metadata!r}"
        )
    return metadata


def _deterministic_embeddings_enabled(environ: Mapping[str, str]) -> bool:
    """Match the runtime switch that selects the deterministic test backend."""
    return environ.get("EPHEMERAL_TEST_EMBEDDINGS") == "1"


def validate_embeddings(vectors: Any, expected_count: int) -> dict[str, Any]:
    """Validate BGE vector count, dimensions, finite values, and L2 norms."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    expected_shape = (expected_count, EXPECTED_EMBEDDING_DIMENSIONS)
    if array.shape != expected_shape:
        raise AssertionError(f"expected embedding shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise AssertionError("embedding vectors contain non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise AssertionError(f"embedding vectors are not L2-normalized: norms={norms.tolist()}")
    return {
        "passed": True,
        "shape": list(array.shape),
        "finite": True,
        "l2_normalized": True,
        "norms": [float(value) for value in norms],
    }


def _record_failure(report: dict[str, Any], stage: str, exc: BaseException) -> None:
    """Add an exception and traceback to a failed smoke report."""
    detail = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if report["failure"] is None:
        report["failure"] = detail
    else:
        report["failure"].setdefault("cleanup_failures", []).append(detail)
    report["status"] = "failed"


def _loaded_embedding_model(engine: Any) -> Any:
    """Return the model owned by the engine after its supported load operation."""
    model = engine._semantic_index.embedding_model
    if model is None:
        raise RuntimeError("engine reported a successful model load but has no loaded embedding model")
    if type(model).__name__ == "_DeterministicTestEmbedding":
        raise RuntimeError("the engine loaded deterministic test embeddings instead of the real model")
    if getattr(model, "model_name", None) != MODEL_NAME:
        raise RuntimeError(
            "the loaded embedding model does not match the expected real model "
            f"{MODEL_NAME!r}: {getattr(model, 'model_name', None)!r}"
        )
    return model


def run_smoke(
    lane: str = "local",
    *,
    environ: Mapping[str, str] | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run embedding and semantic-search checks and return a diagnostic report."""
    environment = os.environ if environ is None else environ
    report = _new_report(lane)
    report["status"] = "failed"
    engine = None
    stage = "deterministic_embedding_guard"
    try:
        if _deterministic_embeddings_enabled(environment):
            raise RuntimeError(
                "EPHEMERAL_TEST_EMBEDDINGS=1 selects deterministic test vectors; "
                "unset it or set it to 0 for this compatibility smoke"
            )
        report["checks"]["deterministic_embedding_guard"] = {
            "passed": True,
            "enabled": False,
        }

        stage = "runtime_model_metadata"
        model_metadata = _runtime_model_metadata()
        report["model"] = model_metadata

        stage = "engine_initialization"
        if engine_factory is None:
            from ephemeral_buffer_mcp.engine import EphemeralEngine

            engine_factory = EphemeralEngine
        engine = engine_factory(
            embedding_model_name=model_metadata["name"],
            embedding_cache_path=(
                environment.get("EPHEMERAL_EMBEDDING_CACHE_DIR")
                or environment.get("EPHEMERAL_FASTEMBED_CACHE_DIR")
            ),
            embedding_warmup=False,
            semantic_prefetch=False,
        )

        stage = "model_load"
        engine.load_embedding_model()
        model = _loaded_embedding_model(engine)

        stage = "embedding_inference"
        vectors = list(model.embed(EMBEDDING_INPUTS))
        report["checks"]["embeddings"] = validate_embeddings(vectors, len(EMBEDDING_INPUTS))

        stage = "semantic_search"
        capture = engine.ingest(CAPTURE_TEXT, label="real-model-compatibility-smoke")
        result = engine.search(
            EMBEDDING_INPUTS[-1],
            mode="semantic",
            capture_id=capture.capture_id,
            top_k=3,
            context_lines=0,
        )
        diagnostics = engine.get_capture_diagnostics(capture.capture_id)
        semantic_coverage = result.get("semantic_coverage")
        semantic_index_state = getattr(diagnostics, "semantic_index_state", None)
        if result.get("status") != "ok":
            raise RuntimeError(f"semantic search returned an error: {result!r}")
        if semantic_coverage != "complete":
            raise RuntimeError(
                "semantic search did not complete semantic coverage: "
                f"{semantic_coverage!r}; result={result!r}"
            )
        if semantic_index_state != "ready":
            raise RuntimeError(
                "semantic search did not leave the capture index ready: "
                f"{semantic_index_state!r}"
            )
        if result.get("match_count", 0) < 1:
            raise RuntimeError(f"semantic search returned no matches: {result!r}")
        report["checks"]["semantic_search"] = {
            "passed": True,
            "status": result.get("status"),
            "mode": result.get("mode"),
            "semantic_coverage": semantic_coverage,
            "semantic_index_state": semantic_index_state,
            "match_count": result.get("match_count"),
        }
        report["status"] = "passed"
    except Exception as exc:
        _record_failure(report, stage, exc)
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as exc:
                _record_failure(report, "engine_shutdown", exc)
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write one indented JSON report, creating its parent directory if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_failure_report(
    path: str | Path,
    *,
    lane: str,
    details: str,
) -> bool:
    """Create a workflow-failure report when installation prevented the smoke."""
    target = Path(path)
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") in {"passed", "failed"}:
            return False

    report = _new_report(lane)
    report["status"] = "failed"
    report["failure"] = {
        "stage": "workflow_setup",
        "type": "WorkflowSetupFailure",
        "message": details or "The model smoke did not produce a report.",
        "traceback": None,
    }
    write_report(report, target)
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default="local", help="dependency lane name for the report")
    parser.add_argument(
        "--report",
        default="real-model-compatibility.json",
        help="path for the JSON diagnostic report",
    )
    parser.add_argument(
        "--ensure-failure-report",
        action="store_true",
        help="write a setup-failure report if a complete report is absent",
    )
    parser.add_argument(
        "--failure-details",
        default="",
        help="workflow step outcomes to include in a fallback report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the smoke or ensure a fallback report exists for a failed lane."""
    args = _parse_args(argv)
    if args.ensure_failure_report:
        created = ensure_failure_report(
            args.report,
            lane=args.lane,
            details=args.failure_details,
        )
        if created:
            print(f"Wrote setup-failure report: {args.report}")
        return 0

    report = run_smoke(args.lane)
    write_report(report, args.report)
    print(f"Real-model compatibility smoke {report['status']}: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
