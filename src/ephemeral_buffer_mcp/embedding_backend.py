"""Small ONNX Runtime adapter for the BGE-small text embedding model.

The semantic backend deliberately uses the model's ONNX export and tokenizer
directly. This keeps the runtime small and avoids installing image-processing
dependencies for text-only search.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import AddedToken, Tokenizer

from .config import BGE_SMALL_HF_REPO, CATALOGUE_EMBEDDING_MODEL, FP32_EMBEDDING_MODEL


_MODEL_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.onnx",
    "onnx/model.onnx",
)
# This is the revision used by the existing FastEmbed BGE model cache. Pinning
# it keeps the model graph and tokenizer in sync and avoids changing vectors
# when the repository's default branch moves.
_BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_SUPPORTED_INPUTS = {"input_ids", "attention_mask", "token_type_ids"}
_DEFAULT_MAX_LENGTH = 512


def _repository_id(model_name: str) -> str:
    """Map supported BGE aliases to their pinned upstream repository."""
    if not isinstance(model_name, str):
        raise ValueError(
            "This backend supports only the BGE-small fp32 model and its catalogue alias."
        )
    if model_name.casefold() in {
        FP32_EMBEDDING_MODEL.casefold(),
        CATALOGUE_EMBEDDING_MODEL.casefold(),
    }:
        return BGE_SMALL_HF_REPO
    raise ValueError(
        f"Unsupported embedding model {model_name!r}; this backend supports only "
        f"{FP32_EMBEDDING_MODEL!r} and its catalogue alias {CATALOGUE_EMBEDDING_MODEL!r}."
    )


def _read_json(path: Path, description: str) -> dict:
    """Read a model metadata object and reject non-object JSON documents."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} from {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{description} in {path} must contain a JSON object")
    return result


def _model_max_length(tokenizer_config: dict, model_config: dict) -> int:
    """Choose a finite tokenizer limit, falling back to the BERT default."""
    limits: list[int] = []
    for key in ("model_max_length", "max_length"):
        value = tokenizer_config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000:
            limits.append(value)
    value = model_config.get("max_position_embeddings")
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000:
        limits.append(value)
    return min(limits) if limits else _DEFAULT_MAX_LENGTH


def _token_content(token: object) -> str | None:
    """Read a token value from either a string or tokenizer JSON object."""
    if isinstance(token, str):
        return token
    if isinstance(token, dict) and isinstance(token.get("content"), str):
        return token["content"]
    return None


def _load_tokenizer(model_dir: Path) -> Tokenizer:
    """Load tokenizer files using the same truncation and padding metadata as the model."""
    model_config = _read_json(model_dir / "config.json", "model config")
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json", "tokenizer config")
    special_tokens = _read_json(model_dir / "special_tokens_map.json", "special token map")
    tokenizer_path = model_dir / "tokenizer.json"
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}: {exc}") from exc

    tokenizer.enable_truncation(max_length=_model_max_length(tokenizer_config, model_config))
    for token in special_tokens.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    if not tokenizer.padding:
        pad_token = _token_content(tokenizer_config.get("pad_token"))
        if pad_token is None:
            pad_token = _token_content(special_tokens.get("pad_token"))
        if pad_token is None:
            raise ValueError("Tokenizer config must define a padding token")
        pad_id = model_config.get("pad_token_id")
        if not isinstance(pad_id, int) or isinstance(pad_id, bool):
            pad_id = tokenizer.token_to_id(pad_token)
        if not isinstance(pad_id, int):
            raise ValueError(f"Tokenizer does not define an ID for padding token {pad_token!r}")
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return tokenizer


class TextEmbedding:
    """Embed text with an ONNX export and its Hugging Face tokenizer."""

    def __init__(
        self,
        model_name: str = FP32_EMBEDDING_MODEL,
        cache_dir: str | None = None,
        threads: int | None = None,
        enable_cpu_mem_arena: bool = False,
    ) -> None:
        self.model_name = model_name
        self.repository_id = _repository_id(model_name)
        model_dir = Path(
            snapshot_download(
                repo_id=self.repository_id,
                revision=_BGE_SMALL_REVISION if self.repository_id == BGE_SMALL_HF_REPO else None,
                cache_dir=cache_dir,
                allow_patterns=list(_MODEL_FILES),
            )
        )
        model_path = next(
            (candidate for candidate in (model_dir / "onnx/model.onnx", model_dir / "model.onnx")
             if candidate.is_file()),
            None,
        )
        if model_path is None:
            raise FileNotFoundError(
                f"Hugging Face model {self.repository_id!r} has no supported ONNX model file"
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = enable_cpu_mem_arena
        if threads is not None:
            session_options.intra_op_num_threads = threads
            session_options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        unsupported = self.input_names - _SUPPORTED_INPUTS
        if "input_ids" not in self.input_names or unsupported:
            raise ValueError(
                "ONNX text models must accept input_ids and may only require "
                f"attention_mask or token_type_ids; found {sorted(self.input_names)}"
            )
        outputs = self.session.get_outputs()
        if not outputs:
            raise ValueError("ONNX text model has no outputs")
        self.output_name = outputs[0].name
        self.tokenizer = _load_tokenizer(model_dir)
        # The engine uses this small compatibility view to estimate token
        # lengths before dispatching bounded embedding batches.
        self.model = SimpleNamespace(tokenizer=self.tokenizer)

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
    ) -> Iterator[np.ndarray]:
        """Yield L2-normalized CLS embeddings in bounded batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        texts = [documents] if isinstance(documents, str) else list(documents)
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            )
            onnx_inputs: dict[str, np.ndarray] = {"input_ids": input_ids}
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                onnx_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
            output = np.asarray(
                self.session.run([self.output_name], onnx_inputs)[0], dtype=np.float32
            )
            if output.ndim == 3:
                embeddings = output[:, 0, :]
            elif output.ndim == 2:
                embeddings = output
            else:
                raise ValueError(f"Unsupported ONNX embedding shape: {output.shape}")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = np.divide(
                embeddings,
                norms,
                out=np.zeros_like(embeddings),
                where=norms > 0,
            )
            yield from normalized
