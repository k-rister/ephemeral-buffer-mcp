"""Unit tests for the text-only ONNX embedding adapter."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ephemeral_buffer_mcp import embedding_backend as backend
from ephemeral_buffer_mcp.config import BGE_SMALL_HF_REPO, CATALOGUE_EMBEDDING_MODEL, FP32_EMBEDDING_MODEL


class FakeTokenizer:
    def __init__(self, padding=None, token_id=0):
        self.padding = padding
        self.token_id = token_id
        self.truncation = None
        self.special_tokens = []
        self.padding_options = None

    def enable_truncation(self, **kwargs):
        self.truncation = kwargs

    def add_special_tokens(self, tokens):
        self.special_tokens.extend(tokens)

    def token_to_id(self, _token):
        return self.token_id

    def enable_padding(self, **kwargs):
        self.padding_options = kwargs
        self.padding = kwargs

    def encode(self, text):
        return SimpleNamespace(ids=list(range(len(text))))

    def encode_batch(self, texts):
        return [
            SimpleNamespace(
                ids=[1, 2],
                attention_mask=[1, 1],
            )
            for _text in texts
        ]


class FakeSession:
    def __init__(self, input_names, output, outputs=None):
        self.inputs = [SimpleNamespace(name=name) for name in input_names]
        self.outputs = outputs if outputs is not None else [SimpleNamespace(name="vectors")]
        self.output = output
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, inputs):
        start = sum(len(call[1]["input_ids"]) for call in self.calls)
        self.calls.append((output_names, inputs))
        count = len(inputs["input_ids"])
        return [self.output[start : start + count]]


class TestEmbeddingBackend(unittest.TestCase):
    def test_repository_aliases_and_validation(self):
        self.assertEqual(backend._repository_id(FP32_EMBEDDING_MODEL), BGE_SMALL_HF_REPO)
        self.assertEqual(backend._repository_id(CATALOGUE_EMBEDDING_MODEL), BGE_SMALL_HF_REPO)
        with self.assertRaisesRegex(ValueError, "supports only"):
            backend._repository_id("org/model")
        with self.assertRaisesRegex(ValueError, "supports only"):
            backend._repository_id("invalid")
        with self.assertRaisesRegex(ValueError, "supports only"):
            backend._repository_id(None)

    def test_read_json_validates_files_and_document_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.json"
            good.write_text('{"key": "value"}', encoding="utf-8")
            self.assertEqual(backend._read_json(good, "metadata"), {"key": "value"})

            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not read"):
                backend._read_json(malformed, "metadata")
            with self.assertRaisesRegex(ValueError, "Could not read"):
                backend._read_json(root / "missing.json", "metadata")

            scalar = root / "scalar.json"
            scalar.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                backend._read_json(scalar, "metadata")

    def test_model_max_length_uses_smallest_valid_limit_or_default(self):
        self.assertEqual(
            backend._model_max_length(
                {"model_max_length": 128, "max_length": 256},
                {"max_position_embeddings": 512},
            ),
            128,
        )
        self.assertEqual(
            backend._model_max_length(
                {"model_max_length": True, "max_length": 1_000_000},
                {"max_position_embeddings": 96},
            ),
            96,
        )
        self.assertEqual(backend._model_max_length({}, {}), backend._DEFAULT_MAX_LENGTH)

    def test_token_content_accepts_strings_and_added_token_objects(self):
        self.assertEqual(backend._token_content("[PAD]"), "[PAD]")
        self.assertEqual(backend._token_content({"content": "[PAD]"}), "[PAD]")
        self.assertIsNone(backend._token_content({"content": 1}))
        self.assertIsNone(backend._token_content(None))

    def _write_tokenizer_metadata(self, directory, *, pad_token="[PAD]", pad_id=0):
        root = Path(directory)
        (root / "config.json").write_text(
            json.dumps({"max_position_embeddings": 512, "pad_token_id": pad_id}),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128, "pad_token": pad_token}),
            encoding="utf-8",
        )
        (root / "special_tokens_map.json").write_text(
            json.dumps({"cls_token": "[CLS]", "extra_token": {"content": "[EXTRA]"}}),
            encoding="utf-8",
        )
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    def test_load_tokenizer_adds_special_tokens_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_tokenizer_metadata(directory)
            tokenizer = FakeTokenizer()
            with (
                patch.object(backend, "Tokenizer", SimpleNamespace(from_file=lambda _path: tokenizer)),
                patch.object(backend, "AddedToken", side_effect=lambda **kwargs: kwargs),
            ):
                loaded = backend._load_tokenizer(Path(directory))

        self.assertIs(loaded, tokenizer)
        self.assertEqual(tokenizer.truncation, {"max_length": 128})
        self.assertEqual(tokenizer.special_tokens[0], "[CLS]")
        self.assertEqual(tokenizer.special_tokens[1], {"content": "[EXTRA]"})
        self.assertEqual(
            tokenizer.padding_options,
            {"pad_id": 0, "pad_token": "[PAD]"},
        )

    def test_load_tokenizer_keeps_existing_padding_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_tokenizer_metadata(directory)
            tokenizer = FakeTokenizer(padding={"pad_id": 0})
            with patch.object(
                backend, "Tokenizer", SimpleNamespace(from_file=lambda _path: tokenizer)
            ):
                self.assertIs(backend._load_tokenizer(Path(directory)), tokenizer)
        self.assertIsNone(tokenizer.padding_options)

    def test_load_tokenizer_reports_missing_padding_token_or_id(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_tokenizer_metadata(directory, pad_token=None, pad_id=None)
            with patch.object(
                backend,
                "Tokenizer",
                SimpleNamespace(from_file=lambda _path: FakeTokenizer(token_id=None)),
            ):
                with self.assertRaisesRegex(ValueError, "must define a padding token"):
                    backend._load_tokenizer(Path(directory))

            self._write_tokenizer_metadata(directory, pad_token="[PAD]", pad_id="invalid")
            with patch.object(
                backend,
                "Tokenizer",
                SimpleNamespace(from_file=lambda _path: FakeTokenizer(token_id=None)),
            ):
                with self.assertRaisesRegex(ValueError, "does not define an ID"):
                    backend._load_tokenizer(Path(directory))

    def test_load_tokenizer_wraps_tokenizer_file_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_tokenizer_metadata(directory)
            with patch.object(
                backend,
                "Tokenizer",
                SimpleNamespace(from_file=lambda _path: (_ for _ in ()).throw(ValueError("bad tokenizer"))),
            ):
                with self.assertRaisesRegex(ValueError, "Could not load tokenizer"):
                    backend._load_tokenizer(Path(directory))

    def _construct_embedding(self, directory, session, *, model_name=FP32_EMBEDDING_MODEL, **kwargs):
        options = SimpleNamespace()
        with (
            patch.object(backend, "snapshot_download", return_value=str(directory)) as download,
            patch.object(backend.ort, "SessionOptions", return_value=options),
            patch.object(backend.ort, "InferenceSession", return_value=session) as create_session,
            patch.object(backend, "_load_tokenizer", return_value=FakeTokenizer()) as load_tokenizer,
        ):
            model = backend.TextEmbedding(model_name=model_name, **kwargs)
        return model, download, options, create_session, load_tokenizer

    def test_model_load_pins_default_and_configures_session(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "onnx" / "model.onnx"
            model_path.parent.mkdir()
            model_path.touch()
            session = FakeSession(
                ["input_ids", "attention_mask", "token_type_ids"],
                np.zeros((1, 2, 3), dtype=np.float32),
            )
            model, download, options, create_session, load_tokenizer = self._construct_embedding(
                directory,
                session,
                cache_dir="/cache",
                threads=3,
                enable_cpu_mem_arena=True,
            )

        self.assertEqual(model.repository_id, BGE_SMALL_HF_REPO)
        self.assertEqual(download.call_args.kwargs["repo_id"], BGE_SMALL_HF_REPO)
        self.assertEqual(download.call_args.kwargs["revision"], backend._BGE_SMALL_REVISION)
        self.assertEqual(download.call_args.kwargs["cache_dir"], "/cache")
        self.assertEqual(options.intra_op_num_threads, 3)
        self.assertEqual(options.inter_op_num_threads, 3)
        self.assertTrue(options.enable_cpu_mem_arena)
        self.assertEqual(create_session.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(model.output_name, "vectors")
        self.assertIs(model.model.tokenizer, model.tokenizer)
        load_tokenizer.assert_called_once()

    def test_model_load_accepts_root_onnx_file_for_supported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.onnx").touch()
            session = FakeSession(["input_ids"], np.zeros((1, 3), dtype=np.float32))
            model, download, options, _session, _tokenizer = self._construct_embedding(directory, session)

        self.assertEqual(model.repository_id, BGE_SMALL_HF_REPO)
        self.assertEqual(download.call_args.kwargs["revision"], backend._BGE_SMALL_REVISION)
        self.assertEqual(options.graph_optimization_level, backend.ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.assertFalse(options.enable_cpu_mem_arena)

    def test_model_load_rejects_unverified_model_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "supports only"):
                self._construct_embedding(
                    directory,
                    FakeSession(["input_ids"], np.zeros((1, 2), dtype=np.float32)),
                    model_name="org/model",
                )

    def test_model_load_rejects_missing_model_files_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no supported ONNX model file"):
                self._construct_embedding(directory, FakeSession(["input_ids"], np.zeros((1, 2))))
            (Path(directory) / "model.onnx").touch()

            with self.assertRaisesRegex(ValueError, "must accept input_ids"):
                self._construct_embedding(directory, FakeSession(["attention_mask"], np.zeros((1, 2))))
            with self.assertRaisesRegex(ValueError, "may only require"):
                self._construct_embedding(
                    directory, FakeSession(["input_ids", "position_ids"], np.zeros((1, 2)))
                )
            with self.assertRaisesRegex(ValueError, "has no outputs"):
                self._construct_embedding(
                    directory,
                    FakeSession(["input_ids"], np.zeros((1, 2)), outputs=[]),
                )

    def test_embed_batches_cls_outputs_normalizes_and_handles_zero_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.onnx").touch()
            output = np.asarray(
                [
                    [[3.0, 4.0], [8.0, 9.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                ],
                dtype=np.float32,
            )
            session = FakeSession(["input_ids", "attention_mask", "token_type_ids"], output)
            model, *_ = self._construct_embedding(directory, session)

        vectors = list(model.embed(["one", "two"], batch_size=1))
        self.assertEqual(len(vectors), 2)
        np.testing.assert_allclose(vectors[0], [0.6, 0.8])
        np.testing.assert_allclose(vectors[1], [0.0, 0.0])
        self.assertEqual(len(session.calls), 2)
        self.assertIn("token_type_ids", session.calls[0][1])
        self.assertEqual(session.calls[0][1]["token_type_ids"].dtype, np.int64)

    def test_embed_accepts_sentence_vectors_and_optional_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.onnx").touch()
            session = FakeSession(["input_ids"], np.asarray([[3.0, 4.0]], dtype=np.float32))
            model, *_ = self._construct_embedding(directory, session)

        vectors = list(model.embed("text"))
        np.testing.assert_allclose(vectors[0], [0.6, 0.8])
        self.assertEqual(set(session.calls[0][1]), {"input_ids"})

    def test_embed_rejects_invalid_batches_and_output_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.onnx").touch()
            session = FakeSession(["input_ids"], np.zeros((1, 1, 1, 1), dtype=np.float32))
            model, *_ = self._construct_embedding(directory, session)

        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            list(model.embed([], batch_size=0))
        self.assertEqual(list(model.embed([])), [])
        with self.assertRaisesRegex(ValueError, "Unsupported ONNX embedding shape"):
            list(model.embed(["text"]))


if __name__ == "__main__":
    unittest.main()
