"""Tests for environment-backed configuration."""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SOCKET_PATH,
    DEFAULT_SEMANTIC_PREFETCH_WORKERS,
    embedding_cache_dir,
    embedding_model_name,
    positive_int_env,
    semantic_prefetch_enabled,
    semantic_prefetch_workers,
    socket_isolation_configured,
    socket_isolation_required,
    socket_path,
    socket_timeout_seconds,
    max_indexed_chunks,
)


class TestPositiveIntEnv(unittest.TestCase):
    def test_missing_value_uses_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 25)

    def test_valid_value_is_used(self):
        with patch.dict(os.environ, {"TEST_LIMIT": "128"}, clear=True):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 128)

    def test_invalid_value_warns_and_uses_default(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, {"TEST_LIMIT": "0"}, clear=True), redirect_stderr(stderr):
            self.assertEqual(positive_int_env("TEST_LIMIT", 25), 25)
        self.assertIn("Ignoring invalid TEST_LIMIT", stderr.getvalue())

    def test_socket_path_can_be_overridden(self):
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/test-ephemeral.sock"}):
            self.assertEqual(socket_path(), "/tmp/test-ephemeral.sock")

    def test_session_id_derives_stable_socket_path(self):
        with patch.dict(os.environ, {"EPHEMERAL_SESSION_ID": "agent-session-1"}, clear=True):
            first_path = socket_path()
            second_path = socket_path()

        self.assertEqual(first_path, second_path)
        self.assertIn("ephemeral_buffer-", first_path)
        self.assertTrue(first_path.endswith(".sock"))
        self.assertNotEqual(first_path, DEFAULT_SOCKET_PATH)

    def test_explicit_socket_path_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_SOCKET_PATH": "/tmp/explicit.sock",
                "EPHEMERAL_SESSION_ID": "agent-session-1",
            },
            clear=True,
        ):
            self.assertEqual(socket_path(), "/tmp/explicit.sock")
        self.assertTrue(DEFAULT_SOCKET_PATH)

    def test_socket_isolation_defaults_to_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(socket_isolation_configured())
            self.assertFalse(socket_isolation_required())

    def test_socket_isolation_can_be_required(self):
        with patch.dict(os.environ, {"EPHEMERAL_REQUIRE_ISOLATION": "true"}, clear=True):
            self.assertTrue(socket_isolation_required())

    def test_session_or_explicit_path_configures_isolation(self):
        with patch.dict(os.environ, {"EPHEMERAL_SESSION_ID": "session-1"}, clear=True):
            self.assertTrue(socket_isolation_configured())
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/eph.sock"}, clear=True):
            self.assertTrue(socket_isolation_configured())

    def test_socket_timeout_defaults_and_accepts_positive_float(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(socket_timeout_seconds(), 10.0)
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_TIMEOUT_SECONDS": "2.5"}, clear=True):
            self.assertEqual(socket_timeout_seconds(), 2.5)

    def test_socket_timeout_rejects_invalid_values(self):
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_TIMEOUT_SECONDS": "0"}, clear=True):
            self.assertEqual(socket_timeout_seconds(), 10.0)

    def test_max_indexed_chunks_defaults_and_reads_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(max_indexed_chunks(), 32768)
        with patch.dict(os.environ, {"EPHEMERAL_MAX_INDEXED_CHUNKS": "12"}, clear=True):
            self.assertEqual(max_indexed_chunks(), 12)

    def test_embedding_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(embedding_model_name(), DEFAULT_EMBEDDING_MODEL)
            self.assertIsNone(embedding_cache_dir())

    def test_embedding_settings_can_be_overridden(self):
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_EMBEDDING_MODEL": "custom/model",
                "EPHEMERAL_FASTEMBED_CACHE_DIR": "/tmp/fastembed",
            },
            clear=True,
        ):
            self.assertEqual(embedding_model_name(), "custom/model")
            self.assertEqual(embedding_cache_dir(), "/tmp/fastembed")

    def test_semantic_prefetch_defaults_disabled_and_one_worker(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(semantic_prefetch_enabled())
            self.assertEqual(semantic_prefetch_workers(), DEFAULT_SEMANTIC_PREFETCH_WORKERS)

    def test_semantic_prefetch_settings_can_be_overridden(self):
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_SEMANTIC_PREFETCH": "true",
                "EPHEMERAL_SEMANTIC_PREFETCH_WORKERS": "2",
            },
            clear=True,
        ):
            self.assertTrue(semantic_prefetch_enabled())
            self.assertEqual(semantic_prefetch_workers(), 2)


if __name__ == "__main__":
    unittest.main()
