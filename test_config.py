"""Tests for environment-backed configuration."""

import io
import math
import os
import runpy
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import config
from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXECUTION_CHECKPOINT_RESERVE_BYTES,
    DEFAULT_EXECUTION_STATE_DIR,
    DEFAULT_EXECUTION_STATE_QUOTA_BYTES,
    DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS,
    DEFAULT_MAX_ACTIVE_TOOL_WORK,
    DEFAULT_MAX_QUEUED_SOCKET_CLIENTS,
    DEFAULT_MAX_QUEUED_TOOL_WORK,
    DEFAULT_SOCKET_PATH,
    DEFAULT_SEMANTIC_PREFETCH_WORKERS,
    DEFAULT_SEMANTIC_WAIT_SECONDS,
    cleanup_default_execution_state_dir,
    codex_mcp_env_config,
    embedding_batch_size,
    embedding_cache_dir,
    embedding_cpu_mem_arena_enabled,
    embedding_max_batch_tokens,
    embedding_model_name,
    embedding_threads,
    embedding_warmup_enabled,
    execution_checkpoint_reserve_bytes,
    execution_state_quota_bytes,
    execution_state_dir,
    optional_positive_int_env,
    positive_int_env,
    runtime_index_budget_adjustment_enabled,
    semantic_prefetch_enabled,
    semantic_prefetch_workers,
    semantic_wait_seconds,
    socket_isolation_configured,
    socket_isolation_required,
    socket_path,
    socket_timeout_seconds,
    max_indexed_chunks,
    max_active_socket_clients,
    max_active_tool_work,
    max_queued_socket_clients,
    max_queued_tool_work,
    load_settings,
    non_negative_int_env,
    semantic_chunk_bytes,
    semantic_chunk_lines,
    semantic_chunk_overlap,
    semantic_max_index_input_bytes,
)


class TestPositiveIntEnv(unittest.TestCase):
    def test_admission_limits_default_override_and_invalid_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(max_active_tool_work(), DEFAULT_MAX_ACTIVE_TOOL_WORK)
            self.assertEqual(max_queued_tool_work(), DEFAULT_MAX_QUEUED_TOOL_WORK)
            self.assertEqual(max_active_socket_clients(), DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS)
            self.assertEqual(max_queued_socket_clients(), DEFAULT_MAX_QUEUED_SOCKET_CLIENTS)

        with patch.dict(os.environ, {
            "EPHEMERAL_MAX_ACTIVE_TOOL_WORK": "3",
            "EPHEMERAL_MAX_QUEUED_TOOL_WORK": "0",
            "EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS": "2",
            "EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS": "5",
        }, clear=True):
            self.assertEqual(max_active_tool_work(), 3)
            self.assertEqual(max_queued_tool_work(), 0)
            self.assertEqual(max_active_socket_clients(), 2)
            self.assertEqual(max_queued_socket_clients(), 5)

        stderr = io.StringIO()
        with patch.dict(os.environ, {
            "EPHEMERAL_MAX_ACTIVE_TOOL_WORK": "0",
            "EPHEMERAL_MAX_QUEUED_TOOL_WORK": "-1",
            "EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS": "invalid",
            "EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS": "-2",
        }, clear=True), redirect_stderr(stderr):
            self.assertEqual(max_active_tool_work(), DEFAULT_MAX_ACTIVE_TOOL_WORK)
            self.assertEqual(max_queued_tool_work(), DEFAULT_MAX_QUEUED_TOOL_WORK)
            self.assertEqual(max_active_socket_clients(), DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS)
            self.assertEqual(max_queued_socket_clients(), DEFAULT_MAX_QUEUED_SOCKET_CLIENTS)
        self.assertIn("EPHEMERAL_MAX_ACTIVE_TOOL_WORK", stderr.getvalue())
        self.assertIn("EPHEMERAL_MAX_QUEUED_TOOL_WORK", stderr.getvalue())
        self.assertIn("EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS", stderr.getvalue())
        self.assertIn("EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS", stderr.getvalue())

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

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(socket_path(), DEFAULT_SOCKET_PATH)

    def test_execution_state_directory_defaults_and_can_be_overridden(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(execution_state_dir(), DEFAULT_EXECUTION_STATE_DIR)
        with patch.dict(os.environ, {"EPHEMERAL_EXECUTION_STATE_DIR": "/tmp/executions"}, clear=True):
            self.assertEqual(execution_state_dir(), "/tmp/executions")
        with patch.dict(os.environ, {"EPHEMERAL_SESSION_ID": "session-1"}, clear=True):
            self.assertIn("ephemeral_buffer_executions-", execution_state_dir())
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/session-a.sock"}, clear=True):
            first = execution_state_dir()
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/session-b.sock"}, clear=True):
            second = execution_state_dir()
        self.assertIn("ephemeral_buffer_executions-socket-", first)
        self.assertNotEqual(first, second)

    def test_execution_storage_limits_default_and_can_be_overridden(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(execution_state_quota_bytes(), DEFAULT_EXECUTION_STATE_QUOTA_BYTES)
            self.assertEqual(
                execution_checkpoint_reserve_bytes(),
                DEFAULT_EXECUTION_CHECKPOINT_RESERVE_BYTES,
            )
        with patch.dict(os.environ, {
            "EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES": "8589934592",
            "EPHEMERAL_EXECUTION_CHECKPOINT_RESERVE_BYTES": "268435456",
        }, clear=True):
            self.assertEqual(execution_state_quota_bytes(), 8 * 1024 * 1024 * 1024)
            self.assertEqual(execution_checkpoint_reserve_bytes(), 256 * 1024 * 1024)

    def test_default_execution_state_cleanup_only_removes_private_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ephemeral_buffer_executions-test")
            os.mkdir(path, 0o700)
            with patch("config.DEFAULT_EXECUTION_STATE_DIR", path), \
                    patch("config.tempfile.gettempdir", return_value=directory), \
                    patch("config.os.getuid", return_value=os.stat(path).st_uid):
                cleanup_default_execution_state_dir()
            self.assertFalse(os.path.exists(path))

    def test_default_execution_state_cleanup_rejects_nonprivate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ephemeral_buffer_executions-test")
            os.mkdir(path, 0o755)
            with patch("config.DEFAULT_EXECUTION_STATE_DIR", path), \
                    patch("config.tempfile.gettempdir", return_value=directory):
                cleanup_default_execution_state_dir()
            self.assertTrue(os.path.isdir(path))

    def test_default_execution_state_cleanup_ignores_unexpected_path_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("config.DEFAULT_EXECUTION_STATE_DIR", os.path.join(directory, "other")), \
                    patch("config.tempfile.gettempdir", return_value=directory):
                cleanup_default_execution_state_dir()
            path = os.path.join(directory, "ephemeral_buffer_executions-test")
            with patch("config.DEFAULT_EXECUTION_STATE_DIR", path), \
                    patch("config.tempfile.gettempdir", return_value=directory), \
                    patch("config.os.stat", side_effect=OSError("gone")):
                cleanup_default_execution_state_dir()

    def test_default_execution_state_cleanup_handles_platform_without_getuid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ephemeral_buffer_executions-test")
            os.mkdir(path, 0o700)
            with patch("config.DEFAULT_EXECUTION_STATE_DIR", path), \
                    patch("config.tempfile.gettempdir", return_value=directory), \
                    patch.object(config.os, "getuid", None):
                cleanup_default_execution_state_dir()
            self.assertTrue(os.path.isdir(path))

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

    def test_explicit_socket_path_is_the_default_state_identity(self):
        environment = {
            "EPHEMERAL_SOCKET_PATH": "/tmp/explicit.sock",
            "EPHEMERAL_SESSION_ID": "agent-session-1",
        }
        with patch.object(config.tempfile, "gettempdir", return_value="/tmp"):
            descriptor = config.resolve_session_descriptor(environment)
            self.assertEqual(descriptor.socket_path, "/tmp/explicit.sock")
            self.assertEqual(descriptor.state_source, "socket:EPHEMERAL_SOCKET_PATH")
            self.assertIn("ephemeral_buffer_executions-socket-", descriptor.state_dir)
            self.assertIn("ephemeral_buffer_executions-", descriptor.legacy_state_dir)

            relative_descriptor = config.resolve_session_descriptor({
                "EPHEMERAL_SOCKET_PATH": "relative.sock",
            })
            self.assertEqual(
                relative_descriptor.legacy_state_dir,
                os.path.join(
                    "/tmp",
                    f"{config.EXECUTION_STATE_SOCKET_PREFIX}"
                    f"{config._identity_digest('relative.sock')}",
                ),
            )

    def test_explicit_state_directory_overrides_resolved_identity(self):
        descriptor = config.resolve_session_descriptor({
            "EPHEMERAL_SOCKET_PATH": "/tmp/explicit.sock",
            "EPHEMERAL_SESSION_ID": "agent-session-1",
            "EPHEMERAL_EXECUTION_STATE_DIR": "/var/tmp/continued-state",
        })
        self.assertEqual(descriptor.state_dir, "/var/tmp/continued-state")
        self.assertEqual(descriptor.state_source, "environment:EPHEMERAL_EXECUTION_STATE_DIR")
        self.assertIsNone(descriptor.legacy_state_dir)

    def test_existing_session_state_reports_explicit_continuation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "EPHEMERAL_SOCKET_PATH": os.path.join(directory, "agent.sock"),
                "EPHEMERAL_SESSION_ID": "legacy-session-1",
            }
            with patch.object(config.tempfile, "gettempdir", return_value=directory):
                legacy_state = config.resolve_session_descriptor(environment).legacy_state_dir
                os.mkdir(legacy_state)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    settings = load_settings(environment)
            self.assertTrue(settings.identity.legacy_state_transition)
            self.assertIn(legacy_state, stderr.getvalue())
            self.assertIn(settings.identity.state_dir, stderr.getvalue())
            self.assertIn("EPHEMERAL_EXECUTION_STATE_DIR", stderr.getvalue())

    def test_settings_snapshot_records_origins_and_invalid_boolean_fallback(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            settings = load_settings({
                "EPHEMERAL_MAX_ACTIVE_TOOL_WORK": "3",
                "EPHEMERAL_SEMANTIC_PREFETCH": "perhaps",
            }, warn_on_legacy_state_transition=False)

        self.assertEqual(settings.max_active_tool_work.value, 3)
        self.assertEqual(settings.max_active_tool_work.source, "environment")
        self.assertEqual(settings.semantic_prefetch_enabled.value, True)
        self.assertEqual(settings.semantic_prefetch_enabled.status, "invalid-fallback")
        self.assertIn("Ignoring invalid EPHEMERAL_SEMANTIC_PREFETCH", stderr.getvalue())
        empty_boolean = load_settings(
            {"EPHEMERAL_SEMANTIC_PREFETCH": ""},
            warn_on_legacy_state_transition=False,
        ).semantic_prefetch_enabled
        self.assertTrue(empty_boolean.value)
        self.assertEqual(empty_boolean.status, "empty-default")
        diagnostics = settings.diagnostics()
        self.assertEqual(
            diagnostics["settings"]["runtime_index_budget_adjustment_enabled"]["sampling_policy"],
            "each MCP request; environment value is re-read; reported value is from startup",
        )

    def test_setting_diagnostics_encode_nonfinite_float_values(self):
        for value, expected in (
            (math.inf, "inf"),
            (-math.inf, "-inf"),
            (math.nan, "nan"),
        ):
            with self.subTest(value=expected):
                setting = config.ConfigSetting("TEST_FLOAT", value, "environment", "accepted")
                self.assertEqual(setting.diagnostics()["value"], expected)

    def test_normalized_relative_socket_preserves_existing_legacy_state(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_socket = "relative.sock"
            legacy_state = os.path.join(
                directory,
                f"{config.EXECUTION_STATE_SOCKET_PREFIX}{config._identity_digest(raw_socket)}",
            )
            os.mkdir(legacy_state)
            with patch.object(config.tempfile, "gettempdir", return_value=directory):
                paths = config.normalized_explicit_identity_paths({
                    "EPHEMERAL_SOCKET_PATH": raw_socket,
                })

        self.assertEqual(paths["EPHEMERAL_SOCKET_PATH"], os.path.abspath(raw_socket))
        self.assertEqual(paths["EPHEMERAL_LEGACY_STATE_DIR"], legacy_state)

    def test_launcher_configuration_main_dispatches_and_rejects_unknown_arguments(self):
        output = io.StringIO()
        with patch.object(config.sys, "argv", ["config.py", "--socket-path"]), \
                patch.object(
                    config, "resolve_session_descriptor",
                    return_value=SimpleNamespace(socket_path="/tmp/config.sock"),
                ), redirect_stdout(output):
            config.main()
        self.assertEqual(output.getvalue(), "/tmp/config.sock\n")

        output = io.StringIO()
        with patch.object(config.sys, "argv", ["config.py", "--codex-env-config"]), \
                patch.object(config, "codex_mcp_env_config", return_value="mcp config"), \
                redirect_stdout(output):
            config.main()
        self.assertEqual(output.getvalue(), "mcp config\n")

        output = io.StringIO()
        with patch.object(config.sys, "argv", ["config.py", "--normalize-explicit-paths"]), \
                patch.object(
                    config, "normalized_explicit_identity_paths",
                    return_value={"EPHEMERAL_SOCKET_PATH": "/tmp/config.sock"},
                ), redirect_stdout(output):
            config.main()
        self.assertEqual(
            output.getvalue(),
            '{"EPHEMERAL_SOCKET_PATH": "/tmp/config.sock"}\n',
        )

        with patch.object(config.sys, "argv", ["config.py", "--unknown"]):
            with self.assertRaisesRegex(SystemExit, "Usage: config.py"):
                config.main()

    def test_config_script_entrypoint_calls_main(self):
        output = io.StringIO()
        with patch.object(config.sys, "argv", [config.__file__, "--socket-path"]), \
                redirect_stdout(output):
            runpy.run_path(config.__file__, run_name="__main__")
        self.assertTrue(output.getvalue().strip().endswith(".sock"))

    def test_codex_environment_config_forwards_identity_and_escapes_strings(self):
        rendered = codex_mcp_env_config({
            "EPHEMERAL_SESSION_ID": "session-1",
            "EPHEMERAL_SOCKET_PATH": '/tmp/a"b.sock',
            "EPHEMERAL_EXECUTION_STATE_DIR": "/tmp/state",
            "EPHEMERAL_SEMANTIC_PREFETCH": "0",
        })
        self.assertIn('EPHEMERAL_SESSION_ID="session-1"', rendered)
        self.assertIn('EPHEMERAL_SOCKET_PATH="/tmp/a\\"b.sock"', rendered)
        self.assertIn('EPHEMERAL_EXECUTION_STATE_DIR="/tmp/state"', rendered)
        self.assertIn('EPHEMERAL_SEMANTIC_PREFETCH="0"', rendered)
        relative_rendered = codex_mcp_env_config({
            "EPHEMERAL_SOCKET_PATH": "relative.sock",
            "EPHEMERAL_EXECUTION_STATE_DIR": "relative-state",
        })
        self.assertIn(f'EPHEMERAL_SOCKET_PATH="{os.path.abspath("relative.sock")}"', relative_rendered)
        self.assertIn(
            f'EPHEMERAL_EXECUTION_STATE_DIR="{os.path.abspath("relative-state")}"',
            relative_rendered,
        )

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

    def test_runtime_index_budget_adjustment_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(runtime_index_budget_adjustment_enabled())
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET": "yes"}, clear=True):
            self.assertTrue(runtime_index_budget_adjustment_enabled())

    def test_embedding_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(embedding_model_name(), DEFAULT_EMBEDDING_MODEL)
            self.assertEqual(DEFAULT_EMBEDDING_MODEL, config.FP32_EMBEDDING_MODEL)
            self.assertEqual(config.CATALOGUE_EMBEDDING_MODEL, config.BGE_SMALL_HF_REPO)
            self.assertIsNone(embedding_cache_dir())
            self.assertIsNone(embedding_threads())

    def test_embedding_threads_accepts_positive_integers_only(self):
        with patch.dict(os.environ, {"EPHEMERAL_EMBEDDING_THREADS": "4"}, clear=True):
            self.assertEqual(embedding_threads(), 4)
        with patch.dict(os.environ, {"EPHEMERAL_EMBEDDING_THREADS": " "}, clear=True):
            self.assertIsNone(embedding_threads())
        for invalid in ("0", "-2", "many"):
            with patch.dict(os.environ, {"EPHEMERAL_EMBEDDING_THREADS": invalid}, clear=True):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertIsNone(optional_positive_int_env("EPHEMERAL_EMBEDDING_THREADS"))
                self.assertIn("Ignoring invalid EPHEMERAL_EMBEDDING_THREADS", stderr.getvalue())

    def test_embedding_memory_settings_defaults_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(embedding_batch_size(), config.DEFAULT_EMBEDDING_BATCH_SIZE)
            self.assertEqual(embedding_max_batch_tokens(), config.DEFAULT_EMBEDDING_MAX_BATCH_TOKENS)
            self.assertFalse(embedding_cpu_mem_arena_enabled())
            self.assertEqual(
                semantic_max_index_input_bytes(),
                config.DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES,
            )
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_EMBEDDING_BATCH_SIZE": "8",
                "EPHEMERAL_EMBEDDING_MAX_BATCH_TOKENS": "2048",
                "EPHEMERAL_EMBEDDING_CPU_MEM_ARENA": "yes",
                "EPHEMERAL_SEMANTIC_MAX_INDEX_INPUT_BYTES": "1024",
            },
            clear=True,
        ):
            self.assertEqual(embedding_batch_size(), 8)
            self.assertEqual(embedding_max_batch_tokens(), 2048)
            self.assertTrue(embedding_cpu_mem_arena_enabled())
            self.assertEqual(semantic_max_index_input_bytes(), 1024)

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

    def test_semantic_prefetch_defaults_enabled_and_one_worker(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(semantic_prefetch_enabled())
            self.assertEqual(semantic_prefetch_workers(), DEFAULT_SEMANTIC_PREFETCH_WORKERS)
        with patch.dict(os.environ, {"EPHEMERAL_SEMANTIC_PREFETCH": "0"}, clear=True):
            self.assertFalse(semantic_prefetch_enabled())

    def test_semantic_wait_seconds_defaults_and_accepts_zero_and_inf(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(semantic_wait_seconds(), DEFAULT_SEMANTIC_WAIT_SECONDS)
        for value, expected in (("0", 0.0), ("2.5", 2.5), ("inf", float("inf"))):
            with patch.dict(os.environ, {"EPHEMERAL_SEMANTIC_WAIT_SECONDS": value}, clear=True):
                self.assertEqual(semantic_wait_seconds(), expected)
        for invalid in ("-1", "nan", "soon"):
            with patch.dict(os.environ, {"EPHEMERAL_SEMANTIC_WAIT_SECONDS": invalid}, clear=True):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(semantic_wait_seconds(), DEFAULT_SEMANTIC_WAIT_SECONDS)
                self.assertIn("Ignoring invalid EPHEMERAL_SEMANTIC_WAIT_SECONDS", stderr.getvalue())

    def test_embedding_warmup_defaults_enabled_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(embedding_warmup_enabled())
        with patch.dict(os.environ, {"EPHEMERAL_EMBEDDING_WARMUP": "off"}, clear=True):
            self.assertFalse(embedding_warmup_enabled())

    def test_semantic_chunk_settings_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(semantic_chunk_lines(), config.DEFAULT_SEMANTIC_CHUNK_LINES)
            self.assertEqual(semantic_chunk_bytes(), config.DEFAULT_SEMANTIC_CHUNK_BYTES)
            self.assertEqual(semantic_chunk_overlap(), config.DEFAULT_SEMANTIC_CHUNK_OVERLAP)
        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_SEMANTIC_CHUNK_LINES": "12",
                "EPHEMERAL_SEMANTIC_CHUNK_BYTES": "2048",
                "EPHEMERAL_SEMANTIC_CHUNK_OVERLAP": "0",
            },
            clear=True,
        ):
            self.assertEqual((semantic_chunk_lines(), semantic_chunk_bytes(), semantic_chunk_overlap()), (12, 2048, 0))
        for invalid in ("-1", "two"):
            with patch.dict(os.environ, {"EPHEMERAL_SEMANTIC_CHUNK_OVERLAP": invalid}, clear=True):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(non_negative_int_env("EPHEMERAL_SEMANTIC_CHUNK_OVERLAP", 3), 3)
                self.assertIn("Ignoring invalid EPHEMERAL_SEMANTIC_CHUNK_OVERLAP", stderr.getvalue())

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
