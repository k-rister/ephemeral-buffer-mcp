"""Unit tests for MCP tool validation, response formatting, and socket IPC."""

import asyncio
import io
import json
import math
import os
import runpy
import socket
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from metrics import LocalMetrics
from socket_protocol import FRAME_HEADER_SIZE, FRAME_MAGIC, decode_header, encode_frame

os.environ.setdefault("EPHEMERAL_DISABLE_SOCKET_SERVER", "1")
import server


class FakeReader:
    def __init__(self, payload):
        self.payload = bytearray(payload)

    async def read(self, limit):
        chunk = bytes(self.payload[:limit])
        del self.payload[:limit]
        return chunk


class ChunkedReader:
    def __init__(self, *chunks):
        self.chunks = iter(chunks)
        self.pending = bytearray()

    async def read(self, limit):
        while not self.pending:
            try:
                self.pending.extend(next(self.chunks))
            except StopIteration:
                return b""
        chunk = bytes(self.pending[:limit])
        del self.pending[:limit]
        return chunk


class FakeWriter:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, payload):
        self.writes.append(payload)

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


def response_json(writer):
    framed = writer.writes[0]
    length = decode_header(framed[:FRAME_HEADER_SIZE])
    return json.loads(framed[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + length])


class TestServerTools(unittest.TestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes
        self.original_index_limit = server.engine.max_indexed_chunks
        self.original_model = server.engine.embedding_model
        server.engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[0.0] * 384 for _ in texts]},
        )()

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
        server.engine.max_indexed_chunks = self.original_index_limit
        server.engine.embedding_model = self.original_model
        server.engine.clear("all")

    def test_tool_instrumentation_logs_content_free_lifecycle(self):
        with patch.object(server, "log_event") as log:
            @server._instrument_tool("probe_tool")
            def successful_tool(query):
                return f"result for {query}"

            self.assertEqual(successful_tool("secret query"), "result for secret query")
            self.assertEqual(
                [call.args[2] for call in log.call_args_list],
                ["mcp_tool_started", "mcp_tool_completed"],
            )
            completed = log.call_args_list[1].kwargs
            self.assertEqual(completed["tool"], "probe_tool")
            self.assertTrue(completed["success"])
            self.assertIsInstance(completed["duration_ms"], float)
            self.assertNotIn("secret query", str(log.call_args_list))

            log.reset_mock()

            @server._instrument_tool("failing_tool")
            def failing_tool():
                raise RuntimeError("private failure")

            with self.assertRaisesRegex(RuntimeError, "private failure"):
                failing_tool()
            self.assertEqual(
                [call.args[2] for call in log.call_args_list],
                ["mcp_tool_started", "mcp_tool_failed"],
            )
            failed = log.call_args_list[1].kwargs
            self.assertEqual(failed["error_type"], "RuntimeError")
            self.assertNotIn("private failure", str(log.call_args_list))

    def test_registered_mcp_tools_use_async_worker_adapter(self):
        async def exercise():
            @server._mcp_tool("blocking_probe")
            @server._instrument_tool("blocking_probe")
            def blocking_probe():
                return "worker result"

            tool = server.mcp._tool_manager._tools["blocking_probe"].fn
            offload = AsyncMock(return_value="worker result")
            with patch.object(server, "to_thread", offload):
                result = await tool()

            self.assertEqual(result, "worker result")
            offload.assert_awaited_once_with(blocking_probe)

        asyncio.run(exercise())

    def test_tool_descriptions_include_agent_routing_and_path_guidance(self):
        capture_text_doc = server.capture_text.__doc__
        capture_file_doc = server.capture_file.__doc__
        execute_doc = server.execute_and_capture.__doc__
        preflight_doc = server.preflight_command.__doc__

        self.assertIn("already-collected text", capture_text_doc)
        self.assertIn("resolve symlinks", capture_file_doc)
        self.assertIn("small, targeted inspection", execute_doc)
        self.assertIn("omitted ``cwd``", execute_doc)
        self.assertIn("inherits the server process directory", execute_doc)
        self.assertIn("filesystem safety", execute_doc)
        self.assertIn("never executed", preflight_doc)
        self.assertIn("Shell expansion", preflight_doc)

    def test_preflight_reports_repo_executable_and_does_not_run_command(self):
        result = json.loads(server.preflight_command("printf 'secret output'", cwd=os.getcwd()))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["working_directory"]["status"], "ok")
        self.assertEqual(result["working_directory"]["source"], "explicit")
        self.assertEqual(result["repository"]["status"], "detected")
        self.assertEqual(result["command"]["executable"]["requested"], "printf")
        self.assertNotIn("secret output", json.dumps(result))
        self.assertIn("not executed", result["limitations"][0])

    def test_preflight_reports_omitted_missing_and_symlinked_cwds(self):
        omitted = json.loads(server.preflight_command("echo hello"))
        self.assertEqual(omitted["working_directory"]["source"], "process-cwd")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            missing = root / "missing"
            dangling = root / "dangling"
            try:
                link.symlink_to(target, target_is_directory=True)
                dangling.symlink_to(root / "gone", target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable on this platform")

            linked = json.loads(server.preflight_command("echo hello", cwd=str(link)))
            self.assertTrue(linked["working_directory"]["is_symlink"])
            self.assertEqual(linked["working_directory"]["status"], "ok")
            self.assertEqual(linked["working_directory"]["symlink_target"], str(target.resolve()))

            missing_result = json.loads(server.preflight_command("echo hello", cwd=str(missing)))
            self.assertEqual(missing_result["working_directory"]["status"], "missing")
            self.assertEqual(missing_result["repository"]["status"], "unavailable")

            dangling_result = json.loads(server.preflight_command("echo hello", cwd=str(dangling)))
            self.assertEqual(dangling_result["working_directory"]["status"], "dangling-symlink")

    def test_preflight_distinguishes_unavailable_command_resolution(self):
        missing = json.loads(server.preflight_command("definitely-not-a-real-command"))
        self.assertEqual(missing["command"]["executable"]["status"], "unavailable")

        malformed = json.loads(server.preflight_command("'unterminated"))
        self.assertIn("unavailable", malformed["command"]["parse_status"])
        self.assertEqual(malformed["command"]["executable"]["status"], "unavailable")

        absolute = json.loads(server.preflight_command(f"{sys.executable} --version"))
        self.assertEqual(absolute["command"]["executable"]["status"], "resolved")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "tool"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            relative = json.loads(server.preflight_command("./tool", cwd=str(root)))
            self.assertEqual(relative["command"]["executable"]["status"], "resolved")
            unavailable = json.loads(server.preflight_command("./missing", cwd=str(root)))
            self.assertEqual(unavailable["command"]["executable"]["status"], "unavailable")

            regular_file = root / "regular-file"
            regular_file.write_text("not a directory", encoding="utf-8")
            not_directory = json.loads(server.preflight_command("echo hello", cwd=str(regular_file)))
            self.assertEqual(not_directory["working_directory"]["status"], "not-a-directory")

    def test_preflight_reports_repository_probe_and_unexpected_failures(self):
        with patch.object(server.subprocess, "run", side_effect=OSError("git unavailable")):
            unavailable = json.loads(server.preflight_command("echo hello", cwd=os.getcwd()))
        self.assertEqual(unavailable["repository"]["status"], "unavailable")
        self.assertEqual(unavailable["repository"]["reason"], "OSError")

        with patch.object(server, "_resolve_preflight_cwd", side_effect=RuntimeError("unexpected")):
            error = json.loads(server.preflight_command("echo hello"))
        self.assertEqual(error, {"reason": "RuntimeError", "status": "error"})

    def test_capture_file_reports_missing_path(self):
        result = server.capture_file("/does/not/exist")

        self.assertIn("does not exist", result)

    def test_capture_file_rejects_limit_above_buffer_budget(self):
        server.engine.max_buffer_bytes = 16
        with tempfile.NamedTemporaryFile() as file_handle:
            result = server.capture_file(file_handle.name, max_bytes=17)

        self.assertIn("exceeds the configured buffer limit", result)

    def test_capture_file_rejects_non_positive_limit(self):
        with tempfile.NamedTemporaryFile() as file_handle:
            result = server.capture_file(file_handle.name, max_bytes=0)

        self.assertIn("max_bytes must be at least 1", result)

    def test_capture_file_reports_read_failure(self):
        with tempfile.NamedTemporaryFile() as file_handle, \
                patch.object(server, "read_file_bounded", side_effect=OSError("permission denied")):
            result = server.capture_file(file_handle.name)

        self.assertIn("Error reading file", result)
        self.assertIn("permission denied", result)

    def test_execute_rejects_limit_above_buffer_budget(self):
        server.engine.max_buffer_bytes = 1024

        result = server.execute_and_capture("printf output", max_output_bytes=1025)

        self.assertIn("max_output_bytes", result)
        self.assertIn("exceeds the configured buffer limit", result)

    def test_execute_rejects_too_small_limit(self):
        result = server.execute_and_capture("printf output", max_output_bytes=100)

        self.assertIn("at least 512", result)

    def test_execute_reports_command_failure(self):
        with patch.object(server, "run_command_bounded", side_effect=OSError("unable to start command")):
            result = server.execute_and_capture("missing-command")

        self.assertIn("Error executing command", result)
        self.assertIn("unable to start command", result)

    def test_execute_reports_timeout(self):
        with patch.object(
            server,
            "run_command_bounded",
            return_value=("partial output", 124, False, 14, True),
        ) as run:
            result = server.execute_and_capture("sleep 10", timeout_seconds=0.5)

        summary = json.loads(result)
        self.assertEqual(summary["status"], "timed_out")
        self.assertTrue(summary["partial"])
        self.assertTrue(summary["timed_out"])
        run.assert_called_once()

    def test_consolidate_captures_preserves_sources_and_is_searchable(self):
        server.capture_text("alpha failure\nalpha detail", label="repo-a")
        server.capture_text("beta success\nbeta detail", label="repo-b")
        source_ids = [item["capture_id"] for item in reversed(server.engine.list_captures())]

        result = json.loads(server.consolidate_captures(source_ids, max_bytes=2048))

        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["source_capture_ids"], source_ids)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(result["record_count"], 4)
        self.assertIn("estimated_tokens", result)
        self.assertIn("retrieval", result)
        self.assertIn(source_ids[0], server.search_capture("alpha", capture_id=result["capture_id"]))
        self.assertIn("source_line", server.get_capture_slice(1, 100, capture_id=result["capture_id"]))

    def test_consolidate_captures_reports_omitted_records_and_missing_ids(self):
        server.capture_text("important failure " * 20, label="large")
        source_id = server.engine.list_captures()[0]["capture_id"]

        result = json.loads(server.consolidate_captures(
            [source_id, "missing"], max_bytes=512, max_captures=2,
        ))

        self.assertEqual(result["selected_capture_count"], 2)
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["missing_capture_ids"], ["missing"])
        self.assertGreater(result["omitted_record_count"], 0)
        self.assertEqual(result["byte_size"] <= 512, True)

    def test_consolidation_rejects_when_sources_cannot_be_retained(self):
        original_max_captures = server.engine.max_captures
        server.engine.max_captures = 1
        try:
            server.capture_text("important failure " * 20, label="source")
            source_id = server.engine.list_captures()[0]["capture_id"]

            result = server.consolidate_captures([source_id], max_bytes=512)

            self.assertIn("protected source captures", result)
            self.assertEqual(server.engine.list_captures()[0]["capture_id"], source_id)
        finally:
            server.engine.max_captures = original_max_captures

    def test_consolidate_captures_validates_limits(self):
        self.assertIn("max_captures must be at least 1", server.consolidate_captures(max_captures=0))
        self.assertIn(
            "max_captures must be at most",
            server.consolidate_captures(max_captures=26),
        )
        self.assertIn("max_bytes must be at least 512", server.consolidate_captures(max_bytes=511))
        self.assertIn(
            "at most 25 capture IDs",
            server.consolidate_captures(["missing"] * 26),
        )
        self.assertIn(
            "capture_ids must be a list",
            server.consolidate_captures("missing"),
        )
        self.assertIn(
            "capture_ids must contain strings",
            server.consolidate_captures([123]),
        )
        server.engine.max_buffer_bytes = 512
        self.assertIn("exceeds the configured buffer limit", server.consolidate_captures(max_bytes=513))

    def test_consolidate_uses_compact_payload_when_pretty_payload_is_too_large(self):
        server.capture_text("compact payload", label="compact")
        source_id = server.engine.list_captures()[0]["capture_id"]

        result = server._consolidated_jsonl([source_id], max_captures=1, max_bytes=512)

        self.assertLessEqual(len(result["content"].encode("utf-8")), 512)
        self.assertNotIn("\n  ", result["content"])

    def test_compact_consolidation_preserves_record_line_boundaries(self):
        source_text = "\n".join(f"record-{index}-" + ("x" * 150) for index in range(4))
        server.capture_text(source_text, label="compact-boundaries")
        source_id = server.engine.list_captures()[0]["capture_id"]

        result = server._consolidated_jsonl([source_id], max_captures=1, max_bytes=1024)
        content = result["content"]
        payload = json.loads(content)
        record_lines = [line for line in content.splitlines() if '"source_line":' in line]

        self.assertLessEqual(len(content.encode("utf-8")), 1024)
        self.assertGreater(result["record_count"], 1)
        self.assertEqual(len(record_lines), result["record_count"])
        self.assertEqual(payload["records"][0]["source_line"], 1)

    def test_consolidate_falls_back_when_source_metadata_exceeds_limit(self):
        server.capture_text("café detail", label="label-" + ("x" * 5000))
        source_id = server.engine.list_captures()[0]["capture_id"]

        result = server._consolidated_jsonl([source_id], max_captures=1, max_bytes=512)

        self.assertLessEqual(len(result["content"].encode("utf-8")), 512)
        payload = json.loads(result["content"])
        self.assertTrue(payload["metadata_omitted"])
        self.assertEqual(payload["records"], [])

    def test_consolidate_reports_ingest_failure(self):
        with patch.object(server.engine, "ingest", side_effect=RuntimeError("storage unavailable")):
            result = server.consolidate_captures([])

        self.assertIn("Error consolidating captures", result)
        self.assertIn("storage unavailable", result)

    def test_buffer_stats_formats_memory_metrics(self):
        server.capture_text("server stats payload", label="server-test")

        result = server.get_buffer_stats()

        self.assertIn("Captures:", result)
        self.assertIn("Embedding bytes:", result)
        self.assertIn("Embedding model:", result)
        self.assertIn("Embedding warm-up:", result)
        self.assertIn("Process RSS:", result)
        self.assertIn("Unaccounted RSS bytes:", result)

    def test_buffer_stats_reports_embedding_warmup_failure(self):
        original_state = server.engine.embedding_warmup_state
        original_failure = server.engine.embedding_warmup_failure
        server.engine.embedding_warmup_state = "failed"
        server.engine.embedding_warmup_failure = "RuntimeError"
        try:
            result = server.get_buffer_stats()
        finally:
            server.engine.embedding_warmup_state = original_state
            server.engine.embedding_warmup_failure = original_failure

        self.assertIn("Embedding warm-up: failed (RuntimeError)", result)

    def test_runtime_diagnostics_reports_content_free_metadata(self):
        server.capture_text("secret command output", label="private-label")

        with patch.dict(
            os.environ,
            {"EPHEMERAL_SESSION_ID": "diagnostic-session"},
            clear=False,
        ):
            result = server.get_runtime_diagnostics()

        self.assertIn("Runtime diagnostics (content-free):", result)
        self.assertIn("Package version: 0.4.0", result)
        self.assertIn("Python:", result)
        self.assertIn("Socket mode: session-derived path", result)
        self.assertIn("Socket lifecycle:", result)
        self.assertIn("Session ID configured: yes", result)
        self.assertIn("Captures: 1/", result)
        self.assertIn("Embedding model:", result)
        self.assertIn("Embedding warm-up:", result)
        self.assertIn("Semantic index budget adjustment:", result)
        self.assertNotIn("secret command output", result)
        self.assertNotIn("private-label", result)

    def test_runtime_index_budget_adjustment_requires_opt_in_and_reports_result(self):
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET": "0"}, clear=False):
            self.assertIn("disabled", server.set_semantic_index_budget(4))
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET": "1"}, clear=False):
            result = json.loads(server.set_semantic_index_budget(4))
        self.assertEqual(result["effective"], 4)
        self.assertEqual(server.engine.get_buffer_stats()["max_indexed_chunks"], 4)
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET": "1"}, clear=False):
            self.assertIn("Error adjusting semantic-index budget", server.set_semantic_index_budget("4"))

    def test_runtime_diagnostics_reports_explicit_and_default_socket_modes(self):
        with patch.dict(os.environ, {"EPHEMERAL_SOCKET_PATH": "/tmp/diagnostic.sock"}, clear=True):
            explicit = server.get_runtime_diagnostics()
        with patch.dict(os.environ, {}, clear=True):
            default = server.get_runtime_diagnostics()

        self.assertIn("Socket mode: explicit path", explicit)
        self.assertIn("Socket mode: shared default path", default)

    def test_mcp_instructions_describe_routing_and_isolation(self):
        with patch.object(server, "socket_isolation_required", return_value=True), \
                patch.object(server, "socket_isolation_configured", return_value=True):
            instructions = server._mcp_instructions()

        self.assertIn("execute_and_capture", instructions)
        self.assertIn("Socket isolation is configured", instructions)

    def test_mcp_instructions_describe_legacy_socket_mode(self):
        with patch.object(server, "socket_isolation_required", return_value=False), \
                patch.object(server, "socket_isolation_configured", return_value=False):
            instructions = server._mcp_instructions()

        self.assertIn("legacy single-session mode", instructions)

    def test_mcp_instructions_report_missing_required_isolation(self):
        with patch.object(server, "socket_isolation_required", return_value=True), \
                patch.object(server, "socket_isolation_configured", return_value=False):
            instructions = server._mcp_instructions()

        self.assertIn("startup must fail", instructions)

    def test_mcp_instructions_report_failed_socket(self):
        with patch.object(server, "_socket_lifecycle", return_value=("failed", "RuntimeError: unavailable")):
            instructions = server._mcp_instructions()

        self.assertIn("Socket lifecycle is failed (RuntimeError: unavailable)", instructions)

    def test_mcp_instructions_refresh_before_serving(self):
        with patch.object(server, "_mcp_instructions", return_value="ready-state guidance"):
            server._refresh_mcp_instructions()

        self.assertEqual(server.mcp.instructions, "ready-state guidance")

    def test_runtime_diagnostics_includes_enabled_local_metrics(self):
        original_metrics = server.METRICS
        original_engine_metrics = server.engine.metrics
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        server.engine.metrics = metrics
        try:
            result = server.get_runtime_diagnostics()
        finally:
            server.METRICS = original_metrics
            server.engine.metrics = original_engine_metrics

        self.assertIn("Local metrics: enabled", result)
        self.assertIn('"enabled": true', result)

    def test_buffer_stats_includes_enabled_local_metrics_without_content(self):
        original_metrics = server.METRICS
        original_engine_metrics = server.engine.metrics
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        server.engine.metrics = metrics
        try:
            server.capture_text("secret metrics payload", label="private metrics label")
            result = server.get_buffer_stats()
        finally:
            server.METRICS = original_metrics
            server.engine.metrics = original_engine_metrics

        self.assertIn("Local metrics: {", result)
        self.assertIn("Data-path bytes: {", result)
        self.assertIn('"captures": 1', result)
        self.assertNotIn("secret metrics payload", result)
        self.assertNotIn("private metrics label", result)

    def test_metrics_snapshot_file_is_opt_in_and_content_free(self):
        original_metrics = server.METRICS
        original_engine_metrics = server.engine.metrics
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        server.engine.metrics = metrics
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"EPHEMERAL_METRICS_FILE": os.path.join(directory, "metrics.json")},
                clear=False,
            ):
                server.capture_text("private snapshot content", label="private snapshot label")
                server._write_metrics_snapshot()
                snapshot = json.loads(Path(directory, "metrics.json").read_text(encoding="utf-8"))
        finally:
            server.METRICS = original_metrics
            server.engine.metrics = original_engine_metrics

        self.assertEqual(snapshot["bytes"]["capture_input_bytes"], len("private snapshot content"))
        self.assertNotIn("private snapshot content", json.dumps(snapshot))
        self.assertNotIn("private snapshot label", json.dumps(snapshot))

    def test_metrics_snapshot_file_skips_disabled_metrics(self):
        original_metrics = server.METRICS
        server.METRICS = LocalMetrics(enabled=False)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"EPHEMERAL_METRICS_FILE": os.path.join(directory, "metrics.json")},
                clear=False,
            ):
                server._write_metrics_snapshot()
                self.assertFalse(Path(directory, "metrics.json").exists())
        finally:
            server.METRICS = original_metrics

    def test_metrics_snapshot_file_write_failure_is_logged(self):
        original_metrics = server.METRICS
        server.METRICS = LocalMetrics(enabled=True)
        try:
            with patch.dict(os.environ, {"EPHEMERAL_METRICS_FILE": "/tmp/metrics.json"}, clear=False), \
                    patch.object(Path, "write_text", side_effect=OSError("read-only")), \
                    self.assertLogs("ephemeral_buffer.server", level="WARNING") as logs:
                server._write_metrics_snapshot()
        finally:
            server.METRICS = original_metrics
        self.assertTrue(any("metrics_snapshot_write_failed" in entry for entry in logs.output))

    def test_runtime_package_version_falls_back_to_source_checkout(self):
        with patch.object(server.Path, "read_text", side_effect=OSError("missing metadata")), \
                patch.object(server, "package_version", side_effect=server.PackageNotFoundError()):
            self.assertEqual(server._runtime_package_version(), "source checkout")

    def test_capture_file_reads_and_labels_content(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "capture.log"
            file_path.write_text("file content", encoding="utf-8")

            result = server.capture_file(str(file_path))

        summary = json.loads(result)
        self.assertEqual(summary["status"], "captured")
        self.assertEqual(summary["source"], "file")
        self.assertEqual(summary["label"], "capture.log")
        self.assertNotIn("previews", summary)

    def test_capture_text_formats_diff_metadata(self):
        diff = """diff --git a/old.txt b/new.txt
--- a/old.txt
+++ b/new.txt
@@ -1 +1 @@
-old
+new
"""
        result = server.capture_text(diff, label="patch", content_type="diff")

        summary = json.loads(result)
        self.assertEqual(summary["content_type"], "diff")
        self.assertIn("new.txt", summary["diff"]["file_map"])

    def test_execute_reports_failed_and_truncated_command(self):
        output = "x" * 700
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(output, 7, True, 700, False),
        ):
            result = server.execute_and_capture("failing-command", max_output_bytes=1024)

        summary = json.loads(result)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["exit_code"], 7)
        self.assertTrue(summary["truncated"])
        self.assertEqual(summary["original_byte_size"], 700)
        self.assertEqual(summary["original_estimated_tokens"], 175)
        self.assertEqual(summary["source"], "command")

    def test_execute_bounds_long_command_metadata(self):
        command = "echo " + ("x" * 10_000)
        with patch.object(
            server,
            "run_command_bounded",
            return_value=("output", 0, False, 6, False),
        ):
            result = json.loads(server.execute_and_capture(command))

        self.assertLessEqual(
            len(result["command"].encode("utf-8")),
            server.SUMMARY_COMMAND_MAX_BYTES,
        )
        self.assertTrue(result["command_truncated"])
        self.assertTrue(result["command"].endswith(server.SUMMARY_COMMAND_TRUNCATION_MARKER))

    def test_invalid_command_metrics_are_rejected_before_execution(self):
        with patch.object(server, "run_command_bounded") as run:
            result = server.execute_and_capture(
                "would-not-run",
                structured_metrics={"value": float("nan")},
            )

        self.assertIn("Error: invalid structured_metrics", result)
        run.assert_not_called()

    def test_invalid_file_metrics_are_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "capture.log"
            file_path.write_text("file content", encoding="utf-8")
            with patch.object(server, "read_file_bounded") as read:
                result = server.capture_file(
                    str(file_path),
                    structured_metrics={"value": float("nan")},
                )

        self.assertIn("Error: invalid structured_metrics", result)
        read.assert_not_called()

    def test_invalid_text_metrics_are_rejected_at_capture_boundary(self):
        result = server.capture_text(
            "would-not-ingest",
            structured_metrics={"value": float("nan")},
        )
        self.assertIn("Error: invalid structured_metrics", result)

    def test_execute_formats_diff_command_response(self):
        diff = "diff --git a/old.txt b/new.txt\n--- a/old.txt\n+++ b/new.txt\n@@ -1 +1 @@\n-old\n+new\n"
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(diff, 0, False, len(diff.encode()), False),
        ):
            result = server.execute_and_capture("git diff", max_output_bytes=1024)

        summary = json.loads(result)
        self.assertEqual(summary["content_type"], "diff")
        self.assertIn("new.txt", summary["diff"]["file_map"])

    def test_search_and_read_tools_report_missing_and_empty_results(self):
        self.assertIn("Search Error", server.search_capture("query", capture_id="missing"))
        with patch.object(
            server.engine,
            "search",
            return_value={"status": "ok", "capture_id": "cap", "label": "label", "matches": []},
        ):
            self.assertIn("No matches found", server.search_capture("query"))
        self.assertIn("Error:", server.get_capture_slice(1, 1, capture_id="missing"))
        self.assertIn("Error:", server.get_capture_summary(capture_id="missing"))

    def test_search_and_read_tools_format_matches_and_diff_summary(self):
        with patch.object(
            server.engine,
            "search",
            return_value={
                "status": "ok",
                "mode": "bm25",
                "capture_id": "cap",
                "label": "label",
                "total_lines": 2,
                "matches": [{
                    "score": 1.0,
                    "matched_range": "1-1",
                    "context_range": "1-2",
                    "snippet": "match",
                }],
            },
        ):
            result = server.search_capture("query", mode="bm25")
        self.assertIn("Match #1", result)
        self.assertIn("match", result)

        with patch.object(
            server.engine,
            "get_slice",
            return_value={
                "status": "ok", "capture_id": "cap", "label": "label",
                "start_line": 1, "end_line": 1, "total_lines": 1, "content": "line",
            },
        ):
            self.assertIn("Lines 1 to 1", server.get_capture_slice(1, 1))
        with patch.object(
            server.engine,
            "get_summary",
            return_value={
                "status": "ok", "capture_id": "cap", "label": "diff",
                "content_type": "diff", "file_map": "new.txt [MODIFIED]",
                "diff_stats": "1 file", "timestamp": 1, "total_lines": 3,
                "byte_size": 10, "truncated": True, "original_byte_size": 20,
                "signals_summary": "None (Clean patch)",
            },
        ):
            summary = json.loads(server.get_capture_summary("cap"))
        self.assertEqual(summary["schema_version"], 1)
        self.assertTrue(summary["truncated"])
        self.assertEqual(summary["original_byte_size"], 20)
        self.assertIn("new.txt", summary["diff"]["file_map"])

        with patch.object(
            server.engine,
            "get_summary",
            return_value={
                "status": "ok", "capture_id": "cap", "label": "log",
                "timestamp": 1, "total_lines": 2, "byte_size": 10,
                "truncated": False, "signals_summary": "None detected",
                "head_preview": "head", "tail_preview": "tail",
            },
        ):
            regular = json.loads(server.get_capture_summary("cap", include_previews=True))
        self.assertEqual(regular["previews"], {"head": "head", "tail": "tail"})

    def test_capture_summary_includes_warnings_metrics_and_token_estimate(self):
        result = server.capture_text(
            "2 tests passed\nWARNING: slow test\nOK",
            label="test-log",
            content_type="log",
            structured_metrics={"tests": 2, "duration_ms": 12.5},
        )

        summary = json.loads(result)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["status"], "captured")
        self.assertEqual(summary["warnings"], [{"type": "warning", "count": 1}])
        self.assertEqual(summary["structured_metrics"]["tests"], 2)
        self.assertGreater(summary["estimated_tokens"], 0)
        self.assertNotIn("previews", summary)

        detailed = json.loads(
            server.get_capture_summary(summary["capture_id"], include_previews=True)
        )
        self.assertIn("previews", detailed)
        self.assertLess(len(result), len(json.dumps(detailed, separators=(",", ":"))))

    def test_summary_response_reduces_prompt_proxy_for_representative_outcomes(self):
        cases = {
            "success": ("completed\n" * 3, 0, False, 27, False),
            "failure": ("ERROR: command failed\n" * 3, 1, False, 66, False),
            "noisy": ("log line with useful context\n" * 500, 0, False, 15_000, False),
            "truncated": ("retained output\n" * 20, 1, True, 80_000, False),
            "timed_out": ("partial output\n" * 3, 124, False, 45, True),
        }
        reductions = {}
        for name, (output, exit_code, truncated, original_size, timed_out) in cases.items():
            with patch.object(
                server,
                "run_command_bounded",
                return_value=(output, exit_code, truncated, original_size, timed_out),
            ):
                compact = server.execute_and_capture(
                    f"representative-{name}",
                    content_type="log",
                    max_output_bytes=1024,
                )
            compact_payload = json.loads(compact)
            detailed = server.get_capture_summary(
                compact_payload["capture_id"],
                include_previews=True,
            )
            compact_proxy = math.ceil(len(compact.encode("utf-8")) / 4)
            detailed_proxy = math.ceil(len(detailed.encode("utf-8")) / 4)
            self.assertLess(compact_proxy, detailed_proxy, name)
            reductions[name] = detailed_proxy - compact_proxy

        self.assertEqual(set(reductions), set(cases))
        self.assertGreater(reductions["noisy"], reductions["success"])

    def test_summary_payload_preserves_engine_errors(self):
        self.assertEqual(
            server._summary_payload({"status": "error", "message": "missing"}),
            {"status": "error", "message": "missing"},
        )

    def test_diff_summary_bounds_large_file_maps(self):
        file_map = "\n".join(
            f"  - file-{index:05d}.txt (+1, -0) | Buffer Lines: L1-L2"
            for index in range(10_000)
        )
        payload = server._summary_payload({
            "status": "ok",
            "schema_version": 1,
            "capture_id": "cap-diff",
            "label": "large diff",
            "source": "capture_text",
            "execution_status": "captured",
            "partial": False,
            "content_type": "diff",
            "timestamp": "now",
            "duration_ms": 1.0,
            "command_exit_code": None,
            "timed_out": False,
            "total_lines": 20_000,
            "byte_size": 500_000,
            "original_byte_size": None,
            "estimated_tokens": 125_000,
            "original_estimated_tokens": None,
            "truncated": False,
            "keyword_signals": {},
            "errors": [],
            "warnings": [],
            "structured_metrics": {},
            "diff_stats": "10,000 files",
            "file_map": file_map,
        })
        self.assertLessEqual(
            len(payload["diff"]["file_map"].encode("utf-8")),
            server.SUMMARY_DIFF_FILE_MAP_MAX_BYTES,
        )
        self.assertTrue(payload["diff"]["file_map_truncated"])
        self.assertGreater(payload["diff"]["omitted_file_count"], 0)

        structured_bounded, structured_omitted = server._bounded_diff_file_map(
            "unused legacy map",
            {
                "files": [
                    {
                        "path": "added.py",
                        "status": "added",
                        "additions": 3,
                        "deletions": 0,
                        "start_line": 1,
                        "end_line": 6,
                    }
                    for _ in range(server.SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES + 1)
                ]
            },
        )
        self.assertIn("added.py [ADDED]", structured_bounded)
        self.assertEqual(structured_omitted, 1)

        bounded, omitted = server._bounded_diff_file_map(
            "x" * (server.SUMMARY_DIFF_FILE_MAP_MAX_BYTES - 1) + "\nsmall"
        )
        self.assertLessEqual(
            len(bounded.encode("utf-8")), server.SUMMARY_DIFF_FILE_MAP_MAX_BYTES
        )
        self.assertEqual(omitted, 2)

        with patch.object(server, "SUMMARY_DIFF_FILE_MAP_MAX_BYTES", 10):
            bounded, omitted = server._bounded_diff_file_map("long line\nsmall")
        self.assertLessEqual(len(bounded.encode("utf-8")), 10)
        self.assertGreater(len(bounded.encode("utf-8")), 0)
        self.assertEqual(omitted, 2)

    def test_search_capture_discloses_semantic_fallback(self):
        response = {
            "status": "ok",
            "capture_id": "cap-fallback",
            "label": "fallback",
            "total_lines": 1,
            "mode": "hybrid",
            "match_count": 1,
            "semantic_fallback": "RuntimeError",
            "matches": [{
                "score": 1.0,
                "matched_range": "L1-L1",
                "context_range": "L1-L1",
                "snippet": ">     1 | lexical result",
            }],
        }
        with patch.object(server.engine, "search", return_value=response):
            result = server.search_capture("query", mode="hybrid")

        self.assertIn("Mode: hybrid; lexical fallback (RuntimeError)", result)
        self.assertIn("Semantic fallback active (RuntimeError)", result)

    def test_search_capture_discloses_pending_semantic_coverage(self):
        response = {
            "status": "ok",
            "capture_id": "cap-pending",
            "label": "pending",
            "total_lines": 9000,
            "mode": "hybrid",
            "match_count": 1,
            "semantic_coverage": "pending",
            "semantic_index_state": "pending",
            "semantic_wait_seconds": 10.0,
            "message": "Semantic index still building for this capture; results are lexical (BM25) only.",
            "matches": [{
                "score": 1.0,
                "matched_range": "L1-L1",
                "context_range": "L1-L1",
                "snippet": ">     1 | lexical result",
            }],
        }
        with patch.object(server.engine, "search", return_value=response):
            result = server.search_capture("query", mode="hybrid")
        self.assertIn("Mode: hybrid; semantic pending (lexical only)", result)
        self.assertIn("Semantic index still building; results are lexical (BM25) only.", result)
        self.assertIn("Repeat the search for hybrid ranking.", result)

        empty = dict(response, matches=[], match_count=0)
        with patch.object(server.engine, "search", return_value=empty):
            result = server.search_capture("query", mode="hybrid")
        self.assertTrue(result.startswith("No matches found for 'query'"))
        self.assertIn("Semantic index still building", result)

    def test_buffer_stats_reports_semantic_wait_budget(self):
        result = server.get_buffer_stats()
        self.assertRegex(result, r"Semantic wait budget: [0-9.]+s \(0 on-demand index jobs running\)")

    def test_context_responses_bound_long_labels(self):
        long_label = "label-" + ("x" * 10_000)
        summary = json.loads(server.capture_text("needle content", label=long_label))
        capture_id = summary["capture_id"]

        responses = [
            server.list_captures(),
            server.search_capture("needle", mode="bm25", capture_id=capture_id),
            server.search_capture("missing", mode="bm25", capture_id=capture_id),
            server.get_capture_slice(1, 1, capture_id=capture_id),
        ]
        for response in responses:
            self.assertLessEqual(
                len(response.encode("utf-8")),
                3 * server.SUMMARY_LABEL_MAX_BYTES,
            )
            self.assertNotIn(long_label, response)
            self.assertIn(server.SUMMARY_LABEL_TRUNCATION_MARKER, response)

        long_query = "q" * 100_000
        for mode in ("bm25", "hybrid", "semantic"):
            response = server.search_capture(long_query, mode=mode, capture_id=capture_id)
            self.assertEqual(
                response,
                f"Search Error: query exceeds the {server.SEARCH_QUERY_MAX_BYTES:,}-byte limit",
            )
        self.assertEqual(
            server.search_capture(None, mode="bm25", capture_id=capture_id),
            "Search Error: query must be a string",
        )

    def test_context_errors_bound_long_capture_ids(self):
        long_id = "z" * 100_000
        summary_error = server.get_capture_summary(long_id)
        consolidation_error = server.consolidate_captures([long_id], max_bytes=512)

        for response in (summary_error, consolidation_error):
            self.assertLessEqual(
                len(response.encode("utf-8")),
                3 * server.SUMMARY_LABEL_MAX_BYTES,
            )
            self.assertNotIn(long_id, response)

    def test_empty_and_populated_capture_listing(self):
        self.assertIn("buffer is empty", server.list_captures())
        server.capture_text("one line", label="listed")
        self.assertIn("listed", server.list_captures())

    def test_search_response_has_a_utf8_budget(self):
        matches = [
            {
                "score": 1.0,
                "matched_range": "1-1",
                "context_range": "1-1",
                "snippet": "x" * (server.SEARCH_RESPONSE_MAX_BYTES // 4),
            }
            for _ in range(8)
        ]
        with patch.object(
            server.engine,
            "search",
            return_value={
                "status": "ok", "mode": "bm25", "capture_id": "cap",
                "label": "label", "total_lines": 1, "matches": matches,
            },
        ):
            result = server.search_capture("query", mode="bm25")

        self.assertLessEqual(len(result.encode("utf-8")), server.SEARCH_RESPONSE_MAX_BYTES)
        self.assertIn("Additional matches omitted", result)

    def test_clear_capture_delegates_success_and_missing_results(self):
        with patch.object(server.engine, "clear", return_value="Cleared capture 'cap'.") as clear:
            self.assertEqual(server.clear_captures("cap"), "Cleared capture 'cap'.")
        clear.assert_called_once_with("cap")

        with patch.object(server.engine, "clear", return_value="Capture 'missing' not found."):
            self.assertIn("not found", server.clear_captures("missing"))

    def test_buffer_stats_handles_unavailable_memory_metrics(self):
        with patch.object(
            server.engine,
            "get_buffer_stats",
            return_value={
                "capture_count": 0, "max_captures": 25, "total_bytes": 0,
                "max_buffer_bytes": 50, "total_lines": 0, "total_chunks": 0,
                "embedding_model": "model", "embedding_model_loaded": False,
                "embedding_cache_dir": None, "embedding_bytes": 0,
                "accounted_bytes": 0, "process_rss_bytes": None,
                "unaccounted_rss_bytes": None,
            },
        ):
            result = server.get_buffer_stats()
        self.assertIn("Process RSS: unavailable", result)
        self.assertIn("Unaccounted RSS bytes: unavailable", result)


class TestServerSocket(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes
        self.original_model = server.engine.embedding_model
        server.engine.max_buffer_bytes = 1024
        server.engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[0.0] * 384 for _ in texts]},
        )()

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
        server.engine.embedding_model = self.original_model
        server.engine.clear("all")

    async def run_handler(self, payload):
        writer = FakeWriter()
        task = server.handle_socket_client(FakeReader(payload), writer)
        await task
        return writer

    async def test_json_payload_returns_success_response(self):
        payload = encode_frame(json.dumps({"label": "socket-test", "text": "hello"}).encode())
        capture = SimpleNamespace(
            capture_id="cap_socket",
            label="socket-test",
            line_count=1,
            byte_size=5,
        )
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, {"status": "error"}]),
        ):
            writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "socket-test")
        self.assertTrue(writer.closed)

    async def test_socket_response_includes_compact_summary(self):
        payload = encode_frame(json.dumps({"label": "socket-summary", "text": "hello"}).encode())
        capture = SimpleNamespace(
            capture_id="cap_socket_summary",
            label="socket-summary",
            line_count=1,
            byte_size=5,
        )
        summary = {
            "status": "ok",
            "schema_version": 1,
            "capture_id": "cap_socket_summary",
            "label": "socket-summary",
            "source": "socket",
            "execution_status": "captured",
            "partial": False,
            "content_type": "text",
            "timestamp": "now",
            "duration_ms": 1.0,
            "command_exit_code": None,
            "timed_out": False,
            "total_lines": 1,
            "byte_size": 5,
            "original_byte_size": None,
            "estimated_tokens": 2,
            "original_estimated_tokens": None,
            "truncated": False,
            "keyword_signals": {},
            "errors": [],
            "warnings": [],
            "structured_metrics": {},
        }
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, summary]),
        ):
            writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertEqual(response["summary"]["schema_version"], 1)
        self.assertEqual(response["summary"]["status"], "captured")
        self.assertNotIn("previews", response["summary"])

    async def test_socket_response_bounds_long_command_label(self):
        long_label = "x" * 10_000
        payload = encode_frame(json.dumps({"label": long_label, "text": "hello"}).encode())
        capture = SimpleNamespace(
            capture_id="cap_long_label",
            label=long_label,
            line_count=1,
            byte_size=5,
        )
        summary = {
            "status": "ok",
            "schema_version": 1,
            "capture_id": "cap_long_label",
            "label": long_label,
            "source": "socket",
            "content_type": "text",
            "total_lines": 1,
            "byte_size": 5,
        }
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, summary]),
        ):
            writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertLessEqual(
            len(response["label"].encode("utf-8")),
            server.SUMMARY_LABEL_MAX_BYTES,
        )
        self.assertTrue(response["summary"]["label_truncated"])
        self.assertLessEqual(
            len(json.dumps(response).encode("utf-8")),
            3 * server.SUMMARY_LABEL_MAX_BYTES,
        )

    async def test_socket_byte_metrics_count_framed_request_and_response(self):
        original_metrics = server.METRICS
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        try:
            payload = encode_frame(json.dumps({"label": "socket-metrics", "text": "hello"}).encode())
            capture = SimpleNamespace(
                capture_id="cap_socket_metrics",
                label="socket-metrics",
                line_count=1,
                byte_size=5,
            )
            with patch.object(
                server,
                "to_thread",
                new=AsyncMock(side_effect=[capture, {"status": "error"}]),
            ):
                writer = await self.run_handler(payload)
        finally:
            server.METRICS = original_metrics

        counters = metrics.snapshot()["bytes"]
        self.assertEqual(counters["socket_request_bytes"], len(payload))
        self.assertEqual(counters["socket_response_bytes"], len(writer.writes[0]))

    async def test_json_payload_is_reassembled_across_socket_reads(self):
        payload = encode_frame(json.dumps({"label": "chunked", "text": "complete payload"}).encode())
        capture = SimpleNamespace(
            capture_id="cap_chunked",
            label="chunked",
            line_count=1,
            byte_size=16,
        )
        writer = FakeWriter()
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, {"status": "error"}]),
        ):
            await server.handle_socket_client(
                ChunkedReader(payload[:7], payload[7:]),
                writer,
            )

        response = response_json(writer)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "chunked")

    async def test_escape_heavy_payload_within_capture_limit_is_accepted(self):
        server.engine.max_buffer_bytes = 100_000
        text = "\x00" * server.engine.max_buffer_bytes
        payload = encode_frame(json.dumps({"label": "escaped", "text": text}).encode())
        self.assertGreater(
            len(payload),
            server.engine.max_buffer_bytes + server.SOCKET_PAYLOAD_OVERHEAD,
        )
        capture = SimpleNamespace(
            capture_id="cap_escaped",
            label="escaped",
            line_count=1,
            byte_size=server.engine.max_buffer_bytes,
        )
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, {"status": "error"}]),
        ):
            writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "escaped")

    async def test_oversized_payload_is_rejected_across_socket_reads(self):
        payload = encode_frame(b"x" * (
            server.engine.max_buffer_bytes * server.SOCKET_JSON_MAX_EXPANSION
            + server.SOCKET_PAYLOAD_OVERHEAD
            + 1
        ))
        writer = FakeWriter()

        await server.handle_socket_client(
            ChunkedReader(payload[:128], payload[128:]),
            writer,
        )

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("exceeds", response["message"])

    async def test_socket_byte_metrics_count_consumed_rejected_request_bytes(self):
        original_metrics = server.METRICS
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        try:
            oversized = encode_frame(b"x" * (
                server.engine.max_buffer_bytes * server.SOCKET_JSON_MAX_EXPANSION
                + server.SOCKET_PAYLOAD_OVERHEAD
                + 1
            ))
            await self.run_handler(oversized)
            truncated_header = FRAME_MAGIC + b"\x01\x00"
            await self.run_handler(truncated_header)
        finally:
            server.METRICS = original_metrics

        self.assertEqual(
            metrics.snapshot()["bytes"]["socket_request_bytes"],
            FRAME_HEADER_SIZE + len(truncated_header),
        )

    async def test_ingest_is_offloaded_from_event_loop(self):
        payload = encode_frame(json.dumps({"label": "offload-test", "text": "hello"}).encode())
        capture = SimpleNamespace(
            capture_id="cap_offload",
            label="offload-test",
            line_count=1,
            byte_size=5,
        )
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, {"status": "error"}]),
        ) as offload:
            await self.run_handler(payload)

        self.assertEqual(offload.await_count, 2)
        summary_callable = offload.await_args_list[1].args[0]
        self.assertIs(summary_callable.__self__, server.engine)
        self.assertIs(summary_callable.__func__, server.engine.get_summary_for_capture.__func__)
        self.assertEqual(offload.await_args_list[1].kwargs, {
            "include_previews": False,
        })

    async def test_oversized_payload_returns_error_response(self):
        payload = encode_frame(b"x" * (
            server.engine.max_buffer_bytes * server.SOCKET_JSON_MAX_EXPANSION
            + server.SOCKET_PAYLOAD_OVERHEAD
            + 1
        ))

        writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("exceeds", response["message"])
        self.assertTrue(writer.closed)

    async def test_empty_payload_closes_without_response(self):
        writer = await self.run_handler(encode_frame(b""))

        self.assertEqual(writer.writes, [])
        self.assertTrue(writer.closed)

    async def test_malformed_payload_falls_back_to_plain_text(self):
        capture = SimpleNamespace(
            capture_id="cap_plain",
            label="CLI pipe",
            line_count=1,
            byte_size=8,
        )
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=[capture, {"status": "error"}]),
        ):
            writer = await self.run_handler(encode_frame(b"not-json"))

        response = response_json(writer)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "CLI pipe")

    async def test_ingest_failure_returns_error_response(self):
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=ValueError("invalid capture")),
        ):
            writer = await self.run_handler(encode_frame(json.dumps({"text": "payload"}).encode()))

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("invalid capture", response["message"])

    async def test_invalid_original_size_returns_error_without_retaining_capture(self):
        original_metrics = server.METRICS
        original_engine_metrics = server.engine.metrics
        metrics = LocalMetrics(enabled=True)
        server.METRICS = metrics
        server.engine.metrics = metrics
        try:
            payload = encode_frame(json.dumps({
                "text": "payload",
                "truncated": True,
                "original_byte_size": "not-an-integer",
            }).encode())
            writer = await self.run_handler(payload)
        finally:
            server.METRICS = original_metrics
            server.engine.metrics = original_engine_metrics

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("original_byte_size", response["message"])
        self.assertEqual(server.engine.captures, {})
        self.assertEqual(metrics.snapshot()["events"]["captures"], 0)

    async def test_invalid_socket_metrics_return_error_without_capturing_envelope(self):
        payload = encode_frame(json.dumps({
            "label": "invalid-metrics",
            "text": "payload",
            "structured_metrics": ["not", "an", "object"],
        }).encode())

        writer = await self.run_handler(payload)

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("JSON object", response["message"])
        self.assertEqual(server.engine.captures, {})

    async def test_non_object_json_payload_returns_error_without_capturing(self):
        writer = await self.run_handler(encode_frame(b"[]"))

        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("JSON object", response["message"])
        self.assertEqual(server.engine.captures, {})

    async def test_truncated_frame_returns_error_response(self):
        writer = await self.run_handler(FRAME_MAGIC + b"\x01\x00")
        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("truncated socket frame", response["message"])

    async def test_unsupported_frame_version_returns_error_response(self):
        writer = await self.run_handler(FRAME_MAGIC + b"\x02\x00\x00\x00\x00")
        response = response_json(writer)
        self.assertEqual(response["status"], "error")
        self.assertIn("unsupported frame version", response["message"])


class TestSocketServerStartup(unittest.TestCase):
    def test_socket_path_lock_closes_fd_when_lock_acquisition_fails(self):
        with patch.object(server.os, "open", return_value=41), \
                patch.object(server.fcntl, "flock", side_effect=OSError("lock failed")), \
                patch.object(server.os, "close") as close:
            with self.assertRaisesRegex(OSError, "lock failed"):
                with server._socket_path_lock("/tmp/ephemeral-buffer-test.sock"):
                    pass
        close.assert_called_once_with(41)

    def test_socket_startup_holds_path_lock_through_binding(self):
        events = []

        class Lock:
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, *_exc_info):
                events.append("exit")
                return False

        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        socket_stat = os.stat_result(
            (stat.S_IFSOCK | 0o600, 1, 2, 1, 0, 0, 0, 0, 0, 0)
        )

        async def start_server(*_args, **_kwargs):
            events.append("bind")
            return Listener()

        with patch.object(server, "SOCKET_PATH", "/tmp/ephemeral-buffer-test.sock"), \
                patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                patch.object(server.asyncio, "set_event_loop"), \
                patch.object(server.os.path, "lexists", return_value=False), \
                patch.object(server.os, "lstat", return_value=socket_stat), \
                patch.object(server.asyncio, "start_unix_server", new=AsyncMock(side_effect=start_server)), \
                patch.object(server.os, "chmod"), \
                patch.object(server, "_socket_path_lock", return_value=Lock()), \
                patch.object(server, "_unlink_socket_if_identity", return_value=True), \
                patch("sys.stderr", new_callable=io.StringIO):
            server.run_socket_server()

        self.assertEqual(events[:2], ["enter", "bind"])
        self.assertEqual(events[-1], "exit")

    def test_disabled_socket_import_marks_startup_event_ready(self):
        with patch.dict(os.environ, {"EPHEMERAL_DISABLE_SOCKET_SERVER": "1"}):
            namespace = runpy.run_path(server.__file__, run_name="server_disabled_import")

        self.assertTrue(namespace["_SOCKET_STARTUP_EVENT"].is_set())

    def test_import_does_not_start_socket_listener(self):
        with patch.dict(os.environ, {
                "EPHEMERAL_DISABLE_SOCKET_SERVER": "0",
                "EPHEMERAL_EMBEDDING_WARMUP": "0",
        }), \
                patch.object(server.threading, "Thread") as thread:
            runpy.run_path(server.__file__, run_name="server_import")

        thread.assert_not_called()

    def test_explicit_socket_startup_honors_disabled_mode(self):
        with patch.dict(os.environ, {"EPHEMERAL_DISABLE_SOCKET_SERVER": "1"}):
            self.assertIsNone(server.start_socket_server())

        self.assertEqual(server._socket_lifecycle()[0], "disabled")

    def test_socket_startup_timeout_is_reported(self):
        event = SimpleNamespace(wait=lambda timeout: False)
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET": "0"}), \
                patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "SOCKET_STARTUP_TIMEOUT_SECONDS", 3):
            with self.assertRaisesRegex(SystemExit, "did not become ready within 3 seconds"):
                server._require_socket_ready()

    def test_socket_startup_timeout_can_continue_stdio_only(self):
        event = SimpleNamespace(wait=lambda timeout: False)
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET": "1"}), \
                patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "SOCKET_STARTUP_TIMEOUT_SECONDS", 3):
            server._require_socket_ready()

    def test_socket_startup_failure_is_reported_to_entrypoint(self):
        event = SimpleNamespace(wait=lambda timeout: True)
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET": "0"}), \
                patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "_socket_lifecycle", return_value=("failed", "RuntimeError: unavailable")):
            with self.assertRaisesRegex(SystemExit, "failed to start: RuntimeError: unavailable"):
                server._require_socket_ready()

    def test_socket_startup_failure_can_continue_stdio_only(self):
        event = SimpleNamespace(wait=lambda timeout: True)
        with patch.dict(os.environ, {"EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET": "true"}), \
                patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "_socket_lifecycle", return_value=("failed", "PermissionError: denied")):
            server._require_socket_ready()

    def test_strict_isolation_rejects_unconfigured_startup(self):
        class FailingLoop:
            def close(self):
                pass

        with patch.object(server, "socket_isolation_required", return_value=True), \
                patch.object(server, "socket_isolation_configured", return_value=False), \
                patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                patch.object(server.asyncio, "set_event_loop"), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            server.run_socket_server()

        self.assertIn("Socket isolation is required", stderr.getvalue())

    def _run_with_existing_socket(self, probe_error, unlink=None):
        class FailingLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                coroutine.close()
                raise RuntimeError("socket unavailable")

        class Probe:
            def connect(self, _path):
                if probe_error is not None:
                    raise probe_error

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "existing.sock")
            stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                stale_listener.bind(socket_path)
            except PermissionError as exc:
                stale_listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            stale_listener.close()
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.socket, "socket", return_value=Probe()), \
                    patch.object(server.os, "unlink", side_effect=unlink) as unlink_mock, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()
        return stderr.getvalue(), unlink_mock

    def test_socket_probe_stale_cleanup_tolerates_missing_path(self):
        stderr, unlink = self._run_with_existing_socket(
            ConnectionRefusedError(),
            FileNotFoundError(),
        )

        unlink.assert_called_once()
        self.assertIn("Socket server error: socket unavailable", stderr)

    def test_socket_probe_file_disappears_during_probe(self):
        stderr, unlink = self._run_with_existing_socket(FileNotFoundError())

        unlink.assert_not_called()
        self.assertIn("Socket server error: socket unavailable", stderr)

    def test_socket_probe_reports_live_listener(self):
        stderr, unlink = self._run_with_existing_socket(None)

        unlink.assert_not_called()
        self.assertIn("Socket already in use", stderr)

    def test_non_socket_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "regular", root / "directory"]
            paths[0].write_text("preserve me", encoding="utf-8")
            paths[1].mkdir()
            symlink = root / "symlink"
            try:
                symlink.symlink_to(paths[0])
                paths.append(symlink)
            except OSError:
                pass

            class FailingLoop:
                def close(self):
                    pass

            for path in paths:
                with self.subTest(path=path), \
                        patch.object(server, "SOCKET_PATH", str(path)), \
                        patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                        patch.object(server.asyncio, "set_event_loop"), \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    server.run_socket_server()

                self.assertTrue(os.path.lexists(path))
                self.assertIn("not a Unix socket", stderr.getvalue())

    def test_socket_replacement_during_probe_is_not_unlinked(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "ephemeral.sock"
            replacement = Path(directory) / "replacement"
            replacement.write_text("preserve replacement", encoding="utf-8")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(socket_path))
            except PermissionError as exc:
                listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            listener.close()

            initial_stat = os.lstat(socket_path)
            replacement_stat = os.lstat(replacement)

            class FailingLoop:
                def close(self):
                    pass

            class RefusingProbe:
                def connect(self, _path):
                    raise ConnectionRefusedError()

                def close(self):
                    pass

            with patch.object(server, "SOCKET_PATH", str(socket_path)), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=True), \
                    patch.object(server.os, "lstat", side_effect=[initial_stat, replacement_stat]), \
                    patch.object(server.socket, "socket", return_value=RefusingProbe()), \
                    patch.object(server.os, "unlink") as unlink, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

            unlink.assert_not_called()
            self.assertTrue(socket_path.exists())
            self.assertIn("changed to a non-socket path", stderr.getvalue())

    def test_socket_disappearing_during_revalidation_is_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "ephemeral.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(socket_path))
            except PermissionError as exc:
                listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            listener.close()
            initial_stat = os.lstat(socket_path)

            class FailingLoop:
                def close(self):
                    pass

                def run_until_complete(self, coroutine):
                    coroutine.close()
                    raise RuntimeError("socket unavailable")

            class RefusingProbe:
                def connect(self, _path):
                    raise ConnectionRefusedError()

                def close(self):
                    pass

            with patch.object(server, "SOCKET_PATH", str(socket_path)), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=True), \
                    patch.object(server.os, "lstat", side_effect=[initial_stat, FileNotFoundError()]), \
                    patch.object(server.socket, "socket", return_value=RefusingProbe()), \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

            self.assertIn("socket unavailable", stderr.getvalue())

    def test_socket_identity_change_during_probe_is_not_unlinked(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "ephemeral.sock"
            replacement_path = Path(directory) / "replacement.sock"
            listeners = []
            for path in (socket_path, replacement_path):
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    listener.bind(str(path))
                except PermissionError as exc:
                    for open_listener in listeners:
                        open_listener.close()
                    listener.close()
                    self.skipTest(f"Unix socket bind unavailable: {exc}")
                listener.close()
            initial_stat = os.lstat(socket_path)
            replacement_stat = os.lstat(replacement_path)

            class FailingLoop:
                def close(self):
                    pass

            class RefusingProbe:
                def connect(self, _path):
                    raise ConnectionRefusedError()

                def close(self):
                    pass

            with patch.object(server, "SOCKET_PATH", str(socket_path)), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=True), \
                    patch.object(server.os, "lstat", side_effect=[initial_stat, replacement_stat]), \
                    patch.object(server.socket, "socket", return_value=RefusingProbe()), \
                    patch.object(server.os, "unlink") as unlink, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

            unlink.assert_not_called()
            self.assertIn("changed while probing", stderr.getvalue())

    def test_socket_probe_os_error_is_reported(self):
        stderr, unlink = self._run_with_existing_socket(OSError("probe failed"))

        unlink.assert_not_called()
        self.assertIn("Unable to verify existing socket", stderr)

    def test_live_socket_is_not_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(socket_path)
            except PermissionError as exc:
                listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            listener.listen()
            try:
                class FailingLoop:
                    def close(self):
                        pass

                    def run_until_complete(self, coroutine):
                        coroutine.close()
                        raise AssertionError("startup should stop before event loop execution")

                with patch.object(server, "SOCKET_PATH", socket_path), \
                        patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                        patch.object(server.asyncio, "set_event_loop"), \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    server.run_socket_server()
                self.assertIn("Socket already in use", stderr.getvalue())
            finally:
                listener.close()

            self.assertTrue(os.path.exists(socket_path))

    def test_stale_socket_is_removed_before_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                stale_listener.bind(socket_path)
            except PermissionError as exc:
                stale_listener.close()
                self.skipTest(f"Unix socket bind unavailable: {exc}")
            stale_listener.close()

            class FailingLoop:
                def close(self):
                    pass

                def run_until_complete(self, coroutine):
                    coroutine.close()
                    raise RuntimeError("socket unavailable")

            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

            self.assertFalse(os.path.exists(socket_path))

    def test_startup_failure_is_reported(self):
        class FailingLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                coroutine.close()
                raise RuntimeError("socket unavailable")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, "SOCKET_PATH", os.path.join(directory, "ephemeral.sock")), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        self.assertIn("Socket server error: socket unavailable", stderr.getvalue())
        self.assertEqual(server._socket_lifecycle()[0], "failed")
        self.assertIn("RuntimeError: socket unavailable", server._socket_lifecycle()[1])

    def test_successful_startup_runs_listener_until_shutdown(self):
        test_case = self

        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                test_case.assertEqual(server._socket_lifecycle()[0], "ready")
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            source_stat = os.stat(__file__)
            socket_stat = os.stat_result(
                (stat.S_IFSOCK | 0o600, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
            )
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())) as start, \
                    patch.object(server.os.path, "lexists", return_value=False), \
                    patch.object(server.os, "lstat", return_value=socket_stat), \
                    patch.object(server.os, "chmod") as chmod, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        start.assert_awaited_once_with(server.handle_socket_client, path=socket_path)
        chmod.assert_called_once_with(socket_path, 0o600)
        self.assertIn("listener stopped", stderr.getvalue())

    def test_shutdown_removes_socket_owned_by_listener(self):
        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        socket_stat = os.stat(__file__)
        socket_stat = os.stat_result((stat.S_IFSOCK | 0o600, socket_stat.st_ino, socket_stat.st_dev, 1, 0, 0, 0, 0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=False), \
                    patch.object(server.os, "lstat", return_value=socket_stat), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())), \
                    patch.object(server.os, "chmod"), \
                    patch.object(server, "_unlink_socket_if_identity", return_value=True) as cleanup, \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

        cleanup.assert_called_once_with(socket_path, (socket_stat.st_dev, socket_stat.st_ino))

    def test_shutdown_tolerates_socket_already_removed(self):
        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        source_stat = os.stat(__file__)
        socket_stat = os.stat_result(
            (stat.S_IFSOCK | 0o600, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=False), \
                    patch.object(server.os, "lstat", side_effect=[socket_stat, socket_stat]), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())), \
                    patch.object(server.os, "chmod"), \
                    patch.object(server, "_unlink_socket_if_identity", side_effect=FileNotFoundError()), \
                    patch.object(server.os, "unlink") as unlink, \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

        unlink.assert_not_called()

    def test_shutdown_tolerates_socket_removed_before_final_revalidation(self):
        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        source_stat = os.stat(__file__)
        socket_stat = os.stat_result(
            (stat.S_IFSOCK | 0o600, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=False), \
                    patch.object(server.os, "lstat", side_effect=[socket_stat, FileNotFoundError()]), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())), \
                    patch.object(server.os, "chmod"), \
                    patch.object(server, "_unlink_socket_if_identity") as cleanup, \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

        cleanup.assert_not_called()

    def test_shutdown_does_not_remove_replacement_socket(self):
        class Listener:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc_info):
                return False

            async def serve_forever(self):
                raise RuntimeError("listener stopped")

        class RunningLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                return asyncio.run(coroutine)

        bound_stat = os.stat(__file__)
        bound_stat = os.stat_result((stat.S_IFSOCK | 0o600, bound_stat.st_ino, bound_stat.st_dev, 1, 0, 0, 0, 0, 0, 0))
        replacement_stat = os.stat_result((stat.S_IFSOCK | 0o600, bound_stat.st_ino + 1, bound_stat.st_dev, 1, 0, 0, 0, 0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "ephemeral.sock")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.os.path, "lexists", return_value=False), \
                    patch.object(server.os, "lstat", side_effect=[bound_stat, replacement_stat]), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())), \
                    patch.object(server.os, "chmod"), \
                    patch.object(server.os, "unlink") as unlink, \
                    patch("sys.stderr", new_callable=io.StringIO):
                server.run_socket_server()

        unlink.assert_not_called()

    def test_socket_probe_failure_is_reported(self):
        class FailingLoop:
            def close(self):
                pass

        class FailingProbe:
            def connect(self, _path):
                raise OSError("probe failed")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "not-a-socket")
            Path(socket_path).write_text("occupied", encoding="utf-8")
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=FailingLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.socket, "socket", return_value=FailingProbe()), \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        self.assertIn("not a Unix socket", stderr.getvalue())

    def test_module_entrypoint_starts_listener_and_runs_mcp(self):
        class ReadyEvent:
            def clear(self):
                pass

            def wait(self, timeout):
                self.timeout = timeout
                return True

        class FakeThread:
            def start(self):
                self.started = True

        fake_thread = FakeThread()
        with patch.dict(os.environ, {"EPHEMERAL_DISABLE_SOCKET_SERVER": "0"}), \
                patch.object(server.threading, "Thread", return_value=fake_thread), \
                patch.object(server.threading, "Event", return_value=ReadyEvent()), \
                patch("mcp.server.fastmcp.FastMCP", return_value=server.mcp), \
                patch.object(server.mcp, "run") as mcp_run:
            runpy.run_path(server.__file__, run_name="__main__")

        self.assertTrue(fake_thread.started)
        mcp_run.assert_called_once_with()

    def test_module_entrypoint_rejects_missing_required_isolation(self):
        with patch.dict(
            os.environ,
            {"EPHEMERAL_REQUIRE_ISOLATION": "1"},
            clear=True,
        ), self.assertRaisesRegex(SystemExit, "Socket isolation is required"):
            runpy.run_path(server.__file__, run_name="__main__")


if __name__ == "__main__":
    unittest.main()
