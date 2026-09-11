"""Unit tests for MCP tool validation, response formatting, and socket IPC."""

import asyncio
import io
import json
import os
import runpy
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from metrics import LocalMetrics

os.environ.setdefault("EPHEMERAL_DISABLE_SOCKET_SERVER", "1")
import server


class FakeReader:
    def __init__(self, payload):
        self.payload = payload
        self.consumed = False

    async def read(self, _limit):
        if self.consumed:
            return b""
        self.consumed = True
        return self.payload


class ChunkedReader:
    def __init__(self, *chunks):
        self.chunks = iter(chunks)

    async def read(self, _limit):
        return next(self.chunks, b"")


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


class TestServerTools(unittest.TestCase):
    def setUp(self):
        server.engine.clear("all")
        self.original_limit = server.engine.max_buffer_bytes
        self.original_model = server.engine.embedding_model
        server.engine.embedding_model = type(
            "TestEmbedding",
            (),
            {"embed": lambda _self, texts: [[0.0] * 384 for _ in texts]},
        )()

    def tearDown(self):
        server.engine.max_buffer_bytes = self.original_limit
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

        self.assertIn("TIMED OUT after 0.5s", result)
        run.assert_called_once()

    def test_consolidate_captures_preserves_sources_and_is_searchable(self):
        server.capture_text("alpha failure\nalpha detail", label="repo-a")
        server.capture_text("beta success\nbeta detail", label="repo-b")
        source_ids = [item["capture_id"] for item in reversed(server.engine.list_captures())]

        result = json.loads(server.consolidate_captures(source_ids, max_bytes=2048))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source_capture_ids"], source_ids)
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(result["record_count"], 4)
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

    def test_consolidate_captures_validates_limits(self):
        self.assertIn("max_captures must be at least 1", server.consolidate_captures(max_captures=0))
        self.assertIn("max_bytes must be at least 512", server.consolidate_captures(max_bytes=511))
        server.engine.max_buffer_bytes = 512
        self.assertIn("exceeds the configured buffer limit", server.consolidate_captures(max_bytes=513))

    def test_consolidate_uses_compact_payload_when_pretty_payload_is_too_large(self):
        server.capture_text("compact payload", label="compact")
        source_id = server.engine.list_captures()[0]["capture_id"]

        result = server._consolidated_jsonl([source_id], max_captures=1, max_bytes=512)

        self.assertLessEqual(len(result["content"].encode("utf-8")), 512)
        self.assertNotIn("\n  ", result["content"])

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
        self.assertIn("Process RSS:", result)
        self.assertIn("Unaccounted RSS bytes:", result)

    def test_runtime_diagnostics_reports_content_free_metadata(self):
        server.capture_text("secret command output", label="private-label")

        with patch.dict(
            os.environ,
            {"EPHEMERAL_SESSION_ID": "diagnostic-session"},
            clear=False,
        ):
            result = server.get_runtime_diagnostics()

        self.assertIn("Runtime diagnostics (content-free):", result)
        self.assertIn("Package version: 0.2.0", result)
        self.assertIn("Python:", result)
        self.assertIn("Socket mode: session-derived path", result)
        self.assertIn("Socket lifecycle:", result)
        self.assertIn("Session ID configured: yes", result)
        self.assertIn("Captures: 1/", result)
        self.assertIn("Embedding model:", result)
        self.assertNotIn("secret command output", result)
        self.assertNotIn("private-label", result)

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

    def test_mcp_instructions_report_missing_required_isolation(self):
        with patch.object(server, "socket_isolation_required", return_value=True), \
                patch.object(server, "socket_isolation_configured", return_value=False):
            instructions = server._mcp_instructions()

        self.assertIn("startup must fail", instructions)

    def test_mcp_instructions_report_failed_socket(self):
        with patch.object(server, "_socket_lifecycle", return_value=("failed", "RuntimeError: unavailable")):
            instructions = server._mcp_instructions()

        self.assertIn("Socket lifecycle is failed (RuntimeError: unavailable)", instructions)

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
        self.assertIn('"captures": 1', result)
        self.assertNotIn("secret metrics payload", result)
        self.assertNotIn("private metrics label", result)

    def test_runtime_package_version_falls_back_to_source_checkout(self):
        with patch.object(server.Path, "read_text", side_effect=OSError("missing metadata")), \
                patch.object(server, "package_version", side_effect=server.PackageNotFoundError()):
            self.assertEqual(server._runtime_package_version(), "source checkout")

    def test_capture_file_reads_and_labels_content(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "capture.log"
            file_path.write_text("file content", encoding="utf-8")

            result = server.capture_file(str(file_path))

        self.assertIn("Captured into ID", result)
        self.assertIn("capture.log", result)

    def test_capture_text_formats_diff_metadata(self):
        diff = """diff --git a/old.txt b/new.txt
--- a/old.txt
+++ b/new.txt
@@ -1 +1 @@
-old
+new
"""
        result = server.capture_text(diff, label="patch", content_type="diff")

        self.assertIn("Unified Diff", result)
        self.assertIn("new.txt", result)

    def test_execute_reports_failed_and_truncated_command(self):
        output = "x" * 700
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(output, 7, True, 700, False),
        ):
            result = server.execute_and_capture("failing-command", max_output_bytes=1024)

        self.assertIn("FAILED (Exit Code 7)", result)
        self.assertIn("truncated from 700 bytes", result)

    def test_execute_formats_diff_command_response(self):
        diff = "diff --git a/old.txt b/new.txt\n--- a/old.txt\n+++ b/new.txt\n@@ -1 +1 @@\n-old\n+new\n"
        with patch.object(
            server,
            "run_command_bounded",
            return_value=(diff, 0, False, len(diff.encode()), False),
        ):
            result = server.execute_and_capture("git diff", max_output_bytes=1024)

        self.assertIn("Type: Unified Diff", result)
        self.assertIn("Modified Files Map", result)
        self.assertIn("new.txt", result)

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
            summary = server.get_capture_summary("cap")
        self.assertIn("Modified Files Map", summary)
        self.assertIn("truncated from 20 bytes", summary)

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
            regular = server.get_capture_summary("cap")
        self.assertIn("Head (First 5 lines)", regular)
        self.assertIn("tail", regular)

    def test_empty_and_populated_capture_listing(self):
        self.assertIn("buffer is empty", server.list_captures())
        server.capture_text("one line", label="listed")
        self.assertIn("listed", server.list_captures())

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
        payload = json.dumps({"label": "socket-test", "text": "hello"}).encode()
        capture = SimpleNamespace(
            capture_id="cap_socket",
            label="socket-test",
            line_count=1,
            byte_size=5,
        )
        with patch.object(server, "to_thread", new=AsyncMock(return_value=capture)):
            writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "socket-test")
        self.assertTrue(writer.closed)

    async def test_json_payload_is_reassembled_across_socket_reads(self):
        payload = json.dumps({"label": "chunked", "text": "complete payload"}).encode()
        capture = SimpleNamespace(
            capture_id="cap_chunked",
            label="chunked",
            line_count=1,
            byte_size=16,
        )
        writer = FakeWriter()
        with patch.object(server, "to_thread", new=AsyncMock(return_value=capture)):
            await server.handle_socket_client(
                ChunkedReader(payload[:7], payload[7:]),
                writer,
            )

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "chunked")

    async def test_escape_heavy_payload_within_capture_limit_is_accepted(self):
        server.engine.max_buffer_bytes = 100_000
        text = "\x00" * server.engine.max_buffer_bytes
        payload = json.dumps({"label": "escaped", "text": text}).encode()
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
        with patch.object(server, "to_thread", new=AsyncMock(return_value=capture)):
            writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "escaped")

    async def test_oversized_payload_is_rejected_across_socket_reads(self):
        payload = b"x" * (
            server.engine.max_buffer_bytes * server.SOCKET_JSON_MAX_EXPANSION
            + server.SOCKET_PAYLOAD_OVERHEAD
        )
        writer = FakeWriter()

        await server.handle_socket_client(
            ChunkedReader(payload[:128], payload[128:]),
            writer,
        )

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "error")
        self.assertIn("exceeds", response["message"])

    async def test_ingest_is_offloaded_from_event_loop(self):
        payload = json.dumps({"label": "offload-test", "text": "hello"}).encode()
        capture = SimpleNamespace(
            capture_id="cap_offload",
            label="offload-test",
            line_count=1,
            byte_size=5,
        )
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(return_value=capture),
        ) as offload:
            await self.run_handler(payload)

        offload.assert_awaited_once()

    async def test_oversized_payload_returns_error_response(self):
        payload = b"x" * (
            server.engine.max_buffer_bytes * server.SOCKET_JSON_MAX_EXPANSION
            + server.SOCKET_PAYLOAD_OVERHEAD
        )

        writer = await self.run_handler(payload)

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "error")
        self.assertIn("exceeds", response["message"])
        self.assertTrue(writer.closed)

    async def test_empty_payload_closes_without_response(self):
        writer = await self.run_handler(b"")

        self.assertEqual(writer.writes, [])
        self.assertTrue(writer.closed)

    async def test_malformed_payload_falls_back_to_plain_text(self):
        capture = SimpleNamespace(
            capture_id="cap_plain",
            label="CLI pipe",
            line_count=1,
            byte_size=8,
        )
        with patch.object(server, "to_thread", new=AsyncMock(return_value=capture)):
            writer = await self.run_handler(b"not-json")

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["label"], "CLI pipe")

    async def test_ingest_failure_returns_error_response(self):
        with patch.object(
            server,
            "to_thread",
            new=AsyncMock(side_effect=ValueError("invalid capture")),
        ):
            writer = await self.run_handler(json.dumps({"text": "payload"}).encode())

        response = json.loads(writer.writes[0])
        self.assertEqual(response["status"], "error")
        self.assertIn("invalid capture", response["message"])


class TestSocketServerStartup(unittest.TestCase):
    def test_import_does_not_start_socket_listener(self):
        with patch.dict(os.environ, {"EPHEMERAL_DISABLE_SOCKET_SERVER": "0"}), \
                patch.object(server.threading, "Thread") as thread:
            runpy.run_path(server.__file__, run_name="server_import")

        thread.assert_not_called()

    def test_explicit_socket_startup_honors_disabled_mode(self):
        with patch.dict(os.environ, {"EPHEMERAL_DISABLE_SOCKET_SERVER": "1"}):
            self.assertIsNone(server.start_socket_server())

        self.assertEqual(server._socket_lifecycle()[0], "disabled")

    def test_socket_startup_timeout_is_reported(self):
        event = SimpleNamespace(wait=lambda timeout: False)
        with patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "SOCKET_STARTUP_TIMEOUT_SECONDS", 3):
            with self.assertRaisesRegex(SystemExit, "did not become ready within 3 seconds"):
                server._require_socket_ready()

    def test_socket_startup_failure_is_reported_to_entrypoint(self):
        event = SimpleNamespace(wait=lambda timeout: True)
        with patch.object(server, "_SOCKET_STARTUP_EVENT", event), \
                patch.object(server, "_socket_lifecycle", return_value=("failed", "RuntimeError: unavailable")):
            with self.assertRaisesRegex(SystemExit, "failed to start: RuntimeError: unavailable"):
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
            Path(socket_path).write_text("occupied", encoding="utf-8")
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
            with patch.object(server, "SOCKET_PATH", socket_path), \
                    patch.object(server.asyncio, "new_event_loop", return_value=RunningLoop()), \
                    patch.object(server.asyncio, "set_event_loop"), \
                    patch.object(server.asyncio, "start_unix_server", new=AsyncMock(return_value=Listener())) as start, \
                    patch.object(server.os, "chmod") as chmod, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                server.run_socket_server()

        start.assert_awaited_once_with(server.handle_socket_client, path=socket_path)
        chmod.assert_called_once_with(socket_path, 0o600)
        self.assertIn("listener stopped", stderr.getvalue())

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

        self.assertIn("Unable to verify existing socket", stderr.getvalue())

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
