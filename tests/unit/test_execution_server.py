"""Tests for resumable-execution MCP tool adapters."""

import asyncio
import errno
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path

from jsonschema import Draft202012Validator

from ephemeral_buffer_mcp.config import DEFAULT_MAX_BUFFER_BYTES

os.environ.setdefault("EPHEMERAL_DISABLE_SOCKET_SERVER", "1")
os.environ.setdefault("EPHEMERAL_TEST_EMBEDDINGS", "1")

import ephemeral_buffer_mcp.server as server
import ephemeral_buffer_mcp.mcp_transport as mcp_transport
import ephemeral_buffer_mcp.socket_transport as socket_transport
import ephemeral_buffer_mcp.tools.execution as execution_tools
import ephemeral_buffer_mcp.tools.execution_responses as execution_responses
from ephemeral_buffer_mcp.execution import (
    ExecutionBusyError,
    ExecutionDiskSpaceError,
    ExecutionPhaseNotFoundError,
    ExecutionQuotaError,
    ExecutionRecordError,
    PhaseExecutionManager,
)


def mcp_result_text(result):
    if hasattr(result, "content"):
        content = result.content
    else:
        content = result[0] if isinstance(result, tuple) else result
    if isinstance(content, str):
        return content
    return content[0].text


class ToolRunner:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def __call__(self, command, cwd, max_output_bytes, timeout_seconds):
        self.calls.append(command)
        return self.results.get(command, (f"{command} output", 0, False, len(command), False))


class TestExecutionTools(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runner = ToolRunner()
        self.manager = PhaseExecutionManager(
            self.directory.name,
            max_output_bytes=server.engine.max_buffer_bytes,
            command_runner=self.runner,
        )
        self.manager_patch = patch.object(server, "execution_manager", self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        server.engine.clear("all")

    def tearDown(self):
        shutdown_result = self.manager.shutdown(grace_seconds=10.0)
        self.assertEqual(
            shutdown_result["unfinished_execution_ids"],
            [],
            "execution manager must finish background state writes before fixture cleanup",
        )
        server.engine.clear("all")
        self.directory.cleanup()

    @staticmethod
    def phase(name, command, **overrides):
        value = {"name": name, "command": command}
        value.update(overrides)
        return value

    @staticmethod
    def public_execution_payload(execution_id, phase_names):
        resume = {
            "available": False,
            "retry_required": False,
            "unsafe_confirmation_required": False,
            "attempt_limit_reached": False,
            "next_phase": None,
            "policy": "safe",
            "blocked_reason": None,
        }
        phases = [
            {
                "name": name,
                "status": "pending",
                "attempts": 0,
                "events": [{"event": "phase_pending", "detail": "x" * 256}],
            }
            for name in phase_names
        ]
        return {
            "status": "ok",
            "schema_version": 1,
            "execution_id": execution_id,
            "label": "large response",
            "execution_status": "running",
            "execution_in_progress": True,
            "partial": False,
            "summary": "running",
            "created_at": "2026-10-28T00:00:00Z",
            "updated_at": "2026-10-28T00:00:00Z",
            "resume": resume,
            "completed_phase_count": 0,
            "phase_count": len(phases),
            "phases": phases,
        }

    def assert_valid_execution_mcp_result(self, result, tool_name):
        payload = json.loads(result.content[0].text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["data"], payload)
        self.assertEqual(
            result.content[0].text,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        self.assertLessEqual(
            mcp_transport._call_tool_result_bytes(
                server.ToolResponseEnvelope(
                    status=result.structured_content["status"],
                    data=result.structured_content["data"],
                    text=result.content[0].text,
                    truncated=result.structured_content["truncated"],
                    next_cursor=result.structured_content["next_cursor"],
                    payload_schema_version=result.structured_content[
                        "payload_schema_version"
                    ],
                )
            ),
            server.MCP_TOOL_RESPONSE_MAX_BYTES - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
        )
        listed_tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
        Draft202012Validator(listed_tools[tool_name].output_schema).validate(
            result.structured_content
        )
        return payload

    def test_start_resume_get_output_and_list_tools(self):
        started = json.loads(
            server.start_execution(
                [self.phase("build", "build"), self.phase("test", "test")],
                execution_id="tool-execution",
                label="tool run",
            )
        )
        self.assertEqual(started["execution_status"], "completed")
        self.assertTrue(started["phases"][0]["result"]["capture_id"].startswith("cap_"))
        self.assertEqual(
            started["phases"][0]["result"]["capture_session_id"],
            server.engine.session_id,
        )
        self.assertTrue(started["phases"][0]["result"]["capture_available"])
        self.assertEqual(json.loads(server.resume_execution("tool-execution"))["execution_status"], "completed")
        self.assertEqual(json.loads(server.get_execution("tool-execution"))["partial"], False)
        with_output = json.loads(server.get_execution("tool-execution", include_output=True))
        self.assertEqual(with_output["phases"][0]["output"], "build output")
        output = json.loads(server.get_execution_output("tool-execution", "test"))
        self.assertEqual(output["phases"][0]["output"], "test output")
        missing_phase = json.loads(
            server.get_execution_output("tool-execution", "missing-phase")
        )
        self.assertEqual(missing_phase["error"]["code"], "phase_not_found")
        listed = json.loads(server.list_executions())
        self.assertEqual(listed["executions"][0]["execution_id"], "tool-execution")

    def test_execution_mcp_payloads_are_versioned_and_match_legacy_json_text(self):
        self.manager.start(
            [self.phase("versioned-phase", "prepared")],
            execution_id="versioned-workflow",
        )

        async def invoke(name, arguments):
            result = await server.mcp.call_tool(name, arguments)
            try:
                payload = json.loads(result.content[0].text)
            except json.JSONDecodeError:
                self.fail(
                    f"{name} did not preserve its JSON text result: "
                    f"{result.structured_content!r}"
                )
            self.assertEqual(
                result.content[0].text,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            self.assertEqual(result.structured_content["data"], payload)
            self.assertEqual(result.structured_content["schema_version"], 1)
            self.assertEqual(result.structured_content["payload_schema_version"], 1)
            return result

        async def exercise():
            started = await invoke(
                "start_execution",
                {
                    "phases": [self.phase("mcp-phase", "started")],
                    "execution_id": "mcp-versioned-workflow",
                },
            )
            start_future = self.manager._background_futures.get(
                "mcp-versioned-workflow"
            )
            if start_future is not None:
                start_future.result(timeout=10)

            responses = [
                started,
                await invoke("resume_execution", {"execution_id": "versioned-workflow"}),
                await invoke("cancel_execution", {"execution_id": "versioned-workflow"}),
                await invoke("get_execution", {"execution_id": "versioned-workflow"}),
                await invoke(
                    "get_execution_output",
                    {"execution_id": "versioned-workflow", "phase_name": "versioned-phase"},
                ),
                await invoke(
                    "get_execution_output",
                    {
                        "execution_id": "missing-output-contract-workflow",
                        "phase_name": "missing-phase",
                    },
                ),
                await invoke("list_executions", {"limit": 25}),
                await invoke("get_execution_capacity", {}),
                await invoke(
                    "retire_executions",
                    {"execution_ids": ["versioned-workflow"], "dry_run": True},
                ),
                await invoke("get_execution", {"execution_id": "missing-contract-workflow"}),
            ]
            return responses

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
            responses = asyncio.run(exercise())
        self.assertTrue(all(response.structured_content["payload_schema_version"] == 1 for response in responses))
        self.assertEqual(responses[-1].structured_content["status"], "error")
        self.assertEqual(responses[-1].structured_content["error"]["code"], "execution_not_found")

        record_path = self.manager.store._path("versioned-workflow")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["schema_version"] = 999
        record_path.write_text(json.dumps(record), encoding="utf-8")
        diagnostic = asyncio.run(
            server.mcp.call_tool(
                "get_execution", {"execution_id": "versioned-workflow"}
            )
        )
        payload = json.loads(diagnostic.content[0].text)
        self.assertEqual(diagnostic.structured_content["payload_schema_version"], 1)
        self.assertEqual(diagnostic.structured_content["schema_version"], 1)
        self.assertEqual(diagnostic.structured_content["data"], payload)
        self.assertEqual(diagnostic.structured_content["data"]["schema_version"], 999)

    def test_service_context_allocates_execution_storage_on_first_use(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = os.path.join(directory, "lazy-execution-state")
            context = server.create_service_context(
                state_dir=state_dir,
                metrics_file=None,
                engine_options={"semantic_prefetch": False, "embedding_warmup": False},
            )
            try:
                self.assertFalse(os.path.exists(state_dir))
                self.assertIsNone(context._execution_manager)

                manager = context.execution_manager
                manager.command_runner = self.runner
                self.assertFalse(os.path.exists(state_dir))

                with server._activate_service_context(context):
                    result = json.loads(
                        server.start_execution(
                            [self.phase("build", "build")],
                            execution_id="context-lazy-state",
                        )
                    )

                self.assertEqual(result["execution_status"], "completed")
                self.assertTrue(os.path.isdir(state_dir))
            finally:
                context.close()

    def test_capacity_and_retirement_tools(self):
        server.start_execution(
            [self.phase("retire", "retire")],
            execution_id="tool-retire",
        )
        capacity = json.loads(server.get_execution_capacity())
        self.assertEqual(capacity["record_count"], 1)
        self.assertEqual(capacity["paired_record_count"], 1)

        preview = json.loads(server.retire_executions(["tool-retire"]))
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["eligible_count"], 1)
        self.assertGreater(preview["projected_reclaimed_bytes"], 0)

        with tempfile.TemporaryDirectory() as archive_directory:
            result = json.loads(server.retire_executions(
                ["tool-retire"],
                archive_path=os.path.join(archive_directory, "tool-record.tar"),
                dry_run=False,
            ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(json.loads(server.list_executions())["executions"], [])

    def test_partial_timeout_is_machine_readable_and_capture_is_searchable(self):
        self.runner.results["timeout"] = ("partial output", 124, True, 2000, True)
        result = json.loads(
            server.start_execution(
                [self.phase("timeout", "timeout")],
                execution_id="tool-timeout",
            )
        )
        self.assertEqual(result["execution_status"], "partial")
        self.assertTrue(result["partial"])
        self.assertIn("partial", result["summary"])
        capture_id = result["phases"][0]["result"]["capture_id"]
        self.assertTrue(capture_id.startswith("cap_"))
        summary = json.loads(server.get_capture_summary(capture_id))
        self.assertEqual(summary["status"], "timed_out")
        self.assertTrue(summary["partial"])

    def test_execution_output_is_retrievable_in_bounded_chunks(self):
        self.runner.results["large"] = ("0123456789" * 200, 0, False, 2000, False)
        started = json.loads(
            server.start_execution(
                [self.phase("large", "large")],
                execution_id="chunked-output",
            )
        )
        chunk = json.loads(
            server.get_execution_output(
                "chunked-output", "large", offset=7, max_bytes=512
            )
        )
        phase = chunk["phases"][0]
        self.assertEqual(len(phase["output"]), 512)
        self.assertTrue(phase["output"].startswith("7890123456"))
        self.assertTrue(phase["truncated"])
        self.assertEqual(phase["next_offset"], 519)

    def test_control_character_output_stays_within_serialized_response_budget(self):
        self.runner.results["control"] = ("\x00" * 8192, 0, False, 8192, False)
        server.start_execution(
            [self.phase("control", "control")],
            execution_id="control-output",
        )
        response = server.get_execution_output("control-output", "control", max_bytes=8192)
        self.assertFalse(response.startswith("Error managing execution: response exceeds"))
        self.assertEqual(len(json.loads(response)["phases"][0]["output"]), 8192)

    def test_mcp_control_character_output_fits_budget_and_remains_pageable(self):
        self.runner.results["mcp-control"] = ("\x00" * 8192, 0, False, 8192, False)
        self.runner.results["mcp-ascii"] = ("a" * 8192, 0, False, 8192, False)
        server.start_execution(
            [self.phase("control", "mcp-control")],
            execution_id="mcp-control-output",
        )
        server.start_execution(
            [self.phase("ascii", "mcp-ascii")],
            execution_id="mcp-ascii-output",
        )

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        async def get_page(execution_id, phase_name, offset=0):
            return await server.mcp.call_tool(
                "get_execution_output",
                {
                    "execution_id": execution_id,
                    "phase_name": phase_name,
                    "offset": offset,
                    "max_bytes": 8192,
                },
            )

        with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
            control = asyncio.run(get_page("mcp-control-output", "control"))
            ascii_result = asyncio.run(get_page("mcp-ascii-output", "ascii"))

        def assert_valid_response(result):
            payload = json.loads(result.content[0].text)
            self.assertFalse(result.is_error)
            self.assertEqual(result.structured_content["data"], payload)
            self.assertEqual(
                result.content[0].text,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            envelope = server.ToolResponseEnvelope(
                status="ok",
                data=payload,
                text=result.content[0].text,
                payload_schema_version=1,
            )
            self.assertLessEqual(
                mcp_transport._call_tool_result_bytes(envelope),
                server.MCP_TOOL_RESPONSE_MAX_BYTES
                - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
            )
            return payload

        control_payload = assert_valid_response(control)
        control_phase = control_payload["phases"][0]
        self.assertLess(len(control_phase["output"].encode("utf-8")), 8192)
        self.assertTrue(control_phase["truncated"])
        self.assertEqual(
            control_phase["next_offset"],
            len(control_phase["output"].encode("utf-8")),
        )

        ascii_payload = assert_valid_response(ascii_result)
        self.assertEqual(len(ascii_payload["phases"][0]["output"]), 8192)
        self.assertFalse(ascii_payload["phases"][0]["truncated"])

        with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
            continuation = asyncio.run(
                get_page(
                    "mcp-control-output",
                    "control",
                    control_phase["next_offset"],
                )
            )
        continuation_payload = assert_valid_response(continuation)
        continuation_phase = continuation_payload["phases"][0]
        self.assertEqual(continuation_phase["offset"], control_phase["next_offset"])
        self.assertEqual(
            continuation_phase["output"],
            "\x00" * len(continuation_phase["output"]),
        )

    def test_mcp_execution_output_returns_valid_error_if_minimum_page_cannot_fit(self):
        too_large = execution_responses._execution_error_json(
            "response_too_large",
            "Execution response exceeds its configured byte limit.",
        )

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(execution_tools, "_execution_json", return_value=too_large):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                result = asyncio.run(
                    server.mcp.call_tool(
                        "get_execution_output",
                        {
                            "execution_id": "minimum-page-too-large",
                            "phase_name": "phase",
                            "max_bytes": 8192,
                        },
                    )
                )

        payload = json.loads(result.content[0].text)
        self.assertTrue(result.is_error)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "response_too_large")
        self.assertEqual(result.structured_content["data"], payload)
        self.assertEqual(
            result.content[0].text,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    def test_mcp_compact_execution_output_fallback_matches_typed_contract(self):
        self.runner.results["compact-control-one"] = (
            "\x00" * 4096,
            0,
            False,
            4096,
            False,
        )
        self.runner.results["compact-control-two"] = (
            "\x00" * 4096,
            0,
            False,
            4096,
            False,
        )
        server.start_execution(
            [
                self.phase("phase-one", "compact-control-one"),
                self.phase("phase-two", "compact-control-two"),
            ],
            execution_id="compact-control-output",
        )

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        # Lower the helper's JSON cap to exercise its real compact-success path
        # deterministically with the multi-phase control-character output.
        with patch.object(execution_tools, "EXECUTION_OUTPUT_RESPONSE_MAX_BYTES", 512):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                result = asyncio.run(
                    server.mcp.call_tool(
                        "get_execution_output",
                        {
                            "execution_id": "compact-control-output",
                            "max_bytes": 8192,
                        },
                    )
                )

        payload = json.loads(result.content[0].text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["data"], payload)
        self.assertTrue(payload["response_truncated"])
        self.assertNotIn("max_bytes", payload)
        self.assertEqual(
            payload["phases"],
            [
                {
                    "name": "phase-one",
                    "status": "completed",
                    "attempts": None,
                    "error": None,
                },
                {
                    "name": "phase-two",
                    "status": "completed",
                    "attempts": None,
                    "error": None,
                },
            ],
        )
        execution_tools.PAYLOAD_CONTRACTS["get_execution_output"].validate_python(
            payload,
            strict=True,
        )
        listed_tools = {
            tool.name: tool for tool in asyncio.run(server.mcp.list_tools())
        }
        Draft202012Validator(
            listed_tools["get_execution_output"].output_schema
        ).validate(result.structured_content)

    def test_mcp_execution_output_preserves_operation_error_during_page_reduction(self):
        self.runner.results["changing-output"] = ("\x00" * 8192, 0, False, 8192, False)
        server.start_execution(
            [self.phase("changing", "changing-output")],
            execution_id="changing-output",
        )
        original_output = self.manager.output
        calls = 0

        def output_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_output(*args, **kwargs)
            raise KeyError("execution changed during paging")

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(self.manager, "output", side_effect=output_then_fail):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                result = asyncio.run(
                    server.mcp.call_tool(
                        "get_execution_output",
                        {
                            "execution_id": "changing-output",
                            "phase_name": "changing",
                            "max_bytes": 8192,
                        },
                    )
                )

        payload = json.loads(result.content[0].text)
        self.assertTrue(result.is_error)
        self.assertEqual(payload["error"]["code"], "execution_not_found")
        self.assertEqual(result.structured_content["data"], payload)
        self.assertGreater(calls, 1)

    def test_durable_execution_marks_session_capture_unavailable_after_restart(self):
        original_engine = server.EphemeralEngine(
            max_buffer_bytes=server.engine.max_buffer_bytes
        )
        fresh_engine = server.EphemeralEngine(
            max_buffer_bytes=server.engine.max_buffer_bytes
        )
        try:
            with patch.object(server, "engine", original_engine):
                started = json.loads(
                    server.start_execution(
                        [self.phase("durable", "durable")],
                        execution_id="durable-capture",
                    )
                )
                phase_result = started["phases"][0]["result"]
                original_capture_id = phase_result["capture_id"]
                self.assertEqual(original_capture_id, "cap_1")
                self.assertEqual(phase_result["capture_session_id"], original_engine.session_id)
                self.assertTrue(phase_result["capture_available"])

                unrelated_capture = fresh_engine.ingest("UNRELATED NEW SERVER OUTPUT")
                self.assertEqual(unrelated_capture.capture_id, original_capture_id)
                self.assertNotEqual(fresh_engine.session_id, original_engine.session_id)

                fresh_manager = PhaseExecutionManager(
                    self.directory.name,
                    max_output_bytes=fresh_engine.max_buffer_bytes,
                    command_runner=self.runner,
                )
                with patch.object(server, "execution_manager", fresh_manager), \
                        patch.object(server, "engine", fresh_engine):
                    result = json.loads(server.get_execution("durable-capture"))
                    durable_output = json.loads(
                        server.get_execution_output("durable-capture", "durable")
                    )
            self.assertFalse(result["phases"][0]["result"]["capture_available"])
            self.assertEqual(durable_output["phases"][0]["output"], "durable output")
        finally:
            original_engine.shutdown()
            fresh_engine.shutdown()

    def test_legacy_execution_capture_reference_is_unavailable(self):
        capture = server.engine.ingest("current session output")
        payload = {"phases": [{"result": {"capture_id": capture.capture_id}}]}

        self.assertIs(execution_responses._execution_public_payload(payload), payload)
        self.assertFalse(payload["phases"][0]["result"]["capture_available"])

    def test_unsafe_side_effect_alias_normalizes_before_manager_validation(self):
        ordinary = server.ExecutionPhaseInput(name="ordinary", command="ordinary")
        self.assertEqual(ordinary.side_effects, "none")
        phase = server.ExecutionPhaseInput(
            name="publish",
            command="publish",
            unsafe_side_effects=True,
        )
        self.assertEqual(phase.side_effects, "unsafe")
        with self.assertRaises(ValueError):
            server.ExecutionPhaseInput(
                name="conflicting",
                command="conflicting",
                side_effects="none",
                unsafe_side_effects=True,
            )
        result = json.loads(
            server.start_execution([phase], execution_id="unsafe-alias")
        )
        self.assertEqual(result["phases"][0]["side_effects"], "unsafe")

    def test_low_engine_buffer_does_not_break_server_import(self):
        environment = os.environ.copy()
        environment.update({
            "EPHEMERAL_DISABLE_SOCKET_SERVER": "1",
            "EPHEMERAL_MAX_BUFFER_BYTES": "256",
            "EPHEMERAL_TEST_EMBEDDINGS": "1",
        })
        completed = subprocess.run(
            [sys.executable, "-c", "import ephemeral_buffer_mcp.server"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_execution_payload_preserves_phases_without_capture_references(self):
        payload = {"phases": [{"result": None}, {"result": {"exit_code": 0}}]}
        self.assertIs(execution_responses._execution_public_payload(payload), payload)
        self.assertNotIn("capture_available", payload["phases"][1]["result"])
        self.assertEqual(execution_responses._json_dumps_with_limit({"ok": True}, None), '{"ok":true}')

    def test_execution_compaction_helpers_cover_optional_payload_shapes(self):
        self.assertEqual(execution_responses._compact_execution_list_payload({}), {})
        compact_list = execution_responses._compact_execution_list_payload(
            {
                "executions": [
                    None,
                    {
                        "status": "invalid_record",
                        "execution_id": None,
                        "record_file_id": "record-file",
                        "schema_version": 2,
                        "reason_code": "MALFORMED_JSON",
                        "read_only": True,
                    },
                    {"status": "invalid_record", "execution_id": None},
                    {
                        "status": "ok",
                        "execution_id": "execution-id",
                        "phase_count": 2,
                    },
                ]
            }
        )
        self.assertEqual(compact_list["returned_count"], 2)
        self.assertEqual(compact_list["omitted_count"], 2)
        self.assertEqual(
            compact_list["executions"][0]["record_file_id"], "record-file"
        )
        self.assertEqual(
            compact_list["executions"][1]["execution_id"], "execution-id"
        )

        self.assertEqual(
            execution_responses._compact_execution_output_payload({"phases": None}),
            {},
        )
        compact_output = execution_responses._compact_execution_output_payload(
            {"phases": [None, {"name": "phase", "status": "completed"}]}
        )
        self.assertEqual(compact_output["phases"], [{"name": "phase", "status": "completed"}])

        self.assertEqual(
            execution_responses._compact_execution_retirement_payload(
                {"items": "malformed"}
            ),
            {},
        )
        compact_retirement = execution_responses._compact_execution_retirement_payload(
            {
                "status": "partial",
                "items": [None, {"execution_id": "retired-id", "eligible": True}],
                "retired": [None, {"execution_id": "retired-id"}],
                "removal_errors": [None, {"execution_id": "failed-id", "error": "OSError"}],
            }
        )
        self.assertEqual(compact_retirement["items"], [{"execution_id": "retired-id", "eligible": True}])
        self.assertEqual(compact_retirement["retired"], [{"execution_id": "retired-id"}])
        self.assertEqual(
            compact_retirement["removal_errors"],
            [{"execution_id": "failed-id", "error": "OSError"}],
        )

        self.assertTrue(
            execution_responses._execution_compact_candidates(
                "start_execution", {"execution_id": "execution-id", "phases": []}
            )
        )
        self.assertTrue(
            execution_responses._execution_compact_candidates(
                "get_execution_output", {"phases": []}
            )
        )
        self.assertEqual(
            execution_responses._execution_compact_candidates(
                "list_executions", {"executions": "malformed"}
            ),
            [],
        )
        self.assertEqual(
            execution_responses._execution_compact_candidates(
                "retire_executions", {"items": "malformed"}
            ),
            [],
        )
        self.assertEqual(
            execution_responses._execution_compact_candidates("unknown", {}), []
        )

    def test_execution_identity_fallbacks_and_output_metadata_selection(self):
        self.assertEqual(execution_responses._execution_identity_details({}), {})
        details = execution_responses._execution_identity_details(
            {
                "execution_id": "top-level-id",
                "executions": [
                    None,
                    {},
                    {"record_file_id": "record-file-id"},
                ],
                "items": [],
                "retired": [{"execution_id": "retired-id"}],
            }
        )
        self.assertEqual(details["execution_id"], "top-level-id")
        self.assertEqual(details["executions"], [{"record_file_id": "record-file-id"}])
        self.assertEqual(details["execution_ids"], ["retired-id"])

        response = server.ToolResponseEnvelope(
            status="ok",
            data={"execution_id": "oversized-id"},
            text="oversized text",
        )
        compact_error = execution_responses._compact_execution_error_response(response)
        self.assertEqual(compact_error.error.code, "response_too_large")
        self.assertEqual(compact_error.error.details["execution_id"], "oversized-id")
        self.assertEqual(json.loads(compact_error.text), compact_error.data)
        self.assertIs(
            execution_responses.fit_execution_mcp_response("get_execution", response),
            response,
        )

        class Manager:
            def __init__(self, metadata, full):
                self.metadata = metadata
                self.full = full
                self.calls = []

            def public(self, execution_id, *, include_output):
                self.calls.append((execution_id, include_output))
                return self.full if include_output else self.metadata

        metadata = {"phases": [None, {}, {"output_bytes": 4}, {"output_bytes": 1.5}]}
        manager = Manager(metadata, {"full_output": True})
        with patch.object(
            execution_responses, "get_active_execution_manager", return_value=manager
        ):
            self.assertIs(execution_responses._execution_get_payload("execution-id", False), metadata)
            self.assertEqual(
                execution_responses._execution_get_payload("execution-id", True),
                {"full_output": True},
            )
        self.assertEqual(manager.calls, [("execution-id", False), ("execution-id", False), ("execution-id", True)])

        large_metadata = {
            "phases": [{"output_bytes": execution_responses.EXECUTION_OUTPUT_RESPONSE_MAX_BYTES + 1}]
        }
        manager = Manager(large_metadata, {"full_output": True})
        with patch.object(
            execution_responses, "get_active_execution_manager", return_value=manager
        ):
            result = execution_responses._execution_get_payload("large-id", True)
        self.assertTrue(result["response_truncated"])
        self.assertEqual(manager.calls, [("large-id", False)])

    def test_socket_probe_paths_are_covered_without_host_socket_permissions(self):
        class FailingLoop:
            def close(self):
                pass

            def run_until_complete(self, coroutine):
                coroutine.close()
                raise RuntimeError("socket unavailable")

        class Probe:
            def __init__(self, error=None):
                self.error = error
                self.path = None

            def connect(self, path):
                self.path = path
                if self.error is not None:
                    raise self.error

            def close(self):
                pass

        source_stat = os.stat(__file__)
        socket_stat = os.stat_result(
            (stat.S_IFSOCK | 0o600, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        replacement_stat = os.stat_result(
            (stat.S_IFSOCK | 0o600, source_stat.st_ino + 1, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        mode_changed_stat = os.stat_result(
            (stat.S_IFSOCK | 0o666, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        regular_stat = os.stat_result(
            (stat.S_IFREG | 0o600, source_stat.st_ino, source_stat.st_dev, 1, 0, 0, 0, 0, 0, 0)
        )
        with patch.object(socket_transport.os, "lstat", return_value=socket_stat):
            self.assertEqual(
                socket_transport._socket_identity("/tmp/socket"),
                (source_stat.st_dev, source_stat.st_ino),
            )
        with patch.object(socket_transport.os, "lstat", return_value=regular_stat):
            self.assertIsNone(socket_transport._socket_identity("/tmp/regular"))
        cases = (
            (Probe(ConnectionRefusedError()), [socket_stat, socket_stat], True, True),
            (Probe(ConnectionRefusedError()), [socket_stat, socket_stat], False, True),
            (Probe(ConnectionRefusedError()), [socket_stat, FileNotFoundError()], None, False),
            (Probe(FileNotFoundError()), [socket_stat], None, False),
            (Probe(OSError("probe failed")), [socket_stat], None, False),
            (Probe(), [socket_stat], None, False),
            (Probe(ConnectionRefusedError()), [socket_stat, regular_stat], None, False),
            (Probe(ConnectionRefusedError()), [socket_stat, replacement_stat], None, False),
            (Probe(ConnectionRefusedError()), [socket_stat, mode_changed_stat], True, True),
        )
        for probe, lstat_results, cleanup_result, expect_cleanup in cases:
            with self.subTest(probe=type(probe.error).__name__ if probe.error else "live"):
                with patch.object(socket_transport, "SOCKET_PATH", "/tmp/ephemeral-buffer-test.sock"), \
                        patch.object(socket_transport.asyncio, "new_event_loop", return_value=FailingLoop()), \
                        patch.object(socket_transport.asyncio, "set_event_loop"), \
                        patch.object(socket_transport.os.path, "lexists", return_value=True), \
                        patch.object(socket_transport.os, "lstat", side_effect=lstat_results), \
                        patch.object(socket_transport.socket, "socket", return_value=probe), \
                        patch.object(socket_transport, "_unlink_socket_if_identity", return_value=cleanup_result) as cleanup, \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    socket_transport.run_socket_server()
                self.assertEqual(socket_transport._socket_lifecycle()[0], "failed")
                if expect_cleanup:
                    cleanup.assert_called_once_with(
                        "/tmp/ephemeral-buffer-test.sock", (source_stat.st_dev, source_stat.st_ino)
                    )
                else:
                    cleanup.assert_not_called()
                self.assertEqual(probe.path, "/tmp/ephemeral-buffer-test.sock")
                if probe.error is not None:
                    self.assertTrue(stderr.getvalue())

    def test_socket_cleanup_does_not_delete_replacement_after_claim(self):
        expected = (1, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "buffer.sock")
            with open(path, "w", encoding="ascii") as stream:
                stream.write("stale")

            def inspect_claimed_inode(claimed_path):
                if claimed_path == path:
                    return expected
                with open(path, "w", encoding="ascii") as stream:
                    stream.write("replacement")
                return expected

            with patch.object(socket_transport, "_socket_identity", side_effect=inspect_claimed_inode):
                self.assertTrue(socket_transport._unlink_socket_if_identity(path, expected))
            with open(path, encoding="ascii") as stream:
                self.assertEqual(stream.read(), "replacement")

    def test_socket_cleanup_rejects_replacement_before_claim(self):
        path = "/tmp/buffer.sock"
        with patch.object(socket_transport.os, "rename") as rename, \
                patch.object(socket_transport, "_socket_identity", return_value=(9, 9)):
            self.assertFalse(socket_transport._unlink_socket_if_identity(path, (1, 2)))
        rename.assert_not_called()

    def test_socket_cleanup_leaves_unexpected_inode_quarantined(self):
        path = "/tmp/buffer.sock"
        with patch.object(socket_transport.os, "rename") as rename, \
                patch.object(socket_transport, "_socket_identity", side_effect=[(1, 2), (9, 9)]), \
                patch.object(socket_transport.os, "getpid", return_value=10), \
                patch.object(server.threading, "get_ident", return_value=11):
            self.assertFalse(socket_transport._unlink_socket_if_identity(path, (1, 2)))
        quarantine = rename.call_args_list[0].args[1]
        self.assertEqual(rename.call_args_list[0].args, (path, quarantine))
        self.assertEqual(len(rename.call_args_list), 1)

    def test_socket_cleanup_leaves_claimed_inode_quarantined_when_unlink_fails(self):
        path = "/tmp/buffer.sock"
        with patch.object(socket_transport.os, "rename") as rename, \
                patch.object(socket_transport, "_socket_identity", side_effect=[(1, 2), (1, 2)]), \
                patch.object(socket_transport.os, "unlink", side_effect=RuntimeError("unlink failed")), \
                patch.object(socket_transport.os, "getpid", return_value=10), \
                patch.object(server.threading, "get_ident", return_value=11):
            with self.assertRaisesRegex(RuntimeError, "unlink failed"):
                socket_transport._unlink_socket_if_identity(path, (1, 2))
        quarantine = rename.call_args_list[0].args[1]
        self.assertEqual(rename.call_args_list[0].args, (path, quarantine))
        self.assertEqual(len(rename.call_args_list), 1)

    def test_socket_cleanup_does_not_overwrite_replacement_during_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "buffer.sock")
            with open(path, "w", encoding="ascii") as stream:
                stream.write("stale")

            def inspect_claimed_inode(claimed_path):
                if claimed_path == path:
                    return (1, 2)
                with open(path, "w", encoding="ascii") as stream:
                    stream.write("replacement")
                return (9, 9)

            with patch.object(socket_transport, "_socket_identity", side_effect=inspect_claimed_inode):
                self.assertFalse(socket_transport._unlink_socket_if_identity(path, (1, 2)))
            with open(path, encoding="ascii") as stream:
                self.assertEqual(stream.read(), "replacement")

    def test_socket_cleanup_tolerates_claimed_inode_already_removed(self):
        path = "/tmp/buffer.sock"
        with patch.object(socket_transport.os, "rename") as rename, \
                patch.object(socket_transport, "_socket_identity", side_effect=[(1, 2), (1, 2)]), \
                patch.object(socket_transport.os, "unlink", side_effect=FileNotFoundError()), \
                patch.object(socket_transport.os, "getpid", return_value=10), \
                patch.object(server.threading, "get_ident", return_value=11):
            self.assertTrue(socket_transport._unlink_socket_if_identity(path, (1, 2)))
        self.assertEqual(len(rename.call_args_list), 1)

    def test_tool_errors_return_stable_json_codes(self):
        self.assertEqual(
            json.loads(server.start_execution([], execution_id="bad"))["error"]["code"],
            "execution_failed",
        )
        self.assertEqual(
            json.loads(server.resume_execution("missing"))["error"]["code"],
            "execution_not_found",
        )
        self.assertEqual(
            json.loads(server.get_execution("missing"))["error"]["code"],
            "execution_not_found",
        )
        self.assertEqual(
            json.loads(server.get_execution_output("missing"))["error"]["code"],
            "execution_not_found",
        )
        self.assertEqual(
            json.loads(execution_responses._execution_json(lambda: "x" * 20, max_response_bytes=10))[
                "error"]["code"],
            "response_too_large",
        )
        with patch.object(server.execution_manager, "public", return_value={"payload": "x" * 70_000}):
            self.assertEqual(
                json.loads(server.get_execution("large", include_output=True))["error"]["code"],
                "response_too_large",
            )
        with patch.object(
            self.manager,
            "request_cancel",
            side_effect=ExecutionBusyError("execution is already leased"),
        ):
            cancellation = json.loads(server.cancel_execution("busy-cancel"))
        self.assertEqual(cancellation["error"]["code"], "execution_busy")

    def test_execution_error_mapping_preserves_typed_details(self):
        def operation_for(error):
            def operation():
                raise error

            return operation

        cases = (
            (ExecutionQuotaError("quota"), "quota_exceeded"),
            (ExecutionDiskSpaceError("space"), "disk_full"),
            (ExecutionPhaseNotFoundError("phase"), "phase_not_found"),
            (OSError(errno.ENOSPC, "disk full"), "disk_full"),
            (OSError(errno.EIO, "storage unavailable"), "execution_storage_error"),
            (ValueError("invalid internal state"), "execution_failed"),
            (RuntimeError("unexpected state"), "execution_failed"),
            (TypeError("unexpected type"), "execution_failed"),
        )
        for error, expected_code in cases:
            with self.subTest(error=type(error).__name__, code=expected_code):
                response = json.loads(execution_responses._execution_json(operation_for(error)))
                self.assertEqual(response["status"], "error")
                self.assertEqual(response["error"]["code"], expected_code)
                self.assertEqual(response["error"]["message"], str(error))

        unsupported = ExecutionRecordError(
            "unsupported version",
            reason_code="UNSUPPORTED_SCHEMA_VERSION",
            schema_version=999,
        )
        response = json.loads(execution_responses._execution_json(operation_for(unsupported)))
        self.assertEqual(response["error"]["code"], "unsupported_schema_version")
        self.assertEqual(
            response["error"]["details"],
            {"reason_code": "UNSUPPORTED_SCHEMA_VERSION", "schema_version": 999},
        )

        malformed = ExecutionRecordError(
            "malformed", reason_code="MALFORMED_RECORD", schema_version="v" * 80
        )
        response = json.loads(execution_responses._execution_json(operation_for(malformed)))
        self.assertEqual(response["error"]["code"], "malformed_record")
        self.assertEqual(response["error"]["details"]["schema_version"], "v" * 64)

        non_finite = ExecutionRecordError(
            "malformed", reason_code="MALFORMED_RECORD", schema_version=float("nan")
        )
        response = json.loads(execution_responses._execution_json(operation_for(non_finite)))
        self.assertNotIn("schema_version", response["error"]["details"])

    def test_record_diagnostics_include_stable_error_codes(self):
        unsupported = {
            "status": "unsupported_record",
            "execution_id": "future",
            "reason_code": "UNSUPPORTED_SCHEMA_VERSION",
            "schema_version": 999,
            "message": "unsupported schema",
        }
        response = json.loads(execution_responses._execution_json(lambda: unsupported))
        self.assertEqual(response["status"], "unsupported_record")
        self.assertEqual(response["error"]["code"], "unsupported_schema_version")
        self.assertEqual(
            response["error"]["details"],
            {"reason_code": "UNSUPPORTED_SCHEMA_VERSION", "schema_version": 999},
        )
        self.assertEqual(response["schema_version"], 999)

        listed = {
            "status": "ok",
            "executions": [
                {
                    "status": "invalid_record",
                    "execution_id": "broken",
                    "reason_code": "MALFORMED_RECORD",
                    "message": "unreadable record",
                }
            ],
        }
        response = json.loads(execution_responses._execution_json(lambda: listed))
        self.assertEqual(
            response["executions"][0]["error"]["code"], "malformed_record"
        )
        self.assertEqual(
            response["executions"][0]["error"]["details"],
            {"reason_code": "MALFORMED_RECORD"},
        )

    def test_tool_reference_documents_stable_error_codes(self):
        documentation = (
            Path(__file__).resolve().parents[2] / "docs" / "tool-reference.md"
        ).read_text(encoding="utf-8")
        codes = (
            "capture_not_found",
            "invalid_range",
            "invalid_cursor",
            "invalid_byte_budget",
            "response_budget_too_small",
            "invalid_query",
            "query_too_large",
            "unsupported_search_mode",
            "invalid_top_k",
            "top_k_limit_exceeded",
            "invalid_context_lines",
            "context_limit_exceeded",
            "invalid_response_budget",
            "execution_busy",
            "execution_not_found",
            "phase_not_found",
            "malformed_record",
            "unsupported_schema_version",
            "quota_exceeded",
            "disk_full",
            "execution_storage_error",
            "response_too_large",
            "execution_failed",
        )
        for code in codes:
            with self.subTest(code=code):
                self.assertIn(f"`{code}`", documentation)

    def test_get_execution_returns_read_only_diagnostics_for_unsupported_schema(self):
        self.manager.create([self.phase("future", "future")], execution_id="future-schema")
        path = self.manager.store._path("future-schema")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["schema_version"] = 999
        path.write_text(json.dumps(record), encoding="utf-8")
        before = path.read_bytes()

        response = json.loads(server.get_execution("future-schema"))

        self.assertEqual(response["status"], "unsupported_record")
        self.assertEqual(response["error"]["code"], "unsupported_schema_version")
        self.assertTrue(response["read_only"])
        self.assertEqual(response["reason_code"], "UNSUPPORTED_SCHEMA_VERSION")
        self.assertEqual(response["schema_version"], 999)
        self.assertEqual(
            response["error"]["details"],
            {"reason_code": "UNSUPPORTED_SCHEMA_VERSION", "schema_version": 999},
        )
        self.assertTrue(response["record_preview"])
        self.assertEqual(path.read_bytes(), before)

        listed = json.loads(server.list_executions())["executions"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["status"], "unsupported_record")
        self.assertEqual(listed[0]["error"]["code"], "unsupported_schema_version")
        self.assertEqual(listed[0]["reason_code"], "UNSUPPORTED_SCHEMA_VERSION")
        self.assertEqual(
            listed[0]["error"]["details"],
            {"reason_code": "UNSUPPORTED_SCHEMA_VERSION", "schema_version": 999},
        )
        self.assertEqual(path.read_bytes(), before)

    def test_oversized_execution_response_keeps_recoverable_id(self):
        phases = [
            self.phase(
                f"phase-{index}",
                f"command-{index}",
                structured_metrics={"payload": "x" * 15_000},
            )
            for index in range(4)
        ]
        response = json.loads(
            server.start_execution(phases, execution_id="oversized-response")
        )
        self.assertEqual(response["execution_id"], "oversized-response")
        self.assertTrue(response["response_truncated"])
        self.assertEqual(response["execution_status"], "completed")

    def test_large_execution_output_is_not_materialized_for_response(self):
        metadata = {
            "status": "ok",
            "execution_id": "large-output",
            "phases": [{"name": "phase", "output_bytes": 70_000}],
        }
        with patch.object(
            server.execution_manager, "public", return_value=metadata
        ) as public:
            response = json.loads(server.get_execution("large-output", include_output=True))
        self.assertTrue(response["response_truncated"])
        self.assertNotIn("output", response["phases"][0])
        public.assert_called_once_with("large-output", include_output=False)

    def test_mcp_get_execution_compacts_oversized_output_without_corrupting_json(self):
        self.runner.results["mcp-nul-output"] = ("\x00" * 8192, 0, False, 8192, False)
        server.start_execution(
            [self.phase("nul-output", "mcp-nul-output")],
            execution_id="mcp-get-execution-nul-output",
        )

        direct_text = server.get_execution(
            "mcp-get-execution-nul-output", include_output=True
        )
        direct_payload = json.loads(direct_text)
        self.assertEqual(direct_payload["phases"][0]["output"], "\x00" * 8192)
        self.assertNotIn("response_truncated", direct_payload)

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
            result = asyncio.run(
                server.mcp.call_tool(
                    "get_execution",
                    {
                        "execution_id": "mcp-get-execution-nul-output",
                        "include_output": True,
                    },
                )
            )

        payload = json.loads(result.content[0].text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["data"], payload)
        self.assertEqual(result.structured_content["payload_schema_version"], 1)
        self.assertTrue(payload["response_truncated"])
        self.assertNotIn("output", payload["phases"][0])
        self.assertEqual(
            result.content[0].text,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        envelope = server.ToolResponseEnvelope(
            status="ok",
            data=payload,
            text=result.content[0].text,
            payload_schema_version=1,
        )
        self.assertLessEqual(
            mcp_transport._call_tool_result_bytes(envelope),
            server.MCP_TOOL_RESPONSE_MAX_BYTES - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
        )
        listed_tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
        Draft202012Validator(listed_tools["get_execution"].output_schema).validate(
            result.structured_content
        )

    def test_execution_mcp_fitter_keeps_oversized_errors_and_fallback_json_valid(self):
        error_payload = {
            "status": "error",
            "error": {
                "code": "malformed_record",
                "message": "x" * 20_000,
                "details": {"execution_id": "malformed-execution"},
            },
        }
        error_text = json.dumps(
            error_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        error_response = server.ToolResponseEnvelope(
            status="error",
            data=error_payload,
            error=server.ToolErrorEnvelope(
                code="malformed_record",
                message="x" * 20_000,
                details={"execution_id": "malformed-execution"},
            ),
            text=error_text,
            payload_schema_version=1,
        )
        compact_error = execution_responses.fit_execution_mcp_response(
            "get_execution", error_response
        )
        parsed_error = json.loads(compact_error.text)
        self.assertEqual(parsed_error["status"], "error")
        self.assertEqual(compact_error.data, parsed_error)
        self.assertEqual(
            parsed_error["error"]["details"]["execution_id"],
            "malformed-execution",
        )
        self.assertLessEqual(
            mcp_transport._call_tool_result_bytes(compact_error),
            server.MCP_TOOL_RESPONSE_MAX_BYTES - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
        )

        oversized_success = {
            "status": "ok",
            "execution_id": "oversized-execution",
            "phases": [],
        }
        success_text = json.dumps(
            oversized_success,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with patch.object(
            execution_responses,
            "call_tool_result_bytes",
            return_value=server.MCP_TOOL_RESPONSE_MAX_BYTES * 2,
        ):
            fallback = execution_responses.fit_execution_mcp_response(
                "get_execution",
                server.ToolResponseEnvelope(
                    status="ok",
                    data=oversized_success,
                    text=success_text,
                    payload_schema_version=1,
                ),
            )
        parsed_fallback = json.loads(fallback.text)
        self.assertEqual(parsed_fallback["status"], "error")
        self.assertEqual(parsed_fallback["error"]["code"], "response_too_large")
        self.assertEqual(fallback.data, parsed_fallback)
        self.assertLessEqual(
            mcp_transport._call_tool_result_bytes(fallback),
            server.MCP_TOOL_RESPONSE_MAX_BYTES - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
        )

    def test_mcp_execution_fitter_compacts_large_start_and_list_payloads(self):
        def phase_name(index):
            prefix = f"phase-{index:02d}-"
            return prefix + ("n" * (239 - len(prefix)))

        start_names = [phase_name(index) for index in range(64)]
        start_payload = self.public_execution_payload("large-start", start_names)

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(self.manager, "start_background", return_value=start_payload):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                start_result = asyncio.run(
                    server.mcp.call_tool(
                        "start_execution",
                        {
                            "phases": [
                                {"name": name, "command": "ignored"}
                                for name in start_names
                            ]
                        },
                    )
                )
        start_data = self.assert_valid_execution_mcp_result(
            start_result, "start_execution"
        )
        self.assertTrue(start_data["response_truncated"])
        self.assertEqual(start_data["execution_id"], "large-start")
        self.assertEqual(start_data["phase_count"], 64)

        listed = [
            self.public_execution_payload(
                f"large-list-{execution_index}",
                [phase_name(index) for index in range(40)],
            )
            for execution_index in range(2)
        ]
        with patch.object(self.manager, "list_public", return_value=listed):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                list_result = asyncio.run(
                    server.mcp.call_tool("list_executions", {"limit": 2})
                )
        list_data = self.assert_valid_execution_mcp_result(
            list_result, "list_executions"
        )
        self.assertTrue(list_data["response_truncated"])
        self.assertEqual(
            [item["execution_id"] for item in list_data["executions"]],
            ["large-list-0", "large-list-1"],
        )
        self.assertEqual(list_data["returned_count"], 2)
        self.assertEqual(list_data["omitted_count"], 0)

    def test_execution_mcp_fitter_handles_all_registered_execution_tools(self):
        public = self.public_execution_payload("contract-execution", ["phase"])
        public["phases"][0]["result"] = {"output": "x" * 40_000}
        output = {
            "status": "ok",
            "execution_id": "contract-execution",
            "execution_status": "completed",
            "partial": False,
            "max_bytes": 40_000,
            "phases": [{
                "name": "phase",
                "status": "completed",
                "output": "x" * 40_000,
                "output_byte_size": 40_000,
                "offset": 0,
                "next_offset": 40_000,
                "truncated": False,
                "result": {},
            }],
        }
        listed_public = dict(public, execution_id="listed-execution")
        listed = {
            "status": "ok",
            "limit": 1,
            "offset": 0,
            "executions": [listed_public],
        }
        capacity = {
            "status": "ok",
            "state_directory": "x" * 50_000,
            "state_directory_exists": True,
            "quota_bytes": 1,
            "checkpoint_reserve_bytes": 1,
            "record_storage_limit_bytes": 1,
            "record_count": 1,
            "summary_count": 1,
            "paired_record_count": 1,
            "unpaired_record_count": 0,
            "unpaired_summary_count": 0,
            "record_bytes": 1,
            "summary_bytes": 1,
            "committed_bytes": 1,
            "temporary_file_bytes": 0,
            "accounted_bytes": 1,
            "active_reserved_bytes": 0,
            "stale_reserved_bytes": 0,
            "available_bytes": 1,
            "capacity_confident": True,
            "active_record_count": 1,
            "fence_pending_record_count": 0,
            "invalid_record_count": 0,
            "invalid_reservation_count": 0,
            "uncertain_lock_count": 0,
            "other_file_count": 0,
            "other_file_bytes": 0,
            "symlink_count": 0,
            "filesystem": {"total_bytes": 1, "free_bytes": 1},
            "background_execution_limit": 8,
            "background_execution_active": 0,
            "background_execution_queued": 0,
        }
        retirement = {
            "status": "partial",
            "dry_run": False,
            "items": [],
            "retired": [{
                "execution_id": "retired-execution",
                "eligible": True,
                "record_bytes": 1,
                "summary_bytes": 1,
                "projected_reclaimed_bytes": 2,
                "label": "x" * 40_000,
            }],
            "retired_count": 1,
            "removal_errors": [{"execution_id": "failed-retirement", "error": "OSError"}],
            "archive_path": "/tmp/" + "x" * 40_000,
            "archive_bytes": 1,
            "archive_is_durable": True,
            "projected_reclaimed_bytes": 2,
        }
        payloads = {
            "start_execution": public,
            "resume_execution": public,
            "cancel_execution": public,
            "get_execution": public,
            "get_execution_output": output,
            "list_executions": listed,
            "get_execution_capacity": capacity,
            "retire_executions": retirement,
        }

        for tool_name, payload in payloads.items():
            with self.subTest(tool=tool_name):
                text = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                response = execution_responses.fit_execution_mcp_response(
                    tool_name,
                    server.ToolResponseEnvelope(
                        status="ok",
                        data=payload,
                        text=text,
                        payload_schema_version=1,
                    ),
                )
                self.assertEqual(json.loads(response.text), response.data)
                execution_tools.PAYLOAD_CONTRACTS[tool_name].validate_python(
                    response.data,
                    strict=True,
                )
                self.assertLessEqual(
                    mcp_transport._call_tool_result_bytes(response),
                    server.MCP_TOOL_RESPONSE_MAX_BYTES
                    - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
                )
                if tool_name == "get_execution_capacity":
                    self.assertEqual(response.status, "error")
                    self.assertEqual(response.error.code, "response_too_large")
                else:
                    self.assertEqual(response.status, "ok")
                    self.assertTrue(response.data["response_truncated"])

    def test_execution_mcp_error_fallback_preserves_list_and_retirement_ids(self):
        cases = (
            (
                "list_executions",
                {
                    "status": "ok",
                    "limit": 2,
                    "offset": 0,
                    "executions": [
                        {"status": "ok", "execution_id": "list-a"},
                        {
                            "status": "invalid_record",
                            "execution_id": None,
                            "record_file_id": "file-b",
                        },
                    ],
                },
                "executions",
            ),
            (
                "retire_executions",
                {
                    "status": "ok",
                    "dry_run": True,
                    "items": [{
                        "execution_id": "retire-a",
                        "eligible": True,
                        "record_bytes": 1,
                        "summary_bytes": 1,
                        "projected_reclaimed_bytes": 2,
                    }],
                    "projected_reclaimed_bytes": 2,
                },
                "execution_ids",
            ),
        )
        for tool_name, payload, identity_key in cases:
            with self.subTest(tool=tool_name):
                text = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                response = server.ToolResponseEnvelope(
                    status="ok",
                    data=payload,
                    text=text,
                    payload_schema_version=1,
                )
                with patch.object(
                    execution_responses,
                    "call_tool_result_bytes",
                    side_effect=[server.MCP_TOOL_RESPONSE_MAX_BYTES * 2,
                                 server.MCP_TOOL_RESPONSE_MAX_BYTES * 2, 0],
                ):
                    compact_error = execution_responses.fit_execution_mcp_response(
                        tool_name, response
                    )
                self.assertEqual(compact_error.status, "error")
                self.assertTrue(compact_error.error.details[identity_key])
                self.assertEqual(json.loads(compact_error.text), compact_error.data)
                self.assertLessEqual(
                    mcp_transport._call_tool_result_bytes(compact_error),
                    server.MCP_TOOL_RESPONSE_MAX_BYTES
                    - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
                )

    def test_oversized_list_response_returns_compact_page_metadata(self):
        entries = [
            {
                "execution_id": f"execution-{index}",
                "label": "label",
                "execution_status": "completed",
                "partial": False,
                "updated_at": "2026-01-01T00:00:00Z",
                "completed_phase_count": 64,
                "phase_count": 64,
                "phases": [{"name": "é" * 256}] * 64,
            }
            for index in range(20)
        ]
        with patch.object(server.execution_manager, "list_public", return_value=entries):
            response = json.loads(server.list_executions())
        self.assertTrue(response["response_truncated"])
        self.assertEqual(response["omitted_count"], 0)
        self.assertEqual(response["executions"][0]["execution_id"], "execution-0")

    def test_oversized_mcp_execution_list_accepts_compact_success_and_diagnostic_items(self):
        resume = {
            "available": False,
            "retry_required": False,
            "unsafe_confirmation_required": False,
            "attempt_limit_reached": False,
            "next_phase": None,
            "policy": "safe",
            "blocked_reason": None,
        }
        phases = [
            {"name": "é" * 256, "status": "completed"}
            for _ in range(64)
        ]
        entries = [
            {
                "status": "ok",
                "schema_version": 1,
                "execution_id": f"execution-{index}",
                "label": "large execution",
                "execution_status": "completed",
                "execution_in_progress": False,
                "partial": False,
                "summary": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "resume": resume,
                "completed_phase_count": len(phases),
                "phase_count": len(phases),
                "phases": phases,
            }
            for index in range(19)
        ]
        entries.append(
            {
                "status": "unsupported_record",
                "execution_id": "unsupported-execution",
                "schema_version": 999,
                "read_only": True,
                "reason_code": "UNSUPPORTED_SCHEMA_VERSION",
                "message": "The record uses a newer schema.",
                "record_bytes": 32,
                "record_preview": "{}",
                "record_preview_sanitized": True,
                "record_preview_redacted": False,
                "record_preview_truncated": False,
            }
        )
        entries.append(
            {
                "status": "invalid_record",
                "execution_id": None,
                "record_file_id": "unrecoverable-listing-id",
                "schema_version": 1,
                "read_only": True,
                "reason_code": "MALFORMED_JSON",
                "message": "The record is malformed.",
                "record_bytes": 8,
                "error": {
                    "code": "malformed_record",
                    "message": "The record is malformed.",
                },
            }
        )

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(self.manager, "list_public", return_value=entries):
            with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
                result = asyncio.run(server.mcp.call_tool("list_executions", {}))

        payload = json.loads(result.content[0].text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "ok")
        self.assertEqual(result.structured_content["payload_schema_version"], 1)
        self.assertEqual(result.structured_content["data"], payload)
        self.assertEqual(
            result.content[0].text,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        self.assertTrue(payload["response_truncated"])
        self.assertEqual(payload["returned_count"], len(entries))
        self.assertEqual(payload["omitted_count"], 0)
        self.assertEqual(payload["executions"][0]["status"], "ok")
        self.assertEqual(payload["executions"][0]["execution_id"], "execution-0")
        self.assertNotIn("created_at", payload["executions"][0])
        self.assertNotIn("resume", payload["executions"][0])
        self.assertNotIn("phases", payload["executions"][0])
        diagnostic = payload["executions"][-1]
        self.assertEqual(diagnostic["status"], "invalid_record")
        self.assertIsNone(diagnostic["execution_id"])
        self.assertEqual(diagnostic["record_file_id"], "unrecoverable-listing-id")
        self.assertEqual(diagnostic["error"]["code"], "malformed_record")
        self.assertEqual(diagnostic["reason_code"], "MALFORMED_JSON")
        self.assertEqual(payload["executions"][-2]["status"], "unsupported_record")
        self.assertEqual(
            payload["executions"][-2]["error"]["code"], "unsupported_schema_version"
        )

    def test_mcp_execution_list_accepts_diagnostic_without_recoverable_id(self):
        store = self.manager.store
        store._ensure_state_dir()
        record_path = store._path("unrecoverable-listing-id")
        record_path.write_text("not json", encoding="utf-8")

        async def run_inline(
            _ticket,
            function,
            /,
            *args,
            _service_context=None,
            _call_id=None,
            **kwargs,
        ):
            return function(*args, **kwargs)

        with patch.object(mcp_transport, "_run_admitted_thread", new=run_inline):
            result = asyncio.run(server.mcp.call_tool("list_executions", {}))

        payload = json.loads(result.content[0].text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["data"], payload)
        self.assertEqual(len(payload["executions"]), 1)
        diagnostic = payload["executions"][0]
        self.assertEqual(diagnostic["status"], "invalid_record")
        self.assertIsNone(diagnostic["execution_id"])
        self.assertEqual(
            diagnostic["record_file_id"], record_path.name.removesuffix(".json")
        )
        self.assertEqual(diagnostic["error"]["code"], "malformed_record")
        execution_tools.PAYLOAD_CONTRACTS["list_executions"].validate_python(
            payload,
            strict=True,
        )
        listed_tools = {
            tool.name: tool for tool in asyncio.run(server.mcp.list_tools())
        }
        Draft202012Validator(
            listed_tools["list_executions"].output_schema
        ).validate(result.structured_content)

    def test_list_compaction_skips_invalid_entries_and_bounds_large_summaries(self):
        entries = [
            None,
            {
                "execution_id": "huge",
                "label": "x" * 70_000,
                "execution_status": "completed",
                "partial": False,
            },
        ]
        with patch.object(server.execution_manager, "list_public", return_value=entries):
            response = json.loads(server.list_executions())
        self.assertTrue(response["response_truncated"])
        self.assertEqual(response["returned_count"], 0)
        self.assertEqual(response["omitted_count"], 2)

    def test_execution_tools_are_registered_with_async_mcp_adapters(self):
        names = {
            tool.name: tool
            for tool in asyncio.run(server.mcp.list_tools())
        }
        for name in (
            "start_execution", "resume_execution", "get_execution",
            "get_execution_output", "list_executions", "get_execution_capacity",
            "retire_executions",
        ):
            self.assertIn(name, names)
        start_schema = names["start_execution"].input_schema
        phase_schema = start_schema["properties"]["phases"]["items"]
        if "$ref" in phase_schema:
            phase_schema = start_schema["$defs"][phase_schema["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(
            set(phase_schema["required"]),
            {"name", "command"},
        )
        self.assertEqual(
            start_schema["properties"]["resume_policy"]["enum"],
            ["safe", "allow-unsafe"],
        )
        self.assertEqual(start_schema["properties"]["phases"]["minItems"], 1)
        self.assertEqual(start_schema["properties"]["phases"]["maxItems"], 64)
        self.assertEqual(
            start_schema["properties"]["timeout_seconds"]["anyOf"][0]["exclusiveMinimum"],
            0,
        )
        self.assertEqual(
            start_schema["properties"]["max_output_bytes"]["anyOf"][0]["minimum"],
            512,
        )
        self.assertNotIn("maximum", start_schema["properties"]["max_output_bytes"]["anyOf"][0])
        list_schema = names["list_executions"].input_schema
        self.assertEqual(list_schema["properties"]["limit"]["minimum"], 1)
        self.assertEqual(list_schema["properties"]["limit"]["maximum"], 100)
        retirement_schema = names["retire_executions"].input_schema
        self.assertEqual(
            retirement_schema["properties"]["execution_ids"]["maxItems"], 20
        )
        output_schema = names["get_execution_output"].input_schema
        self.assertEqual(output_schema["properties"]["max_bytes"]["minimum"], 512)
        self.assertIn(
            "UTF-8 byte offset", output_schema["properties"]["offset"]["description"]
        )
        self.assertEqual(start_schema["properties"]["label"]["maxUtf8Bytes"], 1024)
        self.assertEqual(start_schema["properties"]["cwd"]["anyOf"][0]["maxUtf8Bytes"], 4096)
        for name in ("resume_execution", "get_execution", "get_execution_output"):
            execution_id_schema = names[name].input_schema["properties"]["execution_id"]
            self.assertEqual(execution_id_schema["minLength"], 1)
            self.assertEqual(execution_id_schema["maxUtf8Bytes"], 256)
        self.assertEqual(output_schema["properties"]["phase_name"]["anyOf"][0]["minLength"], 1)
        self.assertEqual(
            output_schema["properties"]["phase_name"]["anyOf"][0]["maxUtf8Bytes"],
            256,
        )
        with self.assertRaises(ValueError):
            server.ExecutionPhaseInput(name="é" * 256, command="echo")
        with self.assertRaises(ValueError):
            server.ExecutionPhaseInput(
                name="output-limit",
                command="echo",
                max_output_bytes=DEFAULT_MAX_BUFFER_BYTES + 1,
            )
        original_limit = server.engine.max_buffer_bytes
        try:
            server.engine.max_buffer_bytes = DEFAULT_MAX_BUFFER_BYTES + 1024
            accepted = server.ExecutionPhaseInput(
                name="configured-limit",
                command="echo",
                max_output_bytes=DEFAULT_MAX_BUFFER_BYTES + 512,
            )
            self.assertEqual(accepted.max_output_bytes, DEFAULT_MAX_BUFFER_BYTES + 512)
        finally:
            server.engine.max_buffer_bytes = original_limit
        self.assertEqual(
            phase_schema["properties"]["structured_metrics"]["maxJsonBytes"],
            16 * 1024,
        )
        with self.assertRaises(ValueError):
            server.ExecutionPhaseInput(
                name="metrics-limit",
                command="echo",
                structured_metrics={"payload": "x" * 20_000},
            )
        self.assertEqual(
            server.ExecutionPhaseInput(
                name="metrics-valid",
                command="echo",
                structured_metrics={"items": 3},
            ).structured_metrics,
            {"items": 3},
        )
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            server.ExecutionPhaseInput(
                name="metrics-invalid",
                command="echo",
                structured_metrics={"items": object()},
            )

        async def invoke():
            worker_started = threading.Event()
            worker_finished = threading.Event()
            original_run = self.manager._run

            def observe_background_run(*args, **kwargs):
                worker_started.set()
                try:
                    return original_run(*args, **kwargs)
                finally:
                    worker_finished.set()

            offload = AsyncMock(
                side_effect=lambda function, *args, **kwargs: function(*args, **kwargs)
            )
            with patch.object(server, "to_thread", offload), patch.object(
                self.manager, "_run", new=observe_background_run
            ):
                started = json.loads(mcp_result_text(await server.mcp.call_tool(
                    "start_execution",
                    {
                        "phases": [self.phase("adapter", "adapter")],
                        "execution_id": "adapter-execution",
                    },
                )))
                self.assertTrue(
                    await asyncio.to_thread(worker_started.wait, 30),
                    "background execution did not start before the deadline",
                )
                self.assertTrue(
                    await asyncio.to_thread(worker_finished.wait, 30),
                    "background execution did not finish before the deadline",
                )
                result = json.loads(
                    mcp_result_text(await server.mcp.call_tool(
                        "get_execution",
                        {"execution_id": "adapter-execution"},
                    ))
                )
                if result.get("background_error"):
                    self.fail(result["background_error"])
                return started, result

        _started, result = asyncio.run(invoke())
        self.assertEqual(result["execution_status"], "completed")


if __name__ == "__main__":
    unittest.main()
