"""Tests for resumable-execution MCP tool adapters."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("EPHEMERAL_DISABLE_SOCKET_SERVER", "1")
os.environ.setdefault("EPHEMERAL_TEST_EMBEDDINGS", "1")

import server
from execution import PhaseExecutionManager


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
        server.engine.clear("all")
        self.directory.cleanup()

    @staticmethod
    def phase(name, command, **overrides):
        value = {"name": name, "command": command}
        value.update(overrides)
        return value

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
        self.assertEqual(json.loads(server.resume_execution("tool-execution"))["execution_status"], "completed")
        self.assertEqual(json.loads(server.get_execution("tool-execution"))["partial"], False)
        with_output = json.loads(server.get_execution("tool-execution", include_output=True))
        self.assertEqual(with_output["phases"][0]["output"], "build output")
        output = json.loads(server.get_execution_output("tool-execution", "test"))
        self.assertEqual(output["phases"][0]["output"], "test output")
        listed = json.loads(server.list_executions())
        self.assertEqual(listed["executions"][0]["execution_id"], "tool-execution")

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

    def test_durable_execution_marks_session_capture_unavailable_after_restart(self):
        started = json.loads(
            server.start_execution(
                [self.phase("durable", "durable")],
                execution_id="durable-capture",
            )
        )
        self.assertTrue(started["phases"][0]["result"]["capture_available"])
        server.engine.clear("all")
        fresh_manager = PhaseExecutionManager(
            self.directory.name,
            max_output_bytes=server.engine.max_buffer_bytes,
            command_runner=self.runner,
        )
        with patch.object(server, "execution_manager", fresh_manager):
            result = json.loads(server.get_execution("durable-capture"))
        self.assertFalse(result["phases"][0]["result"]["capture_available"])

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
            [sys.executable, "-c", "import server"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_execution_payload_preserves_phases_without_capture_references(self):
        payload = {"phases": [{"result": None}, {"result": {"exit_code": 0}}]}
        self.assertIs(server._execution_public_payload(payload), payload)
        self.assertNotIn("capture_available", payload["phases"][1]["result"])

    def test_tool_errors_are_bounded_and_returned_without_raising(self):
        self.assertTrue(server.start_execution([], execution_id="bad").startswith("Error managing execution:"))
        self.assertTrue(server.resume_execution("missing").startswith("Error managing execution:"))
        self.assertTrue(server.get_execution("missing").startswith("Error managing execution:"))
        self.assertTrue(server.get_execution_output("missing").startswith("Error managing execution:"))
        self.assertTrue(
            server._execution_json(lambda: "x" * 20, max_response_bytes=10).startswith(
                "Error managing execution: response exceeds"
            )
        )
        with patch.object(server.execution_manager, "public", return_value={"payload": "x" * 70_000}):
            self.assertTrue(
                server.get_execution("large", include_output=True).startswith(
                    "Error managing execution: response exceeds"
                )
            )

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
        names = server.mcp._tool_manager._tools
        for name in (
            "start_execution", "resume_execution", "get_execution",
            "get_execution_output", "list_executions",
        ):
            self.assertIn(name, names)
        start_schema = names["start_execution"].parameters
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
        list_schema = names["list_executions"].parameters
        self.assertEqual(list_schema["properties"]["limit"]["minimum"], 1)
        self.assertEqual(list_schema["properties"]["limit"]["maximum"], 100)
        output_schema = names["get_execution_output"].parameters
        self.assertEqual(output_schema["properties"]["max_bytes"]["minimum"], 512)
        self.assertEqual(start_schema["properties"]["label"]["maxUtf8Bytes"], 1024)
        self.assertEqual(start_schema["properties"]["cwd"]["anyOf"][0]["maxUtf8Bytes"], 4096)
        for name in ("resume_execution", "get_execution", "get_execution_output"):
            execution_id_schema = names[name].parameters["properties"]["execution_id"]
            self.assertEqual(execution_id_schema["minLength"], 1)
            self.assertEqual(execution_id_schema["maxUtf8Bytes"], 256)
        self.assertEqual(output_schema["properties"]["phase_name"]["anyOf"][0]["minLength"], 1)
        self.assertEqual(
            output_schema["properties"]["phase_name"]["anyOf"][0]["maxUtf8Bytes"],
            256,
        )
        with self.assertRaises(ValueError):
            server.ExecutionPhaseInput(name="é" * 256, command="echo")
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
            adapter = names["start_execution"].fn
            offload = AsyncMock(
                side_effect=lambda function, *args, **kwargs: function(*args, **kwargs)
            )
            with patch.object(server, "to_thread", offload):
                return await adapter(
                    phases=[self.phase("adapter", "adapter")],
                    execution_id="adapter-execution",
                )

        result = json.loads(asyncio.run(invoke()))
        self.assertEqual(result["execution_status"], "completed")


if __name__ == "__main__":
    unittest.main()
