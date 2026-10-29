"""Regression checks for lazy transports and the stable MCP tool contract."""

import asyncio
import ast
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator

from ephemeral_buffer_mcp import create_mcp_server, create_service_context
from ephemeral_buffer_mcp.config import load_settings
from ephemeral_buffer_mcp.mcp_response import (
    ToolResponseEnvelope,
    call_tool_result_bytes,
    structured_tool_output_schema,
)
from ephemeral_buffer_mcp.mcp_transport import _record_tool_response_bytes
import ephemeral_buffer_mcp.server as server
import ephemeral_buffer_mcp.mcp_transport as mcp_transport
import ephemeral_buffer_mcp.tools.retrieval as retrieval_tools
from ephemeral_buffer_mcp.tools.registry import (
    TOOL_CATEGORIES,
    get_tool_categories,
    get_tool_names,
    get_tool_registration,
    get_tool_registrations,
    register_tool,
)
from ephemeral_buffer_mcp.tools.payloads import (
    PAYLOAD_CONTRACTS,
    PAYLOAD_SCHEMA_VERSIONS,
)


class TestTransportDecomposition(unittest.TestCase):
    def test_transports_use_explicit_imports(self):
        package_root = Path(__file__).parents[2] / "src" / "ephemeral_buffer_mcp"
        for name in ("mcp_transport.py", "socket_transport.py"):
            with self.subTest(transport=name):
                tree = ast.parse((package_root / name).read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn("*", [alias.name for alias in node.names])

    def test_server_exports_only_the_supported_embedding_api(self):
        self.assertEqual(
            server.__all__, ["create_mcp_server", "create_service_context"]
        )

    def test_legacy_server_aliases_resolve_to_their_owners(self):
        aliases = {
            "mcp_transport": (
                "mcp", "MCP_TOOL_LIST_CACHE_HINTS", "MCP_TOOL_LIST_CACHE_TTL_MS",
            ),
            "socket_transport": ("handle_socket_client",),
            "tools.execution": (
                "ExecutionPhaseInput", "cancel_execution", "get_execution",
                "get_execution_capacity", "get_execution_output", "list_executions",
                "preflight_command", "resume_execution", "retire_executions",
                "start_execution", "subprocess",
            ),
            "tools.capture": (
                "_consolidated_jsonl", "capture_file", "capture_text",
                "consolidate_captures", "execute_and_capture",
            ),
            "tools.search": ("search_capture", "search_capture_result"),
            "tools.retrieval": (
                "clear_captures", "get_capture_slice", "get_capture_slice_result",
                "get_capture_summary", "list_captures",
            ),
            "tools.diagnostics": (
                "get_buffer_stats", "get_runtime_diagnostics", "get_usage_metrics",
            ),
            "tools.configuration": ("set_semantic_index_budget",),
        }
        for module_name, names in aliases.items():
            module = import_module(f".{module_name}", server.__package__)
            for name in names:
                with self.subTest(alias=name):
                    self.assertIs(getattr(server, name), getattr(module, name))

    def test_unknown_server_names_do_not_load_or_search_modules(self):
        names = (
            "__unknown__", "missing_legacy_name", "_MCP_unknown", "_mcp_unknown",
            "_REGISTERED_MCP_UNKNOWN", "MCP_TOOL_LIST_UNKNOWN",
            "MCP_TOOL_CATEGORY_UNKNOWN", "_register_mcpserver_unknown",
            "_call_tool_result_unknown", "_compact_tool_text_unknown",
            "_legacy_tool_envelope_unknown", "_fit_tool_envelope_unknown",
            "SOCKET_UNKNOWN", "_SOCKET_UNKNOWN", "socket_unknown", "_socket_unknown",
            "start_socket_unknown", "run_socket_unknown", "handle_socket_unknown",
            "_read_socket_unknown", "_write_socket_unknown", "_require_socket_unknown",
            "_unlink_socket_unknown", "_bind_socket_unknown", "_connect_socket_unknown",
            "_verify_bound_socket_unknown",
            # These exist in the owning modules but are not compatibility aliases.
            "_mcp_metrics_scope_key", "_read_socket_payload", "_render_search_text",
            "BaseModel",
        )
        with patch.object(server, "import_module") as importer:
            for name in names:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(AttributeError, name):
                        getattr(server, name)
        importer.assert_not_called()

    def test_mcp_response_byte_metrics_ignore_non_text_content(self):
        record_bytes = Mock()
        metrics = SimpleNamespace(record_bytes=record_bytes)
        _record_tool_response_bytes(SimpleNamespace(metrics=metrics), "capture_text", None)

        record_bytes.assert_not_called()

    def test_tool_modules_use_only_the_named_server_context_bridge(self):
        tools_root = Path(__file__).parents[2] / "src" / "ephemeral_buffer_mcp" / "tools"
        expected_context_accessors = {
            "_active_engine",
            "_active_execution_manager",
            "_active_metrics",
            "_active_service_context",
            "_active_settings",
            "_metrics_snapshot",
            "_request_cancel_event",
            "_server_started_at",
            "_write_metrics_snapshot",
        }
        context_imports = set()
        for path in tools_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if any(alias.name == "*" for alias in node.names):
                    self.fail(f"wildcard import remains in {path.name}")
                if node.module == "server":
                    self.assertEqual(path.name, "_context.py")
                    context_imports.update(alias.name for alias in node.names)
                if node.module == "capture" and any(
                    alias.name.startswith("_") for alias in node.names
                ):
                    self.fail(f"private capture helper imported by {path.name}")
        self.assertEqual(context_imports, expected_context_accessors)

    def test_legacy_server_mcp_lookups_resolve_from_transport_registry(self):
        self.assertEqual(
            server._MCP_TOOL_REGISTRATIONS,
            [
                (item.function, item.name, item.category, item.structured_result_factory)
                for item in get_tool_registrations()
            ],
        )
        self.assertEqual(server._REGISTERED_MCP_TOOL_NAMES, get_tool_names())
        self.assertEqual(server._REGISTERED_MCP_TOOL_CATEGORIES, get_tool_categories())
        self.assertEqual(server.MCP_TOOL_CATEGORY_NAMES, tuple(TOOL_CATEGORIES))
        with self.assertRaisesRegex(AttributeError, "missing_legacy_name"):
            server.missing_legacy_name

    def test_registry_lookup_and_idempotent_registration(self):
        registration = get_tool_registrations()[0]
        self.assertIs(get_tool_registration(registration.name), registration)
        register_tool(
            registration.name,
            registration.category,
            result_policy=registration.result_policy,
            structured_result_factory=registration.structured_result_factory,
            payload_model=registration.payload_model,
            payload_schema_version=registration.payload_schema_version,
            mcp_response_fitter=registration.mcp_response_fitter,
        )(registration.function)
        with self.assertRaisesRegex(ValueError, "duplicate tool registration"):
            register_tool(
                registration.name,
                registration.category,
                result_policy=registration.result_policy,
            )(lambda: None)

    def test_registry_requires_payload_contract_for_mcp_response_fitter(self):
        with self.assertRaisesRegex(
            ValueError, "MCP response fitters require a structured payload contract"
        ):
            register_tool(
                "coverage_missing_fitter_contract",
                "diagnostics",
                result_policy="structured_envelope",
                mcp_response_fitter=lambda _name, response: response,
            )

    def test_every_registered_tool_declares_a_result_policy_and_payload_fixture(self):
        registrations = get_tool_registrations()
        self.assertTrue(registrations)
        for registration in registrations:
            with self.subTest(tool=registration.name):
                self.assertIn(
                    registration.result_policy,
                    {"structured_envelope", "legacy_text"},
                )
                self.assertEqual(
                    registration.payload_model is not None,
                    registration.payload_schema_version is not None,
                )
                if registration.payload_model is not None:
                    self.assertEqual(registration.result_policy, "structured_envelope")

        execution_tool_names = {
            "start_execution", "resume_execution", "cancel_execution", "get_execution",
            "get_execution_output", "list_executions", "get_execution_capacity",
            "retire_executions",
        }
        self.assertEqual(
            {
                registration.name
                for registration in registrations
                if registration.mcp_response_fitter is not None
            },
            execution_tool_names,
        )

        fixture_path = Path(__file__).parents[1] / "fixtures" / "mcp-result-contract.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        actual = {
            name: {
                "payload_schema_version": PAYLOAD_SCHEMA_VERSIONS[name],
                "schema": adapter.json_schema(mode="validation"),
            }
            for name, adapter in sorted(PAYLOAD_CONTRACTS.items())
        }
        self.assertEqual(fixture, actual)
        self.assertEqual(
            {registration.name for registration in registrations if registration.payload_model},
            set(fixture),
        )

        tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
        for registration in registrations:
            if registration.payload_model is None:
                continue
            with self.subTest(output_schema=registration.name):
                output_schema = tools[registration.name].output_schema
                expected_schema = structured_tool_output_schema(
                    registration.payload_model.json_schema(mode="validation"),
                    registration.payload_schema_version,
                )
                self.assertEqual(output_schema, expected_schema)
                self.assertEqual(
                    tools[registration.name].model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )["outputSchema"],
                    output_schema,
                )
                Draft202012Validator.check_schema(output_schema)
                self.assertIsNone(
                    server.mcp._tool_manager._tools[
                        registration.name
                    ].fn_metadata.output_model
                )

    def test_core_socket_wrapper_delegates_to_lazy_transport(self):
        marker = object()
        socket_module = SimpleNamespace(run_socket_server=lambda: marker)
        with patch.object(server, "import_module", return_value=socket_module) as importer:
            self.assertIs(server.run_socket_server(), marker)
        importer.assert_called_once_with(".socket_transport", server.__package__)

    def test_core_import_does_not_load_mcp_sdk_or_transport(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from ephemeral_buffer_mcp.server import *; "
                    "assert create_mcp_server; assert create_service_context; "
                    "assert not any(name == 'mcp' or name.startswith('mcp.') "
                    "for name in sys.modules); "
                    "assert 'ephemeral_buffer_mcp.mcp_transport' not in sys.modules; "
                    "assert 'ephemeral_buffer_mcp.socket_transport' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EPHEMERAL_DISABLE_SOCKET_SERVER": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_tool_lookup_preserves_registration_order_without_loading_sdk(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import ephemeral_buffer_mcp.server as server; "
                    "assert server.capture_text; "
                    "from ephemeral_buffer_mcp.tools.registry import get_tool_names; "
                    "assert get_tool_names()[0] == 'preflight_command'; "
                    "assert not any(name == 'mcp' or name.startswith('mcp.') "
                    "for name in sys.modules); "
                    "assert 'ephemeral_buffer_mcp.mcp_transport' not in sys.modules; "
                    "assert 'ephemeral_buffer_mcp.socket_transport' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "EPHEMERAL_DISABLE_SOCKET_SERVER": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mcp_tool_inputs_match_pre_payload_contract_fixture(self):
        fixture = Path(__file__).parents[1] / "fixtures" / "mcp-tool-contract.json"
        expected = json.loads(fixture.read_text(encoding="utf-8"))
        context = create_service_context(
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        try:
            app = create_mcp_server(context)
            tools = asyncio.run(app.list_tools())
            actual = [
                {
                    key: tool.model_dump(mode="json", by_alias=True, exclude_none=True)[key]
                    for key in ("name", "description", "inputSchema")
                }
                for tool in sorted(tools, key=lambda item: item.name)
            ]
        finally:
            context.close()
        expected_inputs = [
            {key: tool[key] for key in ("name", "description", "inputSchema")}
            for tool in expected
        ]
        self.assertEqual(actual, expected_inputs)

    def test_mcp_success_and_error_response_envelopes_keep_their_shape(self):
        context = create_service_context(
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        try:
            app = create_mcp_server(context)
            success_result = asyncio.run(
                app.call_tool("capture_text", {"content": "contract response"})
            )
            success = success_result.structured_content
            capture_id = success["data"]["capture_id"]
            summary_success = asyncio.run(
                app.call_tool("get_capture_summary", {"capture_id": capture_id})
            ).structured_content
            failure_result = asyncio.run(
                app.call_tool(
                    "get_capture_summary",
                    {"capture_id": "missing-contract-capture"},
                )
            )
            failure = failure_result.structured_content
            tools = {tool.name: tool for tool in asyncio.run(app.list_tools())}
        finally:
            context.close()

        self.assertEqual(
            set(success),
            {"schema_version", "status", "data", "error", "truncated", "next_cursor"},
        )
        self.assertEqual(success["status"], "ok")
        self.assertEqual(success["data"]["status"], "captured")
        self.assertIn("capture_id", success["data"])
        self.assertEqual(
            set(failure),
            {
                "schema_version", "status", "data", "error", "truncated", "next_cursor",
                "payload_schema_version",
            },
        )
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["error"]["code"], "capture_not_found")
        summary_output_schema = tools["get_capture_summary"].output_schema
        self.assertIsNotNone(summary_output_schema)
        Draft202012Validator(summary_output_schema).validate(summary_success)
        Draft202012Validator(summary_output_schema).validate(failure)

    def test_mcp_response_byte_metrics_measure_the_final_adapter_text_once(self):
        settings = load_settings({**os.environ, "EPHEMERAL_METRICS": "1"})
        context = create_service_context(
            settings,
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        try:
            app = create_mcp_server(context)
            capture = asyncio.run(
                app.call_tool("capture_text", {"content": "needle in response metrics"})
            )
            capture_id = capture.structured_content["data"]["capture_id"]
            search = asyncio.run(
                app.call_tool(
                    "search_capture",
                    {"query": "needle", "mode": "bm25", "capture_id": capture_id},
                )
            )
            retrieval = asyncio.run(
                app.call_tool(
                    "get_capture_slice",
                    {"start_line": 1, "end_line": 1, "capture_id": capture_id},
                )
            )

            capture_text = capture.content[0].text
            search_text = search.content[0].text
            retrieval_text = retrieval.content[0].text
            byte_counts = context.metrics.snapshot()["bytes"]
        finally:
            context.close()

        self.assertEqual(
            byte_counts["tool_response_bytes"],
            sum(len(text.encode("utf-8")) for text in (capture_text, search_text, retrieval_text)),
        )
        self.assertEqual(byte_counts["search_response_bytes"], len(search_text.encode("utf-8")))
        self.assertEqual(
            byte_counts["retrieval_response_bytes"],
            len(retrieval_text.encode("utf-8")),
        )

    def test_capture_slice_candidate_budget_includes_payload_version(self):
        settings = load_settings({**os.environ, "EPHEMERAL_TEST_EMBEDDINGS": "1"})
        context = create_service_context(
            settings,
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        content = "\n".join(
            f"line-{index:04d}-" + "x" * 36 for index in range(1500)
        )
        try:
            app = create_mcp_server(context)
            captured = asyncio.run(app.call_tool("capture_text", {"content": content}))
            capture_id = captured.structured_content["data"]["capture_id"]
            measured_versions = []
            measure_response = mcp_transport._call_tool_result_bytes

            def measure_candidate(envelope, *, compact_text=False):
                measured_versions.append(envelope.payload_schema_version)
                return measure_response(envelope, compact_text=compact_text)

            with patch.object(
                retrieval_tools,
                "_call_tool_result_bytes",
                side_effect=measure_candidate,
            ):
                result = asyncio.run(
                    app.call_tool(
                        "get_capture_slice",
                        {
                            "start_line": 1,
                            "end_line": 1500,
                            "capture_id": capture_id,
                        },
                    )
                )
        finally:
            context.close()

        self.assertTrue(measured_versions)
        self.assertEqual(set(measured_versions), {1})
        self.assertFalse(result.is_error)
        structured = result.structured_content
        self.assertEqual(structured["status"], "ok")
        self.assertEqual(structured["payload_schema_version"], 1)
        self.assertTrue(structured["data"]["content"])
        envelope = ToolResponseEnvelope(
            status=structured["status"],
            data=structured["data"],
            text=result.content[0].text,
            truncated=structured["truncated"],
            next_cursor=structured["next_cursor"],
            payload_schema_version=structured["payload_schema_version"],
        )
        self.assertLessEqual(
            call_tool_result_bytes(envelope),
            server.MCP_TOOL_RESPONSE_MAX_BYTES - server.MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
        )

    def test_search_on_empty_capture_returns_typed_success(self):
        settings = load_settings({**os.environ, "EPHEMERAL_TEST_EMBEDDINGS": "1"})
        context = create_service_context(
            settings,
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        try:
            app = create_mcp_server(context)
            captured = asyncio.run(app.call_tool("capture_text", {"content": ""}))
            capture_id = captured.structured_content["data"]["capture_id"]
            result = asyncio.run(
                app.call_tool(
                    "search_capture",
                    {"query": "missing", "mode": "bm25", "capture_id": capture_id},
                )
            )
            tools = {tool.name: tool for tool in asyncio.run(app.list_tools())}
        finally:
            context.close()

        self.assertFalse(result.is_error)
        payload = result.structured_content["data"]
        self.assertEqual(result.structured_content["status"], "ok")
        self.assertEqual(payload["mode"], "bm25")
        self.assertEqual(payload["total_lines"], 0)
        self.assertEqual(payload["matches"], [])
        Draft202012Validator(tools["search_capture"].output_schema).validate(
            result.structured_content
        )


if __name__ == "__main__":
    unittest.main()
