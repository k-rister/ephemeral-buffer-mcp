"""Regression checks for lazy transports and the stable MCP tool contract."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ephemeral_buffer_mcp import create_mcp_server, create_service_context
import ephemeral_buffer_mcp.server as server
from ephemeral_buffer_mcp.tools.registry import (
    TOOL_CATEGORIES,
    get_tool_categories,
    get_tool_names,
    get_tool_registration,
    get_tool_registrations,
    register_tool,
)


class TestTransportDecomposition(unittest.TestCase):
    def test_legacy_server_mcp_lookups_resolve_from_transport_registry(self):
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
            structured_result_factory=registration.structured_result_factory,
        )(registration.function)
        with self.assertRaisesRegex(ValueError, "duplicate tool registration"):
            register_tool(registration.name, registration.category)(lambda: None)

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
                    "import sys; import ephemeral_buffer_mcp.server; "
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

    def test_mcp_tool_schemas_match_pre_decomposition_contract(self):
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
                tool.model_dump(mode="json", by_alias=True, exclude_none=True)
                for tool in sorted(tools, key=lambda item: item.name)
            ]
        finally:
            context.close()
        self.assertEqual(actual, expected)

    def test_mcp_success_and_error_response_envelopes_keep_their_shape(self):
        context = create_service_context(
            metrics_file=None,
            engine_options={"semantic_prefetch": False, "embedding_warmup": False},
        )
        try:
            app = create_mcp_server(context)
            success = asyncio.run(
                app.call_tool("capture_text", {"content": "contract response"})
            ).structured_content
            failure = asyncio.run(
                app.call_tool(
                    "get_capture_summary",
                    {"capture_id": "missing-contract-capture"},
                )
            ).structured_content
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
            {"schema_version", "status", "data", "error", "truncated", "next_cursor"},
        )
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["error"]["code"], "capture_not_found")


if __name__ == "__main__":
    unittest.main()
