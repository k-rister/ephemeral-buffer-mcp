import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mcp import Client

from ephemeral_buffer_mcp import create_mcp_server, create_service_context
from ephemeral_buffer_mcp.mcp_response import ToolErrorEnvelope, ToolResponseEnvelope
from ephemeral_buffer_mcp.response_limits import SEARCH_QUERY_MAX_BYTES
from ephemeral_buffer_mcp.tools import instrumentation
import ephemeral_buffer_mcp.tools.retrieval as retrieval
from ephemeral_buffer_mcp.tools.request_context import current_mcp_supplied_arguments


class _Metrics:
    def measure(self, _name):
        return self

    def __enter__(self):
        return {"success": True}

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False

    def record_bytes(self, _name, _value):
        return None


class TestToolInstrumentation(unittest.TestCase):
    def test_mcp_dispatch_uses_raw_client_argument_names(self):
        context = create_service_context()
        app = create_mcp_server(context)
        events = []
        oversized_query = "x" * (SEARCH_QUERY_MAX_BYTES + 1)

        async def call_through_client():
            async with Client(app) as client:
                return await client.call_tool("list_captures", {"limit": 1})

        with patch.object(instrumentation, "active_metrics", return_value=_Metrics()), patch.object(
            instrumentation, "write_metrics_snapshot"
        ), patch.object(
            retrieval,
            "_active_engine",
            return_value=SimpleNamespace(list_captures_page=lambda **_kwargs: {
                "status": "error",
                "message": "private listing failure",
                "error_code": "listing_failed",
            }),
        ), patch.object(
            instrumentation,
            "log_event",
            side_effect=lambda _logger, level, event, **fields: events.append((event, fields)),
        ):
            search_result = asyncio.run(
                app.call_tool("search_capture", {"query": oversized_query})
            )
            list_result = asyncio.run(call_through_client())

        self.assertTrue(search_result.is_error)
        self.assertTrue(list_result.is_error)
        search_failure = next(
            fields for event, fields in events
            if event == "mcp_tool_result_failed" and fields["tool"] == "search_capture"
        )
        list_failure = next(
            fields for event, fields in events
            if event == "mcp_tool_result_failed" and fields["tool"] == "list_captures"
        )
        self.assertEqual(search_failure["request_shape"], {"query": "string"})
        self.assertFalse(search_failure["capture_id_supplied"])
        self.assertFalse(search_failure["cursor_supplied"])
        self.assertEqual(list_failure["request_shape"], {"limit": "integer"})
        self.assertFalse(list_failure["capture_id_supplied"])
        self.assertFalse(list_failure["cursor_supplied"])
        self.assertIsNone(current_mcp_supplied_arguments())
        self.assertNotIn(oversized_query, str(events))

    def test_structured_failure_retains_safe_diagnostics_without_values(self):
        @instrumentation._instrument_tool("search_capture")
        def search(query: str, capture_id: str = "latest", cursor: str | None = None):
            return ToolResponseEnvelope(
                status="error",
                error=ToolErrorEnvelope(code="invalid_query", message="private query"),
                text="Error: private query",
            )

        events = []
        with patch.object(instrumentation, "active_metrics", return_value=_Metrics()), patch.object(
            instrumentation, "write_metrics_snapshot"
        ), patch.object(
            instrumentation,
            "log_event",
            side_effect=lambda _logger, level, event, **fields: events.append((event, fields)),
        ):
            result = search("private query", capture_id="capture-secret", cursor="cursor-secret")

        self.assertEqual(result.error.code, "invalid_query")
        failure = next(fields for event, fields in events if event == "mcp_tool_result_failed")
        self.assertEqual(failure["error_code"], "invalid_query")
        self.assertEqual(failure["failure_kind"], "result")
        self.assertEqual(failure["result_status"], "error")
        self.assertEqual(failure["request_shape"], {
            "query": "string",
            "capture_id": "string",
            "cursor": "string",
        })
        self.assertTrue(failure["capture_id_supplied"])
        self.assertTrue(failure["cursor_supplied"])
        self.assertNotIn("private query", str(events))
        self.assertNotIn("capture-secret", str(events))
        self.assertNotIn("cursor-secret", str(events))

    def test_exception_failure_uses_stable_category_and_shape_fallback(self):
        @instrumentation._instrument_tool("get_capture_slice")
        def slice_tool(start_line: int):
            raise ValueError("private failure")

        events = []
        with patch.object(instrumentation, "active_metrics", return_value=_Metrics()), patch.object(
            instrumentation, "write_metrics_snapshot"
        ), patch.object(
            instrumentation,
            "log_event",
            side_effect=lambda _logger, level, event, **fields: events.append((event, fields)),
        ):
            with self.assertRaises(TypeError):
                slice_tool(1, 2)

        failure = next(fields for event, fields in events if event == "mcp_tool_failed")
        self.assertEqual(failure["error_code"], "exception_validation")
        self.assertEqual(failure["failure_kind"], "exception")
        self.assertEqual(failure["result_status"], "exception")
        self.assertEqual(failure["request_shape"], {})
        self.assertNotIn("private failure", str(events))

    def test_non_errors_and_text_errors_use_safe_result_classification(self):
        @instrumentation._instrument_tool("get_capture_summary")
        def successful(query: str):
            return ToolResponseEnvelope(status="ok", data={"value": 1})

        @instrumentation._instrument_tool("list_captures")
        def unstructured(query: str):
            return "not json"

        @instrumentation._instrument_tool("capture_file")
        def text_error(path: str):
            return "Error: private failure"

        def run(tool, *args):
            events = []
            with patch.object(
                instrumentation, "active_metrics", return_value=_Metrics()
            ), patch.object(instrumentation, "write_metrics_snapshot"), patch.object(
                instrumentation,
                "log_event",
                side_effect=lambda _logger, level, event, **fields: events.append((event, fields)),
            ):
                result = tool(*args)
            return result, events

        successful_result, successful_events = run(successful, "private query")
        unstructured_result, unstructured_events = run(unstructured, "private query")
        text_result, text_events = run(text_error, "/private/path")

        self.assertEqual(successful_result.status, "ok")
        self.assertNotIn("mcp_tool_result_failed", [event for event, _ in successful_events])
        self.assertEqual(unstructured_result, "not json")
        self.assertNotIn("mcp_tool_result_failed", [event for event, _ in unstructured_events])
        failure = next(fields for event, fields in text_events if event == "mcp_tool_result_failed")
        self.assertEqual(failure["error_code"], "operation_failed")
        self.assertEqual(failure["result_status"], "text_error")
        self.assertEqual(failure["request_shape"], {"path": "string"})
        self.assertNotIn("private failure", str(text_events))
        self.assertNotIn("/private/path", str(text_events))

    def test_request_shape_uses_only_safe_type_names(self):
        self.assertEqual(instrumentation._safe_type_name(b"private"), "bytes")
        self.assertEqual(instrumentation._safe_type_name(object()), "other")

    def test_structured_text_failure_extracts_error_code(self):
        @instrumentation._instrument_tool("execute_and_capture")
        def execute(command: str):
            return '{"status":"error","error":{"code":"operation_failed","message":"private"}}'

        events = []
        with patch.object(instrumentation, "active_metrics", return_value=_Metrics()), patch.object(
            instrumentation, "write_metrics_snapshot"
        ), patch.object(
            instrumentation,
            "log_event",
            side_effect=lambda _logger, level, event, **fields: events.append((event, fields)),
        ):
            execute("private command")

        failure = next(fields for event, fields in events if event == "mcp_tool_result_failed")
        self.assertEqual(failure["error_code"], "operation_failed")
        self.assertEqual(failure["request_shape"], {"command": "string"})
        self.assertNotIn("private command", str(events))


if __name__ == "__main__":
    unittest.main()
