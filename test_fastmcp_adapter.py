"""Compatibility-boundary tests for FastMCP integration."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP
from pydantic import TypeAdapter

import fastmcp_adapter as adapter


class TestFastMCPAdapter(TestCase):
    def setUp(self):
        self._limitations = patch.dict(adapter._VALIDATION_LIMITATIONS, {}, clear=True)
        self._observation_failures = patch.dict(
            adapter._VALIDATION_OBSERVATION_FAILURES, {}, clear=True
        )
        self._limitations.start()
        self._observation_failures.start()
        self._instruction_limitation = adapter._INSTRUCTION_REFRESH_LIMITATION
        adapter._INSTRUCTION_REFRESH_LIMITATION = None

    def tearDown(self):
        self._limitations.stop()
        self._observation_failures.stop()
        adapter._INSTRUCTION_REFRESH_LIMITATION = self._instruction_limitation

    def test_sdk_validation_runs_once_and_reports_invalid_input(self):
        app = FastMCP("validation-observer")
        calls = []
        observations = []

        def sync_probe(value: int) -> str:
            calls.append(value)
            return str(value)

        adapter.register_tool(
            app,
            sync_probe,
            name="sync_probe",
            validation_scope=nullcontext,
            validation_observer=lambda: observations.append("validation"),
        )

        asyncio.run(app.call_tool("sync_probe", {"value": 3}))
        self.assertEqual(calls, [3])
        self.assertEqual(observations, [])

        with self.assertRaises(Exception):
            asyncio.run(app.call_tool("sync_probe", {}))
        self.assertEqual(calls, [3])
        self.assertEqual(observations, ["validation"])

    def test_async_tool_and_tool_validation_error_are_not_misclassified(self):
        app = FastMCP("async-validation-observer")
        calls = []
        observations = []

        async def async_probe(value: int) -> str:
            calls.append(value)
            raise TypeAdapter(int).validate_python("tool-body-error")

        adapter.register_tool(
            app,
            async_probe,
            name="async_probe",
            validation_scope=nullcontext,
            validation_observer=lambda: observations.append("validation"),
        )

        with self.assertRaises(Exception):
            asyncio.run(app.call_tool("async_probe", {"value": 5}))
        self.assertEqual(calls, [5])
        self.assertEqual(observations, [])

    def test_scope_failure_keeps_sdk_validation_and_dispatch(self):
        app = FastMCP("scope-fallback")
        calls = []

        def probe(value: int) -> str:
            calls.append(value)
            return str(value)

        def unavailable_scope():
            raise RuntimeError("private scope detail")

        adapter.register_tool(
            app,
            probe,
            name="scope_probe",
            validation_scope=unavailable_scope,
            validation_observer=lambda: None,
        )
        asyncio.run(app.call_tool("scope_probe", {"value": 7}))

        self.assertEqual(calls, [7])
        status = adapter.compatibility_diagnostics()
        self.assertEqual(status["tools_without_validation_metrics"], ["scope_probe"])
        self.assertNotIn("private scope detail", repr(status))

    def test_observer_failure_does_not_replace_sdk_validation_error(self):
        app = FastMCP("observer-fallback")

        def probe(value: int) -> str:
            return str(value)

        def broken_observer():
            raise RuntimeError("observer detail")

        adapter.register_tool(
            app,
            probe,
            name="observer_probe",
            validation_scope=nullcontext,
            validation_observer=broken_observer,
        )
        with self.assertRaises(Exception):
            asyncio.run(app.call_tool("observer_probe", {}))

        status = adapter.compatibility_diagnostics()
        self.assertEqual(
            status["validation_observation_failures"],
            {"observer_probe": "RuntimeError"},
        )

    def test_registration_fallback_preserves_public_tool_registration(self):
        class PublicOnlyApp:
            def __init__(self):
                self.tools = {}

            def add_tool(self, fn, name=None):
                self.tools[name] = fn

        app = PublicOnlyApp()
        adapter.register_tool(
            app,
            lambda value: value,
            name="fallback_probe",
            validation_scope=nullcontext,
            validation_observer=lambda: None,
        )

        self.assertEqual(app.tools["fallback_probe"](4), 4)
        self.assertEqual(
            adapter.compatibility_diagnostics()["tools_without_validation_metrics"],
            ["fallback_probe"],
        )

    def test_registration_without_validation_instrumentation_uses_public_api(self):
        app = FastMCP("no-observer")

        def probe(value: int) -> str:
            return str(value)

        adapter.register_tool(app, probe, name="public_probe")
        asyncio.run(app.call_tool("public_probe", {"value": 9}))
        self.assertEqual(
            adapter.compatibility_diagnostics()["tools_without_validation_metrics"],
            [],
        )

    def test_unsupported_validation_signature_is_detected(self):
        with patch.object(
            adapter.inspect,
            "signature",
            return_value=SimpleNamespace(parameters={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported argument-validation"):
                adapter._check_supported_validation_hook()

    def test_instruction_refresh_uses_public_setter_when_available(self):
        class PublicInstructions:
            instructions = "initial"

        app = PublicInstructions()
        self.assertTrue(adapter.refresh_instructions(app, "updated"))
        self.assertEqual(app.instructions, "updated")

    def test_instruction_refresh_uses_contained_sdk_compatibility_path(self):
        app = FastMCP("instructions", instructions="initial")
        self.assertTrue(adapter.refresh_instructions(app, "updated"))
        self.assertEqual(app.instructions, "updated")

    def test_instruction_refresh_failure_is_reported_without_raising(self):
        class NoInstructionSetter:
            @property
            def instructions(self):
                return "initial"

        self.assertFalse(adapter.refresh_instructions(NoInstructionSetter(), "updated"))
        status = adapter.compatibility_diagnostics()
        self.assertEqual(status["instruction_refresh"], "unavailable")
        self.assertEqual(status["instruction_refresh_error_type"], "AttributeError")

    def test_compatibility_diagnostics_handles_missing_package_metadata(self):
        with patch.object(adapter, "version", side_effect=adapter.PackageNotFoundError):
            status = adapter.compatibility_diagnostics()
        self.assertEqual(status["sdk_version"], "unknown")


if __name__ == "__main__":
    import unittest

    unittest.main()
