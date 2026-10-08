"""Narrow compatibility boundary for FastMCP SDK integrations.

The application uses FastMCP's public API for normal tool registration,
dispatch, and reading tool metadata. Two capabilities currently lack public
write hooks in the supported SDK: observing argument-validation failures and
refreshing instructions. All SDK-private access is kept in this module, and
failure to install either capability never removes or disables a registered
tool.
"""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

from pydantic import PrivateAttr, ValidationError
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata


_LOGGER = logging.getLogger("fastmcp_adapter")
_STATE_LOCK = threading.Lock()
_VALIDATION_LIMITATIONS: dict[str, str] = {}
_INSTRUCTION_REFRESH_LIMITATION: str | None = None
_VALIDATION_OBSERVATION_FAILURES: dict[str, str] = {}


class _ObservedFuncMetadata(FuncMetadata):
    """Delegate parsing to FastMCP and observe only SDK validation failures."""

    _validation_scope: Callable[[], Any] = PrivateAttr()
    _validation_observer: Callable[[], None] = PrivateAttr()
    _tool_name: str = PrivateAttr()

    @classmethod
    def from_metadata(
        cls,
        metadata: FuncMetadata,
        *,
        tool_name: str,
        validation_scope: Callable[[], Any],
        validation_observer: Callable[[], None],
    ) -> "_ObservedFuncMetadata":
        observed = cls.model_validate(metadata.model_dump())
        observed._tool_name = tool_name
        observed._validation_scope = validation_scope
        observed._validation_observer = validation_observer
        return observed

    async def call_fn_with_arg_validation(
        self,
        fn,
        fn_is_async,
        arguments_to_validate,
        arguments_to_pass_directly,
    ):
        """Run the SDK's validator once, recording its failures when possible."""
        called_tool = False

        if fn_is_async:
            async def observed_fn(*args, **kwargs):
                nonlocal called_tool
                called_tool = True
                return await fn(*args, **kwargs)
        else:
            def observed_fn(*args, **kwargs):
                nonlocal called_tool
                called_tool = True
                return fn(*args, **kwargs)

        scope = ExitStack()
        try:
            scope.enter_context(self._validation_scope())
        except Exception as exc:
            scope.close()
            _record_validation_limitation(self._tool_name, type(exc).__name__)
            # Scope setup belongs to metrics only. Keep FastMCP's normal
            # validation and dispatch path if it cannot be established.
            return await super().call_fn_with_arg_validation(
                fn,
                fn_is_async,
                arguments_to_validate,
                arguments_to_pass_directly,
            )

        try:
            try:
                return await super().call_fn_with_arg_validation(
                    observed_fn,
                    fn_is_async,
                    arguments_to_validate,
                    arguments_to_pass_directly,
                )
            except ValidationError:
                if not called_tool:
                    try:
                        self._validation_observer()
                    except Exception as exc:
                        _record_validation_observation_failure(
                            self._tool_name, type(exc).__name__
                        )
                raise
        finally:
            scope.close()


def register_tool(
    app,
    fn,
    *,
    name: str,
    validation_scope: Callable[[], Any] | None = None,
    validation_observer: Callable[[], None] | None = None,
) -> None:
    """Register a tool publicly and best-effort install validation metrics."""
    app.add_tool(fn, name=name)
    if validation_scope is None or validation_observer is None:
        return

    try:
        _check_supported_validation_hook()
        # FastMCP 1.x does not expose a public hook for its argument validator.
        # Keep this narrowly scoped SDK access here; registration has already
        # succeeded, so any incompatibility leaves ordinary dispatch intact.
        tool = app._tool_manager._tools[name]
        tool.fn_metadata = _ObservedFuncMetadata.from_metadata(
            tool.fn_metadata,
            tool_name=name,
            validation_scope=validation_scope,
            validation_observer=validation_observer,
        )
    except Exception as exc:
        _record_validation_limitation(name, type(exc).__name__)


def _check_supported_validation_hook() -> None:
    parameters = tuple(inspect.signature(FuncMetadata.call_fn_with_arg_validation).parameters)
    expected = (
        "self",
        "fn",
        "fn_is_async",
        "arguments_to_validate",
        "arguments_to_pass_directly",
    )
    if parameters != expected:
        raise RuntimeError("unsupported argument-validation hook signature")


def refresh_instructions(app, instructions: str) -> bool:
    """Update client instructions through a supported setter or contained shim."""
    global _INSTRUCTION_REFRESH_LIMITATION

    try:
        app.instructions = instructions
        with _STATE_LOCK:
            _INSTRUCTION_REFRESH_LIMITATION = None
        return True
    except Exception:
        pass

    try:
        # FastMCP 1.29.1 exposes a read-only instructions property. Isolate the
        # required backing-server update here until the SDK offers a setter.
        app._mcp_server.instructions = instructions
        with _STATE_LOCK:
            _INSTRUCTION_REFRESH_LIMITATION = None
        return True
    except Exception as exc:
        reason = type(exc).__name__
        with _STATE_LOCK:
            _INSTRUCTION_REFRESH_LIMITATION = reason
        _LOGGER.warning(
            "fastmcp_instruction_refresh_unavailable error_type=%s", reason
        )
        return False


def compatibility_diagnostics() -> dict[str, Any]:
    """Return content-free SDK capability status for runtime diagnostics."""
    try:
        sdk_version = version("mcp")
    except PackageNotFoundError:
        sdk_version = "unknown"

    with _STATE_LOCK:
        limitations = dict(sorted(_VALIDATION_LIMITATIONS.items()))
        observation_failures = dict(sorted(_VALIDATION_OBSERVATION_FAILURES.items()))
        instruction_limitation = _INSTRUCTION_REFRESH_LIMITATION

    return {
        "sdk_version": sdk_version,
        "validation_metrics": (
            "partial" if limitations or observation_failures else "available"
        ),
        "tools_without_validation_metrics": list(limitations),
        "validation_observation_failures": observation_failures,
        "instruction_refresh": "unavailable" if instruction_limitation else "available",
        "instruction_refresh_error_type": instruction_limitation,
    }


def _record_validation_limitation(tool_name: str, error_type: str) -> None:
    with _STATE_LOCK:
        _VALIDATION_LIMITATIONS[tool_name] = error_type
    _LOGGER.warning(
        "fastmcp_validation_metrics_unavailable tool=%s error_type=%s",
        tool_name,
        error_type,
    )


def _record_validation_observation_failure(tool_name: str, error_type: str) -> None:
    with _STATE_LOCK:
        _VALIDATION_OBSERVATION_FAILURES[tool_name] = error_type
    _LOGGER.warning(
        "fastmcp_validation_observation_failed tool=%s error_type=%s",
        tool_name,
        error_type,
    )
