"""Content-free tool metrics and lifecycle logging."""

from __future__ import annotations

import json
import inspect
import itertools
import logging
import subprocess
import time
from functools import wraps

from ..logging_utils import get_logger, log_event
from ..mcp_response import ToolResponseEnvelope
from ._context import (
    active_metrics,
    current_request_cancel_event,
    write_metrics_snapshot,
)


LOGGER = get_logger("server")
_TOOL_CALL_IDS = itertools.count(1)
_TIMEOUT_RESULT_TOOLS = {
    "execute_and_capture",
    "start_execution",
    "resume_execution",
}


def _classify_failure_text(text: str) -> str:
    """Map an internal error description to a content-free failure category."""
    normalized = text.lower()
    if any(marker in normalized for marker in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(marker in normalized for marker in (
        "socket",
        "connection refused",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "econn",
    )):
        return "socket"
    if any(marker in normalized for marker in (
        "embedding",
        "semantic index",
        "fastembed",
        "onnx",
    )):
        return "embedding"
    if any(marker in normalized for marker in ("evict", "eviction")):
        return "eviction"
    if any(marker in normalized for marker in (
        "invalid",
        "must ",
        "unsupported",
        "not found",
        "does not exist",
        "exceeds",
        "at least",
        "disabled",
        "cannot ",
        "no capture",
    )):
        return "validation"
    return "other"


def _classify_tool_exception(tool: str, exc: Exception, kwargs: dict) -> str:
    """Classify exceptions without exposing their messages in metrics."""
    if type(exc).__name__.lower().find("evict") >= 0:
        return "eviction"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(exc, (
        ConnectionError,
        BrokenPipeError,
        ConnectionAbortedError,
        ConnectionRefusedError,
        ConnectionResetError,
    )):
        return "socket"
    if tool == "search_capture" and kwargs.get("mode", "hybrid") in {"hybrid", "semantic"}:
        return "embedding"
    if isinstance(exc, (ValueError, TypeError)):
        return "validation"
    return "other"


def _classify_tool_result(tool: str, result) -> str | None:
    """Classify structured or textual tool errors without retaining payloads."""
    if isinstance(result, ToolResponseEnvelope):
        if result.status == "error" and result.error is not None:
            return _classify_failure_text(f"{result.error.code} {result.error.message}")
        return None
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            phases = payload.get("phases", ())
            if not isinstance(phases, list):
                phases = ()
            if tool in _TIMEOUT_RESULT_TOOLS and (
                payload.get("timed_out") is True
                or payload.get("execution_status") in {"timed_out", "timeout"}
                or any(
                    isinstance(phase, dict)
                    and (
                        phase.get("timed_out") is True
                        or phase.get("status") in {"timed_out", "timeout"}
                        or (
                            isinstance(phase.get("result"), dict)
                            and (
                                phase["result"].get("timed_out") is True
                                or phase["result"].get("status") in {"timed_out", "timeout"}
                            )
                        )
                    )
                    for phase in phases
                )
            ):
                return "timeout"
            if payload.get("status") == "error":
                return _classify_failure_text(str(payload.get("message", "")))
        if result.startswith(("Error", "Search Error")):
            return _classify_failure_text(result)
    return None


def _instrument_tool(name):
    """Decorate a tool with opt-in content-free call metrics and timing logs."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            invocation_mode = kwargs.pop("__invocation_mode", "direct")
            supplied_call_id = kwargs.pop("__call_id", None)
            persist_snapshot = invocation_mode != "mcp"
            if "__invocation_mode" in inspect.signature(function).parameters:
                kwargs["__invocation_mode"] = invocation_mode
            call_id = supplied_call_id if supplied_call_id is not None else next(_TOOL_CALL_IDS)
            started = time.perf_counter()
            log_event(LOGGER, logging.INFO, "mcp_tool_started", call_id=call_id, tool=name)
            try:
                metrics = active_metrics()
                with metrics.measure(name) as state:
                    try:
                        result = function(*args, **kwargs)
                    except Exception as exc:
                        state["failure_category"] = _classify_tool_exception(name, exc, kwargs)
                        raise
                    response_text = result.text if isinstance(result, ToolResponseEnvelope) else result
                    if invocation_mode != "mcp" and isinstance(response_text, str):
                        response_bytes = len(response_text.encode("utf-8"))
                        metrics.record_bytes("tool_response_bytes", response_bytes)
                        if name == "search_capture":
                            metrics.record_bytes("search_response_bytes", response_bytes)
                        elif name in {"get_capture_slice", "get_capture_summary"}:
                            metrics.record_bytes("retrieval_response_bytes", response_bytes)
                    failure_category = _classify_tool_result(name, result)
                    if failure_category is not None:
                        state["success"] = False
                        state["failure_category"] = failure_category
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                log_event(
                    LOGGER,
                    logging.INFO,
                    "mcp_tool_completed",
                    call_id=call_id,
                    duration_ms=duration_ms,
                    success=state["success"],
                    tool=name,
                )
                return result
            except Exception as exc:
                persist_snapshot = True
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "mcp_tool_failed",
                    call_id=call_id,
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                    tool=name,
                )
                raise
            finally:
                cancel_event = current_request_cancel_event()
                if cancel_event is not None and cancel_event.is_set():
                    persist_snapshot = True
                if persist_snapshot:
                    # MCP adapters persist successful calls after response fitting.
                    # Direct calls and worker failures persist here.
                    write_metrics_snapshot()
        wrapper._supports_invocation_mode = True
        wrapper._supports_call_id = True
        return wrapper
    return decorator
