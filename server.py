"""
MCP Server for Ephemeral Command Output Hybrid Search.
Supports stdio MCP protocol and Unix Domain Socket IPC for CLI piping.
"""

import os
import atexit
import dataclasses
import sys
import json
import asyncio
import inspect
from contextvars import ContextVar, copy_context
import socket
import stat
import threading
import logging
import itertools
import platform
import re
import shlex
import shutil
import subprocess
import time
import tempfile
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from config import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    cleanup_execution_state_dir,
    load_settings,
    new_default_execution_state_dir,
    runtime_index_budget_adjustment_enabled,
    startup_settings,
)

SETTINGS = startup_settings()
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from fastmcp_adapter import (
    compatibility_diagnostics as _fastmcp_compatibility_diagnostics,
    refresh_instructions as _refresh_fastmcp_instructions,
    register_tool as _register_fastmcp_tool,
)
from engine import (
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    MAX_CAPTURE_SLICE_CONTENT_BYTES,
    CAPTURE_SLICE_SEGMENT_MAX_BYTES,
    MAX_SEARCH_CONTEXT_LINES,
    MAX_SEARCH_TOP_K,
    EphemeralEngine,
    normalize_structured_metrics,
)
from capture_utils import read_file_bounded, run_command_bounded
from execution import (
    ExecutionBusyError,
    MAX_EXECUTION_ID_BYTES,
    MAX_EXECUTION_RETIRE_BATCH,
    MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
    MAX_EXECUTION_PHASES,
    MAX_STRUCTURED_METRICS_BYTES,
    MAX_PHASE_NAME_BYTES,
    PhaseExecutionManager,
)
from logging_utils import get_logger, log_event
from metrics import LocalMetrics
from socket_protocol import FRAME_HEADER_SIZE, decode_header, encode_frame
from admission import (
    ADMISSION_METRICS,
    AdmissionBusy,
    MAX_ACTIVE_DIAGNOSTIC_WORK,
    MAX_ACTIVE_SOCKET_CLIENTS,
    MAX_ACTIVE_TOOL_WORK,
    MAX_QUEUED_SOCKET_CLIENTS,
    admission_gate,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no Unix socket backend.
    fcntl = None

SOCKET_PATH = SETTINGS.identity.socket_path
SOCKET_PAYLOAD_OVERHEAD = 64 * 1024
SOCKET_CLIENT_READ_TIMEOUT_SECONDS = SETTINGS.socket_timeout_seconds.value


def socket_isolation_required() -> bool:
    """Return the startup-snapshotted isolation policy."""
    return SETTINGS.socket_require_isolation.value


def socket_isolation_configured() -> bool:
    """Return whether the startup descriptor has an isolated identity."""
    return SETTINGS.identity.isolation_configured


# JSON string escaping can expand a UTF-8 capture by at most six bytes per
# source byte (for example, a control character encoded as ``\u0000``).
SOCKET_JSON_MAX_EXPANSION = 6
EXECUTION_OUTPUT_RESPONSE_MAX_BYTES = 64 * 1024
SEARCH_RESPONSE_MAX_BYTES = 64 * 1024
MCP_TOOL_RESPONSE_MAX_BYTES = 64 * 1024
MCP_JSONRPC_ENVELOPE_RESERVE_BYTES = 1024
SEARCH_STRUCTURED_MATCH_MAX_BYTES = 2 * 1024
SEARCH_STRUCTURED_CONTEXT_MAX_BYTES = 1024
SEARCH_SNIPPET_TRUNCATION_MARKER = (
    "... [snippet truncated; use the Context range above with get_capture_slice for full content]"
)
SEARCH_MATCHES_OMITTED_MARKER = (
    "... [Additional matches omitted; the search response reached its byte budget]"
)
SEARCH_SNIPPET_AND_MATCHES_TRUNCATION_MARKER = (
    "... [snippet truncated; Additional matches omitted. "
    "Use the Context range above with get_capture_slice for full content]"
)
SUMMARY_DIFF_FILE_MAP_MAX_BYTES = 8 * 1024
SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES = 100

# The MCP and socket gates together bound submissions to this process-owned
# executor, so EB work does not accumulate in asyncio's default executor queue.
_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=(
        MAX_ACTIVE_TOOL_WORK
        + MAX_ACTIVE_SOCKET_CLIENTS
        + MAX_ACTIVE_DIAGNOSTIC_WORK
    ),
    thread_name_prefix="ephemeral-work",
)


async def to_thread(function, /, *args, **kwargs):
    """Run sync work on EB's bounded worker pool with context propagation."""
    loop = asyncio.get_running_loop()
    context = copy_context()
    call = partial(function, *args, **kwargs)
    return await loop.run_in_executor(_WORK_EXECUTOR, context.run, call)


async def _run_admitted_thread(
    ticket,
    function,
    /,
    *args,
    _service_context=None,
    _call_id=None,
    **kwargs,
):
    """Keep admission reserved until worker work finishes, including cancellation."""
    task = asyncio.create_task(to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancellation_event = _REQUEST_CANCEL_EVENT.get()
        if cancellation_event is not None:
            cancellation_event.set()
        ticket.defer_release_until(task)
        if _service_context is not None and _call_id is not None:
            _service_context.defer_tool_call_until(_call_id, task)
        raise


_REQUEST_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "ephemeral_request_cancel_event", default=None
)
_MCP_TOOL_INVOCATION: ContextVar[bool] = ContextVar(
    "ephemeral_mcp_tool_invocation", default=False
)
SUMMARY_COMMAND_MAX_BYTES = 1024
SUMMARY_COMMAND_TRUNCATION_MARKER = "... [command truncated]"
SUMMARY_LABEL_MAX_BYTES = 1024
SUMMARY_LABEL_TRUNCATION_MARKER = "... [label truncated]"
SEARCH_QUERY_MAX_BYTES = 1024
SEARCH_QUERY_TRUNCATION_MARKER = "... [query truncated]"
CONSOLIDATION_MAX_CAPTURES = 25
CONSOLIDATION_CAPTURE_ID_MAX_BYTES = 256
SOCKET_STARTUP_TIMEOUT_SECONDS = SETTINGS.socket_startup_timeout_seconds.value
SERVER_STARTED_AT = time.time()
LOGGER = get_logger("server")
METRICS = LocalMetrics(enabled=SETTINGS.metrics_enabled.value)
_REGISTERED_MCP_TOOL_NAMES: list[str] = []
_REGISTERED_MCP_TOOL_CATEGORIES: dict[str, str] = {}
_MCP_TOOL_REGISTRATIONS: list[tuple[Any, str, str, Any]] = []
MCP_TOOL_CATEGORY_NAMES = (
    "capture",
    "configuration",
    "diagnostics",
    "execution",
    "lifecycle",
    "retrieval",
    "search",
)
_TOOL_CALL_IDS = itertools.count(1)
_TIMEOUT_RESULT_TOOLS = {
    "execute_and_capture",
    "start_execution",
    "resume_execution",
}
_SOCKET_STATE_LOCK = threading.Lock()
_SOCKET_STATE = "disabled" if SETTINGS.disable_socket_server.value else "not-started"
_SOCKET_FAILURE = None
_SOCKET_STARTUP_EVENT = threading.Event()
_SOCKET_PATH_LOCKS = {}
_SOCKET_PATH_LOCKS_GUARD = threading.Lock()


class ToolErrorEnvelope(BaseModel):
    """Stable machine-readable error information for public tool results."""

    code: str
    message: str


class ToolResponseEnvelope(BaseModel):
    """Versioned structured result paired with readable MCP text content."""

    schema_version: Literal[1] = 1
    status: Literal["ok", "error"]
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ToolErrorEnvelope] = None
    text: Optional[str] = None
    truncated: bool = False
    next_cursor: Optional[str] = None
_SOCKET_PATH_LOCK_DEPTH = threading.local()
_MCP_SESSION_SCOPE_LOCK = threading.Lock()
_MCP_SESSION_SCOPES: weakref.WeakKeyDictionary[Any, tuple[str, dict[str, str]]] = (
    weakref.WeakKeyDictionary()
)
MAX_MCP_SESSION_SCOPE_FALLBACK = 128
_MCP_SESSION_SCOPE_FALLBACK: OrderedDict[
    int, tuple[Any, tuple[str, dict[str, str]]]
] = OrderedDict()
if _SOCKET_STATE == "disabled":
    _SOCKET_STARTUP_EVENT.set()


@contextmanager
def _bind_mcp_metrics_scope(app=None, service_context: "ServiceContext | None" = None) -> Any:
    """Bind metrics to the current MCP transport session without exposing it."""
    try:
        request_context = (app or mcp).get_context().request_context
        session = request_context.session
    except (AttributeError, LookupError, ValueError):
        yield
        return

    with _MCP_SESSION_SCOPE_LOCK:
        try:
            scope = _MCP_SESSION_SCOPES.get(session)
            if scope is None:
                scope_id = f"mcp_{uuid.uuid4().hex[:16]}"
                attribution = {"kind": "mcp_session", "mode": "private", "id": scope_id}
                scope = (scope_id, attribution)
                _MCP_SESSION_SCOPES[session] = scope
        except TypeError:
            # Keep compatibility with transports whose session object is not
            # weak-referenceable or hashable. Retain the actual session object
            # so a reused id cannot inherit a prior client's scope; bound this
            # fallback because such objects cannot provide a cleanup callback.
            session_key = id(session)
            entry = _MCP_SESSION_SCOPE_FALLBACK.get(session_key)
            if entry is not None and entry[0] is session:
                _MCP_SESSION_SCOPE_FALLBACK.move_to_end(session_key)
                scope = entry[1]
            else:
                scope_id = f"mcp_{uuid.uuid4().hex[:16]}"
                attribution = {"kind": "mcp_session", "mode": "private", "id": scope_id}
                scope = (scope_id, attribution)
                _MCP_SESSION_SCOPE_FALLBACK[session_key] = (session, scope)
                _MCP_SESSION_SCOPE_FALLBACK.move_to_end(session_key)
                while len(_MCP_SESSION_SCOPE_FALLBACK) > MAX_MCP_SESSION_SCOPE_FALLBACK:
                    _MCP_SESSION_SCOPE_FALLBACK.popitem(last=False)

    metrics = (
        service_context.metrics
        if service_context is not None and service_context is not DEFAULT_SERVICES
        else _active_metrics()
    )
    with metrics.bind_scope(scope[0], attribution=scope[1]):
        yield


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


def _classify_tool_exception(tool: str, exc: Exception, kwargs: dict[str, Any]) -> str:
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
    if (
        tool == "search_capture"
        and kwargs.get("mode", "hybrid") in {"hybrid", "semantic"}
    ):
        return "embedding"
    if isinstance(exc, (ValueError, TypeError)):
        return "validation"
    return "other"


def _classify_tool_result(tool: str, result: Any) -> str | None:
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
            if (
                tool in _TIMEOUT_RESULT_TOOLS
                and (
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
                )
            ):
                return "timeout"
            if payload.get("status") == "error":
                return _classify_failure_text(str(payload.get("message", "")))
        if result.startswith(("Error", "Search Error")):
            return _classify_failure_text(result)
    return None


def _set_socket_state(state, failure=None):
    """Publish socket lifecycle state for diagnostics and the module entrypoint."""
    global _SOCKET_STATE, _SOCKET_FAILURE
    with _SOCKET_STATE_LOCK:
        _SOCKET_STATE = state
        _SOCKET_FAILURE = failure
        if state == "starting":
            _SOCKET_STARTUP_EVENT.clear()
        if state in {"ready", "failed", "disabled"}:
            _SOCKET_STARTUP_EVENT.set()


def _socket_lifecycle():
    with _SOCKET_STATE_LOCK:
        return _SOCKET_STATE, _SOCKET_FAILURE


def _socket_identity(path):
    """Return a socket's filesystem identity without following symlinks."""
    path_stat = os.lstat(path)
    if not stat.S_ISSOCK(path_stat.st_mode):
        return None
    return path_stat.st_dev, path_stat.st_ino


@contextmanager
def _socket_path_lock(path):
    """Serialize socket probing and cleanup for one pathname."""
    key = os.path.abspath(path)
    with _SOCKET_PATH_LOCKS_GUARD:
        lock = _SOCKET_PATH_LOCKS.setdefault(key, threading.RLock())
    with lock:
        held_keys = getattr(_SOCKET_PATH_LOCK_DEPTH, "keys", set())
        if key in held_keys:
            yield
            return
        held_keys = set(held_keys)
        held_keys.add(key)
        _SOCKET_PATH_LOCK_DEPTH.keys = held_keys
        lock_fd = None
        try:
            if fcntl is not None:
                lock_path = f"{key}.lock"
                lock_fd = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT,
                    0o600,
                )
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(lock_fd)
                    lock_fd = None
                    raise
            yield
        finally:
            try:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
            finally:
                held_keys.remove(key)
                _SOCKET_PATH_LOCK_DEPTH.keys = held_keys


def _unlink_socket_if_identity(path, expected_identity):
    with _socket_path_lock(path):
        return _unlink_socket_if_identity_unlocked(path, expected_identity)


def _unlink_socket_if_identity_unlocked(path, expected_identity):
    """Remove only the socket inode that was atomically claimed.

    Unlinking a pathname after an identity check has a replacement race.  Move
    the name to a private quarantine name first; any replacement created at
    the original path is then independent of the object being removed.
    """
    parent = os.path.dirname(path) or "."
    quarantine = os.path.join(
        parent,
        f".{os.path.basename(path)}.cleanup-{os.getpid()}-"
        f"{threading.get_ident()}-{uuid.uuid4().hex}",
    )
    try:
        if _socket_identity(path) != expected_identity:
            return False
        os.rename(path, quarantine)
    except FileNotFoundError:
        return False
    try:
        if _socket_identity(quarantine) != expected_identity:
            # Leave the claimed inode quarantined.  Restoring it with rename
            # after checking the path would let a concurrent replacement be
            # overwritten between the check and the restore.
            return False
        try:
            os.unlink(quarantine)
        except FileNotFoundError:
            # Another cleanup actor removed the claimed inode.  The original
            # pathname is still independent, so cleanup is complete.
            return True
        return True
    except BaseException:
        # Do not restore the quarantined inode: a replacement may have
        # appeared at the original pathname since any earlier inspection.
        raise


def _allow_stdio_without_socket() -> bool:
    """Return whether MCP stdio may continue when the socket cannot start."""
    return SETTINGS.allow_stdio_without_socket.value


def _require_socket_ready():
    """Require socket readiness unless explicit stdio-only fallback is enabled."""
    if not _SOCKET_STARTUP_EVENT.wait(timeout=SOCKET_STARTUP_TIMEOUT_SECONDS):
        if _allow_stdio_without_socket():
            log_event(
                LOGGER, logging.WARNING, "socket_unavailable_stdio_continues",
                reason="startup timeout",
            )
            return
        raise SystemExit(
            f"Socket server did not become ready within {SOCKET_STARTUP_TIMEOUT_SECONDS} seconds"
        )
    socket_state, socket_failure = _socket_lifecycle()
    if socket_state == "failed":
        if _allow_stdio_without_socket():
            log_event(
                LOGGER, logging.WARNING, "socket_unavailable_stdio_continues",
                reason=socket_failure,
            )
            return
        raise SystemExit(f"Socket server failed to start: {socket_failure}")


def _instrument_tool(name):
    """Decorate a tool with opt-in content-free call metrics and timing logs."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            call_id = next(_TOOL_CALL_IDS)
            started = time.perf_counter()
            log_event(LOGGER, logging.INFO, "mcp_tool_started", call_id=call_id, tool=name)
            try:
                metrics = _active_metrics()
                with metrics.measure(name) as state:
                    try:
                        result = function(*args, **kwargs)
                    except Exception as exc:
                        state["failure_category"] = _classify_tool_exception(
                            name, exc, kwargs
                        )
                        raise
                    response_text = result.text if isinstance(result, ToolResponseEnvelope) else result
                    if isinstance(response_text, str):
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
                    LOGGER, logging.INFO, "mcp_tool_completed",
                    call_id=call_id, duration_ms=duration_ms,
                    success=state["success"], tool=name,
                )
                return result
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                log_event(
                    LOGGER, logging.ERROR, "mcp_tool_failed",
                    call_id=call_id, duration_ms=duration_ms,
                    error_type=type(exc).__name__, tool=name,
                )
                raise
            finally:
                # MCP clients may terminate the server process without running
                # Python's atexit handlers. Persist the current counters after
                # each tool call so benchmark runners can still collect them.
                _write_metrics_snapshot()
        return wrapper
    return decorator


def _call_tool_result(envelope: ToolResponseEnvelope) -> CallToolResult:
    """Create an MCP result with readable text and its structured envelope."""
    return CallToolResult(
        content=[TextContent(type="text", text=envelope.text or "")],
        structuredContent=envelope.model_dump(mode="json", exclude={"text"}),
    )


def _call_tool_result_bytes(envelope: ToolResponseEnvelope) -> int:
    """Measure the JSON-encoded MCP result body, including both result channels."""
    return len(_call_tool_result(envelope).model_dump_json(by_alias=True).encode("utf-8"))


def _legacy_tool_envelope(tool: str, result: Any) -> ToolResponseEnvelope:
    """Wrap an existing Python text result in the versioned MCP result shape."""
    text = result if isinstance(result, str) else str(result)
    data: Dict[str, Any] = {}
    error = None
    status: Literal["ok", "error"] = "ok"
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        if payload.get("status") == "error":
            status = "error"
            if payload.get("error_code"):
                error_code = str(payload["error_code"])
            elif tool in {"get_capture_summary", "clear_captures"}:
                error_code = "capture_not_found"
            elif tool in {"capture_text", "capture_file", "execute_and_capture"}:
                error_code = "capture_failed"
            else:
                error_code = f"{tool}_failed"
            error = ToolErrorEnvelope(
                code=error_code,
                message=str(payload.get("message") or "The operation failed."),
            )
        else:
            data = payload
    elif text.startswith(("Error", "Search Error")) or (
        tool == "clear_captures" and "not found" in text.lower()
    ):
        status = "error"
        if tool == "get_capture_summary":
            code = "capture_not_found" if "not found" in text.lower() else "retrieval_failed"
        elif tool == "search_capture":
            code = "invalid_search_request" if any(
                marker in text.lower()
                for marker in ("query must", "query exceeds", "unsupported", "top_k", "context_lines")
            ) else "search_failed"
        elif tool.startswith("capture_") or tool == "execute_and_capture":
            code = "capture_failed"
        elif tool == "clear_captures" and "not found" in text.lower():
            code = "capture_not_found"
        else:
            code = "operation_failed"
        error = ToolErrorEnvelope(code=code, message=text)
    else:
        data = {"message": text}

    envelope = ToolResponseEnvelope(status=status, data=data, error=error, text=text)
    return _fit_tool_envelope(envelope)


def _fit_tool_envelope(
    envelope: ToolResponseEnvelope,
    max_bytes: int = MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
) -> ToolResponseEnvelope:
    """Keep the complete structured MCP result within the configured byte budget."""
    if _call_tool_result_bytes(envelope) <= max_bytes:
        return envelope

    marker = "\n... [response truncated; use the structured status and follow-up tools]"
    shortened = envelope.model_copy(update={"data": {}, "truncated": True})
    low = 0
    high = max_bytes
    best = shortened.model_copy(update={"text": marker[:max(0, max_bytes // 4)]})
    while low <= high:
        middle = (low + high) // 2
        candidate_text = _truncate_utf8_with_marker(envelope.text, middle, marker)
        if candidate_text is None:
            candidate_text = ""
        candidate = shortened.model_copy(update={"text": candidate_text})
        if _call_tool_result_bytes(candidate) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _mcp_tool(name, category, *, structured_result_factory=None):
    """Register a synchronous tool implementation behind an async MCP adapter.

    The synchronous function remains the public Python API, while FastMCP sees
    an async callable and therefore does not execute blocking work on its event
    loop.  The complete instrumented call runs in a worker thread so command
    execution, model loading, and searches all share the same non-blocking
    boundary. ``category`` is the tool's primary descriptive capability group.
    """
    if category not in MCP_TOOL_CATEGORY_NAMES:
        raise ValueError(f"unknown MCP tool category: {category}")

    def decorator(function):
        _MCP_TOOL_REGISTRATIONS.append(
            (function, name, category, structured_result_factory)
        )
        _register_mcp_tool_for_app(
            mcp,
            function,
            name,
            category,
            structured_result_factory,
            DEFAULT_SERVICES,
        )
        if name not in _REGISTERED_MCP_TOOL_NAMES:
            _REGISTERED_MCP_TOOL_NAMES.append(name)
        _REGISTERED_MCP_TOOL_CATEGORIES[name] = category
        return function

    return decorator


def create_mcp_server(context: "ServiceContext | None" = None):
    """Build an MCP application whose tools use the supplied service context."""
    service_context = context or DEFAULT_SERVICES
    app = FastMCP(
        "ephemeral-buffer",
        instructions=_mcp_instructions(service_context),
    )
    for function, name, category, structured_result_factory in _MCP_TOOL_REGISTRATIONS:
        _register_mcp_tool_for_app(
            app,
            function,
            name,
            category,
            structured_result_factory,
            service_context,
        )
    return app


def _register_mcp_tool_for_app(
    app,
    function,
    name,
    category,
    structured_result_factory,
    service_context,
):
    """Register one shared tool implementation with context-bound adapters."""
    # Reuse the registration behavior installed by the decorator factory.
    @wraps(function)
    async def adapter(*args, **kwargs):
        with _activate_service_context(service_context), _bind_mcp_metrics_scope(
            app, service_context
        ):
            registered_call = service_context.register_tool_call(name)
            if registered_call is None:
                message = "The service is shutting down; retry after it restarts."
                if structured_result_factory is not None or category in {
                    "capture", "retrieval", "search", "lifecycle"
                }:
                    response = ToolResponseEnvelope(
                        status="error",
                        error=ToolErrorEnvelope(code="server_shutting_down", message=message),
                        text=f"Error: {message}",
                    )
                    return _call_tool_result(_fit_tool_envelope(response))
                if category == "execution":
                    return json.dumps(
                        {"status": "error", "error": {"code": "server_shutting_down", "message": message}},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                return f"Error: server_shutting_down: {message}"
            call_id, cancellation_event = registered_call
            try:
                lane = "diagnostics" if name == "get_buffer_stats" else "mcp"
                ticket = await admission_gate(lane).acquire(name)
            except AdmissionBusy:
                service_context.finish_tool_call(call_id)
                message = "Foreground work capacity is full; retry this tool call shortly."
                log_event(LOGGER, logging.WARNING, "mcp_tool_rejected_busy", tool=name)
                if structured_result_factory is not None or category in {
                    "capture", "retrieval", "search", "lifecycle"
                }:
                    response = ToolResponseEnvelope(
                        status="error",
                        error=ToolErrorEnvelope(code="server_busy", message=message),
                        text=f"Error: {message}",
                    )
                    return _call_tool_result(_fit_tool_envelope(response))
                if category == "execution":
                    return json.dumps(
                        {"status": "error", "error": {"code": "server_busy", "message": message}},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                return f"Error: server_busy: {message}"
            except BaseException:
                service_context.finish_tool_call(call_id)
                raise

            if service_context.closing:
                ticket.release()
                service_context.finish_tool_call(call_id)
                message = "The service is shutting down; retry after it restarts."
                if structured_result_factory is not None or category in {
                    "capture", "retrieval", "search", "lifecycle"
                }:
                    response = ToolResponseEnvelope(
                        status="error",
                        error=ToolErrorEnvelope(code="server_shutting_down", message=message),
                        text=f"Error: {message}",
                    )
                    return _call_tool_result(_fit_tool_envelope(response))
                if category == "execution":
                    return json.dumps(
                        {"status": "error", "error": {"code": "server_shutting_down", "message": message}},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                return f"Error: server_shutting_down: {message}"

            cancel_token = None
            mcp_invocation_token = _MCP_TOOL_INVOCATION.set(True)
            try:
                cancel_token = (
                    _REQUEST_CANCEL_EVENT.set(cancellation_event)
                    if cancellation_event is not None
                    else None
                )
                if structured_result_factory is not None:
                    response = await _run_admitted_thread(
                        ticket,
                        structured_result_factory,
                        *args,
                        _service_context=service_context,
                        _call_id=call_id,
                        **kwargs,
                    )
                    if not isinstance(response, ToolResponseEnvelope):
                        raise TypeError("structured_result_factory must return ToolResponseEnvelope")
                else:
                    result = await _run_admitted_thread(
                        ticket,
                        function,
                        *args,
                        _service_context=service_context,
                        _call_id=call_id,
                        **kwargs,
                    )
                    if category not in {"capture", "retrieval", "search", "lifecycle"}:
                        return result
                    response = _legacy_tool_envelope(name, result)
                return _call_tool_result(_fit_tool_envelope(response))
            finally:
                if cancel_token is not None:
                    _REQUEST_CANCEL_EVENT.reset(cancel_token)
                _MCP_TOOL_INVOCATION.reset(mcp_invocation_token)
                ticket.release()
                service_context.finish_tool_call(call_id)

    if structured_result_factory is not None or category in {
        "capture", "retrieval", "search", "lifecycle"
    }:
        signature = inspect.signature(function)
        adapter.__signature__ = signature.replace(return_annotation=ToolResponseEnvelope)
        adapter.__annotations__ = {
            **getattr(function, "__annotations__", {}),
            "return": ToolResponseEnvelope,
        }
    @contextmanager
    def validation_scope():
        with _activate_service_context(service_context), _bind_mcp_metrics_scope(
            app, service_context
        ):
            yield

    def observe_validation_failure():
        with _active_metrics().measure(name) as state:
            state["success"] = False
            state["failure_category"] = "validation"
        _write_metrics_snapshot(service_context)

    _register_fastmcp_tool(
        app,
        adapter,
        name=name,
        validation_scope=validation_scope,
        validation_observer=observe_validation_failure,
    )

def _mcp_instructions(context: "ServiceContext | None" = None) -> str:
    """Return client-visible operating guidance for this server instance."""
    if context is None:
        isolation_required = socket_isolation_required()
        isolation_configured = socket_isolation_configured()
    else:
        isolation_required = context.settings.socket_require_isolation.value
        isolation_configured = context.settings.identity.isolation_configured
    if isolation_required and not isolation_configured:
        isolation = "Socket isolation is required but not configured; startup must fail."
    elif isolation_configured:
        isolation = "Socket isolation is configured for this session."
    else:
        isolation = "This is legacy single-session mode; configure EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH for concurrency."
    socket_state, socket_failure = _socket_lifecycle()
    if socket_state == "failed":
        socket_status = f"Socket lifecycle is failed ({socket_failure})."
    else:
        socket_status = f"Socket lifecycle is {socket_state}."
    return (
        "Use execute_and_capture for large, noisy, or uncertain command output and for workflows "
        "that need later search or follow-up retrieval. Use direct command execution for small, "
        "targeted inspections. " + isolation + " " + socket_status
    )


# Initialize FastMCP
mcp = FastMCP("ephemeral-buffer", instructions=_mcp_instructions())


def _refresh_mcp_instructions(app=None, context: "ServiceContext | None" = None) -> None:
    """Refresh client guidance immediately before MCP request serving."""
    target = app or mcp
    _refresh_fastmcp_instructions(target, _mcp_instructions(context))


def _runtime_package_version() -> str:
    """Return the version for the source or installed server being run."""
    pyproject = Path(__file__).with_name("pyproject.toml")
    try:
        source_text = pyproject.read_text(encoding="utf-8")
    except OSError:
        source_text = ""
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']\s*$', source_text, re.MULTILINE)
    if match:
        return match.group(1)

    try:
        return package_version("ephemeral-buffer-mcp")
    except PackageNotFoundError:
        return "source checkout"


_UNSET = object()


class ServiceContext:
    """Own the mutable resources for one embedded or server service instance."""

    def __init__(
        self,
        settings,
        *,
        metrics: LocalMetrics | None = None,
        engine: EphemeralEngine | None = None,
        engine_options: dict[str, Any] | None = None,
        metrics_file: str | None | object = _UNSET,
    ) -> None:
        if metrics_file is _UNSET:
            self.metrics_file = settings.metrics_file.value
        else:
            self.metrics_file = metrics_file
            metrics_file_setting = dataclasses.replace(
                settings.metrics_file,
                value=metrics_file,
                source="context",
                status="accepted" if metrics_file else "disabled",
                invalid_value_behavior="not applicable",
            )
            settings = dataclasses.replace(settings, metrics_file=metrics_file_setting)
        self.settings = settings
        self.metrics = metrics or LocalMetrics(enabled=settings.metrics_enabled.value)
        self.engine = engine or EphemeralEngine(
            settings=settings,
            metrics=self.metrics,
            **(engine_options or {}),
        )
        self._execution_manager: PhaseExecutionManager | None = None
        self._execution_manager_lock = threading.Lock()
        self._metrics_snapshot_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._close_complete = threading.Event()
        self._close_error: BaseException | None = None
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_condition = threading.Condition(self._lifecycle_lock)
        self._closing = False
        self._active_tool_calls: Dict[str, tuple[str, threading.Event | None]] = {}
        self._deferred_tool_calls: set[str] = set()
        self.last_shutdown_report: Dict[str, Any] = {}
        self.engine.set_metrics_snapshot_callback(lambda: _write_metrics_snapshot(self))

    @property
    def execution_manager(self) -> PhaseExecutionManager:
        """Construct durable execution storage machinery only when requested."""
        with self._execution_manager_lock:
            with self._lifecycle_lock:
                if self._closing:
                    raise RuntimeError("service is shutting down")
            if self._execution_manager is None:
                self._execution_manager = PhaseExecutionManager(
                    self.settings.identity.state_dir,
                    max_output_bytes=max(512, self.engine.max_buffer_bytes),
                    quota_bytes=self.settings.execution_state_quota_bytes.value,
                    checkpoint_reserve_bytes=(
                        self.settings.execution_checkpoint_reserve_bytes.value
                    ),
                )
        return self._execution_manager

    def start(self) -> None:
        """Start optional background engine work owned by this context."""
        self.engine.start_embedding_warmup()

    def register_tool_call(
        self, name: str
    ) -> tuple[str, threading.Event | None] | None:
        """Track admitted work and provide a request cancellation signal for commands."""
        with self._lifecycle_lock:
            if self._closing:
                return None
            call_id = uuid.uuid4().hex
            cancellation_event = threading.Event() if name == "execute_and_capture" else None
            self._active_tool_calls[call_id] = (name, cancellation_event)
            return call_id, cancellation_event

    def finish_tool_call(self, call_id: str) -> None:
        with self._lifecycle_condition:
            if call_id in self._deferred_tool_calls:
                return
            self._active_tool_calls.pop(call_id, None)
            self._lifecycle_condition.notify_all()

    def defer_tool_call_until(
        self,
        call_id: str,
        task: asyncio.Future[Any],
    ) -> None:
        """Keep context work active until detached worker work completes."""
        with self._lifecycle_condition:
            if call_id not in self._active_tool_calls:
                return
            self._deferred_tool_calls.add(call_id)

        def finish_deferred(_task: asyncio.Future[Any]) -> None:
            with self._lifecycle_condition:
                self._deferred_tool_calls.discard(call_id)
                self._active_tool_calls.pop(call_id, None)
                self._lifecycle_condition.notify_all()

        task.add_done_callback(finish_deferred)

    def wait_for_tool_calls(self, timeout: float) -> bool:
        """Wait only for work registered to this service context."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lifecycle_condition:
            while self._active_tool_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle_condition.wait(remaining)
            return True

    @property
    def closing(self) -> bool:
        with self._lifecycle_lock:
            return self._closing

    def close(self) -> Dict[str, Any]:
        """Stop admission, cancel owned work, and report anything beyond the grace period."""
        with self._close_lock:
            if self._closed:
                should_wait = True
            else:
                should_wait = False
                self._closed = True
                with self._lifecycle_lock:
                    self._closing = True
                    for _name, cancellation_event in self._active_tool_calls.values():
                        if cancellation_event is not None:
                            cancellation_event.set()
        if should_wait:
            self._close_complete.wait()
            if self._close_error is not None:
                raise self._close_error
            return dict(self.last_shutdown_report)
        try:
            return self._close_resources()
        except BaseException as exc:
            self._close_error = exc
            raise
        finally:
            self._close_complete.set()

    def _close_resources(self) -> Dict[str, Any]:
        grace_setting = getattr(self.settings, "shutdown_grace_seconds", None)
        grace_seconds = (
            grace_setting.value
            if grace_setting is not None
            else DEFAULT_SHUTDOWN_GRACE_SECONDS
        )
        deadline = time.monotonic() + grace_seconds
        execution_shutdown = {
            "unfinished_execution_ids": [],
            "unfinished_execution_count": 0,
        }
        execution_grace = grace_seconds / 2
        engine_grace = grace_seconds / 4
        with self._execution_manager_lock:
            execution_manager = self._execution_manager
        if execution_manager is not None:
            execution_shutdown = execution_manager.shutdown(execution_grace)
        admission_wait = max(
            0.0,
            deadline - time.monotonic() - engine_grace,
        )
        admission_drained = self.wait_for_tool_calls(
            admission_wait
        )
        engine_shutdown = self.engine.shutdown(
            timeout_seconds=max(0.0, deadline - time.monotonic())
        )
        admission_shutdown = ADMISSION_METRICS.snapshot()
        try:
            _write_metrics_snapshot(self)
        finally:
            with self._lifecycle_lock:
                active_names = [name for name, _event in self._active_tool_calls.values()]
            is_private_state_dir = (
                self.settings.identity.state_source
                == "process:private temporary directory"
            )
            state_dir_cleanup_skipped = is_private_state_dir and (
                not admission_drained
                or bool(active_names)
                or execution_shutdown.get("unfinished_execution_count", 0) > 0
            )
            self.last_shutdown_report = {
                "admission_drained": admission_drained,
                "active_tool_calls": len(active_names),
                "active_tool_types": sorted(set(active_names)),
                "admission_active": admission_shutdown["admission_active"],
                "admission_queued": admission_shutdown["admission_queued"],
                "admission_active_by_type": admission_shutdown["admission_active_by_type"],
                "admission_queued_by_type": admission_shutdown["admission_queued_by_type"],
                "execution_state_dir_cleanup_skipped": state_dir_cleanup_skipped,
                **execution_shutdown,
                **engine_shutdown,
            }
            log_event(
                LOGGER,
                logging.WARNING
                if (
                    not admission_drained
                    or bool(active_names)
                    or self.last_shutdown_report.get("unfinished_execution_count", 0)
                    or self.last_shutdown_report.get("unfinished_engine_work")
                )
                else logging.INFO,
                "service_shutdown_complete",
                grace_seconds=grace_seconds,
                admission_drained=admission_drained,
                active_tool_calls=len(active_names),
                active_tool_types=sorted(set(active_names)),
                admission_active=admission_shutdown["admission_active"],
                admission_queued=admission_shutdown["admission_queued"],
                admission_active_by_type=admission_shutdown["admission_active_by_type"],
                admission_queued_by_type=admission_shutdown["admission_queued_by_type"],
                execution_state_dir_cleanup_skipped=state_dir_cleanup_skipped,
                unfinished_execution_count=execution_shutdown.get(
                    "unfinished_execution_count", 0
                ),
                unfinished_engine_work=engine_shutdown.get("unfinished_work", []),
            )
        if is_private_state_dir and not state_dir_cleanup_skipped:
            cleanup_execution_state_dir(self.settings.identity.state_dir)
        return dict(self.last_shutdown_report)


_SERVICE_CONTEXT_OVERRIDE: ContextVar[ServiceContext | None] = ContextVar(
    "ephemeral_server_service_context", default=None
)


@contextmanager
def _activate_service_context(context: ServiceContext) -> Any:
    token = _SERVICE_CONTEXT_OVERRIDE.set(context)
    try:
        yield context
    finally:
        _SERVICE_CONTEXT_OVERRIDE.reset(token)


def create_service_context(
    settings=None,
    *,
    metrics_file: str | None = None,
    state_dir: str | None = None,
    engine_options: dict[str, Any] | None = None,
) -> ServiceContext:
    """Create an independently owned service context for embedding or benchmarks.

    When settings are resolved here, the process-private default state path is
    unique to this context and is only created by the first durable execution.
    Pass ``metrics_file=None`` to keep metrics in memory without persistence.
    """
    if settings is None:
        settings = load_settings(
            default_state_dir=state_dir or new_default_execution_state_dir()
        )
    elif (
        state_dir is None
        and settings.identity.state_source == "process:private temporary directory"
    ):
        identity = dataclasses.replace(
            settings.identity,
            state_dir=new_default_execution_state_dir(),
        )
        settings = dataclasses.replace(settings, identity=identity)
    if state_dir is not None:
        identity = dataclasses.replace(settings.identity, state_dir=os.path.abspath(state_dir))
        identity = dataclasses.replace(
            identity,
            state_source="context:state_dir",
            durability="on-disk; retained across normal shutdowns until explicitly retired",
            legacy_state_dir=None,
        )
        settings = dataclasses.replace(settings, identity=identity)
    return ServiceContext(
        settings,
        metrics_file=metrics_file,
        engine_options=engine_options,
    )


DEFAULT_SERVICES = ServiceContext(SETTINGS, metrics=METRICS)
engine = DEFAULT_SERVICES.engine


class _LazyExecutionManagerProxy:
    """Keep the module-level compatibility API lazy until execution is used."""

    def __getattr__(self, name):
        return getattr(DEFAULT_SERVICES.execution_manager, name)


execution_manager = _LazyExecutionManagerProxy()
_ENGINE_OVERRIDE: ContextVar[Optional[EphemeralEngine]] = ContextVar(
    "ephemeral_server_engine_override",
    default=None,
)


def _active_service_context() -> ServiceContext:
    return _SERVICE_CONTEXT_OVERRIDE.get() or DEFAULT_SERVICES


def _active_settings():
    context = _SERVICE_CONTEXT_OVERRIDE.get()
    return SETTINGS if context is None or context is DEFAULT_SERVICES else context.settings


def _active_metrics() -> LocalMetrics:
    context = _SERVICE_CONTEXT_OVERRIDE.get()
    return METRICS if context is None or context is DEFAULT_SERVICES else context.metrics


def _active_engine() -> EphemeralEngine:
    """Return the active context engine or the process engine."""
    override = _ENGINE_OVERRIDE.get()
    if override is not None:
        return override
    context = _SERVICE_CONTEXT_OVERRIDE.get()
    return engine if context is None or context is DEFAULT_SERVICES else context.engine


def _active_execution_manager():
    context = _SERVICE_CONTEXT_OVERRIDE.get()
    return execution_manager if context is None or context is DEFAULT_SERVICES else context.execution_manager


def _capture_execution_phase(
    phase: Dict[str, Any],
    output: str,
    result: Dict[str, Any],
) -> Optional[str]:
    """Retain a phase result in the searchable ring as a convenience snapshot."""
    active_engine = _active_engine()
    capture = active_engine.ingest(
        output,
        label=f"execution phase: {phase['name']}",
        content_type="auto",
        truncated=bool(result.get("truncated", False)),
        original_byte_size=(
            result.get("original_byte_size") if result.get("truncated", False) else None
        ),
        command_exit_code=result.get("exit_code"),
        timed_out=bool(result.get("timed_out", False)),
        source="execution",
        duration_ms=result.get("duration_ms"),
        structured_metrics=phase.get("structured_metrics", {}),
    )
    result["capture_session_id"] = active_engine.session_id
    return capture.capture_id


def _background_execution_output_handler():
    """Preserve the calling service context for phase captures on worker threads."""
    context = copy_context()

    def handle(phase, output, result):
        return context.run(_capture_execution_phase, phase, output, result)

    return handle


def _execution_json(operation, *, max_response_bytes: Optional[int] = None) -> str:
    """Run an execution operation and return a compact machine-readable result."""
    try:
        payload = operation()
        result = _json_dumps_with_limit(payload, max_response_bytes)
        if result is None:
            if isinstance(payload, dict) and isinstance(payload.get("executions"), list):
                compact = {
                    key: payload[key]
                    for key in ("status", "limit", "offset")
                    if key in payload
                }
                compact["response_truncated"] = True
                compact["executions"] = []
                for execution in payload["executions"]:
                    if not isinstance(execution, dict):
                        continue
                    summary = {
                        key: execution[key]
                        for key in (
                            "execution_id", "label", "execution_status", "partial",
                            "execution_in_progress", "cancellation_requested",
                            "background_error", "updated_at",
                            "completed_phase_count", "phase_count",
                        )
                        if key in execution
                    }
                    candidate = dict(compact)
                    candidate["executions"] = [*compact["executions"], summary]
                    candidate["returned_count"] = len(candidate["executions"])
                    candidate["omitted_count"] = len(payload["executions"]) - candidate["returned_count"]
                    encoded_candidate = _json_dumps_with_limit(candidate, max_response_bytes)
                    if encoded_candidate is None:
                        break
                    compact["executions"].append(summary)
                compact["returned_count"] = len(compact["executions"])
                compact["omitted_count"] = len(payload["executions"]) - len(compact["executions"])
                compact_result = _json_dumps_with_limit(compact, max_response_bytes)
                if compact_result is not None:
                    return compact_result
            if isinstance(payload, dict) and payload.get("execution_id"):
                compact = {
                    key: payload[key]
                    for key in (
                        "status", "schema_version", "execution_id", "label",
                        "execution_status", "execution_in_progress",
                        "cancellation_requested", "partial", "summary", "created_at",
                        "background_error", "updated_at", "resume",
                        "completed_phase_count", "phase_count",
                    )
                    if key in payload
                }
                compact["response_truncated"] = True
                compact["phases"] = [
                    {
                        key: phase.get(key)
                        for key in ("name", "status", "attempts", "error")
                    }
                    for phase in payload.get("phases", [])
                    if isinstance(phase, dict)
                ]
                compact_result = _json_dumps_with_limit(compact, max_response_bytes)
                if compact_result is not None:
                    return compact_result
            return (
                "Error managing execution: response exceeds the "
                f"{max_response_bytes:,}-byte limit; request a smaller output chunk"
            )
        return result
    except ExecutionBusyError as exc:
        return json.dumps(
            {
                "status": "error",
                "error": {"code": "execution_busy", "message": str(exc)},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        return f"Error managing execution: {exc}"


def _json_dumps_with_limit(payload: Any, max_bytes: Optional[int]) -> Optional[str]:
    """Encode incrementally so an oversized response never builds in full."""
    if max_bytes is None:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    chunks = []
    total_bytes = 0
    for chunk in encoder.iterencode(payload):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > max_bytes:
            return None
        chunks.append(chunk)
    return "".join(chunks)


def _execution_public_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate session-local capture references before returning durable state."""
    active_engine = _active_engine()
    for phase in payload.get("phases", []):
        result = phase.get("result")
        if not isinstance(result, dict) or "capture_id" not in result:
            continue
        result["capture_available"] = (
            result.get("capture_session_id") == active_engine.session_id
            and active_engine.get_capture(result["capture_id"]) is not None
        )
    return payload


def _execution_get_payload(execution_id: str, include_output: bool) -> Dict[str, Any]:
    """Avoid copying large durable output before the response budget is known."""
    metadata = _active_execution_manager().public(execution_id, include_output=False)
    if not include_output:
        return metadata
    output_bytes = sum(
        phase.get("output_bytes", 0)
        for phase in metadata.get("phases", [])
        if isinstance(phase, dict) and isinstance(phase.get("output_bytes", 0), int)
    )
    if output_bytes > EXECUTION_OUTPUT_RESPONSE_MAX_BYTES:
        metadata["response_truncated"] = True
        return metadata
    return _active_execution_manager().public(execution_id, include_output=True)


def _metrics_snapshot(
    *,
    service_context: ServiceContext | None = None,
    since_snapshot: Optional[str] = None,
    include_snapshot_token: bool = False,
    scope_key: str | None = None,
) -> Dict[str, Any]:
    """Return metrics using the live EB MCP registration inventory."""
    metrics = (
        service_context.metrics
        if service_context is not None and service_context is not DEFAULT_SERVICES
        else _active_metrics()
    )
    return metrics.snapshot(
        available_tools=_REGISTERED_MCP_TOOL_NAMES,
        tool_categories=_REGISTERED_MCP_TOOL_CATEGORIES,
        since_snapshot=since_snapshot,
        include_snapshot_token=include_snapshot_token,
        scope_key=scope_key,
    )


def _write_metrics_snapshot(service_context: ServiceContext | None = None) -> None:
    """Persist an opt-in, content-free metrics snapshot for benchmark runners."""
    context = service_context or _active_service_context()
    path = (
        _active_settings().metrics_file.value
        if context is DEFAULT_SERVICES
        else context.metrics_file
    )
    metrics = _active_metrics() if context is DEFAULT_SERVICES else context.metrics
    if not path or not metrics.enabled:
        return
    with context._metrics_snapshot_lock:
        temporary_path = None
        try:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(json.dumps(
                    _metrics_snapshot(service_context=context, scope_key="process"),
                    sort_keys=True,
                ))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        except OSError as exc:
            log_event(LOGGER, logging.WARNING, "metrics_snapshot_write_failed", error_type=type(exc).__name__)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


atexit.register(DEFAULT_SERVICES.close)


# --- MCP Tools ---


class ExecutionPhaseInput(BaseModel):
    """Schema contract for one phase exposed through the MCP tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES})
    command: str = Field(min_length=1, json_schema_extra={"maxUtf8Bytes": 16 * 1024})
    cwd: Optional[str] = Field(
        default=None, min_length=1, json_schema_extra={"maxUtf8Bytes": 4096}
    )
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    max_output_bytes: Optional[int] = Field(default=None, ge=512)
    structured_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"maxJsonBytes": MAX_STRUCTURED_METRICS_BYTES},
    )
    side_effects: Literal["none", "unsafe"] = "none"
    unsafe_side_effects: Optional[bool] = None
    idempotency_key: Optional[str] = Field(
        default=None, min_length=1,
        json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES},
    )

    @field_validator("name", "command", "cwd", "idempotency_key")
    @classmethod
    def validate_utf8_byte_limit(cls, value, info):
        """Apply the same UTF-8 byte bounds as durable execution records."""
        limits = {
            "name": MAX_PHASE_NAME_BYTES,
            "command": 16 * 1024,
            "cwd": 4096,
            "idempotency_key": MAX_PHASE_NAME_BYTES,
        }
        if value is not None and len(value.encode("utf-8")) > limits[info.field_name]:
            raise ValueError(
                f"{info.field_name} exceeds the {limits[info.field_name]:,}-byte limit"
            )
        return value

    @field_validator("structured_metrics")
    @classmethod
    def validate_structured_metrics_size(cls, value):
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("structured_metrics must contain JSON-compatible values") from exc
        if len(encoded) > MAX_STRUCTURED_METRICS_BYTES:
            raise ValueError(
                f"structured_metrics exceeds the {MAX_STRUCTURED_METRICS_BYTES:,}-byte limit"
            )
        return value

    @field_validator("max_output_bytes")
    @classmethod
    def validate_output_limit(cls, value):
        if value is not None and value > _active_engine().max_buffer_bytes:
            raise ValueError(
                f"max_output_bytes cannot exceed the configured {_active_engine().max_buffer_bytes:,}-byte limit"
            )
        return value

    @model_validator(mode="after")
    def normalize_side_effect_alias(self):
        """Treat the boolean alias as an override only when the enum was omitted."""
        if "unsafe_side_effects" not in self.model_fields_set:
            return self
        alias_side_effects = "unsafe" if self.unsafe_side_effects else "none"
        if "side_effects" in self.model_fields_set and self.side_effects != alias_side_effects:
            raise ValueError("side_effects and unsafe_side_effects must agree")
        self.side_effects = alias_side_effects
        return self


def _resolve_preflight_cwd(cwd: Optional[str]) -> dict[str, Any]:
    """Resolve a requested working directory without executing user commands."""
    requested = cwd if cwd is not None else os.getcwd()
    path = Path(requested).expanduser()
    resolved = path.resolve(strict=False)
    exists = os.path.lexists(path)
    is_symlink = path.is_symlink()
    target_exists = resolved.exists() if is_symlink else exists
    if is_symlink and not target_exists:
        status = "dangling-symlink"
    elif not exists:
        status = "missing"
    elif not resolved.is_dir():
        status = "not-a-directory"
    else:
        status = "ok"
    return {
        "input": requested,
        "resolved": str(resolved),
        "source": "explicit" if cwd is not None else "process-cwd",
        "status": status,
        "exists": exists,
        "is_directory": resolved.is_dir(),
        "is_symlink": is_symlink,
        "symlink_target": str(resolved) if is_symlink else None,
        "symlink_target_exists": target_exists if is_symlink else None,
    }


def _resolve_preflight_executable(tokens: list[str], resolved_cwd: str) -> dict[str, Any]:
    """Resolve only the first parsed token; shell expansion remains out of scope."""
    if not tokens:
        return {"status": "unavailable", "reason": "command contains no parseable tokens"}
    token = tokens[0]
    if os.path.sep in token or (os.altsep and os.altsep in token):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = Path(resolved_cwd) / candidate
        candidate = candidate.resolve(strict=False)
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return {"status": "resolved", "requested": token, "resolved": str(candidate)}
        return {"status": "unavailable", "requested": token, "reason": "path is not an executable file"}

    search_path = os.environ.get("PATH", os.defpath)
    search_entries = [
        entry if os.path.isabs(entry) else os.path.abspath(os.path.join(resolved_cwd, entry))
        for entry in search_path.split(os.pathsep)
    ]
    resolved = shutil.which(token, path=os.pathsep.join(search_entries))
    if resolved:
        return {"status": "resolved", "requested": token, "resolved": str(Path(resolved).resolve(strict=False))}
    return {"status": "unavailable", "requested": token, "reason": "executable was not found on PATH"}


@_mcp_tool("preflight_command", "diagnostics")
@_instrument_tool("preflight_command")
def preflight_command(command: str, cwd: Optional[str] = None) -> str:
    """Return content-free path and executable diagnostics without running ``command``.

    This resolves the working directory, symlink target, detectable Git
    repository root, and first executable token. Relative and empty PATH entries
    are resolved against the requested working directory. Shell expansion, aliases,
    pipelines, redirections, environment changes, and arbitrary shell logic
    cannot be verified here. The requested command is never executed.
    """
    try:
        working_directory = _resolve_preflight_cwd(cwd)
        try:
            tokens = shlex.split(command)
            parse_status = "ok" if tokens else "empty"
        except ValueError as exc:
            tokens = []
            parse_status = f"unavailable: {exc}"

        executable = _resolve_preflight_executable(tokens, working_directory["resolved"])
        repository: dict[str, Any] = {
            "status": "unavailable",
            "root": None,
            "reason": "working directory is not an existing directory",
        }
        if working_directory["status"] == "ok":
            try:
                result = subprocess.run(
                    ["git", "-C", working_directory["resolved"], "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    repository = {"status": "detected", "root": str(Path(result.stdout.strip()).resolve())}
                else:
                    repository = {"status": "not-detected", "root": None, "reason": "not inside a Git work tree"}
            except (OSError, subprocess.TimeoutExpired) as exc:
                repository = {"status": "unavailable", "root": None, "reason": type(exc).__name__}

        return json.dumps(
            {
                "status": "ok",
                "command": {
                    "parse_status": parse_status,
                    "token_count": len(tokens),
                    "executable": executable,
                },
                "working_directory": working_directory,
                "repository": repository,
                "limitations": [
                    "The requested command was not executed.",
                    "Shell expansion, aliases, pipelines, redirections, and environment changes are not resolved.",
                    "Repository detection reports only a local Git work-tree root; it does not verify user intent or remotes.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "reason": type(exc).__name__})


@_mcp_tool("start_execution", "execution")
@_instrument_tool("start_execution")
def start_execution(
    phases: Annotated[List[ExecutionPhaseInput], Field(min_length=1, max_length=MAX_EXECUTION_PHASES)],
    execution_id: Optional[Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )]] = None,
    label: Annotated[str, Field(json_schema_extra={"maxUtf8Bytes": 1024})] = "",
    resume_policy: Literal["safe", "allow-unsafe"] = "safe",
    cwd: Optional[Annotated[str, Field(json_schema_extra={"maxUtf8Bytes": 4096})]] = None,
    timeout_seconds: Annotated[Optional[float], Field(gt=0)] = None,
    max_output_bytes: Annotated[
        Optional[int], Field(ge=512)
    ] = None,
) -> str:
    """Start a sequential, durably checkpointed set of command phases.

    Returns a durable execution ID promptly; use get_execution to follow
    progress. Caller disconnection detaches from this work. Use cancel_execution
    to request intentional termination.

    Each phase is an object with ``name`` and ``command`` plus optional
    ``cwd``, ``timeout_seconds``, ``max_output_bytes``, ``structured_metrics``,
    and ``side_effects`` (``none`` or ``unsafe``). A completed phase is never
    rerun by ``resume_execution``. An unsafe phase that must be retried after
    failure, timeout, or interruption requires ``confirm_unsafe=True`` or the
    explicit ``resume_policy='allow-unsafe'``. Outputs and phase event history
    are stored under ``EPHEMERAL_EXECUTION_STATE_DIR``.
    """
    phase_payloads = [
        phase.model_dump(exclude_none=True) if isinstance(phase, ExecutionPhaseInput) else phase
        for phase in phases
    ]
    manager = _active_execution_manager()
    start_method = (
        manager.start_background
        if _MCP_TOOL_INVOCATION.get()
        else manager.start
    )
    return _execution_json(
        lambda: _execution_public_payload(
            start_method(
                phase_payloads,
                execution_id=execution_id,
                label=label,
                resume_policy=resume_policy,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                output_handler=_background_execution_output_handler(),
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("resume_execution", "execution")
@_instrument_tool("resume_execution")
def resume_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    retry_failed: bool = False,
    confirm_unsafe: bool = False,
) -> str:
    """Resume an execution from its first incomplete phase and return promptly.

    Use get_execution to follow progress. Caller disconnection detaches from
    this durable work.

    Completed phases are skipped. Failed and timed-out phases require
    ``retry_failed=True``; a safe phase recovered as interrupted resumes on
    the normal call. Retries of unsafe phases additionally need
    ``confirm_unsafe=True`` unless the execution was created with the explicit
    ``allow-unsafe`` resume policy.
    """
    manager = _active_execution_manager()
    resume_method = (
        manager.resume_background
        if _MCP_TOOL_INVOCATION.get()
        else manager.resume
    )
    return _execution_json(
        lambda: _execution_public_payload(
            resume_method(
                execution_id,
                retry_failed=retry_failed,
                confirm_unsafe=confirm_unsafe,
                output_handler=_background_execution_output_handler(),
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("cancel_execution", "execution")
@_instrument_tool("cancel_execution")
def cancel_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
) -> str:
    """Request cancellation of an active durable execution by its ID."""
    return _execution_json(
        lambda: _active_execution_manager().request_cancel(execution_id),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("get_execution", "execution")
@_instrument_tool("get_execution")
def get_execution(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    include_output: bool = False,
) -> str:
    """Return durable phase metadata and event history for one execution."""
    return _execution_json(
        lambda: _execution_public_payload(
            _execution_get_payload(execution_id, include_output)
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("get_execution_output", "execution")
@_instrument_tool("get_execution_output")
def get_execution_output(
    execution_id: Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_EXECUTION_ID_BYTES}
    )],
    phase_name: Optional[Annotated[str, Field(
        min_length=1, json_schema_extra={"maxUtf8Bytes": MAX_PHASE_NAME_BYTES}
    )]] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    max_bytes: Annotated[int, Field(ge=512, le=MAX_EXECUTION_OUTPUT_CHUNK_BYTES)] = MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
) -> str:
    """Retrieve one bounded output chunk for all phases or one named phase."""
    return _execution_json(
        lambda: _execution_public_payload(
            _active_execution_manager().output(
                execution_id,
                phase_name,
                offset=offset,
                max_bytes=max_bytes,
            )
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("list_executions", "execution")
@_instrument_tool("list_executions")
def list_executions(
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> str:
    """List durable executions with compact partial/completion summaries."""
    return _execution_json(
        lambda: {
            "status": "ok",
            "limit": limit,
            "offset": offset,
            "executions": _active_execution_manager().list_public(limit=limit, offset=offset),
        },
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("get_execution_capacity", "execution")
@_instrument_tool("get_execution_capacity")
def get_execution_capacity() -> str:
    """Report durable storage diagnostics and active/queued background execution counts."""
    return _execution_json(
        lambda: _active_execution_manager().capacity(),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


@_mcp_tool("retire_executions", "execution")
@_instrument_tool("retire_executions")
def retire_executions(
    execution_ids: Annotated[
        List[str], Field(min_length=1, max_length=MAX_EXECUTION_RETIRE_BATCH)
    ],
    archive_path: Optional[Annotated[str, Field(
        json_schema_extra={"maxUtf8Bytes": 4096}
    )]] = None,
    dry_run: bool = True,
) -> str:
    """Preview selected durable records or archive and retire them.

    A dry run is read-only and reports eligibility and projected space reclaimed.
    Actual retirement requires an archive path outside the state directory; the
    archived bundle is durably written before the source record pair is removed.
    Active and fence-pending executions are never retired.
    """
    return _execution_json(
        lambda: _active_execution_manager().retire(
            execution_ids,
            archive_path=archive_path,
            dry_run=dry_run,
        ),
        max_response_bytes=EXECUTION_OUTPUT_RESPONSE_MAX_BYTES,
    )


def _bounded_diff_file_map(
    file_map: str,
    diff_meta: Optional[Dict[str, Any]] = None,
) -> tuple[str, int]:
    """Bound diff file-map text and return it with the omitted-entry count."""
    retained: List[str] = []
    files = diff_meta.get("files") if isinstance(diff_meta, dict) else None
    if isinstance(files, list):
        total_entries = len(files)
        # Format only the bounded prefix from structured metadata. The full
        # raw diff remains available through get_capture_slice.
        source_lines = []
        for file_info in files:
            status_tag = f" [{file_info['status'].upper()}]" if file_info["status"] != "modified" else ""
            source_lines.append(
                f"  - {file_info['path']}{status_tag} (+{file_info['additions']}, -{file_info['deletions']}) "
                f"| Buffer Lines: L{file_info['start_line']}-L{file_info['end_line']}"
            )
            if len(source_lines) >= SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES:
                break
    else:
        total_entries = file_map.count("\n") + (1 if file_map else 0)
        source_lines = []
        start = 0
        for _ in range(SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES):
            end = file_map.find("\n", start)
            if end < 0:
                if start < len(file_map):
                    source_lines.append(file_map[start:])
                break
            source_lines.append(file_map[start:end])
            start = end + 1

    for line in source_lines:
        candidate = "\n".join([*retained, line])
        if len(candidate.encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            break
        retained.append(line)

    omitted = total_entries - len(retained)
    if omitted:
        marker = (
            f"... [{omitted:,} diff file-map entries omitted; "
            "use get_capture_slice for complete diff details] ..."
        )
        while retained and len(("\n".join([*retained, marker])).encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            retained.pop()
            omitted += 1
        bounded = "\n".join([*retained, marker])
        if len(bounded.encode("utf-8")) > SUMMARY_DIFF_FILE_MAP_MAX_BYTES:
            bounded = marker[:SUMMARY_DIFF_FILE_MAP_MAX_BYTES]
        return bounded, omitted
    return "\n".join(retained), 0


def _bounded_summary_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    """Bound text metadata included alongside an execution summary."""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    marker_bytes = marker.encode("utf-8")
    prefix = encoded[: max_bytes - len(marker_bytes)].decode("utf-8", errors="ignore")
    return prefix + marker, True


def _bounded_json_text(value: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    """Bound the escaped JSON representation of metadata strings."""
    escaped_bytes = lambda text: len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2
    if escaped_bytes(value) <= max_bytes:
        return value, False

    marker_bytes = escaped_bytes(marker)
    if marker_bytes > max_bytes:
        return "", True
    prefix_budget = max_bytes - marker_bytes
    prefix = []
    prefix_bytes = 0
    for char in value:
        char_bytes = escaped_bytes(char)
        if prefix_bytes + char_bytes > prefix_budget:
            break
        prefix.append(char)
        prefix_bytes += char_bytes
    return "".join(prefix) + marker, True


def _truncate_utf8_with_marker(value: str, max_bytes: int, marker: str) -> Optional[str]:
    """Truncate UTF-8 text to a byte limit while retaining a visible marker."""
    marker_bytes = marker.encode("utf-8")
    if max_bytes < len(marker_bytes):
        return None
    prefix_budget = max_bytes - len(marker_bytes)
    prefix = []
    prefix_bytes = 0
    for char in value:
        char_bytes = len(char.encode("utf-8"))
        if prefix_bytes + char_bytes > prefix_budget:
            break
        prefix.append(char)
        prefix_bytes += char_bytes
    return "".join(prefix) + marker


def _bounded_command(command: str) -> tuple[str, bool]:
    """Bound command metadata included alongside an execution summary."""
    return _bounded_summary_text(
        command,
        SUMMARY_COMMAND_MAX_BYTES,
        SUMMARY_COMMAND_TRUNCATION_MARKER,
    )


def _summary_payload(summary: Dict[str, Any], include_previews: bool = False) -> Dict[str, Any]:
    """Project an engine summary into the compact, versioned MCP schema."""
    if summary.get("status") == "error":
        return dict(summary)

    execution_status = summary.get("execution_status")
    if execution_status is None:
        execution_status = "captured" if summary.get("command_exit_code") is None else (
            "success" if summary.get("command_exit_code") == 0 else "failed"
        )
    bounded_label, label_truncated = _bounded_summary_text(
        summary["label"],
        SUMMARY_LABEL_MAX_BYTES,
        SUMMARY_LABEL_TRUNCATION_MARKER,
    )
    payload: Dict[str, Any] = {
        "schema_version": summary.get("schema_version", 1),
        "capture_id": summary["capture_id"],
        "label": bounded_label,
        "label_truncated": label_truncated,
        "source": summary.get("source", "capture"),
        "status": execution_status,
        "partial": summary.get("partial", summary.get("timed_out", False)),
        "content_type": summary.get("content_type", "text"),
        "timestamp": summary.get("timestamp"),
        "duration_ms": summary.get("duration_ms"),
        "exit_code": summary.get("command_exit_code"),
        "timed_out": summary.get("timed_out", False),
        "total_lines": summary.get("total_lines", 0),
        "byte_size": summary.get("byte_size", 0),
        "original_byte_size": summary.get("original_byte_size"),
        "estimated_tokens": summary.get("estimated_tokens"),
        "original_estimated_tokens": summary.get("original_estimated_tokens"),
        "truncated": summary.get("truncated", False),
        "signals": summary.get("keyword_signals", {}),
        "errors": summary.get("errors", []),
        "warnings": summary.get("warnings", []),
        "structured_metrics": summary.get("structured_metrics", {}),
        "retrieval": {
            "search": f"search_capture(query='...', capture_id='{summary['capture_id']}')",
            "slice": f"get_capture_slice(start_line=..., end_line=..., capture_id='{summary['capture_id']}')",
        },
    }
    if summary.get("content_type") == "diff":
        bounded_file_map, omitted_files = _bounded_diff_file_map(
            summary.get("file_map", ""),
            summary.get("diff_meta"),
        )
        payload["diff"] = {
            "stats": summary.get("diff_stats", ""),
            "file_map": bounded_file_map,
            "file_map_truncated": omitted_files > 0,
            "omitted_file_count": omitted_files,
        }
    if include_previews:
        payload["previews"] = {
            "head": summary.get("head_preview", ""),
            "tail": summary.get("tail_preview", ""),
        }
    return payload


def _summary_json(
    capture_id: str,
    include_previews: bool = False,
    capture: Optional[Any] = None,
) -> str:
    """Return a compact JSON summary suitable for an agent response."""
    if capture is None:
        summary = _active_engine().get_summary(capture_id, include_previews=include_previews)
    else:
        summary = _active_engine().get_summary_for_capture(capture, include_previews=include_previews)
    if summary.get("status") == "error":
        message, _ = _bounded_summary_text(
            str(summary.get("message", "capture summary unavailable")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error: {message}"
    return json.dumps(
        _summary_payload(summary, include_previews=include_previews),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _capture_text(
    content: str,
    label: str = "",
    content_type: str = "auto",
    source: str = "capture_text",
    duration_ms: Optional[float] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ingests raw text output directly into the ephemeral search index.
    Automatically detects diffs, logs, and text structures.
    Returns a compact JSON summary while preserving the complete capture for later retrieval.
    """
    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"
    cap = _active_engine().ingest(
        content,
        label=label,
        content_type=content_type,
        source=source,
        duration_ms=duration_ms,
        structured_metrics=structured_metrics,
    )
    return _summary_json(cap.capture_id, capture=cap)


@_mcp_tool("capture_text", "capture")
@_instrument_tool("capture_text")
def capture_text(
    content: str,
    label: str = "",
    content_type: str = "auto",
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Ingest already-collected text and return capture metadata.

    Use this when the caller already has output to index. For a noisy or
    potentially long command, use ``execute_and_capture`` so output remains
    bounded before it reaches the agent context. The return value is a compact
    versioned JSON summary; optional named metrics are retained in
    ``structured_metrics``.
    """
    return _capture_text(
        content,
        label=label,
        content_type=content_type,
        structured_metrics=structured_metrics,
    )


@_mcp_tool("capture_file", "capture")
@_instrument_tool("capture_file")
def capture_file(
    file_path: str,
    label: str = "",
    content_type: str = "auto",
    max_bytes: Optional[int] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Reads a file or log output from disk and ingests it into the ephemeral
    search index. Returns a compact versioned JSON summary and preserves
    optional named metrics in ``structured_metrics``.

    Validate the intended file path before calling: resolve symlinks when path
    identity matters, confirm the file belongs to the expected workspace, and
    use an explicit bounded ``max_bytes`` for large or untrusted files. Capture
    limits control output handling; they do not validate filesystem intent.
    """
    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        active_engine = _active_engine()
        if max_bytes is not None and max_bytes > active_engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "capture_file_limit_rejected",
                requested_bytes=max_bytes,
                max_buffer_bytes=active_engine.max_buffer_bytes,
            )
            return (
                f"Error: max_bytes ({max_bytes:,}) exceeds the configured "
                f"buffer limit ({active_engine.max_buffer_bytes:,})."
            )
        read_limit = active_engine.max_buffer_bytes if max_bytes is None else max_bytes
        if read_limit < 1:
            return "Error: max_bytes must be at least 1."
        if not stat.S_ISREG(os.stat(file_path).st_mode):
            return f"Error: File '{file_path}' is not a regular file."
        content = read_file_bounded(file_path, read_limit)
        if not label:
            label = os.path.basename(file_path)
        return _capture_text(
            content,
            label=label,
            content_type=content_type,
            source="file",
            structured_metrics=structured_metrics,
        )
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@_mcp_tool("execute_and_capture", "capture")
@_instrument_tool("execute_and_capture")
def execute_and_capture(
    command: str,
    cwd: Optional[str] = None,
    label: str = "",
    content_type: str = "auto",
    max_output_bytes: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    structured_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Runs a shell command, captures stdout/stderr, indexes it, and returns a
    compact versioned JSON summary without flooding the prompt context with
    thousands of lines. The summary includes status, duration, sizes,
    approximate token counts, truncation, signals, and optional named metrics.

    Use this for noisy tests, builds, logs, and other output that benefits from
    bounded capture and later search. Direct command execution is usually
    faster for a small, targeted inspection; use capture once output may be
    noisy, large, or uncertain. This is an advisory routing heuristic, not an
    enforced threshold. Before running, verify the
    command, intended repository, and working directory: an omitted ``cwd``
    inherits the server process directory, and symlinks or shell expansion can
    target a different path than expected. This tool bounds output but does
    not validate command intent, path identity, or filesystem safety.
    
    Args:
        command: Shell command line to execute.
        cwd: Optional working directory for command execution; pass an explicit
            validated path for repository-sensitive commands.
        label: Optional human-readable description/label for this capture.
        content_type: Content type hint - 'auto' (default, detects diff/log/text), 'diff', 'log', or 'text'.
        max_output_bytes: Maximum command output retained (default: configured buffer byte limit).
        timeout_seconds: Optional maximum runtime; timed-out commands return exit code 124.
    """
    if not label:
        label = command[:40] + ("..." if len(command) > 40 else "")

    try:
        structured_metrics = normalize_structured_metrics(structured_metrics)
    except ValueError as exc:
        return f"Error: invalid structured_metrics: {exc}"

    try:
        active_engine = _active_engine()
        if max_output_bytes is not None and max_output_bytes > active_engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "command_output_limit_rejected",
                requested_bytes=max_output_bytes,
                max_buffer_bytes=active_engine.max_buffer_bytes,
            )
            return (
                f"Error: max_output_bytes ({max_output_bytes:,}) exceeds the configured "
                f"buffer limit ({active_engine.max_buffer_bytes:,})."
            )
        output_limit = active_engine.max_buffer_bytes if max_output_bytes is None else max_output_bytes
        command_started = time.perf_counter()
        command_result = run_command_bounded(
            command,
            cwd,
            output_limit,
            timeout_seconds,
            cancellation_event=_REQUEST_CANCEL_EVENT.get(),
        )
        output, exit_code, truncated, original_byte_size, timed_out = command_result
        duration_ms = round((time.perf_counter() - command_started) * 1000, 3)
        
        cap = active_engine.ingest(
            output,
            label=f"cmd: {label}",
            content_type=content_type,
            truncated=truncated,
            original_byte_size=original_byte_size if truncated else None,
            command_exit_code=exit_code,
            timed_out=timed_out,
            source="command",
            duration_ms=duration_ms,
            structured_metrics=structured_metrics,
        )
        payload = _summary_payload(
            active_engine.get_summary_for_capture(cap, include_previews=False)
        )
        bounded_command, command_truncated = _bounded_command(command)
        payload["command"] = bounded_command
        payload["command_truncated"] = command_truncated
        payload["command_cancelled"] = bool(getattr(command_result, "cancelled", False))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception as e:
        log_event(LOGGER, logging.ERROR, "command_execution_failed", error_type=type(e).__name__)
        return f"Error executing command: {str(e)}"


def _consolidated_jsonl(
    capture_ids: Optional[List[str]],
    max_captures: int,
    max_bytes: int,
) -> Dict[str, Any]:
    """Compatibility wrapper around the engine's consolidation implementation."""
    return _active_engine().consolidate(capture_ids, max_captures, max_bytes)


@_mcp_tool("consolidate_captures", "capture")
@_instrument_tool("consolidate_captures")
def consolidate_captures(
    capture_ids: Optional[List[str]] = None,
    label: str = "consolidated captures",
    max_captures: int = 25,
    max_bytes: Optional[int] = None,
) -> str:
    """Create one bounded, searchable JSON capture from multiple captures.

    The consolidated capture keeps source capture IDs and source line numbers.
    If records do not fit, the complete source captures remain available through
    their original IDs until normal LRU eviction. The operation fails if the
    consolidated capture cannot be admitted while retaining those sources.
    """
    try:
        if max_captures > CONSOLIDATION_MAX_CAPTURES:
            return (
                f"Error consolidating captures: max_captures must be at most "
                f"{CONSOLIDATION_MAX_CAPTURES}"
            )
        if capture_ids is not None:
            if not isinstance(capture_ids, list):
                return "Error consolidating captures: capture_ids must be a list or null"
            if len(capture_ids) > CONSOLIDATION_MAX_CAPTURES:
                return (
                    f"Error consolidating captures: at most "
                    f"{CONSOLIDATION_MAX_CAPTURES} capture IDs may be requested"
                )
            for capture_id in capture_ids:
                if not isinstance(capture_id, str):
                    return "Error consolidating captures: capture_ids must contain strings"
                if len(capture_id.encode("utf-8")) > CONSOLIDATION_CAPTURE_ID_MAX_BYTES:
                    return "Error consolidating captures: capture ID exceeds the 256-byte limit"
        active_engine = _active_engine()
        result = active_engine.consolidate(capture_ids, max_captures, max_bytes)
        capture = active_engine.ingest(
            result["content"],
            label=label,
            content_type="text",
            source="consolidated",
            protected_capture_ids=result["source_capture_ids"],
        )
        payload = _summary_payload(
            active_engine.get_summary_for_capture(capture, include_previews=False)
        )
        payload.update({
            "source_capture_ids": result["source_capture_ids"],
            "requested_capture_count": result["requested_capture_count"],
            "selected_capture_count": result["selected_capture_count"],
            "source_count": result["source_count"],
            "record_count": result["record_count"],
            "omitted_record_count": result["omitted_record_count"],
            "missing_capture_ids": result["missing_capture_ids"],
            "next_steps": {
                "search": f"search_capture(query='...', capture_id='{capture.capture_id}')",
                "slice": f"get_capture_slice(start_line=..., end_line=..., capture_id='{capture.capture_id}')",
                "source_detail": "Use the original source capture IDs for complete omitted records.",
            },
        })
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        log_event(LOGGER, logging.ERROR, "capture_consolidation_failed", error_type=type(exc).__name__)
        message, _ = _bounded_summary_text(
            str(exc),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        return f"Error consolidating captures: {message}"


def _render_search_text(
    result: Dict[str, Any],
    matches: List[Dict[str, Any]],
    *,
    omitted: bool,
    query: Optional[str] = None,
) -> str:
    query = str(result.get("query", "") if query is None else query)
    display_query, _ = _bounded_summary_text(
        query,
        SEARCH_QUERY_MAX_BYTES,
        SEARCH_QUERY_TRUNCATION_MARKER,
    )
    label, _ = _bounded_summary_text(
        str(result.get("label", "")),
        SUMMARY_LABEL_MAX_BYTES,
        SUMMARY_LABEL_TRUNCATION_MARKER,
    )
    fallback_note = (
        f"Semantic fallback active ({result['semantic_fallback']})."
        if result.get("semantic_fallback")
        else None
    )
    pending_note = (
        "Semantic index still building; results are lexical (BM25) only. Repeat the search for hybrid ranking."
        if result.get("semantic_coverage") == "pending"
        else None
    )
    if not matches:
        if omitted:
            return (
                "Matches were found but omitted to stay within the structured search response budget. "
                "Use search_capture with a smaller top_k or get_capture_slice for full content."
            )
        return (
            f"No matches found for '{display_query}' in capture '{result.get('capture_id')}' ({label})."
            + (f" {fallback_note}" if fallback_note else "")
            + (f" {pending_note}" if pending_note else "")
        )

    mode_label = str(result.get("mode", "unknown"))
    if result.get("semantic_fallback"):
        mode_label += f"; lexical fallback ({result['semantic_fallback']})"
    elif result.get("semantic_coverage") == "pending":
        mode_label += "; semantic pending (lexical only)"
    out = [
        f'Search Results for: "{display_query}" [Mode: {mode_label}]',
        f"Capture: `{result.get('capture_id')}` ({label}, {result.get('total_lines', 0)} total lines)",
        f"Found {len(matches)} relevant section(s):\n",
    ]
    if fallback_note:
        out.insert(1, fallback_note)
    elif pending_note:
        out.insert(1, pending_note)

    for index, match in enumerate(matches, start=1):
        out.extend([
            f"### Match #{index} (Score: {match.get('score', 0)}, Range: {match.get('matched_range', 'unknown')}, Context: {match.get('context_range', 'unknown')})",
            "```text",
            str(match.get("snippet", "")),
            "```\n",
        ])
    if omitted:
        out.append(
            "Additional matches omitted because the structured search response reached "
            f"its {SEARCH_RESPONSE_MAX_BYTES:,}-byte budget. Use get_capture_slice for full content."
        )
    return "\n".join(out)


@_instrument_tool("search_capture")
def _search_capture_response(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3,
) -> ToolResponseEnvelope:
    if not isinstance(query, str):
        message = "Search Error: query must be a string"
        return ToolResponseEnvelope(
            status="error", error=ToolErrorEnvelope(code="invalid_query", message=message), text=message
        )
    if len(query.encode("utf-8")) > SEARCH_QUERY_MAX_BYTES:
        message = f"Search Error: query exceeds the {SEARCH_QUERY_MAX_BYTES:,}-byte limit"
        return ToolResponseEnvelope(
            status="error", error=ToolErrorEnvelope(code="query_too_large", message=message), text=message
        )
    response_budget = MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
    res = _active_engine().search(
        query=query,
        mode=mode,
        capture_id=capture_id,
        top_k=top_k,
        context_lines=context_lines,
        response_budget_bytes=response_budget,
    )
    if res.get("status") == "error":
        message, _ = _bounded_summary_text(
            str(res.get("message", "search failed")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        text = f"Search Error: {message}"
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(
                code=str(res.get("error_code") or "search_failed"),
                message=message,
            ),
            text=text,
        )

    engine_matches = res.get("matches", [])
    construction_omitted = bool(res.get("response_matches_omitted", False))
    _active_metrics().record_result_count("search_capture", len(engine_matches))
    metadata = {
        key: res[key]
        for key in (
            "capture_id", "label", "total_lines", "mode", "query", "semantic_coverage",
            "semantic_fallback", "semantic_index_state", "semantic_wait_seconds", "message",
        )
        if key in res
    }
    metadata.setdefault("query", query)
    text_result = dict(res)
    text_result.setdefault("query", query)
    structured_matches = []
    for match in engine_matches:
        snippet, snippet_truncated = _bounded_summary_text(
            str(match.get("snippet", "")),
            SEARCH_STRUCTURED_MATCH_MAX_BYTES,
            SEARCH_SNIPPET_TRUNCATION_MARKER,
        )
        context, context_truncated = _bounded_summary_text(
            str(match.get("context", "")),
            SEARCH_STRUCTURED_CONTEXT_MAX_BYTES,
            "... [raw context truncated; use get_capture_slice for full content] ...",
        )
        structured_matches.append({
            key: match[key]
            for key in (
                "chunk_id", "chunk_index", "score", "matched_range", "context_range",
                "context_start_line", "context_end_line",
            )
            if key in match
        } | {
            "snippet": snippet,
            "snippet_truncated": snippet_truncated or bool(match.get("snippet_truncated")),
            "context": context,
            "context_truncated": context_truncated or bool(match.get("context_truncated")),
        })

    selected: List[Dict[str, Any]] = []
    best: Optional[ToolResponseEnvelope] = None
    for index, match in enumerate(structured_matches):
        candidate_matches = selected + [match]
        omitted = construction_omitted or index < len(structured_matches) - 1
        data = {
            **metadata,
            "matches": candidate_matches,
            "match_count": len(candidate_matches),
            "available_match_count": len(structured_matches),
        }
        candidate = ToolResponseEnvelope(
            status="ok",
            data=data,
            text=_render_search_text(text_result, candidate_matches, omitted=omitted, query=query),
            truncated=omitted,
        )
        if _call_tool_result_bytes(candidate) > response_budget:
            break
        selected.append(match)
        best = candidate

    if best is None and engine_matches:
        # The normal per-match caps leave room for the first match, but retain a
        # stable bounded response if a future metadata field changes that.
        data = {**metadata, "matches": [], "match_count": 0, "available_match_count": len(engine_matches)}
        text = _render_search_text(text_result, [], omitted=True, query=query)
        best = ToolResponseEnvelope(status="ok", data=data, text=text, truncated=True)
    if best is None:
        text = _render_search_text(
            text_result,
            [],
            omitted=construction_omitted,
            query=query,
        )
        best = ToolResponseEnvelope(
            status="ok",
            data={**metadata, "matches": [], "match_count": 0},
            text=text,
            truncated=construction_omitted,
        )
    return best


@_mcp_tool("search_capture", "search", structured_result_factory=_search_capture_response)
def search_capture(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3,
) -> str:
    """Search a capture and return the compatibility text from a typed result."""
    return search_capture_result(query, mode, capture_id, top_k, context_lines).text or ""


def search_capture_result(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3,
) -> ToolResponseEnvelope:
    """Return a typed search result while preserving ``search_capture`` text callers."""
    return _search_capture_response(query, mode, capture_id, top_k, context_lines)


@_instrument_tool("get_capture_slice")
def _capture_slice_response(
    start_line: int,
    end_line: int,
    capture_id: str = "latest",
    max_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
    cursor: Optional[str] = None,
) -> ToolResponseEnvelope:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 4096 <= max_bytes <= MCP_TOOL_RESPONSE_MAX_BYTES:
        message = f"max_bytes must be between 4,096 and {MCP_TOOL_RESPONSE_MAX_BYTES:,}."
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(code="invalid_byte_budget", message=message),
            text=f"Error: {message}",
        )

    response_budget = min(max_bytes, MCP_TOOL_RESPONSE_MAX_BYTES) - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
    active_engine = _active_engine()
    capture = active_engine.get_capture(capture_id)
    label_wire_budget = min(
        SUMMARY_LABEL_MAX_BYTES,
        max(128, response_budget // 12),
    )
    label_bytes = 0
    if capture is not None:
        safe_label, _ = _bounded_json_text(
            str(capture.label),
            label_wire_budget,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        label_bytes = len(json.dumps(safe_label, ensure_ascii=False).encode("utf-8")) - 2
    # The wire result repeats returned text in both channels. Limit individual
    # segments conservatively so even JSON-escaped control characters can fit
    # in the smallest accepted response budget.
    max_segment_bytes = max(
        4,
        min(
            CAPTURE_SLICE_SEGMENT_MAX_BYTES,
            max(4, response_budget - 1536 - (2 * label_bytes)) // 12,
        ),
    )
    page = active_engine.get_slice_page(
        start_line,
        end_line,
        capture_id=capture_id,
        max_content_bytes=min(MAX_CAPTURE_SLICE_CONTENT_BYTES, max_bytes // 2),
        cursor=cursor,
        max_segment_bytes=max_segment_bytes,
    )
    if page.get("status") != "ok":
        message = str(page.get("message", "capture slice unavailable"))
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(
                code=str(page.get("error_code") or "retrieval_failed"),
                message=message,
            ),
            text=f"Error: {message}",
        )

    selected: List[Dict[str, Any]] = []
    best: Optional[ToolResponseEnvelope] = None
    segments = page["segments"]
    for index, segment in enumerate(segments):
        candidate_segments = selected + [segment]
        candidate_cursor = segment["cursor_after"]
        content = "".join(item["content_piece"] for item in candidate_segments)
        rendered_segments = []
        current_line = None
        current_text: List[str] = []
        current_complete = False
        for item in candidate_segments:
            if current_line is not None and item["line"] != current_line:
                suffix = "" if current_complete else " [continued]"
                rendered_segments.append(f"  {current_line:5d} | {''.join(current_text)}{suffix}")
                current_text = []
            current_line = item["line"]
            current_text.append(item["text"])
            current_complete = item["line_complete"]
        if current_line is not None:
            suffix = "" if current_complete else " [continued]"
            rendered_segments.append(f"  {current_line:5d} | {''.join(current_text)}{suffix}")
        is_truncated = candidate_cursor is not None
        display_content = "\n".join(rendered_segments)
        if is_truncated:
            display_content += (
                "\n... [slice truncated; continue with next_cursor='"
                + str(candidate_cursor)
                + "']"
            )
        label, _ = _bounded_json_text(
            str(page["label"]),
            label_wire_budget,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        text = (
            f"Capture: `{page['capture_id']}` ({label}) | Lines {page['start_line']} to "
            f"{segment['line']} of {page['total_lines']}\n```text\n{display_content}\n```"
        )
        data = {
            "capture_id": page["capture_id"],
            "label": label,
            "requested_start_line": page["requested_start_line"],
            "requested_end_line": page["requested_end_line"],
            "start_line": candidate_segments[0]["line"],
            "end_line": segment["line"],
            "total_lines": page["total_lines"],
            "content": content,
            "content_bytes": len(content.encode("utf-8")),
            "segments": [
                {
                    key: item[key]
                    for key in ("line", "byte_offset", "byte_length", "line_complete", "separator_after")
                }
                for item in candidate_segments
            ],
        }
        candidate = ToolResponseEnvelope(
            status="ok",
            data=data,
            text=text,
            truncated=is_truncated,
            next_cursor=candidate_cursor,
        )
        if _call_tool_result_bytes(candidate) > response_budget:
            break
        selected.append(segment)
        best = candidate

    if best is None:
        message = "The response byte budget is too small to include capture content and its continuation metadata."
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(code="response_budget_too_small", message=message),
            text=f"Error: {message}",
        )
    return best


@_mcp_tool("get_capture_slice", "retrieval", structured_result_factory=_capture_slice_response)
def get_capture_slice(
    start_line: int,
    end_line: int,
    capture_id: str = "latest",
    max_bytes: Annotated[int, Field(ge=4096, le=MCP_TOOL_RESPONSE_MAX_BYTES)] = MCP_TOOL_RESPONSE_MAX_BYTES,
    cursor: Optional[str] = None,
) -> str:
    """Fetch a byte-bounded page of 1-indexed capture lines.

    Repeat the range and capture ID from the first call, then pass the returned
    opaque ``next_cursor`` to continue. Pages never split a Unicode character;
    the cursor's line offset is a zero-based UTF-8 byte offset.
    """
    return get_capture_slice_result(start_line, end_line, capture_id, max_bytes, cursor).text or ""


def get_capture_slice_result(
    start_line: int,
    end_line: int,
    capture_id: str = "latest",
    max_bytes: int = MAX_CAPTURE_SLICE_CONTENT_BYTES,
    cursor: Optional[str] = None,
) -> ToolResponseEnvelope:
    """Return a typed paged retrieval result for Python callers."""
    return _capture_slice_response(start_line, end_line, capture_id, max_bytes, cursor)


@_mcp_tool("get_capture_summary", "retrieval")
@_instrument_tool("get_capture_summary")
def get_capture_summary(capture_id: str = "latest", include_previews: bool = False) -> str:
    """
    Returns a compact versioned JSON summary. Set ``include_previews`` when
    head and tail samples are needed; full output remains available through
    ``get_capture_slice``.
    """
    return _summary_json(capture_id, include_previews=include_previews)


@_instrument_tool("list_captures")
def _list_captures_response() -> ToolResponseEnvelope:
    captures = []
    text_lines = ["Active Captures in Ephemeral Buffer:"]
    for capture in _active_engine().list_captures():
        label, label_truncated = _bounded_summary_text(
            capture["label"],
            SUMMARY_LABEL_MAX_BYTES,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
        captures.append({
            "capture_id": capture["capture_id"],
            "label": label,
            "label_truncated": label_truncated,
            "total_lines": capture["total_lines"],
            "byte_size": capture["byte_size"],
            "timestamp": capture["timestamp"],
        })
        text_lines.append(
            f"- `{capture['capture_id']}`: \"{label}\" | {capture['total_lines']:,} lines | "
            f"{capture['byte_size']:,} bytes | {capture['timestamp']}"
        )
    if not captures:
        text = "Ephemeral buffer is empty. No captures currently stored."
    else:
        text = "\n".join(text_lines)
    return ToolResponseEnvelope(
        status="ok",
        data={"captures": captures, "capture_count": len(captures)},
        text=text,
    )


@_mcp_tool("list_captures", "retrieval", structured_result_factory=_list_captures_response)
def list_captures() -> str:
    """List active captures and return the compatibility text rendering."""
    return list_captures_result().text or ""


def list_captures_result() -> ToolResponseEnvelope:
    """Return a typed list of active captures for Python callers."""
    return _list_captures_response()


@_mcp_tool("clear_captures", "lifecycle")
@_instrument_tool("clear_captures")
def clear_captures(capture_id: str = "all") -> str:
    """
    Clears all or a specific capture from the ephemeral buffer to free memory.
    """
    return _active_engine().clear(capture_id)


@_mcp_tool("get_buffer_stats", "diagnostics")
@_instrument_tool("get_buffer_stats")
def get_buffer_stats() -> str:
    """Return aggregate capture, admission, accounting, and process RSS metrics."""
    stats = _active_engine().get_buffer_stats()
    rss = stats["process_rss_bytes"]
    unaccounted = stats["unaccounted_rss_bytes"]
    rss_line = "Process RSS: unavailable" if rss is None else f"Process RSS: {rss:,} bytes"
    unaccounted_line = (
        "Unaccounted RSS bytes: unavailable"
        if unaccounted is None
        else f"Unaccounted RSS bytes: {unaccounted:,}"
    )
    model_state = "loaded" if stats["embedding_model_loaded"] else "not loaded"
    model_line = f"Embedding model: {stats['embedding_model']} ({model_state})"
    warmup_state = stats.get("embedding_warmup_state", "not-started")
    warmup_failure = stats.get("embedding_warmup_failure")
    warmup_line = f"Embedding warm-up: {warmup_state}"
    if warmup_failure:
        warmup_line += f" ({warmup_failure})"
    cache_line = f"Embedding cache: {stats['embedding_cache_dir'] or 'default'}"
    indexed_chunks = stats.get("indexed_chunks", stats["total_chunks"])
    max_indexed_chunks = stats.get("max_indexed_chunks", indexed_chunks)
    remaining_indexed_chunks = stats.get(
        "remaining_indexed_chunks", max_indexed_chunks - indexed_chunks
    )
    admission_active_by_type = json.dumps(
        stats.get("admission_active_by_type", {}), sort_keys=True
    )
    admission_queued_by_type = json.dumps(
        stats.get("admission_queued_by_type", {}), sort_keys=True
    )
    admission_rejected_by_type = json.dumps(
        stats.get("admission_rejected_by_type", {}), sort_keys=True
    )
    result = (
        f"Captures: {stats['capture_count']}/{stats['max_captures']}\n"
        f"Content bytes: {stats['total_bytes']:,}/{stats['max_buffer_bytes']:,}\n"
        f"Foreground work: {stats.get('admission_active', 0)} active, "
        f"{stats.get('admission_queued', 0)} queued, "
        f"{stats.get('admission_rejected', 0)} rejected\n"
        f"Tool capacity: {stats.get('admission_max_active_tool_work', 0)} active, "
        f"{stats.get('admission_max_queued_tool_work', 0)} queued; reserved stats slot: "
        f"{stats.get('admission_max_active_diagnostic_work', 0)}\n"
        f"Socket capacity: {stats.get('admission_max_active_socket_clients', 0)} active, "
        f"{stats.get('admission_max_queued_socket_clients', 0)} queued\n"
        f"Foreground work by type: active={admission_active_by_type}, "
        f"queued={admission_queued_by_type}, rejected={admission_rejected_by_type}\n"
        f"Deferred storage: {stats.get('deferred_storage_capture_count', 0)} captures, "
        f"{stats.get('deferred_storage_readers', 0)} readers, "
        f"{stats.get('deferred_storage_bytes', 0):,} accounted bytes\n"
        f"Lines: {stats['total_lines']:,}\n"
        f"Chunks: {stats['total_chunks']:,}\n"
        f"Indexed chunks: {indexed_chunks:,}/{max_indexed_chunks:,} "
        f"({remaining_indexed_chunks:,} remaining)\n"
        f"Index budget adjustment: {json.dumps(stats.get('last_index_budget_adjustment', {}), sort_keys=True)}\n"
        f"{model_line}\n"
        f"{warmup_line}\n"
        f"{cache_line}\n"
        f"Embedding bytes: {stats['embedding_bytes']:,}\n"
        f"Semantic prefetch: {'enabled' if stats.get('semantic_prefetch_enabled', False) else 'disabled'} "
        f"({stats.get('semantic_prefetch_pending', 0)} pending, {stats.get('semantic_prefetch_failed', 0)} failed, "
        f"{stats.get('semantic_index_budget_exceeded', 0)} over budget)\n"
        f"Semantic memory limits: {stats['semantic_max_index_input_bytes']:,} input bytes/capture, "
        f"batch {stats['embedding_batch_size']}, {stats['embedding_max_batch_tokens']:,} padded token slots, "
        f"CPU arena {'enabled' if stats['embedding_cpu_mem_arena_enabled'] else 'disabled'}\n"
        f"Semantic wait budget: {stats.get('semantic_wait_seconds', 0.0):g}s "
        f"({stats.get('semantic_index_on_demand_running', 0)} on-demand index jobs running, "
        f"{stats.get('semantic_index_on_demand_queued', 0)} queued)\n"
        f"Accounted bytes: {stats['accounted_bytes']:,}\n"
        f"{rss_line}\n"
        f"{unaccounted_line}"
    )
    if _active_metrics().enabled:
        snapshot = _metrics_snapshot()
        result += f"\nData-path bytes: {json.dumps(snapshot['bytes'], sort_keys=True)}"
        result += f"\nLocal metrics: {json.dumps(snapshot, sort_keys=True)}"
    return result


@_mcp_tool("get_runtime_diagnostics", "diagnostics")
@_instrument_tool("get_runtime_diagnostics")
def get_runtime_diagnostics() -> str:
    """Returns opt-in runtime metadata without exposing captured content."""
    stats = _active_engine().get_buffer_stats()
    installed_version = _runtime_package_version()
    settings = _active_settings()
    identity = settings.identity
    socket_mode = {
        "environment:EPHEMERAL_SOCKET_PATH": "explicit path",
        "session:EPHEMERAL_SESSION_ID": "session-derived path",
        "default:shared legacy socket": "shared default path",
    }[identity.socket_source]

    uptime_seconds = max(0, int(time.time() - SERVER_STARTED_AT))
    socket_state, socket_failure = _socket_lifecycle()
    file_handler = next(
        (handler for handler in LOGGER.handlers if isinstance(handler, logging.FileHandler)),
        None,
    )
    configured_log_file = os.environ.get("EPHEMERAL_LOG_FILE", "").strip()
    if file_handler is not None:
        log_file = file_handler.baseFilename
    elif configured_log_file:
        log_file = f"unavailable (configured path: {configured_log_file})"
    else:
        log_file = "not configured"
    log_level = logging.getLevelName(LOGGER.getEffectiveLevel())
    rss = stats["process_rss_bytes"]
    unaccounted = stats["unaccounted_rss_bytes"]
    lines = [
        "Runtime diagnostics (content-free):",
        f"Package version: {installed_version}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Uptime: {uptime_seconds:,} seconds",
        f"Socket mode: {socket_mode}",
        f"Socket path: {identity.socket_path}",
        f"Socket identity source: {identity.socket_source}",
        f"Socket lifecycle: {socket_state}",
        *( [f"Socket failure: {socket_failure}"] if socket_failure else [] ),
        f"Log file: {log_file}",
        f"Log level: {log_level}",
        f"Session ID configured: {'yes' if identity.session_id is not None else 'no'}",
        f"Session ID source: {identity.session_source}",
        f"Session fingerprint: {identity.session_fingerprint or 'not configured'}",
        f"Execution state directory: {identity.state_dir}",
        f"Execution state source: {identity.state_source}",
        f"Execution state durability: {identity.durability}",
        *(
            [
                "Existing state namespace needs explicit continuation: "
                f"{identity.legacy_state_dir}; set EPHEMERAL_EXECUTION_STATE_DIR to continue."
            ]
            if identity.legacy_state_transition
            else []
        ),
        f"Startup settings snapshot: {json.dumps(settings.diagnostics(), sort_keys=True)}",
        f"Captures: {stats['capture_count']}/{stats['max_captures']}",
        f"Content bytes: {stats['total_bytes']:,}/{stats['max_buffer_bytes']:,}",
        f"Embedding model: {stats['embedding_model']} ({'loaded' if stats['embedding_model_loaded'] else 'not loaded'})",
        f"Embedding warm-up: {stats.get('embedding_warmup_state', 'not-started')}"
        + (f" ({stats['embedding_warmup_failure']})" if stats.get("embedding_warmup_failure") else ""),
        f"Lexical search backend: {stats['lexical_backend']}",
        f"Embedding cache: {stats['embedding_cache_dir'] or 'default'}",
        f"Semantic index budget adjustment: {json.dumps(stats.get('last_index_budget_adjustment', {}), sort_keys=True)}",
        f"Semantic prefetch: {'enabled' if stats.get('semantic_prefetch_enabled', False) else 'disabled'} "
        f"({stats.get('semantic_prefetch_pending', 0)} pending, {stats.get('semantic_prefetch_failed', 0)} failed, "
        f"{stats.get('semantic_index_budget_exceeded', 0)} over budget)",
        f"Semantic memory limits: {stats['semantic_max_index_input_bytes']:,} input bytes/capture, "
        f"batch {stats['embedding_batch_size']}, {stats['embedding_max_batch_tokens']:,} padded token slots, "
        f"CPU arena {'enabled' if stats['embedding_cpu_mem_arena_enabled'] else 'disabled'}",
        f"Process RSS: {'unavailable' if rss is None else f'{rss:,} bytes'}",
        f"Unaccounted RSS: {'unavailable' if unaccounted is None else f'{unaccounted:,} bytes'}",
        f"Local metrics: {'enabled' if _active_metrics().enabled else 'disabled'}",
        "Captured content, labels, and command arguments are not included.",
    ]
    fastmcp_status = _fastmcp_compatibility_diagnostics()
    limitations = []
    if fastmcp_status["tools_without_validation_metrics"]:
        limitations.append(
            "validation metrics unavailable for "
            + ", ".join(fastmcp_status["tools_without_validation_metrics"])
        )
    if fastmcp_status["validation_observation_failures"]:
        limitations.append(
            "validation metrics failed during observation for "
            + ", ".join(fastmcp_status["validation_observation_failures"])
        )
    if fastmcp_status["instruction_refresh"] == "unavailable":
        limitations.append("runtime instruction refresh unavailable")
    if limitations:
        lines.append(
            f"FastMCP {fastmcp_status['sdk_version']} compatibility limitation: "
            + "; ".join(limitations)
            + ". Tool dispatch remains available."
        )
    if _active_metrics().enabled:
        snapshot = _metrics_snapshot()
        lines.append(f"Data-path bytes: {json.dumps(snapshot['bytes'], sort_keys=True)}")
        lines.append(f"Metrics summary: {json.dumps(snapshot, sort_keys=True)}")
    return "\n".join(lines)


@_mcp_tool("get_usage_metrics", "diagnostics")
@_instrument_tool("get_usage_metrics")
def get_usage_metrics(since: Optional[str] = None) -> str:
    """Return content-free usage metrics, optionally since a snapshot token."""
    return json.dumps(
        {
            "schema_version": 2,
            **_metrics_snapshot(
                since_snapshot=since,
                include_snapshot_token=True,
            ),
        },
        sort_keys=True,
    )


@_mcp_tool("set_semantic_index_budget", "configuration")
@_instrument_tool("set_semantic_index_budget")
def set_semantic_index_budget(max_indexed_chunks: int) -> str:
    """Adjust this session's semantic-index chunk budget when explicitly enabled."""
    if not runtime_index_budget_adjustment_enabled():
        return (
            "Error: runtime semantic-index budget adjustment is disabled; set "
            "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1 at server startup to enable it."
        )
    try:
        return json.dumps(_active_engine().set_max_indexed_chunks(max_indexed_chunks), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return f"Error adjusting semantic-index budget: {exc}"


# --- Unix Domain Socket IPC for CLI piping (ephbuf) ---

async def _read_exact(
    reader: asyncio.StreamReader,
    size: int,
    byte_counter: Optional[str] = None,
) -> bytes:
    """Read exactly ``size`` bytes, tolerating fragmented stream reads."""
    chunks = []
    remaining = size
    while remaining:
        chunk = await reader.read(remaining)
        if not chunk:
            raise ValueError("truncated socket frame")
        if byte_counter is not None:
            METRICS.record_bytes(byte_counter, len(chunk))
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def _read_socket_payload(reader: asyncio.StreamReader, read_limit: int) -> bytes:
    """Read one versioned length-prefixed request within the payload limit."""
    header = await _read_exact(reader, FRAME_HEADER_SIZE, "socket_request_bytes")
    payload_length = decode_header(header)
    if payload_length > read_limit:
        log_event(
            LOGGER,
            logging.WARNING,
            "socket_payload_limit_rejected",
            payload_bytes=payload_length,
            max_payload_bytes=read_limit,
        )
        raise ValueError(f"CLI payload exceeds the {engine.max_buffer_bytes:,}-byte capture limit")
    return await _read_exact(reader, payload_length, "socket_request_bytes")


async def _write_socket_error(
    writer: asyncio.StreamWriter,
    code: str,
    message: str,
) -> None:
    response_frame = encode_frame(
        json.dumps(
            {"status": "error", "code": code, "message": message},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    METRICS.record_bytes("socket_response_bytes", len(response_frame))
    writer.write(response_frame)
    await writer.drain()


def handle_socket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    async def _handle():
        registered_call = DEFAULT_SERVICES.register_tool_call("socket")
        if registered_call is None:
            try:
                await _write_socket_error(
                    writer,
                    "server_shutting_down",
                    "The service is shutting down; retry after it restarts.",
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return
        call_id, _cancellation_event = registered_call
        try:
            ticket = await admission_gate("socket").acquire("socket")
        except AdmissionBusy:
            DEFAULT_SERVICES.finish_tool_call(call_id)
            log_event(LOGGER, logging.WARNING, "socket_client_rejected_busy")
            try:
                await _write_socket_error(
                    writer,
                    "server_busy",
                    "Socket work capacity is full; retry this request shortly.",
                )
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
            return
        except BaseException:
            DEFAULT_SERVICES.finish_tool_call(call_id)
            raise

        if DEFAULT_SERVICES.closing:
            ticket.release()
            DEFAULT_SERVICES.finish_tool_call(call_id)
            try:
                await _write_socket_error(
                    writer,
                    "server_shutting_down",
                    "The service is shutting down; retry after it restarts.",
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return

        try:
            # Read and validate one complete framed payload before decoding it.
            read_limit = (
                engine.max_buffer_bytes * SOCKET_JSON_MAX_EXPANSION
                + SOCKET_PAYLOAD_OVERHEAD
            )
            try:
                data = await asyncio.wait_for(
                    _read_socket_payload(reader, read_limit),
                    timeout=SOCKET_CLIENT_READ_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log_event(LOGGER, logging.WARNING, "socket_client_read_timeout")
                await _write_socket_error(
                    writer,
                    "socket_read_timeout",
                    "Socket request frame was not completed before the read timeout.",
                )
                return
            if not data:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                label = "CLI pipe"
                text = data.decode("utf-8", errors="replace")
                content_type = "auto"
                truncated = False
                original_byte_size = None
                command_exit_code = None
                timed_out = False
                duration_ms = None
                structured_metrics = None
            else:
                if not isinstance(payload, dict):
                    raise ValueError("socket payload must be a JSON object")
                label = payload.get("label", "CLI pipe")
                text = payload.get("text", "")
                content_type = payload.get("content_type", "auto")
                truncated = bool(payload.get("truncated", False))
                original_byte_size = payload.get("original_byte_size")
                command_exit_code = payload.get("command_exit_code")
                timed_out = bool(payload.get("timed_out", False))
                duration_ms = payload.get("duration_ms")
                structured_metrics = normalize_structured_metrics(payload.get("structured_metrics"))

            cap = await _run_admitted_thread(
                ticket,
                engine.ingest,
                text,
                label=label,
                content_type=content_type,
                truncated=truncated,
                original_byte_size=original_byte_size,
                command_exit_code=command_exit_code,
                timed_out=timed_out,
                source="socket",
                duration_ms=duration_ms,
                structured_metrics=structured_metrics,
                _service_context=DEFAULT_SERVICES,
                _call_id=call_id,
            )
            resp = {
                "status": "ok",
                "capture_id": cap.capture_id,
                "label": _bounded_summary_text(
                    cap.label,
                    SUMMARY_LABEL_MAX_BYTES,
                    SUMMARY_LABEL_TRUNCATION_MARKER,
                )[0],
                "line_count": cap.line_count,
                "byte_size": cap.byte_size
            }
            summary = await _run_admitted_thread(
                ticket,
                engine.get_summary_for_capture,
                cap,
                _service_context=DEFAULT_SERVICES,
                _call_id=call_id,
                include_previews=False,
            )
            if summary.get("status") == "ok":
                resp["summary"] = _summary_payload(summary)
            response_frame = encode_frame(json.dumps(resp).encode("utf-8"))
            METRICS.record_bytes("socket_response_bytes", len(response_frame))
            writer.write(response_frame)
            await writer.drain()
        except Exception as e:
            log_event(LOGGER, logging.ERROR, "socket_client_failed", error_type=type(e).__name__)
            err_resp = {"status": "error", "message": str(e)}
            response_frame = encode_frame(json.dumps(err_resp).encode("utf-8"))
            METRICS.record_bytes("socket_response_bytes", len(response_frame))
            writer.write(response_frame)
            await writer.drain()
        finally:
            ticket.release()
            DEFAULT_SERVICES.finish_tool_call(call_id)
            writer.close()
            await writer.wait_closed()

    # Return the task as well as scheduling it so embedders and tests can
    # await completion when they need deterministic cleanup.
    return asyncio.create_task(_handle())


def run_socket_server():
    """Runs a Unix domain socket server in a separate thread so CLI tools can pipe to it."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _set_socket_state("starting")
    bound_socket_identity = None
    socket_path_lock = None
    lock_acquired = False

    try:
        if socket_isolation_required() and not socket_isolation_configured():
            raise RuntimeError(
                "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
            )
        socket_path_lock = _socket_path_lock(SOCKET_PATH)
        socket_path_lock.__enter__()
        lock_acquired = True
        if os.path.lexists(SOCKET_PATH):
            initial_path_stat = os.lstat(SOCKET_PATH)
            if not stat.S_ISSOCK(initial_path_stat.st_mode):
                raise RuntimeError(
                    f"Socket path exists but is not a Unix socket: {SOCKET_PATH}"
                )
            initial_socket_identity = (
                initial_path_stat.st_dev,
                initial_path_stat.st_ino,
            )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(SOCKET_PATH)
            except ConnectionRefusedError:
                try:
                    current_path_stat = os.lstat(SOCKET_PATH)
                except FileNotFoundError:
                    current_path_stat = None
                if current_path_stat is not None:
                    current_socket_identity = (
                        current_path_stat.st_dev,
                        current_path_stat.st_ino,
                    )
                    if not stat.S_ISSOCK(current_path_stat.st_mode):
                        raise RuntimeError(
                            f"Socket path changed to a non-socket path: {SOCKET_PATH}"
                        )
                    if current_socket_identity != initial_socket_identity:
                        raise RuntimeError(
                            f"Socket path changed while probing: {SOCKET_PATH}"
                        )
                    # No listener accepted the connection, so this is a stale
                    # socket. Claim it before removal so a replacement cannot
                    # be deleted by a pathname-only unlink.
                    if not _unlink_socket_if_identity(
                        SOCKET_PATH, current_socket_identity
                    ) and os.path.lexists(SOCKET_PATH):
                        raise RuntimeError(
                            f"Socket path changed while cleaning stale socket: {SOCKET_PATH}"
                        )
            except FileNotFoundError:
                pass
            except OSError as exc:
                log_event(LOGGER, logging.ERROR, "socket_probe_failed", socket_path=SOCKET_PATH, error_type=type(exc).__name__)
                raise RuntimeError(f"Unable to verify existing socket {SOCKET_PATH}: {exc}") from exc
            else:
                log_event(LOGGER, logging.ERROR, "socket_conflict", socket_path=SOCKET_PATH)
                raise RuntimeError(f"Socket already in use at {SOCKET_PATH}")
            finally:
                probe.close()

        async def _main():
            nonlocal bound_socket_identity
            socket_backlog = (
                MAX_ACTIVE_SOCKET_CLIENTS + MAX_QUEUED_SOCKET_CLIENTS
            )
            server = await asyncio.start_unix_server(
                handle_socket_client,
                path=SOCKET_PATH,
                backlog=socket_backlog,
            )
            bound_path_stat = os.lstat(SOCKET_PATH)
            if stat.S_ISSOCK(bound_path_stat.st_mode):
                bound_socket_identity = (
                    bound_path_stat.st_dev,
                    bound_path_stat.st_ino,
                )
            os.chmod(SOCKET_PATH, 0o600)
            _set_socket_state("ready")
            async with server:
                await server.serve_forever()

        loop.run_until_complete(_main())
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
        _set_socket_state("failed", failure)
        log_event(LOGGER, logging.ERROR, "socket_server_failed", error_type=type(e).__name__)
        LOGGER.exception("socket_server_exception")
        # Preserve the established stderr diagnostic for callers and tests
        # that capture the server's direct error stream.
        print(f"Socket server error: {e}", file=sys.stderr)
    finally:
        try:
            if bound_socket_identity is not None:
                try:
                    current_path_stat = os.lstat(SOCKET_PATH)
                except FileNotFoundError:
                    pass
                else:
                    current_socket_identity = (
                        current_path_stat.st_dev,
                        current_path_stat.st_ino,
                    )
                    if (
                        stat.S_ISSOCK(current_path_stat.st_mode)
                        and current_socket_identity == bound_socket_identity
                    ):
                        try:
                            _unlink_socket_if_identity(
                                SOCKET_PATH, bound_socket_identity
                            )
                        except FileNotFoundError:
                            pass
        finally:
            if lock_acquired and socket_path_lock is not None:
                socket_path_lock.__exit__(None, None, None)
            loop.close()


def start_socket_server():
    """Start the IPC listener explicitly and return its background thread."""
    if SETTINGS.disable_socket_server.value:
        _set_socket_state("disabled")
        return None
    _set_socket_state("starting")
    socket_thread = threading.Thread(target=run_socket_server, daemon=True)
    socket_thread.start()
    return socket_thread


if __name__ == "__main__":
    if socket_isolation_required() and not socket_isolation_configured():
        raise SystemExit(
            "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
        )
    if not SETTINGS.disable_socket_server.value:
        start_socket_server()
        _require_socket_ready()
    try:
        DEFAULT_SERVICES.start()
        _refresh_mcp_instructions()
        mcp.run()
    finally:
        DEFAULT_SERVICES.close()
