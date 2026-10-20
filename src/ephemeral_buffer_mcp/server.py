"""
MCP Server for Ephemeral Command Output Hybrid Search.
Supports stdio MCP protocol and Unix Domain Socket IPC for CLI piping.
"""

import errno
import atexit
import dataclasses
import asyncio
import json
import inspect
import math
import os
import sys
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
from importlib import import_module
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
from .config import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    cleanup_execution_state_dir,
    load_settings,
    new_default_execution_state_dir,
    runtime_index_budget_adjustment_enabled,
    startup_settings,
)

if __name__ == "__main__":
    # ``python -m`` executes this file as ``__main__``. Delegate immediately
    # so transports and service lifecycle use the canonical package module.
    import_module(".server", __package__)._run_server()
    raise SystemExit(0)

SETTINGS = startup_settings()
from .engine import (
    MAX_CAPTURE_EVICTION_DETAILS,
    MAX_CAPTURE_SLICE_CONTENT_BYTES,
    CAPTURE_SLICE_SEGMENT_MAX_BYTES,
    EphemeralEngine,
    normalize_structured_metrics,
)
from .tools.formatting import (
    SEARCH_QUERY_MAX_BYTES,
    SEARCH_QUERY_TRUNCATION_MARKER,
    SUMMARY_COMMAND_MAX_BYTES,
    SUMMARY_COMMAND_TRUNCATION_MARKER,
    SUMMARY_DIFF_FILE_MAP_MAX_BYTES,
    SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES,
    SUMMARY_LABEL_MAX_BYTES,
    SUMMARY_LABEL_TRUNCATION_MARKER,
)
from .capture_utils import read_file_bounded, run_command_bounded
from .execution import (
    DEFAULT_EXECUTION_LIST_LIMIT,
    ExecutionBusyError,
    ExecutionDiskSpaceError,
    ExecutionPhaseNotFoundError,
    ExecutionQuotaError,
    ExecutionRecordError,
    MAX_EXECUTION_ID_BYTES,
    MAX_EXECUTION_LIST_LIMIT,
    MAX_EXECUTION_RETIRE_BATCH,
    MAX_EXECUTION_OUTPUT_CHUNK_BYTES,
    MAX_EXECUTION_PHASES,
    MAX_STRUCTURED_METRICS_BYTES,
    MAX_PHASE_NAME_BYTES,
    PhaseExecutionManager,
)
from .logging_utils import get_logger, log_event
from .metrics import LocalMetrics
from .schema_versions import (
    CAPTURE_SUMMARY_SCHEMA_VERSION,
    TOOL_RESPONSE_SCHEMA_VERSION,
)
from .admission import (
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
MCP_TOOL_RESPONSE_MAX_BYTES = 64 * 1024
MCP_JSONRPC_ENVELOPE_RESERVE_BYTES = 1024
SEARCH_RESPONSE_MAX_BYTES = (
    MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
)
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


async def _run_admitted_tool_call(
    service_context,
    tool_name: str,
    lane: str,
    operation,
    *,
    admission_gate_factory,
    work_type: str,
    on_rejection,
    on_admission_error,
    shutdown_message: str,
    busy_message: str,
):
    """Share service tracking, admission, cancellation scope, and cleanup."""
    registered_call = service_context.register_tool_call(tool_name)
    if registered_call is None:
        return await on_rejection("server_shutting_down", shutdown_message)

    call_id, cancellation_event = registered_call
    try:
        ticket = await admission_gate_factory(lane).acquire(work_type)
    except AdmissionBusy:
        service_context.finish_tool_call(call_id)
        log_event(
            LOGGER,
            logging.WARNING,
            "tool_call_rejected_busy",
            tool=tool_name,
            lane=lane,
        )
        return await on_rejection("server_busy", busy_message)
    except BaseException as exc:
        service_context.finish_tool_call(call_id)
        return await on_admission_error(exc)

    if service_context.closing:
        try:
            ticket.release()
        finally:
            service_context.finish_tool_call(call_id)
        return await on_rejection("server_shutting_down", shutdown_message)

    cancel_token = (
        _REQUEST_CANCEL_EVENT.set(cancellation_event)
        if cancellation_event is not None
        else None
    )
    try:
        return await operation(ticket, call_id)
    finally:
        if cancel_token is not None:
            _REQUEST_CANCEL_EVENT.reset(cancel_token)
        try:
            ticket.release()
        finally:
            service_context.finish_tool_call(call_id)


_REQUEST_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "ephemeral_request_cancel_event", default=None
)
CONSOLIDATION_MAX_CAPTURES = 25
CONSOLIDATION_CAPTURE_ID_MAX_BYTES = 256
SOCKET_STARTUP_TIMEOUT_SECONDS = SETTINGS.socket_startup_timeout_seconds.value
SERVER_STARTED_AT = time.time()
LOGGER = get_logger("server")
METRICS = LocalMetrics(enabled=SETTINGS.metrics_enabled.value)
_TOOL_CALL_IDS = itertools.count(1)
_TIMEOUT_RESULT_TOOLS = {
    "execute_and_capture",
    "start_execution",
    "resume_execution",
}


class ToolErrorEnvelope(BaseModel):
    """Stable machine-readable error information for public tool results."""

    code: str
    message: str


class ToolResponseEnvelope(BaseModel):
    """Versioned structured result paired with readable MCP text content."""

    schema_version: Literal[TOOL_RESPONSE_SCHEMA_VERSION] = (
        TOOL_RESPONSE_SCHEMA_VERSION
    )
    status: Literal["ok", "error"]
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ToolErrorEnvelope] = None
    text: Optional[str] = None
    truncated: bool = False
    next_cursor: Optional[str] = None


def _call_tool_result_bytes(
    envelope: ToolResponseEnvelope,
    *,
    compact_text: bool = False,
) -> int:
    """Measure the MCP result wire shape without importing the MCP SDK."""
    text = envelope.text or ""
    if compact_text:
        module = import_module(".mcp_transport", __package__)
        text = module._compact_tool_text(envelope)
    result = {
        "_meta": None,
        "content": [
            {"type": "text", "text": text, "annotations": None, "_meta": None}
        ],
        "structuredContent": envelope.model_dump(
            mode="json", by_alias=True, exclude={"text"}
        ),
        "isError": False,
        "resultType": "complete",
    }
    return len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
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


def _compact_tool_results_enabled_for_mcp(invocation_mode="direct") -> bool:
    """Read the compact-result preference only for explicit MCP dispatch."""
    if invocation_mode != "mcp":
        return False
    context = _active_service_context()
    setting = getattr(
        getattr(context, "settings", SETTINGS),
        "compact_tool_results_enabled",
        None,
    )
    return bool(getattr(setting, "value", False))


def _instrument_tool(name):
    """Decorate a tool with opt-in content-free call metrics and timing logs."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            invocation_mode = kwargs.pop("__invocation_mode", "direct")
            if "__invocation_mode" in inspect.signature(function).parameters:
                kwargs["__invocation_mode"] = invocation_mode
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
                    response_text = (
                        result.text if isinstance(result, ToolResponseEnvelope) else result
                    )
                    if invocation_mode == "mcp" and _compact_tool_results_enabled_for_mcp(
                        invocation_mode
                    ):
                        mcp_transport = import_module(".mcp_transport", __package__)
                        categories = mcp_transport.get_tool_categories()
                    else:
                        mcp_transport = None
                        categories = {}
                    if (
                        mcp_transport is not None
                        and categories.get(name)
                        in {"capture", "retrieval", "search", "lifecycle"}
                    ):
                        # Keep byte metrics inside this measured call while using
                        # the same compact, budget-fitted text the adapter returns.
                        measured_envelope = result
                        if not isinstance(measured_envelope, ToolResponseEnvelope):
                            measured_envelope = mcp_transport._legacy_tool_envelope(
                                name, result, compact_text=True
                            )
                        measured_envelope = mcp_transport._fit_tool_envelope(
                            measured_envelope, compact_text=True
                        )
                        response_text = measured_envelope.text
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
        wrapper._supports_invocation_mode = True
        return wrapper
    return decorator


_COMPACT_RESULT_SUMMARY_FIELDS = (
    "capture_id",
    "execution_id",
    "label",
    "query",
    "mode",
    "source",
    "total_lines",
    "byte_size",
    "original_byte_size",
    "content_bytes",
    "capture_count",
    "match_count",
    "available_match_count",
    "returned_count",
    "omitted_count",
    "requested_start_line",
    "requested_end_line",
    "start_line",
    "end_line",
    "partial",
    "exit_code",
)


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


def _execution_error_json(
    code: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    error: Dict[str, Any] = {"code": code, "message": message or "Execution operation failed."}
    if details:
        error["details"] = details
    return json.dumps(
        {"status": "error", "error": error},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _execution_exception_json(exc: Exception) -> str:
    """Map execution exceptions to stable codes without inspecting messages."""
    details = None
    if isinstance(exc, ExecutionBusyError):
        code = "execution_busy"
    elif isinstance(exc, ExecutionRecordError):
        code = (
            "unsupported_schema_version"
            if exc.reason_code == "UNSUPPORTED_SCHEMA_VERSION"
            else "malformed_record"
        )
        details = _execution_record_error_details(
            exc.reason_code,
            exc.schema_version,
        )
    elif isinstance(exc, ExecutionQuotaError):
        code = "quota_exceeded"
    elif isinstance(exc, ExecutionDiskSpaceError):
        code = "disk_full"
    elif isinstance(exc, ExecutionPhaseNotFoundError):
        code = "phase_not_found"
    elif isinstance(exc, KeyError):
        code = "execution_not_found"
    elif isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        code = "disk_full"
    elif isinstance(exc, OSError):
        code = "execution_storage_error"
    else:
        code = "execution_failed"
    return _execution_error_json(code, str(exc), details=details)


def _execution_record_error_details(
    reason_code: Any,
    schema_version: Any,
) -> Dict[str, Any]:
    """Return bounded, JSON-safe diagnostic fields for an invalid record."""
    details: Dict[str, Any] = {}
    if isinstance(reason_code, str):
        details["reason_code"] = reason_code
    if isinstance(schema_version, (str, int, float, bool)) and (
        not isinstance(schema_version, float) or math.isfinite(schema_version)
    ):
        if isinstance(schema_version, str):
            schema_version = schema_version.encode("utf-8", errors="replace")[:64].decode(
                "utf-8", errors="ignore"
            )
        details["schema_version"] = schema_version
    return details


def _execution_payload_with_error_codes(payload: Any) -> Any:
    """Add stable error codes to read-only malformed-record diagnostics."""
    if not isinstance(payload, dict):
        return payload

    def annotate(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
        status = diagnostic.get("status")
        if status not in {"invalid_record", "unsupported_record"}:
            return diagnostic
        code = (
            "unsupported_schema_version"
            if diagnostic.get("reason_code") == "UNSUPPORTED_SCHEMA_VERSION"
            or status == "unsupported_record"
            else "malformed_record"
        )
        annotated = dict(diagnostic)
        error = {
            "code": code,
            "message": str(diagnostic.get("message") or "Execution record is invalid."),
        }
        details = _execution_record_error_details(
            diagnostic.get("reason_code"),
            diagnostic.get("schema_version"),
        )
        if details:
            error["details"] = details
        annotated["error"] = error
        return annotated

    annotated = annotate(payload)
    executions = payload.get("executions")
    if isinstance(executions, list):
        annotated = dict(annotated)
        annotated["executions"] = [
            annotate(item) if isinstance(item, dict) else item for item in executions
        ]
    return annotated


def _execution_json(operation, *, max_response_bytes: Optional[int] = None) -> str:
    """Run an execution operation and return a compact machine-readable result."""
    try:
        payload = _execution_payload_with_error_codes(operation())
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
                            "status", "execution_id", "label", "execution_status", "partial",
                            "execution_in_progress", "cancellation_requested",
                            "background_error", "updated_at",
                            "completed_phase_count", "phase_count",
                            "reason_code", "schema_version", "read_only", "error",
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
                        "completed_phase_count", "phase_count", "reason_code",
                        "read_only", "error", "record_bytes", "record_preview",
                        "record_preview_sanitized", "record_preview_redacted",
                        "record_preview_truncated",
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
            return _execution_error_json(
                "response_too_large",
                "Execution response exceeds the configured byte limit; request a smaller output chunk.",
            )
        return result
    except Exception as exc:
        return _execution_exception_json(exc)


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
    from .tools import load_tool_handlers
    from .tools.registry import get_tool_categories, get_tool_names

    load_tool_handlers()
    return metrics.snapshot(
        available_tools=get_tool_names(),
        tool_categories=get_tool_categories(),
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


def create_mcp_server(context: "ServiceContext | None" = None):
    """Build the MCP transport lazily around the shared service context."""
    module = import_module(".mcp_transport", __package__)
    return module.create_mcp_server(context)


def start_socket_server():
    """Start the Unix socket transport without importing MCP SDK modules."""
    module = import_module(".socket_transport", __package__)
    return module.start_socket_server()


def run_socket_server():
    """Run the Unix socket transport in the current worker thread."""
    module = import_module(".socket_transport", __package__)
    return module.run_socket_server()


def _require_socket_ready():
    module = import_module(".socket_transport", __package__)
    return module._require_socket_ready()


def __getattr__(name: str):
    """Resolve legacy internal names from their owning domain module."""
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "mcp" or name.startswith(
        (
            "_MCP",
            "_mcp",
            "_REGISTERED_MCP_",
            "MCP_TOOL_LIST_",
            "MCP_TOOL_CATEGORY_",
            "_register_mcpserver",
            "_call_tool_result",
            "_compact_tool_text",
            "_legacy_tool_envelope",
            "_fit_tool_envelope",
        )
    ):
        module = import_module(".mcp_transport", __package__)
        if name == "_MCP_TOOL_REGISTRATIONS":
            return [
                (item.function, item.name, item.category, item.structured_result_factory)
                for item in module.load_tool_handlers()
            ]
        if name == "_REGISTERED_MCP_TOOL_NAMES":
            return [item.name for item in module.load_tool_handlers()]
        if name == "_REGISTERED_MCP_TOOL_CATEGORIES":
            return {item.name: item.category for item in module.load_tool_handlers()}
        if name == "MCP_TOOL_CATEGORY_NAMES":
            return tuple(module.TOOL_CATEGORIES)
        if hasattr(module, name):
            return getattr(module, name)
    if name.startswith(
        (
            "SOCKET_",
            "_SOCKET_",
            "socket_",
            "_socket_",
            "start_socket",
            "run_socket",
            "handle_socket",
            "_read_socket",
            "_write_socket",
            "_require_socket",
            "_unlink_socket",
            "_bind_socket",
            "_connect_socket",
            "_verify_bound_socket",
        )
    ):
        module = import_module(".socket_transport", __package__)
        if hasattr(module, name):
            return getattr(module, name)
    from .tools import load_tool_handlers
    load_tool_handlers()
    for module_name in ("execution", "capture", "search", "retrieval", "diagnostics", "configuration"):
        module = import_module(f".tools.{module_name}", __package__)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in globals() if not name.startswith("__")]


def _run_server():
    if socket_isolation_required() and not socket_isolation_configured():
        raise SystemExit(
            "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
        )
    if not SETTINGS.disable_socket_server.value:
        start_socket_server()
        _require_socket_ready()
    try:
        DEFAULT_SERVICES.start()
        mcp_module = import_module(".mcp_transport", __package__)
        mcp_app = mcp_module.create_mcp_server(DEFAULT_SERVICES)
        mcp_module._refresh_mcp_instructions(mcp_app, DEFAULT_SERVICES)
        mcp_app.run()
    finally:
        DEFAULT_SERVICES.close()
