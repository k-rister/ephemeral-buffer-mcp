"""MCP transport assembly, middleware, and protocol result conversion."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from typing import Any, Optional

from mcp.server import CacheHint, MCPServer
from mcp.server.context import ServerRequestContext
from mcp.types import CallToolResult, TextContent

from . import server as _runtime
from .server import *  # noqa: F403
from .mcpserver_adapter import (
    compatibility_diagnostics as _mcpserver_compatibility_diagnostics,
    refresh_instructions as _refresh_mcpserver_instructions,
    register_tool as _register_mcpserver_tool,
)
from .tools import load_tool_handlers
from .tools.formatting import bounded_summary_text, truncate_utf8_with_marker
from .tools.registry import (
    TOOL_CATEGORIES,
    get_tool_categories,
    get_tool_registration,
)

load_tool_handlers()

_MCP_SESSION_SCOPE_LOCK = threading.Lock()
_MCP_SESSION_SCOPES: weakref.WeakKeyDictionary[Any, tuple[str, dict[str, str]]] = (
    weakref.WeakKeyDictionary()
)
MAX_MCP_SESSION_SCOPE_FALLBACK = 128
MCP_TOOL_LIST_CACHE_TTL_MS = 300_000
MCP_TOOL_LIST_CACHE_HINTS = {
    "tools/list": CacheHint(ttl_ms=MCP_TOOL_LIST_CACHE_TTL_MS, scope="public"),
}
_MCP_SESSION_SCOPE_FALLBACK: OrderedDict[int, tuple[Any, tuple[str, dict[str, str]]]] = (
    OrderedDict()
)


@contextmanager
def _bind_mcp_metrics_scope(
    session: Any = None,
    service_context: "ServiceContext | None" = None,
) -> Any:
    """Bind metrics to a stable MCP connection object when the transport has one."""
    if session is None:
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
            # Keep compatibility with transports whose connection object is not
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


def _mcp_metrics_scope_key(request_context: ServerRequestContext) -> Any:
    """Return the stable connection identity used for request-scoped metrics."""
    session = getattr(request_context, "session", None)
    # MCPServer 2.3.0 creates one ServerSession proxy per request; its
    # connection is the lifetime-stable identity on stateful transports.
    return getattr(session, "_connection", session)


class _MCPRequestMetricsMiddleware:
    """Bind MCP tool calls to their isolated metrics scope."""

    def __init__(self, service_context: "ServiceContext") -> None:
        self._service_context = service_context

    async def __call__(self, request_context, call_next):
        if request_context.method != "tools/call":
            return await call_next(request_context)
        with _activate_service_context(self._service_context):
            with _bind_mcp_metrics_scope(
                _mcp_metrics_scope_key(request_context), self._service_context
            ):
                return await call_next(request_context)


def _compact_tool_text(envelope: ToolResponseEnvelope) -> str:
    """Render concise MCP text without repeating structured payload fields."""
    parts = [f"status={envelope.status}"]
    if envelope.error is not None:
        parts.append(f"error_code={envelope.error.code}")

    data = envelope.data if isinstance(envelope.data, dict) else {}
    if data.get("status") not in (None, envelope.status):
        parts.append(f"result_status={data['status']}")
    for key in _COMPACT_RESULT_SUMMARY_FIELDS:
        value = data.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, str):
            value, _ = bounded_summary_text(value, 160, "... [truncated]")
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={value}")
    if envelope.error is None and isinstance(data.get("message"), str):
        message, _ = bounded_summary_text(
            data["message"], 160, "... [truncated]"
        )
        parts.append(f"message={json.dumps(message, ensure_ascii=False)}")

    if envelope.truncated or data.get("truncated"):
        parts.append("truncated=true")
    next_cursor = envelope.next_cursor or data.get("next_cursor")
    if next_cursor:
        # The cursor is a bounded opaque continuation handle; preserve it exactly.
        parts.append(f"next_cursor={json.dumps(next_cursor, ensure_ascii=False)}")

    matches = data.get("matches")
    if isinstance(matches, list) and matches:
        ranges = [
            str(match["matched_range"])
            for match in matches[:5]
            if isinstance(match, dict) and match.get("matched_range") is not None
        ]
        if ranges:
            parts.append(f"matched_ranges={json.dumps(ranges, ensure_ascii=False)}")
        parts.append("match_details=structuredContent.data.matches")
        parts.append("Use get_capture_slice for full content.")
    elif "content" in data:
        parts.append("content=structuredContent.data.content")
    elif isinstance(data.get("captures"), list):
        capture_ids = [
            capture["capture_id"]
            for capture in data["captures"][:5]
            if isinstance(capture, dict) and capture.get("capture_id") is not None
        ]
        if capture_ids:
            parts.append(f"capture_ids={json.dumps(capture_ids, ensure_ascii=False)}")
        parts.append("capture_details=structuredContent.data.captures")
    elif envelope.error is not None:
        parts.append("details=structuredContent.error.message")
    else:
        parts.append("details=structuredContent")

    parts.append("Set EPHEMERAL_COMPACT_TOOL_RESULTS=false for legacy text output.")
    return "Compact result: " + "; ".join(parts)


def _call_tool_result(envelope: ToolResponseEnvelope) -> CallToolResult:
    """Create an MCP result with readable text and its structured envelope."""
    return CallToolResult(
        content=[TextContent(type="text", text=envelope.text or "")],
        structured_content=envelope.model_dump(mode="json", exclude={"text"}),
    )


def _call_tool_result_bytes(
    envelope: ToolResponseEnvelope,
    *,
    compact_text: bool = False,
) -> int:
    """Measure the JSON-encoded MCP result body, including both result channels."""
    if compact_text:
        envelope = envelope.model_copy(update={"text": _compact_tool_text(envelope)})
    return len(_call_tool_result(envelope).model_dump_json(by_alias=True).encode("utf-8"))


def _legacy_tool_envelope(
    tool: str,
    result: Any,
    *,
    compact_text: bool = False,
    legacy_error_code: str | None = None,
    legacy_error_mode: str | None = None,
) -> ToolResponseEnvelope:
    """Wrap an existing Python text result in the versioned MCP result shape."""
    text = result if isinstance(result, str) else str(result)
    if legacy_error_code is None or legacy_error_mode is None:
        try:
            registration = get_tool_registration(tool)
        except KeyError:
            registration = None
        if registration is not None:
            if legacy_error_code is None:
                legacy_error_code = registration.legacy_error_code
            if legacy_error_mode is None:
                legacy_error_mode = registration.legacy_error_mode
    legacy_error_mode = legacy_error_mode or "default"
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
            elif legacy_error_code is not None:
                error_code = legacy_error_code
            else:
                error_code = f"{tool}_failed"
            error = ToolErrorEnvelope(
                code=error_code,
                message=str(payload.get("message") or "The operation failed."),
            )
        else:
            data = payload
    elif text.startswith(("Error", "Search Error")) or (
        legacy_error_mode == "clear" and "not found" in text.lower()
    ):
        status = "error"
        if legacy_error_mode == "retrieval":
            code = "capture_not_found" if "not found" in text.lower() else "retrieval_failed"
        elif legacy_error_mode == "search":
            code = "invalid_search_request" if any(
                marker in text.lower()
                for marker in ("query must", "query exceeds", "unsupported", "top_k", "context_lines")
            ) else "search_failed"
        elif legacy_error_mode == "capture":
            code = "capture_failed"
        elif legacy_error_mode == "clear" and "not found" in text.lower():
            code = "capture_not_found"
        else:
            code = "operation_failed"
        error = ToolErrorEnvelope(code=code, message=text)
    else:
        data = {"message": text}

    envelope = ToolResponseEnvelope(status=status, data=data, error=error, text=text)
    return _fit_tool_envelope(envelope, compact_text=compact_text)


def _fit_tool_envelope(
    envelope: ToolResponseEnvelope,
    max_bytes: int = MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES,
    *,
    compact_text: bool = False,
) -> ToolResponseEnvelope:
    """Keep the complete structured MCP result within the configured byte budget."""
    if compact_text:
        envelope = envelope.model_copy(update={"text": _compact_tool_text(envelope)})
    if _call_tool_result_bytes(envelope) <= max_bytes:
        return envelope

    marker = "\n... [response truncated; use the structured status and follow-up tools]"
    shortened = envelope.model_copy(update={"data": {}, "truncated": True})
    low = 0
    high = max_bytes
    best = shortened.model_copy(update={"text": marker[:max(0, max_bytes // 4)]})
    while low <= high:
        middle = (low + high) // 2
        candidate_text = truncate_utf8_with_marker(
            envelope.text or "", middle, marker
        )
        if candidate_text is None:
            candidate_text = ""
        candidate = shortened.model_copy(update={"text": candidate_text})
        if _call_tool_result_bytes(candidate) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _service_rejection_result(
    code: str,
    message: str,
    *,
    structured: bool,
    compact_text: bool = False,
) -> CallToolResult | str:
    """Return a consistent JSON error body for busy and shutdown failures."""
    error_text = json.dumps(
        {"status": "error", "error": {"code": code, "message": message}},
        sort_keys=True,
        separators=(",", ":"),
    )
    if not structured:
        return error_text
    response = ToolResponseEnvelope(
        status="error",
        error=ToolErrorEnvelope(code=code, message=message),
        text=error_text,
    )
    return _call_tool_result(_fit_tool_envelope(response, compact_text=compact_text))


def create_mcp_server(context: "ServiceContext | None" = None):
    """Build an MCP application from the shared domain-tool registry."""
    service_context = context or DEFAULT_SERVICES
    app = MCPServer(
        "ephemeral-buffer",
        instructions=_mcp_instructions(service_context),
        cache_hints=MCP_TOOL_LIST_CACHE_HINTS,
        middleware=[_MCPRequestMetricsMiddleware(service_context)],
    )
    for registration in load_tool_handlers():
        _register_mcp_tool_for_app(
            app,
            registration.function,
            registration.name,
            registration.category,
            registration.structured_result_factory,
            service_context,
            mcp_admission_lane=registration.mcp_admission_lane,
            background_execution=registration.background_execution,
            legacy_error_code=registration.legacy_error_code,
            legacy_error_mode=registration.legacy_error_mode,
        )
    return app


def _register_mcp_tool_for_app(
    app,
    function,
    name,
    category,
    structured_result_factory,
    service_context,
    *,
    mcp_admission_lane=None,
    background_execution=None,
    legacy_error_code=None,
    legacy_error_mode=None,
):
    """Register one shared tool implementation with context-bound adapters."""
    try:
        registration = get_tool_registration(name)
    except KeyError:
        registration = None
    if registration is not None:
        if mcp_admission_lane is None:
            mcp_admission_lane = registration.mcp_admission_lane
        if background_execution is None:
            background_execution = registration.background_execution
        if legacy_error_code is None:
            legacy_error_code = registration.legacy_error_code
        if legacy_error_mode is None:
            legacy_error_mode = registration.legacy_error_mode
    mcp_admission_lane = mcp_admission_lane or "mcp"
    background_execution = bool(background_execution)
    legacy_error_mode = legacy_error_mode or "default"
    # Reuse the registration behavior installed by the decorator factory.
    @wraps(function)
    async def adapter(*args, **kwargs):
        with _activate_service_context(service_context):
            compact_text = bool(
                getattr(
                    getattr(
                        getattr(service_context, "settings", None),
                        "compact_tool_results_enabled",
                        None,
                    ),
                    "value",
                    False,
                )
            )
            structured = (
                structured_result_factory is not None
                or category in {"capture", "retrieval", "search", "lifecycle"}
            )

            async def on_rejection(code, message):
                return _service_rejection_result(
                    code,
                    message,
                    structured=structured,
                    compact_text=compact_text,
                )

            async def on_admission_error(exc):
                raise exc

            async def invoke(ticket, call_id):
                if structured_result_factory is not None:
                    factory_kwargs = kwargs
                    if getattr(
                        structured_result_factory,
                        "_supports_invocation_mode",
                        False,
                    ):
                        factory_kwargs = {**kwargs, "__invocation_mode": "mcp"}
                    response = await _run_admitted_thread(
                        ticket,
                        structured_result_factory,
                        *args,
                        _service_context=service_context,
                        _call_id=call_id,
                        **factory_kwargs,
                    )
                    if not isinstance(response, ToolResponseEnvelope):
                        raise TypeError("structured_result_factory must return ToolResponseEnvelope")
                else:
                    dispatch_function = function
                    dispatch_kwargs = kwargs
                    if getattr(dispatch_function, "_supports_invocation_mode", False):
                        dispatch_kwargs = {**kwargs, "__invocation_mode": "mcp"}
                    if background_execution:
                        dispatch_kwargs = {**dispatch_kwargs, "__background": True}
                    result = await _run_admitted_thread(
                        ticket,
                        dispatch_function,
                        *args,
                        _service_context=service_context,
                        _call_id=call_id,
                        **dispatch_kwargs,
                    )
                    if category not in {"capture", "retrieval", "search", "lifecycle"}:
                        return result
                    response = _legacy_tool_envelope(
                        name,
                        result,
                        compact_text=compact_text,
                        legacy_error_code=legacy_error_code,
                        legacy_error_mode=legacy_error_mode,
                    )
                return _call_tool_result(
                    _fit_tool_envelope(response, compact_text=compact_text)
                )

            return await _run_admitted_tool_call(
                service_context,
                name,
                mcp_admission_lane,
                invoke,
                admission_gate_factory=admission_gate,
                work_type=name,
                on_rejection=on_rejection,
                on_admission_error=on_admission_error,
                shutdown_message="The service is shutting down; retry after it restarts.",
                busy_message="Foreground work capacity is full; retry this tool call shortly.",
            )

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
        with _activate_service_context(service_context):
            yield

    def observe_validation_failure():
        with _active_metrics().measure(name) as state:
            state["success"] = False
            state["failure_category"] = "validation"
        _write_metrics_snapshot(service_context)

    _register_mcpserver_tool(
        app,
        adapter,
        name=name,
        validation_scope=validation_scope,
        validation_observer=observe_validation_failure,
    )

def _mcp_instructions(context: "ServiceContext | None" = None) -> str:
    """Return the session-specific socket state for client initialization."""
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
    socket_transport = importlib.import_module(".socket_transport", __package__)
    socket_state, socket_failure = socket_transport._socket_lifecycle()
    if socket_state == "failed":
        socket_status = f"Socket lifecycle is failed ({socket_failure})."
    else:
        socket_status = f"Socket lifecycle is {socket_state}."
    return f"{isolation} {socket_status}"


def _refresh_mcp_instructions(app=None, context: "ServiceContext | None" = None) -> None:
    """Refresh client guidance immediately before MCP request serving."""
    target = app or mcp
    _refresh_mcpserver_instructions(target, _mcp_instructions(context))


mcp = create_mcp_server()
