"""Named bridge from domain tools to the active service context."""

from __future__ import annotations

from ..server import (
    _active_engine,
    _active_execution_manager,
    _active_metrics,
    _active_service_context,
    _active_settings,
    _metrics_snapshot,
    _request_cancel_event,
    _server_started_at,
    _write_metrics_snapshot,
)


def active_engine():
    return _active_engine()


def active_execution_manager():
    return _active_execution_manager()


def active_metrics():
    return _active_metrics()


def active_service_context():
    return _active_service_context()


def active_settings():
    return _active_settings()


def current_request_cancel_event():
    return _request_cancel_event()


def server_started_at() -> float:
    return _server_started_at()


def metrics_snapshot(**kwargs):
    return _metrics_snapshot(**kwargs)


def write_metrics_snapshot():
    _write_metrics_snapshot()


def compact_results_enabled(invocation_mode="direct") -> bool:
    """Return the current context's compact-result setting for MCP calls."""
    if invocation_mode != "mcp":
        return False
    setting = getattr(active_settings(), "compact_tool_results_enabled", None)
    return bool(getattr(setting, "value", False))
