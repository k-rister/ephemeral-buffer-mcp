"""Diagnostics domain handlers."""

import importlib
import json
import logging
import os
import platform
import re
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from ._context import (
    active_engine as _active_engine,
    active_metrics as _active_metrics,
    active_settings as _active_settings,
    metrics_snapshot as _metrics_snapshot,
    server_started_at as _server_started_at,
)
from .instrumentation import _instrument_tool
from .registry import register_tool

LOGGER = get_logger("server")


def _runtime_package_version() -> str:
    """Return the version for the source or installed server being run."""
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        source_text = pyproject.read_text(encoding="utf-8")
    except OSError:
        source_text = ""
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']\s*$', source_text, re.MULTILINE)
    if match:
        return match.group(1)
    for distribution in (
        "ephemeral-buffer-mcp",
        "ephemeral-buffer-mcp-lexical",
        "ephemeral-buffer-mcp-core",
    ):
        try:
            return package_version(distribution)
        except PackageNotFoundError:
            continue
    return "source checkout"


def _socket_lifecycle():
    return importlib.import_module("..socket_transport", __package__)._socket_lifecycle()


def _mcpserver_compatibility_diagnostics():
    adapter = importlib.import_module("..mcpserver_adapter", __package__)
    return adapter.compatibility_diagnostics()


@register_tool(
    "get_buffer_stats", "diagnostics", mcp_admission_lane="diagnostics"
)
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
    semantic_backend_line = (
        "Semantic backend: available"
        if stats.get("semantic_backend_available", True)
        else "Semantic backend: unavailable (install ephemeral-buffer-mcp for semantic search)"
    )
    load_failure = stats.get("embedding_model_load_failure")
    retry_after = stats.get("embedding_model_retry_after_seconds", 0.0)
    if load_failure and retry_after > 0:
        load_backoff_line = (
            f"Embedding model load backoff: {load_failure} "
            f"(retry in {retry_after:g}s)"
        )
    elif load_failure:
        load_backoff_line = f"Embedding model load retry: available ({load_failure})"
    else:
        load_backoff_line = "Embedding model load backoff: inactive"
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
    capture_removal_counts = stats.get(
        "capture_removals_by_reason",
        {"count": 0, "bytes": 0, "chunks": 0, "clear": 0, "budget": 0},
    )
    capture_rejection_counts = stats.get(
        "capture_rejections_by_reason",
        {"bytes": 0, "chunks": 0},
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
        f"Capture removals by reason: {json.dumps(capture_removal_counts, sort_keys=True)}\n"
        f"Capture rejections by reason: {json.dumps(capture_rejection_counts, sort_keys=True)}\n"
        f"Deferred storage: {stats.get('deferred_storage_capture_count', 0)} captures, "
        f"{stats.get('deferred_storage_readers', 0)} readers, "
        f"{stats.get('deferred_storage_bytes', 0):,} accounted bytes\n"
        f"Lines: {stats['total_lines']:,}\n"
        f"Chunks: {stats['total_chunks']:,}\n"
        f"Indexed chunks: {indexed_chunks:,}/{max_indexed_chunks:,} "
        f"({remaining_indexed_chunks:,} remaining)\n"
        f"Index budget adjustment: {json.dumps(stats.get('last_index_budget_adjustment', {}), sort_keys=True)}\n"
        f"{model_line}\n"
        f"{semantic_backend_line}\n"
        f"{load_backoff_line}\n"
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


@register_tool("get_runtime_diagnostics", "diagnostics")
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

    uptime_seconds = max(0, int(time.time() - _server_started_at()))
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
        "Semantic backend: "
        + ("available" if stats.get("semantic_backend_available", True) else "unavailable"),
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
    mcpserver_status = _mcpserver_compatibility_diagnostics()
    limitations = []
    if mcpserver_status["tools_without_validation_metrics"]:
        limitations.append(
            "validation metrics unavailable for "
            + ", ".join(mcpserver_status["tools_without_validation_metrics"])
        )
    if mcpserver_status["validation_observation_failures"]:
        limitations.append(
            "validation metrics failed during observation for "
            + ", ".join(mcpserver_status["validation_observation_failures"])
        )
    if mcpserver_status["instruction_refresh"] == "unavailable":
        limitations.append("runtime instruction refresh unavailable")
    if limitations:
        lines.append(
            f"MCPServer {mcpserver_status['sdk_version']} compatibility limitation: "
            + "; ".join(limitations)
            + ". Tool dispatch remains available."
        )
    if _active_metrics().enabled:
        snapshot = _metrics_snapshot()
        lines.append(f"Data-path bytes: {json.dumps(snapshot['bytes'], sort_keys=True)}")
        lines.append(f"Metrics summary: {json.dumps(snapshot, sort_keys=True)}")
    return "\n".join(lines)


@register_tool("get_usage_metrics", "diagnostics")
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
