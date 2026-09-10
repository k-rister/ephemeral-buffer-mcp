"""
MCP Server for Ephemeral Command Output Hybrid Search.
Supports stdio MCP protocol and Unix Domain Socket IPC for CLI piping.
"""

import os
import atexit
import sys
import json
import asyncio
import socket
import threading
import logging
import itertools
import platform
import re
import shlex
import shutil
import subprocess
import time
from functools import wraps
from importlib.metadata import PackageNotFoundError, version as package_version
from asyncio import to_thread
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import positive_int_env, socket_isolation_configured, socket_isolation_required, socket_path
from mcp.server.fastmcp import FastMCP
from engine import (
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    EphemeralEngine,
)
from capture_utils import read_file_bounded, run_command_bounded
from logging_utils import get_logger, log_event
from metrics import LocalMetrics

SOCKET_PATH = socket_path()
SOCKET_PAYLOAD_OVERHEAD = 64 * 1024
SERVER_STARTED_AT = time.time()
LOGGER = get_logger("server")
METRICS = LocalMetrics()
_TOOL_CALL_IDS = itertools.count(1)


def _instrument_tool(name):
    """Decorate a tool with opt-in content-free call metrics and timing logs."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            call_id = next(_TOOL_CALL_IDS)
            started = time.perf_counter()
            log_event(LOGGER, logging.INFO, "mcp_tool_started", call_id=call_id, tool=name)
            try:
                with METRICS.measure(name) as state:
                    result = function(*args, **kwargs)
                    if isinstance(result, str) and result.startswith(("Error", "Search Error")):
                        state["success"] = False
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
        return wrapper
    return decorator

def _mcp_instructions() -> str:
    """Return client-visible operating guidance for this server instance."""
    if socket_isolation_required() and not socket_isolation_configured():
        isolation = "Socket isolation is required but not configured; startup must fail."
    elif socket_isolation_configured():
        isolation = "Socket isolation is configured for this session."
    else:
        isolation = "This is legacy single-session mode; configure EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH for concurrency."
    return (
        "Use execute_and_capture for large, noisy, or uncertain command output and for workflows "
        "that need later search or follow-up retrieval. Use direct command execution for small, "
        "targeted inspections. " + isolation
    )


# Initialize FastMCP
mcp = FastMCP("ephemeral-buffer", instructions=_mcp_instructions())


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


engine = EphemeralEngine(
    max_captures=positive_int_env("EPHEMERAL_MAX_CAPTURES", DEFAULT_MAX_CAPTURES),
    max_buffer_bytes=positive_int_env("EPHEMERAL_MAX_BUFFER_BYTES", DEFAULT_MAX_BUFFER_BYTES),
    metrics=METRICS,
)
atexit.register(engine.shutdown)


# --- MCP Tools ---


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
    resolved = shutil.which(token)
    if resolved:
        return {"status": "resolved", "requested": token, "resolved": str(Path(resolved).resolve(strict=False))}
    return {"status": "unavailable", "requested": token, "reason": "executable was not found on PATH"}


@mcp.tool()
@_instrument_tool("preflight_command")
def preflight_command(command: str, cwd: Optional[str] = None) -> str:
    """Return content-free path and executable diagnostics without running ``command``.

    This resolves the working directory, symlink target, detectable Git
    repository root, and first executable token. Shell expansion, aliases,
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

def _capture_text(content: str, label: str = "", content_type: str = "auto") -> str:
    """
    Ingests raw text output directly into the ephemeral search index.
    Automatically detects diffs, logs, and text structures.
    Returns capture metadata (ID, line count, byte size, diff summary if applicable).
    """
    cap = engine.ingest(content, label=label, content_type=content_type)
    summary = engine.get_summary(cap.capture_id)
    
    if summary.get("content_type") == "diff" and summary.get("file_map"):
        return (
            f"Captured into ID '{cap.capture_id}' ({cap.label})\n"
            f"- Type: Unified Diff ({summary.get('diff_stats')})\n"
            f"- Lines: {cap.line_count:,} | Bytes: {cap.byte_size:,} | Chunks: {len(cap.chunks)}\n"
            f"- Detected Signals: {summary.get('signals_summary')}\n\n"
            f"--- Modified Files Map ---\n{summary['file_map']}\n\n"
            f"Use `search_capture` or `get_capture_slice` with capture_id='{cap.capture_id}' to query."
        )

    return (
        f"Captured into ID '{cap.capture_id}' ({cap.label})\n"
        f"- Lines: {cap.line_count:,}\n"
        f"- Bytes: {cap.byte_size:,}\n"
        f"- Chunks: {len(cap.chunks)}\n"
        f"Use `search_capture` with capture_id='{cap.capture_id}' or 'latest' to query."
    )


@mcp.tool()
@_instrument_tool("capture_text")
def capture_text(content: str, label: str = "", content_type: str = "auto") -> str:
    """Ingest already-collected text and return capture metadata.

    Use this when the caller already has output to index. For a noisy or
    potentially long command, use ``execute_and_capture`` so output remains
    bounded before it reaches the agent context.
    """
    return _capture_text(content, label=label, content_type=content_type)


@mcp.tool()
@_instrument_tool("capture_file")
def capture_file(
    file_path: str,
    label: str = "",
    content_type: str = "auto",
    max_bytes: Optional[int] = None
) -> str:
    """
    Reads a file or log output from disk and ingests it into the ephemeral search index.

    Validate the intended file path before calling: resolve symlinks when path
    identity matters, confirm the file belongs to the expected workspace, and
    use an explicit bounded ``max_bytes`` for large or untrusted files. Capture
    limits control output handling; they do not validate filesystem intent.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        if max_bytes is not None and max_bytes > engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "capture_file_limit_rejected",
                requested_bytes=max_bytes,
                max_buffer_bytes=engine.max_buffer_bytes,
            )
            return (
                f"Error: max_bytes ({max_bytes:,}) exceeds the configured "
                f"buffer limit ({engine.max_buffer_bytes:,})."
            )
        read_limit = engine.max_buffer_bytes if max_bytes is None else max_bytes
        if read_limit < 1:
            return "Error: max_bytes must be at least 1."
        content = read_file_bounded(file_path, read_limit)
        if not label:
            label = os.path.basename(file_path)
        return _capture_text(content, label=label, content_type=content_type)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@mcp.tool()
@_instrument_tool("execute_and_capture")
def execute_and_capture(
    command: str,
    cwd: Optional[str] = None,
    label: str = "",
    content_type: str = "auto",
    max_output_bytes: Optional[int] = None,
    timeout_seconds: Optional[float] = None
) -> str:
    """
    Runs a shell command, captures stdout/stderr, indexes it, and returns a concise summary
    (exit code, line count, diff map or error signals, head/tail preview) WITHOUT flooding
    your prompt context with thousands of lines.

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
        if max_output_bytes is not None and max_output_bytes > engine.max_buffer_bytes:
            log_event(
                LOGGER,
                logging.WARNING,
                "command_output_limit_rejected",
                requested_bytes=max_output_bytes,
                max_buffer_bytes=engine.max_buffer_bytes,
            )
            return (
                f"Error: max_output_bytes ({max_output_bytes:,}) exceeds the configured "
                f"buffer limit ({engine.max_buffer_bytes:,})."
            )
        output_limit = engine.max_buffer_bytes if max_output_bytes is None else max_output_bytes
        output, exit_code, truncated, original_byte_size, timed_out = run_command_bounded(
            command, cwd, output_limit, timeout_seconds
        )
        
        cap = engine.ingest(
            output,
            label=f"cmd: {label}",
            content_type=content_type,
            truncated=truncated,
            original_byte_size=original_byte_size if truncated else None,
            command_exit_code=exit_code,
            timed_out=timed_out,
        )
        summary = engine.get_summary(cap.capture_id)
        
        if timed_out:
            status_str = f"TIMED OUT after {timeout_seconds:g}s"
        else:
            status_str = "SUCCESS" if exit_code == 0 else f"FAILED (Exit Code {exit_code})"
        truncation_str = ""
        if summary.get("truncated"):
            truncation_str = f"\nOutput: truncated from {summary['original_byte_size']:,} bytes\n"
        
        if summary.get("content_type") == "diff" and summary.get("file_map"):
            diff_stats = summary.get("diff_stats", "")
            file_map = summary.get("file_map", "")
            signals_str = summary.get("signals_summary", "None (Clean patch)")
            return (
                f"Command: `{command}`\n"
                f"Status: {status_str} | Type: Unified Diff ({diff_stats})\n"
                f"Captured ID: `{cap.capture_id}` ({cap.line_count:,} lines, {cap.byte_size:,} bytes)\n"
                f"{truncation_str}"
                f"Detected Signals: {signals_str}\n\n"
                f"--- Modified Files Map ---\n"
                f"{file_map}\n\n"
                f"Query details using `search_capture(query='...', capture_id='{cap.capture_id}')` or slice lines with `get_capture_slice(start_line=..., end_line=...)`."
            )
            
        signals_str = summary.get("signals_summary", "None detected")
        return (
            f"Command: `{command}`\n"
            f"Status: {status_str}\n"
            f"Captured ID: `{cap.capture_id}` ({cap.line_count:,} lines, {cap.byte_size:,} bytes)\n"
            f"{truncation_str}"
            f"Detected Signals: {signals_str}\n\n"
            f"--- Head (First 5 lines) ---\n{summary['head_preview']}\n\n"
            f"--- Tail (Last 5 lines) ---\n{summary['tail_preview']}\n\n"
            f"Query details using `search_capture(query='...', capture_id='{cap.capture_id}')`."
        )
    except Exception as e:
        log_event(LOGGER, logging.ERROR, "command_execution_failed", error_type=type(e).__name__)
        return f"Error executing command: {str(e)}"


def _consolidated_jsonl(
    capture_ids: Optional[List[str]],
    max_captures: int,
    max_bytes: int,
) -> Dict[str, Any]:
    """Compatibility wrapper around the engine's consolidation implementation."""
    return engine.consolidate(capture_ids, max_captures, max_bytes)


@mcp.tool()
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
    their original IDs and the response reports how many records were omitted.
    """
    try:
        result = engine.consolidate(capture_ids, max_captures, max_bytes)
        capture = engine.ingest(
            result["content"],
            label=label,
            content_type="text",
        )
        return json.dumps({
            "status": "ok",
            "capture_id": capture.capture_id,
            "label": capture.label,
            "source_capture_ids": result["source_capture_ids"],
            "requested_capture_count": result["requested_capture_count"],
            "selected_capture_count": result["selected_capture_count"],
            "source_count": result["source_count"],
            "record_count": result["record_count"],
            "omitted_record_count": result["omitted_record_count"],
            "missing_capture_ids": result["missing_capture_ids"],
            "byte_size": capture.byte_size,
            "next_steps": {
                "search": f"search_capture(query='...', capture_id='{capture.capture_id}')",
                "slice": f"get_capture_slice(start_line=..., end_line=..., capture_id='{capture.capture_id}')",
                "source_detail": "Use the original source capture IDs for complete omitted records.",
            },
        }, ensure_ascii=False)
    except Exception as exc:
        log_event(LOGGER, logging.ERROR, "capture_consolidation_failed", error_type=type(exc).__name__)
        return f"Error consolidating captures: {exc}"


@mcp.tool()
@_instrument_tool("search_capture")
def search_capture(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3
) -> str:
    """
    Searches the captured command output using BM25, Semantic embedding, or Hybrid (RRF) ranking.
    When opt-in semantic prefetch is enabled, semantic and hybrid searches wait
    for the relevant background index job; failed jobs fall back to synchronous
    lazy indexing.
    
    Args:
        query: Search keywords or natural language question (e.g. 'auth failure', 'ECONNREFUSED', 'why did the build fail?').
        mode: Search mode - 'hybrid' (recommended, lexically weighted BM25 + Semantic), 'bm25' (keyword terms), or 'semantic' (vector concepts).
        capture_id: The capture ID to query (defaults to 'latest').
        top_k: Number of matching snippets to return (default: 5).
        context_lines: Number of surrounding lines of context to include with each match (default: 3; must be non-negative).
    """
    res = engine.search(
        query=query,
        mode=mode,
        capture_id=capture_id,
        top_k=top_k,
        context_lines=context_lines
    )
    
    if res.get("status") == "error":
        return f"Search Error: {res.get('message')}"
        
    matches = res.get("matches", [])
    METRICS.record_result_count("search_capture", len(matches))
    if not matches:
        return f"No matches found for '{query}' in capture '{res.get('capture_id')}' ({res.get('label')})."
        
    out = [
        f"Search Results for: \"{query}\" [Mode: {res['mode']}]",
        f"Capture: `{res['capture_id']}` ({res['label']}, {res['total_lines']} total lines)",
        f"Found {len(matches)} relevant section(s):\n"
    ]
    
    for i, m in enumerate(matches, 1):
        out.append(f"### Match #{i} (Score: {m['score']}, Range: {m['matched_range']}, Context: {m['context_range']})")
        out.append("```text")
        out.append(m["snippet"])
        out.append("```\n")
        
    return "\n".join(out)


@mcp.tool()
@_instrument_tool("get_capture_slice")
def get_capture_slice(start_line: int, end_line: int, capture_id: str = "latest") -> str:
    """
    Fetches an exact range of lines (1-indexed) from a capture to inspect full context around a match.
    """
    res = engine.get_slice(start_line, end_line, capture_id=capture_id)
    if res.get("status") == "error":
        return f"Error: {res.get('message')}"
        
    return (
        f"Capture: `{res['capture_id']}` ({res['label']}) | Lines {res['start_line']} to {res['end_line']} of {res['total_lines']}\n"
        f"```text\n{res['content']}\n```"
    )


@mcp.tool()
@_instrument_tool("get_capture_summary")
def get_capture_summary(capture_id: str = "latest") -> str:
    """
    Returns quick diagnostics for a capture: total lines, byte size, diff file map or error signals, and previews.
    """
    res = engine.get_summary(capture_id)
    if res.get("status") == "error":
        return f"Error: {res.get('message')}"
        
    signals = res.get("signals_summary", "None detected")
    
    if res.get("content_type") == "diff" and res.get("file_map"):
        return (
            f"Capture: `{res['capture_id']}` ({res['label']})\n"
            f"Type: Unified Diff ({res.get('diff_stats')})\n"
            f"Timestamp: {res['timestamp']}\n"
            f"Total Lines: {res['total_lines']:,} | Size: {res['byte_size']:,} bytes\n"
            f"Output: {'truncated from ' + format(res['original_byte_size'], ',') + ' bytes' if res.get('truncated') else 'complete'}\n"
            f"Detected Signals: {signals}\n\n"
            f"--- Modified Files Map ---\n"
            f"{res['file_map']}"
        )

    return (
        f"Capture: `{res['capture_id']}` ({res['label']})\n"
        f"Timestamp: {res['timestamp']}\n"
        f"Total Lines: {res['total_lines']:,} | Size: {res['byte_size']:,} bytes\n"
        f"Output: {'truncated from ' + format(res['original_byte_size'], ',') + ' bytes' if res.get('truncated') else 'complete'}\n"
        f"Detected Keyword Signals: {signals}\n\n"
        f"--- Head (First 5 lines) ---\n{res['head_preview']}\n\n"
        f"--- Tail (Last 5 lines) ---\n{res['tail_preview']}"
    )


@mcp.tool()
@_instrument_tool("list_captures")
def list_captures() -> str:
    """
    Lists all captures currently retained in the ephemeral ring buffer.
    """
    caps = engine.list_captures()
    if not caps:
        return "Ephemeral buffer is empty. No captures currently stored."
        
    out = ["Active Captures in Ephemeral Buffer:"]
    for c in caps:
        out.append(f"- `{c['capture_id']}`: \"{c['label']}\" | {c['total_lines']:,} lines | {c['byte_size']:,} bytes | {c['timestamp']}")
    return "\n".join(out)


@mcp.tool()
@_instrument_tool("clear_captures")
def clear_captures(capture_id: str = "all") -> str:
    """
    Clears all or a specific capture from the ephemeral buffer to free memory.
    """
    return engine.clear(capture_id)


@mcp.tool()
@_instrument_tool("get_buffer_stats")
def get_buffer_stats() -> str:
    """Returns aggregate capture, accounting, prefetch, and process RSS metrics."""
    stats = engine.get_buffer_stats()
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
    cache_line = f"Embedding cache: {stats['embedding_cache_dir'] or 'default'}"
    result = (
        f"Captures: {stats['capture_count']}/{stats['max_captures']}\n"
        f"Content bytes: {stats['total_bytes']:,}/{stats['max_buffer_bytes']:,}\n"
        f"Lines: {stats['total_lines']:,}\n"
        f"Chunks: {stats['total_chunks']:,}\n"
        f"{model_line}\n"
        f"{cache_line}\n"
        f"Embedding bytes: {stats['embedding_bytes']:,}\n"
        f"Semantic prefetch: {'enabled' if stats.get('semantic_prefetch_enabled', False) else 'disabled'} "
        f"({stats.get('semantic_prefetch_pending', 0)} pending, {stats.get('semantic_prefetch_failed', 0)} failed)\n"
        f"Accounted bytes: {stats['accounted_bytes']:,}\n"
        f"{rss_line}\n"
        f"{unaccounted_line}"
    )
    if METRICS.enabled:
        result += f"\nLocal metrics: {json.dumps(METRICS.snapshot(), sort_keys=True)}"
    return result


@mcp.tool()
@_instrument_tool("get_runtime_diagnostics")
def get_runtime_diagnostics() -> str:
    """Returns opt-in runtime metadata without exposing captured content."""
    stats = engine.get_buffer_stats()
    installed_version = _runtime_package_version()

    if os.environ.get("EPHEMERAL_SOCKET_PATH"):
        socket_mode = "explicit path"
    elif os.environ.get("EPHEMERAL_SESSION_ID"):
        socket_mode = "session-derived path"
    else:
        socket_mode = "shared default path"

    uptime_seconds = max(0, int(time.time() - SERVER_STARTED_AT))
    rss = stats["process_rss_bytes"]
    unaccounted = stats["unaccounted_rss_bytes"]
    lines = [
        "Runtime diagnostics (content-free):",
        f"Package version: {installed_version}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Uptime: {uptime_seconds:,} seconds",
        f"Socket mode: {socket_mode}",
        f"Socket path: {SOCKET_PATH}",
        f"Session ID configured: {'yes' if os.environ.get('EPHEMERAL_SESSION_ID') else 'no'}",
        f"Captures: {stats['capture_count']}/{stats['max_captures']}",
        f"Content bytes: {stats['total_bytes']:,}/{stats['max_buffer_bytes']:,}",
        f"Embedding model: {stats['embedding_model']} ({'loaded' if stats['embedding_model_loaded'] else 'not loaded'})",
        f"Embedding cache: {stats['embedding_cache_dir'] or 'default'}",
        f"Semantic prefetch: {'enabled' if stats.get('semantic_prefetch_enabled', False) else 'disabled'} "
        f"({stats.get('semantic_prefetch_pending', 0)} pending, {stats.get('semantic_prefetch_failed', 0)} failed)",
        f"Process RSS: {'unavailable' if rss is None else f'{rss:,} bytes'}",
        f"Unaccounted RSS: {'unavailable' if unaccounted is None else f'{unaccounted:,} bytes'}",
        f"Local metrics: {'enabled' if METRICS.enabled else 'disabled'}",
        "Captured content, labels, and command arguments are not included.",
    ]
    if METRICS.enabled:
        lines.append(f"Metrics summary: {json.dumps(METRICS.snapshot(), sort_keys=True)}")
    return "\n".join(lines)


# --- Unix Domain Socket IPC for CLI piping (ephbuf) ---

async def _read_socket_payload(reader: asyncio.StreamReader, read_limit: int) -> bytes:
    """Read one EOF-delimited request while enforcing the payload limit."""
    chunks = []
    payload_bytes = 0
    while True:
        chunk = await reader.read(min(65536, read_limit - payload_bytes + 1))
        if not chunk:
            break
        chunks.append(chunk)
        payload_bytes += len(chunk)
        if payload_bytes >= read_limit:
            log_event(
                LOGGER,
                logging.WARNING,
                "socket_payload_limit_rejected",
                payload_bytes=payload_bytes,
                max_payload_bytes=read_limit,
            )
            raise ValueError(f"CLI payload exceeds the {engine.max_buffer_bytes:,}-byte capture limit")
    return b"".join(chunks)


def handle_socket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    async def _handle():
        try:
            # Read the EOF-delimited payload completely before decoding it.
            read_limit = engine.max_buffer_bytes + SOCKET_PAYLOAD_OVERHEAD
            data = await _read_socket_payload(reader, read_limit)
            if not data:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
                label = payload.get("label", "CLI pipe")
                text = payload.get("text", "")
                content_type = payload.get("content_type", "auto")
                truncated = bool(payload.get("truncated", False))
                original_byte_size = payload.get("original_byte_size")
                command_exit_code = payload.get("command_exit_code")
                timed_out = bool(payload.get("timed_out", False))
            except Exception:
                label = "CLI pipe"
                text = data.decode("utf-8", errors="replace")
                content_type = "auto"
                truncated = False
                original_byte_size = None
                command_exit_code = None
                timed_out = False

            cap = await to_thread(
                engine.ingest,
                text,
                label=label,
                content_type=content_type,
                truncated=truncated,
                original_byte_size=original_byte_size,
                command_exit_code=command_exit_code,
                timed_out=timed_out,
            )
            resp = {
                "status": "ok",
                "capture_id": cap.capture_id,
                "label": cap.label,
                "line_count": cap.line_count,
                "byte_size": cap.byte_size
            }
            writer.write(json.dumps(resp).encode("utf-8"))
            await writer.drain()
        except Exception as e:
            log_event(LOGGER, logging.ERROR, "socket_client_failed", error_type=type(e).__name__)
            err_resp = {"status": "error", "message": str(e)}
            writer.write(json.dumps(err_resp).encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    # Return the task as well as scheduling it so embedders and tests can
    # await completion when they need deterministic cleanup.
    return asyncio.create_task(_handle())


def run_socket_server():
    """Runs a Unix domain socket server in a separate thread so CLI tools can pipe to it."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if socket_isolation_required() and not socket_isolation_configured():
            raise RuntimeError(
                "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
            )
        if os.path.lexists(SOCKET_PATH):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(SOCKET_PATH)
            except ConnectionRefusedError:
                # No listener accepted the connection, so this is a stale socket.
                try:
                    os.unlink(SOCKET_PATH)
                except FileNotFoundError:
                    pass
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
            server = await asyncio.start_unix_server(handle_socket_client, path=SOCKET_PATH)
            os.chmod(SOCKET_PATH, 0o600)
            async with server:
                await server.serve_forever()

        loop.run_until_complete(_main())
    except Exception as e:
        log_event(LOGGER, logging.ERROR, "socket_server_failed", error_type=type(e).__name__)
        LOGGER.exception("socket_server_exception")
        # Preserve the established stderr diagnostic for callers and tests
        # that capture the server's direct error stream.
        print(f"Socket server error: {e}", file=sys.stderr)
    finally:
        loop.close()


# Start IPC socket background listener thread
socket_thread = threading.Thread(target=run_socket_server, daemon=True)
# Unit tests can disable the listener because they exercise the handler and
# startup paths directly; real server and end-to-end processes leave it on.
if os.environ.get("EPHEMERAL_DISABLE_SOCKET_SERVER") != "1":
    socket_thread.start()


if __name__ == "__main__":
    if socket_isolation_required() and not socket_isolation_configured():
        raise SystemExit(
            "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
        )
    mcp.run()
