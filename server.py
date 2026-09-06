"""
MCP Server for Ephemeral Command Output Hybrid Search.
Supports stdio MCP protocol and Unix Domain Socket IPC for CLI piping.
"""

import os
import sys
import json
import asyncio
import socket
import threading
import logging
import platform
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from asyncio import to_thread
from typing import Optional
from config import positive_int_env, socket_path
from mcp.server.fastmcp import FastMCP
from engine import (
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_CAPTURES,
    EphemeralEngine,
)
from capture_utils import read_file_bounded, run_command_bounded
from logging_utils import get_logger, log_event

SOCKET_PATH = socket_path()
SOCKET_PAYLOAD_OVERHEAD = 64 * 1024
SERVER_STARTED_AT = time.time()
LOGGER = get_logger("server")

# Initialize FastMCP
mcp = FastMCP("ephemeral-buffer")


engine = EphemeralEngine(
    max_captures=positive_int_env("EPHEMERAL_MAX_CAPTURES", DEFAULT_MAX_CAPTURES),
    max_buffer_bytes=positive_int_env("EPHEMERAL_MAX_BUFFER_BYTES", DEFAULT_MAX_BUFFER_BYTES),
)


# --- MCP Tools ---

@mcp.tool()
def capture_text(content: str, label: str = "", content_type: str = "auto") -> str:
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
def capture_file(
    file_path: str,
    label: str = "",
    content_type: str = "auto",
    max_bytes: Optional[int] = None
) -> str:
    """
    Reads a file or log output from disk and ingests it into the ephemeral search index.
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
        return capture_text(content, label=label, content_type=content_type)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@mcp.tool()
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
    
    Args:
        command: Shell command line to execute.
        cwd: Optional working directory for command execution.
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


@mcp.tool()
def search_capture(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3
) -> str:
    """
    Searches the captured command output using BM25, Semantic embedding, or Hybrid (RRF) ranking.
    
    Args:
        query: Search keywords or natural language question (e.g. 'auth failure', 'ECONNREFUSED', 'why did the build fail?').
        mode: Search mode - 'hybrid' (recommended, combines BM25 + Semantic), 'bm25' (exact keywords/errors), or 'semantic' (vector concepts).
        capture_id: The capture ID to query (defaults to 'latest').
        top_k: Number of matching snippets to return (default: 5).
        context_lines: Number of surrounding lines of context to include with each match (default: 3).
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
def clear_captures(capture_id: str = "all") -> str:
    """
    Clears all or a specific capture from the ephemeral buffer to free memory.
    """
    return engine.clear(capture_id)


@mcp.tool()
def get_buffer_stats() -> str:
    """Returns aggregate capture, accounting, and process RSS metrics."""
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
    return (
        f"Captures: {stats['capture_count']}/{stats['max_captures']}\n"
        f"Content bytes: {stats['total_bytes']:,}/{stats['max_buffer_bytes']:,}\n"
        f"Lines: {stats['total_lines']:,}\n"
        f"Chunks: {stats['total_chunks']:,}\n"
        f"{model_line}\n"
        f"{cache_line}\n"
        f"Embedding bytes: {stats['embedding_bytes']:,}\n"
        f"Accounted bytes: {stats['accounted_bytes']:,}\n"
        f"{rss_line}\n"
        f"{unaccounted_line}"
    )


@mcp.tool()
def get_runtime_diagnostics() -> str:
    """Returns opt-in runtime metadata without exposing captured content."""
    stats = engine.get_buffer_stats()
    try:
        installed_version = package_version("ephemeral-buffer-mcp")
    except PackageNotFoundError:
        installed_version = "source checkout"

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
        f"Process RSS: {'unavailable' if rss is None else f'{rss:,} bytes'}",
        f"Unaccounted RSS: {'unavailable' if unaccounted is None else f'{unaccounted:,} bytes'}",
        "Captured content, labels, and command arguments are not included.",
    ]
    return "\n".join(lines)


# --- Unix Domain Socket IPC for CLI piping (ephbuf) ---

def handle_socket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    async def _handle():
        try:
            # Read payload (simple json line or framed message)
            read_limit = engine.max_buffer_bytes + SOCKET_PAYLOAD_OVERHEAD
            data = await reader.read(read_limit)
            if not data:
                return
            if len(data) >= read_limit:
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "socket_payload_limit_rejected",
                    payload_bytes=len(data),
                    max_payload_bytes=read_limit,
                )
                raise ValueError(f"CLI payload exceeds the {engine.max_buffer_bytes:,}-byte capture limit")
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
    mcp.run()
