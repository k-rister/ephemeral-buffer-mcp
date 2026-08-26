"""
MCP Server for Ephemeral Command Output Hybrid Search.
Supports stdio MCP protocol and Unix Domain Socket IPC for CLI piping.
"""

import os
import sys
import json
import asyncio
import subprocess
import threading
from typing import Optional
from mcp.server.fastmcp import FastMCP
from engine import EphemeralEngine

SOCKET_PATH = "/tmp/ephemeral_buffer.sock"

# Initialize FastMCP
mcp = FastMCP("ephemeral-buffer")
engine = EphemeralEngine()


# --- MCP Tools ---

@mcp.tool()
def capture_text(content: str, label: str = "") -> str:
    """
    Ingests raw text output directly into the ephemeral search index.
    Returns capture metadata (ID, line count, byte size).
    """
    cap = engine.ingest(content, label=label)
    return (
        f"Captured into ID '{cap.capture_id}' ({cap.label})\n"
        f"- Lines: {cap.line_count}\n"
        f"- Bytes: {cap.byte_size:,}\n"
        f"- Chunks: {len(cap.chunks)}\n"
        f"Use `search_capture` with capture_id='{cap.capture_id}' or 'latest' to query."
    )


@mcp.tool()
def capture_file(file_path: str, label: str = "") -> str:
    """
    Reads a file or log output from disk and ingests it into the ephemeral search index.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if not label:
            label = os.path.basename(file_path)
        return capture_text(content, label=label)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@mcp.tool()
def execute_and_capture(command: str, cwd: Optional[str] = None, label: str = "") -> str:
    """
    Runs a shell command in the background, captures stdout and stderr, indexes it,
    and returns a concise summary (exit code, line count, error signals, preview)
    WITHOUT flooding your prompt context with thousands of lines.
    """
    if not label:
        label = command[:40] + ("..." if len(command) > 40 else "")
        
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace"
        )
        output = proc.stdout
        exit_code = proc.returncode
        
        cap = engine.ingest(output, label=f"cmd: {label}")
        summary = engine.get_summary(cap.capture_id)
        
        status_str = "SUCCESS" if exit_code == 0 else f"FAILED (Exit Code {exit_code})"
        
        signals_str = ", ".join(f"{k}: {v}" for k, v in summary["keyword_signals"].items())
        if not signals_str:
            signals_str = "None detected"
            
        return (
            f"Command: `{command}`\n"
            f"Status: {status_str}\n"
            f"Captured ID: `{cap.capture_id}` ({cap.line_count} lines, {cap.byte_size:,} bytes)\n"
            f"Detected Signals: {signals_str}\n\n"
            f"--- Head (First 5 lines) ---\n{summary['head_preview']}\n\n"
            f"--- Tail (Last 5 lines) ---\n{summary['tail_preview']}\n\n"
            f"Query details using `search_capture(query='...', capture_id='{cap.capture_id}')`."
        )
    except Exception as e:
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
    Returns quick diagnostics for a capture: total lines, byte size, detected error patterns, and head/tail previews.
    """
    res = engine.get_summary(capture_id)
    if res.get("status") == "error":
        return f"Error: {res.get('message')}"
        
    signals = ", ".join(f"{k}: {v}" for k, v in res["keyword_signals"].items()) or "None"
    return (
        f"Capture: `{res['capture_id']}` ({res['label']})\n"
        f"Timestamp: {res['timestamp']}\n"
        f"Total Lines: {res['total_lines']:,} | Size: {res['byte_size']:,} bytes\n"
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


# --- Unix Domain Socket IPC for CLI piping (agy-cap) ---

def handle_socket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    async def _handle():
        try:
            # Read payload (simple json line or framed message)
            data = await reader.read()
            if not data:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
                label = payload.get("label", "CLI pipe")
                text = payload.get("text", "")
            except Exception:
                label = "CLI pipe"
                text = data.decode("utf-8", errors="replace")

            cap = engine.ingest(text, label=label)
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
            err_resp = {"status": "error", "message": str(e)}
            writer.write(json.dumps(err_resp).encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    asyncio.create_task(_handle())


def run_socket_server():
    """Runs a Unix domain socket server in a separate thread so CLI tools can pipe to it."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    async def _main():
        server = await asyncio.start_unix_server(handle_socket_client, path=SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o777)
        async with server:
            await server.serve_forever()

    try:
        loop.run_until_complete(_main())
    except Exception as e:
        print(f"Socket server error: {e}", file=sys.stderr)


# Start IPC socket background listener thread
socket_thread = threading.Thread(target=run_socket_server, daemon=True)
socket_thread.start()


if __name__ == "__main__":
    mcp.run()
