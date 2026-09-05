#!/usr/bin/env python3
"""
CLI Helper tool for piping command output into Ephemeral Buffer MCP Server.
Usage:
  # Pipe stdout/stderr directly:
  pytest -v 2>&1 | ephbuf --label "pytest run"

  # Force unified-diff parsing when automatic detection is ambiguous:
  git diff HEAD~3 | ephbuf --label "feature diff" --type diff
  
  # Or wrap a command execution:
  ephbuf --label "build" -- make all

  --type accepts: auto (default), diff, log, or text.
"""

import sys
import os
import socket
import json
import argparse
from capture_utils import DEFAULT_MAX_OUTPUT_BYTES, bound_chunks, run_command_bounded
from config import positive_int_env, socket_path

SOCKET_PATH = socket_path()


def send_to_mcp(
    text: str,
    label: str = "",
    content_type: str = "auto",
    truncated: bool = False,
    original_byte_size: int = 0,
) -> dict:
    if not os.path.exists(SOCKET_PATH):
        return {
            "status": "error",
            "message": f"MCP server socket not found at {SOCKET_PATH}. Is the ephemeral-buffer MCP server running?"
        }

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        
        payload = json.dumps({
            "label": label,
            "text": text,
            "content_type": content_type,
            "truncated": truncated,
            "original_byte_size": original_byte_size if truncated else None,
        }).encode("utf-8")
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        
        resp_data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp_data += chunk
            
        sock.close()
        return json.loads(resp_data.decode("utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to communicate with MCP server: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Pipe output into Ephemeral Buffer MCP server for hybrid search and agent analysis."
    )
    parser.add_argument("--label", "-l", default="", help="Optional descriptive label for this capture")
    parser.add_argument("--type", "-t", choices=["auto", "diff", "log", "text"], default="auto", help="Optional content type hint (default: auto)")
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=positive_int_env("EPHEMERAL_MAX_BUFFER_BYTES", DEFAULT_MAX_OUTPUT_BYTES),
        help="Maximum output retained (default: EPHEMERAL_MAX_BUFFER_BYTES or 50 MiB)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Maximum runtime for a wrapped command; timed-out commands exit with status 124",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional command to execute and capture")

    args = parser.parse_args()

    # If a command was passed after '--'
    if args.command:
        cmd_list = args.command
        if cmd_list and cmd_list[0] == "--":
            cmd_list = cmd_list[1:]
            
        cmd_str = " ".join(cmd_list)
        label = args.label or cmd_str
        print(f"[ephbuf] Executing: {cmd_str}")
        
        try:
            output, exit_code, truncated, original_byte_size, timed_out = run_command_bounded(
                cmd_str, None, args.max_output_bytes, args.timeout_seconds
            )
        except ValueError as e:
            parser.error(str(e))
        # Also print output locally so user can see it if desired
        sys.stdout.write(output)
        sys.stdout.flush()
        
        res = send_to_mcp(
            output,
            label=label,
            content_type=args.type,
            truncated=truncated,
            original_byte_size=original_byte_size,
        )
        if res.get("status") == "ok":
            print(f"\n[ephbuf] Successfully captured {res['line_count']:,} lines into buffer `{res['capture_id']}` ({res['label']})", file=sys.stderr)
        else:
            print(f"\n[ephbuf] Warning: {res.get('message')}", file=sys.stderr)
        if timed_out:
            print(f"\n[ephbuf] Command timed out after {args.timeout_seconds:g}s", file=sys.stderr)
        sys.exit(exit_code)

    # Otherwise read from stdin (piped input)
    if not sys.stdin.isatty():
        input_stream = getattr(sys.stdin, "buffer", sys.stdin)
        if input_stream is sys.stdin:
            input_chunks = (chunk.encode("utf-8") for chunk in iter(lambda: sys.stdin.read(65536), ""))
        else:
            input_chunks = iter(lambda: input_stream.read(65536), b"")
        input_text, truncated, original_byte_size = bound_chunks(input_chunks, args.max_output_bytes)
        label = args.label or "Piped STDIN"
        res = send_to_mcp(
            input_text,
            label=label,
            content_type=args.type,
            truncated=truncated,
            original_byte_size=original_byte_size,
        )
        if res.get("status") == "ok":
            print(f"[ephbuf] Successfully captured {res['line_count']:,} lines into buffer `{res['capture_id']}` ({res['label']})", file=sys.stderr)
        else:
            print(f"[ephbuf] Warning: {res.get('message')}", file=sys.stderr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
