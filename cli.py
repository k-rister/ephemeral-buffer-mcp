#!/usr/bin/env python3
"""
CLI Helper tool for piping command output into Ephemeral Buffer MCP Server.
Usage:
  # Pipe stdout/stderr directly:
  pytest -v 2>&1 | agy-cap --label "pytest run"
  
  # Or wrap a command execution:
  agy-cap --label "build" -- make all
"""

import sys
import os
import socket
import json
import argparse
import subprocess

SOCKET_PATH = "/tmp/ephemeral_buffer.sock"


def send_to_mcp(text: str, label: str = "") -> dict:
    if not os.path.exists(SOCKET_PATH):
        return {
            "status": "error",
            "message": f"MCP server socket not found at {SOCKET_PATH}. Is the ephemeral-buffer MCP server running?"
        }

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        
        payload = json.dumps({"label": label, "text": text}).encode("utf-8")
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
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional command to execute and capture")

    args = parser.parse_args()

    # If a command was passed after '--'
    if args.command:
        cmd_list = args.command
        if cmd_list and cmd_list[0] == "--":
            cmd_list = cmd_list[1:]
            
        cmd_str = " ".join(cmd_list)
        label = args.label or cmd_str
        print(f"[agy-cap] Executing: {cmd_str}")
        
        proc = subprocess.run(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace"
        )
        output = proc.stdout
        # Also print output locally so user can see it if desired
        sys.stdout.write(output)
        sys.stdout.flush()
        
        res = send_to_mcp(output, label=label)
        if res.get("status") == "ok":
            print(f"\n[agy-cap] Successfully captured {res['line_count']:,} lines into buffer `{res['capture_id']}` ({res['label']})", file=sys.stderr)
        else:
            print(f"\n[agy-cap] Warning: {res.get('message')}", file=sys.stderr)
        sys.exit(proc.returncode)

    # Otherwise read from stdin (piped input)
    if not sys.stdin.isatty():
        input_text = sys.stdin.read()
        label = args.label or "Piped STDIN"
        res = send_to_mcp(input_text, label=label)
        if res.get("status") == "ok":
            print(f"[agy-cap] Successfully captured {res['line_count']:,} lines into buffer `{res['capture_id']}` ({res['label']})", file=sys.stderr)
        else:
            print(f"[agy-cap] Warning: {res.get('message')}", file=sys.stderr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
