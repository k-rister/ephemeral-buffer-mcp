#!/usr/bin/env bash
# shellcheck shell=bash
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="$dir/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -x "$dir/.venv/bin/python" ]; then
    exec "$dir/.venv/bin/python" -m ephemeral_buffer_mcp.server
fi
if [ -x "$dir/venv/bin/python" ]; then
    exec "$dir/venv/bin/python" -m ephemeral_buffer_mcp.server
fi
if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1 || [ -x "$PYTHON" ]; then
        exec "$PYTHON" -m ephemeral_buffer_mcp.server
    fi
    echo "Configured PYTHON interpreter was not found or is not executable: $PYTHON" >&2
    exit 127
fi
if command -v python3 >/dev/null 2>&1; then
    exec "$(command -v python3)" -m ephemeral_buffer_mcp.server
fi
echo "No Python interpreter found; create .venv or set PYTHON to an executable interpreter." >&2
exit 127
