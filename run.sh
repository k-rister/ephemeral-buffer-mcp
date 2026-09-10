#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -x "$DIR/.venv/bin/python" ]; then
    exec "$DIR/.venv/bin/python" "$DIR/server.py"
fi
if [ -x "$DIR/venv/bin/python" ]; then
    exec "$DIR/venv/bin/python" "$DIR/server.py"
fi
if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1 || [ -x "$PYTHON" ]; then
        exec "$PYTHON" "$DIR/server.py"
    fi
    echo "Configured PYTHON interpreter was not found or is not executable: $PYTHON" >&2
    exit 127
fi
if command -v python3 >/dev/null 2>&1; then
    exec "$(command -v python3)" "$DIR/server.py"
fi
echo "No Python interpreter found; create .venv or set PYTHON to an executable interpreter." >&2
exit 127
