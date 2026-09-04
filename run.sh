#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -x "$DIR/venv/bin/python" ]; then
    exec "$DIR/venv/bin/python" "$DIR/server.py"
fi
exec "${PYTHON:-python3}" "$DIR/server.py"
