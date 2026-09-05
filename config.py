"""Environment-backed server configuration helpers."""

import os
import sys
import tempfile
import hashlib


DEFAULT_MAX_CAPTURES = 25
DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = DEFAULT_MAX_BUFFER_BYTES
DEFAULT_SOCKET_PATH = os.path.join(tempfile.gettempdir(), "ephemeral_buffer.sock")
SESSION_SOCKET_PREFIX = "ephemeral_buffer-"


def socket_path() -> str:
    """Return the shared server/CLI socket path for the current session."""
    configured_path = os.environ.get("EPHEMERAL_SOCKET_PATH")
    if configured_path:
        return configured_path

    session_id = os.environ.get("EPHEMERAL_SESSION_ID")
    if session_id:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        return os.path.join(tempfile.gettempdir(), f"{SESSION_SOCKET_PREFIX}{digest}.sock")

    return DEFAULT_SOCKET_PATH


def positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment setting or its safe default."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 1:
            raise ValueError
        return parsed
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}; using {default}", file=sys.stderr)
        return default
