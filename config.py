"""Environment-backed server configuration helpers."""

import os
import sys
import tempfile
import hashlib
import math


DEFAULT_MAX_CAPTURES = 25
DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_INDEXED_CHUNKS = 32768
DEFAULT_MAX_OUTPUT_BYTES = DEFAULT_MAX_BUFFER_BYTES
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SEMANTIC_PREFETCH_WORKERS = 1
DEFAULT_SOCKET_PATH = os.path.join(tempfile.gettempdir(), "ephemeral_buffer.sock")
DEFAULT_SOCKET_TIMEOUT_SECONDS = 10.0
SESSION_SOCKET_PREFIX = "ephemeral_buffer-"


def socket_isolation_configured() -> bool:
    """Return whether this process has an explicit session/socket identity."""
    return bool(os.environ.get("EPHEMERAL_SOCKET_PATH") or os.environ.get("EPHEMERAL_SESSION_ID"))


def socket_isolation_required() -> bool:
    """Return whether shared legacy socket fallback is forbidden."""
    return os.environ.get("EPHEMERAL_REQUIRE_ISOLATION", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def socket_timeout_seconds() -> float:
    """Return the bounded CLI socket operation timeout in seconds."""
    name = "EPHEMERAL_SOCKET_TIMEOUT_SECONDS"
    value = os.environ.get(name)
    if value is None:
        return DEFAULT_SOCKET_TIMEOUT_SECONDS
    try:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        print(
            f"Ignoring invalid {name}={value!r}; using {DEFAULT_SOCKET_TIMEOUT_SECONDS}",
            file=sys.stderr,
        )
        return DEFAULT_SOCKET_TIMEOUT_SECONDS


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


def embedding_model_name() -> str:
    """Return the configured FastEmbed model name."""
    return os.environ.get("EPHEMERAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_cache_dir() -> str | None:
    """Return the optional FastEmbed model cache directory."""
    return os.environ.get("EPHEMERAL_FASTEMBED_CACHE_DIR") or None


def semantic_prefetch_enabled() -> bool:
    """Return whether post-ingestion semantic indexing is enabled."""
    return os.environ.get("EPHEMERAL_SEMANTIC_PREFETCH", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def semantic_prefetch_workers() -> int:
    """Return the bounded number of semantic prefetch workers."""
    return positive_int_env("EPHEMERAL_SEMANTIC_PREFETCH_WORKERS", DEFAULT_SEMANTIC_PREFETCH_WORKERS)


def max_indexed_chunks() -> int:
    """Return the maximum total number of indexed chunks retained in memory."""
    return positive_int_env("EPHEMERAL_MAX_INDEXED_CHUNKS", DEFAULT_MAX_INDEXED_CHUNKS)


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
