"""Environment-backed server configuration helpers."""

import dataclasses
import hashlib
import math
import os
import shutil
import stat
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar


DEFAULT_MAX_CAPTURES = 25
DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_INDEXED_CHUNKS = 32768
DEFAULT_MAX_OUTPUT_BYTES = DEFAULT_MAX_BUFFER_BYTES
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_MAX_BATCH_TOKENS = 4096
DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES = 4 * 1024 * 1024
# The FastEmbed catalogue entry for bge-small-en-v1.5 downloads a reduced-precision
# ONNX file whose matrix kernels do not parallelize on common CPU hosts.  The
# engine registers the upstream fp32 export under an alias and uses it by
# default; the catalogue file stays selectable by its catalogue name.
BGE_SMALL_HF_REPO = "BAAI/bge-small-en-v1.5"
CATALOGUE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FP32_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5-fp32"
DEFAULT_EMBEDDING_MODEL = FP32_EMBEDDING_MODEL
DEFAULT_SEMANTIC_PREFETCH_WORKERS = 1
# Hybrid search waits at most this long for a capture's semantic index before
# answering with lexical results and a pending marker; ``inf`` waits forever.
DEFAULT_SEMANTIC_WAIT_SECONDS = 10.0
DEFAULT_SEMANTIC_CHUNK_LINES = 8
DEFAULT_SEMANTIC_CHUNK_BYTES = 1024
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 0
DEFAULT_SOCKET_PATH = os.path.join(tempfile.gettempdir(), "ephemeral_buffer.sock")
DEFAULT_SOCKET_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ACTIVE_TOOL_WORK = 8
DEFAULT_MAX_QUEUED_TOOL_WORK = 16
DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS = 4
DEFAULT_MAX_QUEUED_SOCKET_CLIENTS = 8
DEFAULT_SOCKET_STARTUP_TIMEOUT_SECONDS = 5
DEFAULT_SHUTDOWN_GRACE_SECONDS = 10.0
SESSION_SOCKET_PREFIX = "ephemeral_buffer-"
EXECUTION_STATE_SESSION_PREFIX = "ephemeral_buffer_executions-"
EXECUTION_STATE_SOCKET_PREFIX = "ephemeral_buffer_executions-socket-"
DEFAULT_EXECUTION_STATE_QUOTA_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_EXECUTION_CHECKPOINT_RESERVE_BYTES = 128 * 1024 * 1024


def new_default_execution_state_dir() -> str:
    """Return a unique private-state path without creating it."""
    return os.path.join(
        tempfile.gettempdir(),
        f"{EXECUTION_STATE_SESSION_PREFIX}{uuid.uuid4().hex}",
    )


DEFAULT_EXECUTION_STATE_DIR = new_default_execution_state_dir()

T = TypeVar("T")


@dataclass(frozen=True)
class ConfigSetting(Generic[T]):
    """One typed configuration value and how startup resolved it."""

    name: str
    value: T
    source: str
    status: str
    invalid_value_behavior: str = "warn and use the default"
    sampling_policy: str = "startup"

    def diagnostics(self) -> dict[str, Any]:
        value: Any = self.value
        if isinstance(value, float) and not math.isfinite(value):
            value = "inf" if value > 0 else "-inf" if value < 0 else "nan"
        return {
            "name": self.name,
            "value": value,
            "source": self.source,
            "status": self.status,
            "invalid_value_behavior": self.invalid_value_behavior,
            "sampling_policy": self.sampling_policy,
        }


@dataclass(frozen=True)
class SessionDescriptor:
    """Resolved transport and durable-state identity for one process."""

    session_id: str | None
    session_source: str
    socket_path: str
    socket_source: str
    state_dir: str
    state_source: str
    durability: str
    legacy_state_dir: str | None
    isolation_configured: bool

    @property
    def session_fingerprint(self) -> str | None:
        if self.session_id is None:
            return None
        return hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:12]

    @property
    def legacy_state_transition(self) -> bool:
        return bool(
            self.legacy_state_dir
            and self.legacy_state_dir != self.state_dir
            and os.path.isdir(self.legacy_state_dir)
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "session_id_configured": self.session_id is not None,
            "session_source": self.session_source,
            "session_fingerprint": self.session_fingerprint,
            "socket_path": self.socket_path,
            "socket_source": self.socket_source,
            "state_dir": self.state_dir,
            "state_source": self.state_source,
            "durability": self.durability,
            "legacy_state_dir": self.legacy_state_dir if self.legacy_state_transition else None,
            "isolation_configured": self.isolation_configured,
        }


@dataclass(frozen=True)
class SettingsSnapshot:
    """Typed startup settings shared by the server, CLI, and launch helpers."""

    identity: SessionDescriptor
    max_captures: ConfigSetting[int]
    max_buffer_bytes: ConfigSetting[int]
    socket_timeout_seconds: ConfigSetting[float]
    shutdown_grace_seconds: ConfigSetting[float]
    socket_startup_timeout_seconds: ConfigSetting[int]
    socket_require_isolation: ConfigSetting[bool]
    allow_stdio_without_socket: ConfigSetting[bool]
    disable_socket_server: ConfigSetting[bool]
    metrics_enabled: ConfigSetting[bool]
    metrics_file: ConfigSetting[str | None]
    max_active_tool_work: ConfigSetting[int]
    max_queued_tool_work: ConfigSetting[int]
    max_active_socket_clients: ConfigSetting[int]
    max_queued_socket_clients: ConfigSetting[int]
    execution_state_quota_bytes: ConfigSetting[int]
    execution_checkpoint_reserve_bytes: ConfigSetting[int]
    embedding_model_name: ConfigSetting[str]
    embedding_cache_dir: ConfigSetting[str | None]
    embedding_threads: ConfigSetting[int | None]
    embedding_batch_size: ConfigSetting[int]
    embedding_max_batch_tokens: ConfigSetting[int]
    embedding_cpu_mem_arena_enabled: ConfigSetting[bool]
    embedding_warmup_enabled: ConfigSetting[bool]
    semantic_max_index_input_bytes: ConfigSetting[int]
    semantic_prefetch_enabled: ConfigSetting[bool]
    semantic_prefetch_workers: ConfigSetting[int]
    semantic_wait_seconds: ConfigSetting[float]
    semantic_chunk_lines: ConfigSetting[int]
    semantic_chunk_bytes: ConfigSetting[int]
    semantic_chunk_overlap: ConfigSetting[int]
    max_indexed_chunks: ConfigSetting[int]
    runtime_index_budget_adjustment_enabled: ConfigSetting[bool]

    def diagnostics(self) -> dict[str, Any]:
        settings = {
            field.name: getattr(self, field.name).diagnostics()
            for field in dataclasses.fields(self)
            if field.name != "identity"
        }
        return {
            "schema_version": 1,
            "identity": self.identity.diagnostics(),
            "settings": settings,
        }


def cleanup_execution_state_dir(path: str) -> None:
    """Remove only a private temporary execution-state directory."""
    path = os.path.abspath(path)
    temp_dir = os.path.abspath(tempfile.gettempdir())
    if (
        os.path.dirname(path) != temp_dir
        or not os.path.basename(path).startswith(EXECUTION_STATE_SESSION_PREFIX)
    ):
        return
    try:
        info = os.stat(path, follow_symlinks=False)
        get_uid = getattr(os, "getuid", None)
        if (
            not stat.S_ISDIR(info.st_mode)
            or get_uid is None
            or info.st_uid != get_uid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            return
        shutil.rmtree(path)
    except (OSError, ValueError):
        return


def cleanup_default_execution_state_dir() -> None:
    """Remove this process's private default execution state, if it exists."""
    cleanup_execution_state_dir(DEFAULT_EXECUTION_STATE_DIR)


def socket_isolation_configured() -> bool:
    """Return whether this process has an explicit session/socket identity."""
    return bool(os.environ.get("EPHEMERAL_SOCKET_PATH") or os.environ.get("EPHEMERAL_SESSION_ID"))


def socket_isolation_required() -> bool:
    """Return whether shared legacy socket fallback is forbidden."""
    return _read_bool_setting("EPHEMERAL_REQUIRE_ISOLATION", False, os.environ).value


def socket_timeout_seconds() -> float:
    """Return the bounded CLI operation and server request-read timeout."""
    return _read_float_setting(
        "EPHEMERAL_SOCKET_TIMEOUT_SECONDS",
        DEFAULT_SOCKET_TIMEOUT_SECONDS,
        os.environ,
        minimum=0.0,
        inclusive=False,
        allow_infinity=False,
    ).value


def max_active_tool_work() -> int:
    """Return the maximum MCP tool calls admitted to worker threads."""
    return _read_int_setting("EPHEMERAL_MAX_ACTIVE_TOOL_WORK", DEFAULT_MAX_ACTIVE_TOOL_WORK, os.environ, minimum=1).value


def max_queued_tool_work() -> int:
    """Return the bounded number of MCP tool calls waiting for worker capacity."""
    return _read_int_setting("EPHEMERAL_MAX_QUEUED_TOOL_WORK", DEFAULT_MAX_QUEUED_TOOL_WORK, os.environ, minimum=0).value


def max_active_socket_clients() -> int:
    """Return the maximum socket clients admitted to request processing."""
    return _read_int_setting("EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS", DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS, os.environ, minimum=1).value


def max_queued_socket_clients() -> int:
    """Return the bounded number of socket clients waiting for request capacity."""
    return _read_int_setting("EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS", DEFAULT_MAX_QUEUED_SOCKET_CLIENTS, os.environ, minimum=0).value


def socket_path() -> str:
    """Return the shared server/CLI socket path for the current session."""
    return resolve_session_descriptor(os.environ).socket_path


def execution_state_dir() -> str:
    """Return the state directory resolved from the process session descriptor."""
    return resolve_session_descriptor(os.environ).state_dir


def _identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def resolve_session_descriptor(
    environ: Mapping[str, str] | None = None,
    *,
    default_state_dir: str | None = None,
) -> SessionDescriptor:
    """Resolve socket and state identity with one shared precedence contract.

    Explicit state storage wins for persistence. Otherwise, an explicit socket
    path is the shared identity for both the socket and derived state. A
    session ID derives both paths when no explicit socket is present.
    """
    values = os.environ if environ is None else environ
    session_id = values.get("EPHEMERAL_SESSION_ID") or None
    raw_explicit_socket = values.get("EPHEMERAL_SOCKET_PATH") or None
    explicit_socket = (
        os.path.abspath(os.path.expanduser(raw_explicit_socket))
        if raw_explicit_socket
        else None
    )
    explicit_state = values.get("EPHEMERAL_EXECUTION_STATE_DIR") or None

    if explicit_socket:
        effective_socket = explicit_socket
        socket_source = "environment:EPHEMERAL_SOCKET_PATH"
    elif session_id:
        digest = _identity_digest(session_id)
        effective_socket = os.path.join(
            tempfile.gettempdir(), f"{SESSION_SOCKET_PREFIX}{digest}.sock"
        )
        socket_source = "session:EPHEMERAL_SESSION_ID"
    else:
        effective_socket = DEFAULT_SOCKET_PATH
        socket_source = "default:shared legacy socket"

    legacy_state_dir = None
    if explicit_state:
        effective_state = os.path.abspath(os.path.expanduser(explicit_state))
        state_source = "environment:EPHEMERAL_EXECUTION_STATE_DIR"
    elif explicit_socket:
        digest = _identity_digest(explicit_socket)
        effective_state = os.path.abspath(os.path.join(
            tempfile.gettempdir(), f"{EXECUTION_STATE_SOCKET_PREFIX}{digest}"
        ))
        state_source = "socket:EPHEMERAL_SOCKET_PATH"
        if session_id:
            legacy_digest = _identity_digest(session_id)
            legacy_state_dir = os.path.join(
                tempfile.gettempdir(), f"{EXECUTION_STATE_SESSION_PREFIX}{legacy_digest}"
            )
        else:
            legacy_state_dir = values.get("EPHEMERAL_LEGACY_STATE_DIR")
            if legacy_state_dir is None and raw_explicit_socket != explicit_socket:
                legacy_digest = _identity_digest(raw_explicit_socket)
                legacy_state_dir = os.path.join(
                    tempfile.gettempdir(), f"{EXECUTION_STATE_SOCKET_PREFIX}{legacy_digest}"
                )
    elif session_id:
        digest = _identity_digest(session_id)
        effective_state = os.path.abspath(os.path.join(
            tempfile.gettempdir(), f"{EXECUTION_STATE_SESSION_PREFIX}{digest}"
        ))
        state_source = "session:EPHEMERAL_SESSION_ID"
    else:
        effective_state = default_state_dir or DEFAULT_EXECUTION_STATE_DIR
        state_source = "process:private temporary directory"

    if state_source == "process:private temporary directory":
        durability = "process-private; removed on normal shutdown where secure cleanup is available"
    else:
        durability = "on-disk; retained across normal shutdowns until explicitly retired"

    descriptor = SessionDescriptor(
        session_id=session_id,
        session_source="environment:EPHEMERAL_SESSION_ID" if session_id else "not configured",
        socket_path=effective_socket,
        socket_source=socket_source,
        state_dir=effective_state,
        state_source=state_source,
        durability=durability,
        legacy_state_dir=legacy_state_dir,
        isolation_configured=bool(explicit_socket or session_id),
    )
    return descriptor


def execution_state_quota_bytes() -> int:
    """Return the aggregate on-disk envelope for durable execution state."""
    return _read_int_setting("EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES", DEFAULT_EXECUTION_STATE_QUOTA_BYTES, os.environ, minimum=1).value


def execution_checkpoint_reserve_bytes() -> int:
    """Return disk headroom kept available for atomic execution checkpoints."""
    return _read_int_setting("EPHEMERAL_EXECUTION_CHECKPOINT_RESERVE_BYTES", DEFAULT_EXECUTION_CHECKPOINT_RESERVE_BYTES, os.environ, minimum=1).value


def embedding_model_name() -> str:
    """Return the configured FastEmbed model name."""
    return _read_string_setting("EPHEMERAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL, os.environ).value


def embedding_cache_dir() -> str | None:
    """Return the optional FastEmbed model cache directory."""
    return _read_string_setting("EPHEMERAL_FASTEMBED_CACHE_DIR", None, os.environ, empty_is_default=True).value


def embedding_threads() -> int | None:
    """Return the optional ONNX Runtime thread count for embedding inference.

    ``None`` leaves ONNX Runtime's own default in place.
    """
    return _read_optional_positive_int_setting("EPHEMERAL_EMBEDDING_THREADS", os.environ).value


def embedding_batch_size() -> int:
    """Return the maximum number of semantic chunks sent to one embed call."""
    return _read_int_setting("EPHEMERAL_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE, os.environ, minimum=1).value


def embedding_max_batch_tokens() -> int:
    """Return the maximum padded token slots allowed in one embedding batch."""
    return _read_int_setting("EPHEMERAL_EMBEDDING_MAX_BATCH_TOKENS", DEFAULT_EMBEDDING_MAX_BATCH_TOKENS, os.environ, minimum=1).value


def embedding_cpu_mem_arena_enabled() -> bool:
    """Return whether ONNX Runtime's CPU memory arena is enabled for embeddings."""
    return _read_bool_setting("EPHEMERAL_EMBEDDING_CPU_MEM_ARENA", False, os.environ).value


def semantic_max_index_input_bytes() -> int:
    """Return the UTF-8 input-byte budget for one capture's semantic index."""
    return _read_int_setting("EPHEMERAL_SEMANTIC_MAX_INDEX_INPUT_BYTES", DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES, os.environ, minimum=1).value


def embedding_warmup_enabled() -> bool:
    """Return whether the embedding model is warmed in the background at startup."""
    return _read_bool_setting("EPHEMERAL_EMBEDDING_WARMUP", True, os.environ).value


def semantic_prefetch_enabled() -> bool:
    """Return whether post-ingestion semantic indexing is enabled (default on)."""
    return _read_bool_setting("EPHEMERAL_SEMANTIC_PREFETCH", True, os.environ).value


def semantic_prefetch_workers() -> int:
    """Return the bounded number of semantic prefetch workers."""
    return _read_int_setting("EPHEMERAL_SEMANTIC_PREFETCH_WORKERS", DEFAULT_SEMANTIC_PREFETCH_WORKERS, os.environ, minimum=1).value


def semantic_wait_seconds() -> float:
    """Return how long hybrid search waits for a semantic index before going lexical-first.

    ``0`` never waits, ``inf`` waits until the index is ready; negative, NaN,
    and unparsable values fall back to the default.
    """
    return _read_float_setting(
        "EPHEMERAL_SEMANTIC_WAIT_SECONDS",
        DEFAULT_SEMANTIC_WAIT_SECONDS,
        os.environ,
        minimum=0.0,
        allow_infinity=True,
    ).value


def semantic_chunk_lines() -> int:
    """Return the maximum number of lines packed into one semantic chunk."""
    return _read_int_setting("EPHEMERAL_SEMANTIC_CHUNK_LINES", DEFAULT_SEMANTIC_CHUNK_LINES, os.environ, minimum=1).value


def semantic_chunk_bytes() -> int:
    """Return the UTF-8 byte budget that closes a semantic chunk early."""
    return _read_int_setting("EPHEMERAL_SEMANTIC_CHUNK_BYTES", DEFAULT_SEMANTIC_CHUNK_BYTES, os.environ, minimum=1).value


def semantic_chunk_overlap() -> int:
    """Return how many trailing lines consecutive semantic chunks share."""
    return _read_int_setting("EPHEMERAL_SEMANTIC_CHUNK_OVERLAP", DEFAULT_SEMANTIC_CHUNK_OVERLAP, os.environ, minimum=0).value


def max_indexed_chunks() -> int:
    """Return the maximum total number of indexed chunks retained in memory."""
    return _read_int_setting("EPHEMERAL_MAX_INDEXED_CHUNKS", DEFAULT_MAX_INDEXED_CHUNKS, os.environ, minimum=1).value


def runtime_index_budget_adjustment_enabled() -> bool:
    """Read the deliberately mutable gate once per adjustment request."""
    return _read_bool_setting(
        "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET", False, os.environ,
        sampling_policy="each MCP request",
    ).value


def optional_positive_int_env(name: str) -> int | None:
    """Return a positive integer environment setting, or ``None`` when unset or invalid."""
    return _read_optional_positive_int_setting(name, os.environ).value


def non_negative_int_env(name: str, default: int) -> int:
    """Return a non-negative integer environment setting or its safe default."""
    return _read_int_setting(name, default, os.environ, minimum=0).value


def positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment setting or its safe default."""
    return _read_int_setting(name, default, os.environ, minimum=1).value


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _report_invalid(name: str, raw: str, default: Any) -> None:
    display_default = "unset" if default is None else repr(default)
    print(
        f"Ignoring invalid {name}={raw!r}; using {display_default}",
        file=sys.stderr,
    )


def _read_int_setting(
    name: str,
    default: int,
    environ: Mapping[str, str],
    *,
    minimum: int,
) -> ConfigSetting[int]:
    raw = environ.get(name)
    if raw is None:
        return ConfigSetting(name, default, "default", "unset")
    try:
        value = int(raw)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError):
        _report_invalid(name, raw, default)
        return ConfigSetting(name, default, "default", "invalid-fallback")
    return ConfigSetting(name, value, "environment", "accepted")


def _read_optional_positive_int_setting(
    name: str,
    environ: Mapping[str, str],
) -> ConfigSetting[int | None]:
    raw = environ.get(name)
    if raw is None:
        return ConfigSetting(
            name, None, "default", "unset", invalid_value_behavior="warn and use unset default"
        )
    if not raw.strip():
        return ConfigSetting(
            name,
            None,
            "default",
            "empty-default",
            invalid_value_behavior="treat empty as unset",
        )
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
    except (TypeError, ValueError):
        _report_invalid(name, raw, None)
        return ConfigSetting(name, None, "default", "invalid-fallback")
    return ConfigSetting(name, value, "environment", "accepted")


def _read_float_setting(
    name: str,
    default: float,
    environ: Mapping[str, str],
    *,
    minimum: float,
    inclusive: bool = True,
    allow_infinity: bool = False,
) -> ConfigSetting[float]:
    raw = environ.get(name)
    if raw is None:
        return ConfigSetting(name, default, "default", "unset")
    try:
        value = float(raw)
        if math.isnan(value) or (not allow_infinity and not math.isfinite(value)):
            raise ValueError
        if value < minimum or (not inclusive and value == minimum):
            raise ValueError
    except (TypeError, ValueError):
        _report_invalid(name, raw, default)
        return ConfigSetting(name, default, "default", "invalid-fallback")
    return ConfigSetting(name, value, "environment", "accepted")


def _read_bool_setting(
    name: str,
    default: bool,
    environ: Mapping[str, str],
    *,
    sampling_policy: str = "startup",
) -> ConfigSetting[bool]:
    raw = environ.get(name)
    if raw is None:
        return ConfigSetting(name, default, "default", "unset", sampling_policy=sampling_policy)
    normalized = raw.strip().lower()
    if not normalized:
        return ConfigSetting(
            name,
            default,
            "default",
            "empty-default",
            invalid_value_behavior="treat empty as unset",
            sampling_policy=sampling_policy,
        )
    if normalized in _TRUE_VALUES:
        return ConfigSetting(name, True, "environment", "accepted", sampling_policy=sampling_policy)
    if normalized in _FALSE_VALUES:
        return ConfigSetting(name, False, "environment", "accepted", sampling_policy=sampling_policy)
    _report_invalid(name, raw, default)
    return ConfigSetting(
        name,
        default,
        "default",
        "invalid-fallback",
        sampling_policy=sampling_policy,
    )


def boolean_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment value, warning when it is invalid."""
    return _read_bool_setting(name, default, os.environ).value


def _read_string_setting(
    name: str,
    default: str | None,
    environ: Mapping[str, str],
    *,
    empty_is_default: bool = False,
) -> ConfigSetting[str | None]:
    raw = environ.get(name)
    if raw is None or (empty_is_default and not raw.strip()):
        status = "unset" if raw is None else "empty-default"
        return ConfigSetting(name, default, "default", status, invalid_value_behavior="not applicable")
    return ConfigSetting(name, raw, "environment", "accepted", invalid_value_behavior="not applicable")


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    warn_on_legacy_state_transition: bool = True,
    default_state_dir: str | None = None,
) -> SettingsSnapshot:
    """Parse one immutable startup settings snapshot from an environment."""
    values = os.environ if environ is None else environ
    identity = resolve_session_descriptor(values, default_state_dir=default_state_dir)
    if warn_on_legacy_state_transition and identity.legacy_state_transition:
        print(
            "Execution state identity now follows the resolved socket identity. "
            f"Existing state is at {identity.legacy_state_dir}; the effective directory is "
            f"{identity.state_dir}. Set EPHEMERAL_EXECUTION_STATE_DIR to the existing path "
            "to continue using those records.",
            file=sys.stderr,
        )

    def integer(name: str, default: int, minimum: int = 1) -> ConfigSetting[int]:
        return _read_int_setting(name, default, values, minimum=minimum)

    def boolean(name: str, default: bool) -> ConfigSetting[bool]:
        return _read_bool_setting(name, default, values)

    return SettingsSnapshot(
        identity=identity,
        max_captures=integer("EPHEMERAL_MAX_CAPTURES", DEFAULT_MAX_CAPTURES),
        max_buffer_bytes=integer("EPHEMERAL_MAX_BUFFER_BYTES", DEFAULT_MAX_BUFFER_BYTES),
        socket_timeout_seconds=_read_float_setting(
            "EPHEMERAL_SOCKET_TIMEOUT_SECONDS", DEFAULT_SOCKET_TIMEOUT_SECONDS,
            values, minimum=0.0, inclusive=False, allow_infinity=False,
        ),
        shutdown_grace_seconds=_read_float_setting(
            "EPHEMERAL_SHUTDOWN_GRACE_SECONDS", DEFAULT_SHUTDOWN_GRACE_SECONDS,
            values, minimum=0.0, inclusive=True, allow_infinity=False,
        ),
        socket_startup_timeout_seconds=integer(
            "EPHEMERAL_SOCKET_STARTUP_TIMEOUT_SECONDS", DEFAULT_SOCKET_STARTUP_TIMEOUT_SECONDS,
        ),
        socket_require_isolation=boolean("EPHEMERAL_REQUIRE_ISOLATION", False),
        allow_stdio_without_socket=boolean("EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET", False),
        disable_socket_server=boolean("EPHEMERAL_DISABLE_SOCKET_SERVER", False),
        metrics_enabled=boolean("EPHEMERAL_METRICS", False),
        metrics_file=_read_string_setting(
            "EPHEMERAL_METRICS_FILE", None, values, empty_is_default=True,
        ),
        max_active_tool_work=integer("EPHEMERAL_MAX_ACTIVE_TOOL_WORK", DEFAULT_MAX_ACTIVE_TOOL_WORK),
        max_queued_tool_work=integer("EPHEMERAL_MAX_QUEUED_TOOL_WORK", DEFAULT_MAX_QUEUED_TOOL_WORK, minimum=0),
        max_active_socket_clients=integer("EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS", DEFAULT_MAX_ACTIVE_SOCKET_CLIENTS),
        max_queued_socket_clients=integer("EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS", DEFAULT_MAX_QUEUED_SOCKET_CLIENTS, minimum=0),
        execution_state_quota_bytes=integer(
            "EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES", DEFAULT_EXECUTION_STATE_QUOTA_BYTES,
        ),
        execution_checkpoint_reserve_bytes=integer(
            "EPHEMERAL_EXECUTION_CHECKPOINT_RESERVE_BYTES", DEFAULT_EXECUTION_CHECKPOINT_RESERVE_BYTES,
        ),
        embedding_model_name=_read_string_setting(
            "EPHEMERAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL, values,
        ),
        embedding_cache_dir=_read_string_setting(
            "EPHEMERAL_FASTEMBED_CACHE_DIR", None, values, empty_is_default=True,
        ),
        embedding_threads=_read_optional_positive_int_setting("EPHEMERAL_EMBEDDING_THREADS", values),
        embedding_batch_size=integer("EPHEMERAL_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
        embedding_max_batch_tokens=integer(
            "EPHEMERAL_EMBEDDING_MAX_BATCH_TOKENS", DEFAULT_EMBEDDING_MAX_BATCH_TOKENS,
        ),
        embedding_cpu_mem_arena_enabled=boolean("EPHEMERAL_EMBEDDING_CPU_MEM_ARENA", False),
        embedding_warmup_enabled=boolean("EPHEMERAL_EMBEDDING_WARMUP", True),
        semantic_max_index_input_bytes=integer(
            "EPHEMERAL_SEMANTIC_MAX_INDEX_INPUT_BYTES", DEFAULT_SEMANTIC_MAX_INDEX_INPUT_BYTES,
        ),
        semantic_prefetch_enabled=boolean("EPHEMERAL_SEMANTIC_PREFETCH", True),
        semantic_prefetch_workers=integer(
            "EPHEMERAL_SEMANTIC_PREFETCH_WORKERS", DEFAULT_SEMANTIC_PREFETCH_WORKERS,
        ),
        semantic_wait_seconds=_read_float_setting(
            "EPHEMERAL_SEMANTIC_WAIT_SECONDS", DEFAULT_SEMANTIC_WAIT_SECONDS,
            values, minimum=0.0, allow_infinity=True,
        ),
        semantic_chunk_lines=integer("EPHEMERAL_SEMANTIC_CHUNK_LINES", DEFAULT_SEMANTIC_CHUNK_LINES),
        semantic_chunk_bytes=integer("EPHEMERAL_SEMANTIC_CHUNK_BYTES", DEFAULT_SEMANTIC_CHUNK_BYTES),
        semantic_chunk_overlap=integer(
            "EPHEMERAL_SEMANTIC_CHUNK_OVERLAP", DEFAULT_SEMANTIC_CHUNK_OVERLAP, minimum=0,
        ),
        max_indexed_chunks=integer("EPHEMERAL_MAX_INDEXED_CHUNKS", DEFAULT_MAX_INDEXED_CHUNKS),
        runtime_index_budget_adjustment_enabled=_read_bool_setting(
            "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET", False, values,
            sampling_policy=(
                "each MCP request; environment value is re-read; reported value is from startup"
            ),
        ),
    )


_STARTUP_SETTINGS: SettingsSnapshot | None = None
_STARTUP_SETTINGS_LOCK = threading.Lock()


def startup_settings() -> SettingsSnapshot:
    """Return the process-wide startup snapshot, parsing and reporting once."""
    global _STARTUP_SETTINGS
    if _STARTUP_SETTINGS is None:
        with _STARTUP_SETTINGS_LOCK:
            if _STARTUP_SETTINGS is None:
                _STARTUP_SETTINGS = load_settings()
    return _STARTUP_SETTINGS


CODEX_MCP_ENV_NAMES = (
    "EPHEMERAL_SESSION_ID",
    "EPHEMERAL_SOCKET_PATH",
    "EPHEMERAL_EXECUTION_STATE_DIR",
    "EPHEMERAL_LEGACY_STATE_DIR",
    "EPHEMERAL_REQUIRE_ISOLATION",
    "EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET",
    "EPHEMERAL_DISABLE_SOCKET_SERVER",
    "EPHEMERAL_METRICS",
    "EPHEMERAL_METRICS_FILE",
    "EPHEMERAL_LOG_LEVEL",
    "EPHEMERAL_LOG_FILE",
    "EPHEMERAL_MAX_CAPTURES",
    "EPHEMERAL_MAX_BUFFER_BYTES",
    "EPHEMERAL_SOCKET_TIMEOUT_SECONDS",
    "EPHEMERAL_SOCKET_STARTUP_TIMEOUT_SECONDS",
    "EPHEMERAL_MAX_ACTIVE_TOOL_WORK",
    "EPHEMERAL_MAX_QUEUED_TOOL_WORK",
    "EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS",
    "EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS",
    "EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES",
    "EPHEMERAL_EXECUTION_CHECKPOINT_RESERVE_BYTES",
    "EPHEMERAL_EMBEDDING_MODEL",
    "EPHEMERAL_FASTEMBED_CACHE_DIR",
    "EPHEMERAL_EMBEDDING_THREADS",
    "EPHEMERAL_EMBEDDING_BATCH_SIZE",
    "EPHEMERAL_EMBEDDING_MAX_BATCH_TOKENS",
    "EPHEMERAL_EMBEDDING_CPU_MEM_ARENA",
    "EPHEMERAL_EMBEDDING_WARMUP",
    "EPHEMERAL_SEMANTIC_MAX_INDEX_INPUT_BYTES",
    "EPHEMERAL_SEMANTIC_PREFETCH",
    "EPHEMERAL_SEMANTIC_PREFETCH_WORKERS",
    "EPHEMERAL_SEMANTIC_WAIT_SECONDS",
    "EPHEMERAL_SEMANTIC_CHUNK_LINES",
    "EPHEMERAL_SEMANTIC_CHUNK_BYTES",
    "EPHEMERAL_SEMANTIC_CHUNK_OVERLAP",
    "EPHEMERAL_MAX_INDEXED_CHUNKS",
    "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET",
)


def codex_mcp_env_config(environ: Mapping[str, str] | None = None) -> str:
    """Build Codex's MCP environment table from the same known EB settings."""
    import json

    values = dict(os.environ if environ is None else environ)
    values.update(normalized_explicit_identity_paths(values))
    entries = []
    for name in CODEX_MCP_ENV_NAMES:
        value = values.get(name)
        if value is not None:
            entries.append(f"{name}={json.dumps(value, ensure_ascii=False)}")
    return "mcp_servers.ephemeral-buffer.env={" + ", ".join(entries) + "}"


def normalized_explicit_identity_paths(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return absolute forms for explicitly configured socket/state paths."""
    values = os.environ if environ is None else environ
    normalized = {}
    for name in ("EPHEMERAL_SOCKET_PATH", "EPHEMERAL_EXECUTION_STATE_DIR"):
        value = values.get(name)
        if value:
            normalized[name] = os.path.abspath(os.path.expanduser(value))
    raw_socket = values.get("EPHEMERAL_SOCKET_PATH") or None
    if (
        raw_socket
        and not values.get("EPHEMERAL_EXECUTION_STATE_DIR")
        and not values.get("EPHEMERAL_SESSION_ID")
    ):
        resolved_socket = normalized["EPHEMERAL_SOCKET_PATH"]
        if raw_socket != resolved_socket:
            legacy_digest = _identity_digest(raw_socket)
            legacy_state_dir = os.path.join(
                tempfile.gettempdir(), f"{EXECUTION_STATE_SOCKET_PREFIX}{legacy_digest}"
            )
            if os.path.isdir(legacy_state_dir):
                normalized["EPHEMERAL_LEGACY_STATE_DIR"] = legacy_state_dir
    return normalized


def main() -> None:
    """Run the small configuration interface used by the shell launchers."""
    if sys.argv[1:] == ["--socket-path"]:
        print(resolve_session_descriptor().socket_path)
    elif sys.argv[1:] == ["--codex-env-config"]:
        print(codex_mcp_env_config())
    elif sys.argv[1:] == ["--normalize-explicit-paths"]:
        import json

        print(json.dumps(normalized_explicit_identity_paths(), ensure_ascii=False))
    else:
        raise SystemExit(
            "Usage: config.py --socket-path | --codex-env-config | --normalize-explicit-paths"
        )


if __name__ == "__main__":
    main()
