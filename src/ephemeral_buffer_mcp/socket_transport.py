"""Unix-domain socket transport lifecycle and framed request handling."""

from __future__ import annotations

import importlib

from .server import *  # noqa: F403
from .tools import load_tool_handlers
from .socket_protocol import (
    FRAME_HEADER_SIZE,
    SocketRequestError,
    decode_header,
    decode_socket_request,
    encode_frame,
    encode_socket_response,
)

load_tool_handlers()
_capture_handlers = importlib.import_module(".tools.capture", __package__)

_SOCKET_STATE_LOCK = threading.Lock()
_SOCKET_STATE = "disabled" if SETTINGS.disable_socket_server.value else "not-started"
_SOCKET_FAILURE = None
_SOCKET_STARTUP_EVENT = threading.Event()
_SOCKET_PATH_LOCKS = {}
_SOCKET_PATH_LOCKS_GUARD = threading.Lock()
_SOCKET_PATH_LOCK_DEPTH = threading.local()
SOCKET_PATH = SETTINGS.identity.socket_path
SOCKET_PAYLOAD_OVERHEAD = 64 * 1024
SOCKET_CLIENT_READ_TIMEOUT_SECONDS = SETTINGS.socket_timeout_seconds.value
_SOCKET_CLIENT_TASKS: set[asyncio.Task[Any]] = set()
_SOCKET_STARTUP_GATED = False
_SOCKET_STARTUP_PROBE_ADDRESS: Optional[str] = None
_SOCKET_STARTUP_PROBE_TOKEN: Optional[bytes] = None
SOCKET_STARTUP_TIMEOUT_SECONDS = SETTINGS.socket_startup_timeout_seconds.value

if _SOCKET_STATE == "disabled":
    _SOCKET_STARTUP_EVENT.set()


def _set_socket_state(state, failure=None):
    """Publish socket lifecycle state for diagnostics and the module entrypoint."""
    global _SOCKET_STATE, _SOCKET_FAILURE
    with _SOCKET_STATE_LOCK:
        _SOCKET_STATE = state
        _SOCKET_FAILURE = failure
        if state == "starting":
            _SOCKET_STARTUP_EVENT.clear()
        if state in {"ready", "failed", "disabled"}:
            _SOCKET_STARTUP_EVENT.set()


def _socket_lifecycle():
    with _SOCKET_STATE_LOCK:
        return _SOCKET_STATE, _SOCKET_FAILURE


def _socket_identity(path):
    """Return a socket's filesystem identity without following symlinks."""
    path_stat = os.lstat(path)
    if not stat.S_ISSOCK(path_stat.st_mode):
        return None
    return path_stat.st_dev, path_stat.st_ino


def _new_socket_probe():
    """Create the short-lived client used to verify a Unix socket path."""
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


def _bind_socket_probe(probe, socket_path):
    """Bind a short-lived probe using a path that fits AF_UNIX limits."""
    directories = (
        "/tmp",
        "/var/tmp",
        tempfile.gettempdir(),
        os.path.dirname(socket_path) or ".",
    )
    seen_directories = set()
    last_error = OSError("No Unix socket probe address candidates are available")
    for directory in directories:
        if directory in seen_directories:
            continue
        seen_directories.add(directory)
        probe_address = os.path.join(directory, f"ebp-{uuid.uuid4().hex}")
        try:
            probe.bind(probe_address)
        except OSError as exc:
            last_error = exc
            continue
        return probe_address

    raise last_error


async def _connect_socket_probe(probe, path):
    """Connect a bound local probe socket and wrap it as asyncio streams."""
    loop = asyncio.get_running_loop()
    await loop.sock_connect(probe, path)
    return await asyncio.open_connection(sock=probe)


async def _verify_bound_socket_path(bound_socket, path):
    """Return the path identity only after it is verified against the listener."""
    global _SOCKET_STARTUP_PROBE_ADDRESS, _SOCKET_STARTUP_PROBE_TOKEN

    bound_fd_stat = os.fstat(bound_socket.fileno())
    if not stat.S_ISSOCK(bound_fd_stat.st_mode):
        raise RuntimeError("Bound Unix listener descriptor is not a socket")
    if bound_socket.getsockname() != path:
        raise RuntimeError(f"Bound Unix listener address does not match path: {path}")

    path_stat_before = os.lstat(path)
    if not stat.S_ISSOCK(path_stat_before.st_mode):
        raise RuntimeError(f"Bound socket path is not a socket: {path}")

    # On Linux, fstat(socket_fd) identifies the sockfs socket object while
    # lstat(path) identifies its filesystem node; those inode values differ.
    # Verify the path through asyncio's normal accept callback so startup
    # never consumes a queued client connection itself.
    probe = None
    writer = None
    probe_address = None
    probe_bound = False
    old_probe_address = _SOCKET_STARTUP_PROBE_ADDRESS
    old_probe_token = _SOCKET_STARTUP_PROBE_TOKEN
    probe_token = os.urandom(32)
    try:
        probe = _new_socket_probe()
        probe.setblocking(False)
        probe_address = _bind_socket_probe(probe, path)
        probe_bound = True
        _SOCKET_STARTUP_PROBE_ADDRESS = probe.getsockname()
        _SOCKET_STARTUP_PROBE_TOKEN = probe_token
        try:
            reader, writer = await asyncio.wait_for(
                _connect_socket_probe(probe, path), timeout=1.0
            )
            # The asyncio transport now owns the connected probe socket.
            probe = None
            header = await asyncio.wait_for(
                _read_exact(reader, FRAME_HEADER_SIZE), timeout=1.0
            )
            response_length = decode_header(header)
            response = await asyncio.wait_for(
                _read_exact(reader, response_length), timeout=1.0
            )
            if response != probe_token:
                raise RuntimeError(
                    f"Socket path did not acknowledge the bound listener: {path}"
                )
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
    except OSError as exc:
        raise RuntimeError(f"Unable to verify bound socket path {path}: {exc}") from exc
    finally:
        _SOCKET_STARTUP_PROBE_ADDRESS = old_probe_address
        _SOCKET_STARTUP_PROBE_TOKEN = old_probe_token
        if probe is not None:
            probe.close()
        if probe_bound:
            try:
                os.unlink(probe_address)
            except FileNotFoundError:
                pass

    path_stat_after = os.lstat(path)
    path_identity_before = (path_stat_before.st_dev, path_stat_before.st_ino)
    path_identity_after = (path_stat_after.st_dev, path_stat_after.st_ino)
    if (
        not stat.S_ISSOCK(path_stat_after.st_mode)
        or path_identity_before != path_identity_after
    ):
        raise RuntimeError(f"Socket path changed while verifying the bound listener: {path}")
    return path_identity_after


@contextmanager
def _socket_path_lock(path):
    """Serialize socket probing and cleanup for one pathname."""
    key = os.path.abspath(path)
    with _SOCKET_PATH_LOCKS_GUARD:
        lock = _SOCKET_PATH_LOCKS.setdefault(key, threading.RLock())
    with lock:
        held_keys = getattr(_SOCKET_PATH_LOCK_DEPTH, "keys", set())
        if key in held_keys:
            yield
            return
        held_keys = set(held_keys)
        held_keys.add(key)
        _SOCKET_PATH_LOCK_DEPTH.keys = held_keys
        lock_fd = None
        try:
            if fcntl is not None:
                lock_path = f"{key}.lock"
                lock_fd = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT,
                    0o600,
                )
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(lock_fd)
                    lock_fd = None
                    raise
            yield
        finally:
            try:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
            finally:
                held_keys.remove(key)
                _SOCKET_PATH_LOCK_DEPTH.keys = held_keys


def _unlink_socket_if_identity(path, expected_identity):
    with _socket_path_lock(path):
        return _unlink_socket_if_identity_unlocked(path, expected_identity)


def _unlink_socket_if_identity_unlocked(path, expected_identity):
    """Remove only the socket inode that was atomically claimed.

    Unlinking a pathname after an identity check has a replacement race.  Move
    the name to a private quarantine name first; any replacement created at
    the original path is then independent of the object being removed.
    """
    parent = os.path.dirname(path) or "."
    quarantine = os.path.join(
        parent,
        f".{os.path.basename(path)}.cleanup-{os.getpid()}-"
        f"{threading.get_ident()}-{uuid.uuid4().hex}",
    )
    try:
        if _socket_identity(path) != expected_identity:
            return False
        os.rename(path, quarantine)
    except FileNotFoundError:
        return False
    try:
        if _socket_identity(quarantine) != expected_identity:
            # Leave the claimed inode quarantined.  Restoring it with rename
            # after checking the path would let a concurrent replacement be
            # overwritten between the check and the restore.
            return False
        try:
            os.unlink(quarantine)
        except FileNotFoundError:
            # Another cleanup actor removed the claimed inode.  The original
            # pathname is still independent, so cleanup is complete.
            return True
        return True
    except BaseException:
        # Do not restore the quarantined inode: a replacement may have
        # appeared at the original pathname since any earlier inspection.
        raise


def _allow_stdio_without_socket() -> bool:
    """Return whether MCP stdio may continue when the socket cannot start."""
    return SETTINGS.allow_stdio_without_socket.value


def _require_socket_ready():
    """Require socket readiness unless explicit stdio-only fallback is enabled."""
    if not _SOCKET_STARTUP_EVENT.wait(timeout=SOCKET_STARTUP_TIMEOUT_SECONDS):
        if _allow_stdio_without_socket():
            log_event(
                LOGGER, logging.WARNING, "socket_unavailable_stdio_continues",
                reason="startup timeout",
            )
            return
        raise SystemExit(
            f"Socket server did not become ready within {SOCKET_STARTUP_TIMEOUT_SECONDS} seconds"
        )
    socket_state, socket_failure = _socket_lifecycle()
    if socket_state == "failed":
        if _allow_stdio_without_socket():
            log_event(
                LOGGER, logging.WARNING, "socket_unavailable_stdio_continues",
                reason=socket_failure,
            )
            return
        raise SystemExit(f"Socket server failed to start: {socket_failure}")


async def _read_exact(
    reader: asyncio.StreamReader,
    size: int,
    byte_counter: Optional[str] = None,
) -> bytes:
    """Read exactly ``size`` bytes, tolerating fragmented stream reads."""
    chunks = []
    remaining = size
    while remaining:
        chunk = await reader.read(remaining)
        if not chunk:
            raise ValueError("truncated socket frame")
        if byte_counter is not None:
            METRICS.record_bytes(byte_counter, len(chunk))
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def _read_socket_payload(reader: asyncio.StreamReader, read_limit: int) -> bytes:
    """Read one versioned length-prefixed request within the payload limit."""
    header = await _read_exact(reader, FRAME_HEADER_SIZE, "socket_request_bytes")
    payload_length = decode_header(header)
    if payload_length > read_limit:
        log_event(
            LOGGER,
            logging.WARNING,
            "socket_payload_limit_rejected",
            payload_bytes=payload_length,
            max_payload_bytes=read_limit,
        )
        raise ValueError(f"CLI payload exceeds the {read_limit:,}-byte socket payload limit")
    return await _read_exact(reader, payload_length, "socket_request_bytes")


async def _write_socket_error(
    writer: asyncio.StreamWriter,
    code: str,
    message: str,
) -> None:
    response_frame = encode_frame(
        encode_socket_response({"status": "error", "code": code, "message": message})
    )
    METRICS.record_bytes("socket_response_bytes", len(response_frame))
    writer.write(response_frame)
    await writer.drain()


def handle_socket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle one framed CLI request and keep its task alive until completion."""
    # Snapshot startup state when the connection callback runs. A queued
    # connection may not start its task until after verification has cleared
    # the globals, but it still arrived while ordinary requests were gated.
    startup_gated = _SOCKET_STARTUP_GATED
    startup_probe_address = _SOCKET_STARTUP_PROBE_ADDRESS
    startup_probe_token = _SOCKET_STARTUP_PROBE_TOKEN

    async def _handle():
        probe_address = startup_probe_address
        probe_token = startup_probe_token
        if probe_address is not None and probe_token is not None:
            try:
                peer_address = writer.get_extra_info("socket").getpeername()
            except (AttributeError, OSError):
                peer_address = None
            if peer_address == probe_address:
                response_frame = encode_frame(probe_token)
                METRICS.record_bytes("socket_response_bytes", len(response_frame))
                try:
                    writer.write(response_frame)
                    await writer.drain()
                finally:
                    writer.close()
                    await writer.wait_closed()
                return

        if startup_gated:
            try:
                await _write_socket_error(
                    writer,
                    "server_starting",
                    "The service is starting; retry this request shortly.",
                )
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
            return

        registered_call = DEFAULT_SERVICES.register_tool_call("socket")
        if registered_call is None:
            try:
                await _write_socket_error(
                    writer,
                    "server_shutting_down",
                    "The service is shutting down; retry after it restarts.",
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return
        call_id, _cancellation_event = registered_call
        try:
            ticket = await admission_gate("socket").acquire("socket")
        except AdmissionBusy:
            DEFAULT_SERVICES.finish_tool_call(call_id)
            log_event(LOGGER, logging.WARNING, "socket_client_rejected_busy")
            try:
                await _write_socket_error(
                    writer,
                    "server_busy",
                    "Socket work capacity is full; retry this request shortly.",
                )
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
            return
        except BaseException as admission_error:
            DEFAULT_SERVICES.finish_tool_call(call_id)
            try:
                if isinstance(admission_error, Exception):
                    try:
                        await _write_socket_error(
                            writer,
                            "socket_admission_failed",
                            "Unable to admit this socket request; retry shortly.",
                        )
                    except Exception:
                        pass
            finally:
                try:
                    writer.close()
                except Exception:
                    pass
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            raise

        if DEFAULT_SERVICES.closing:
            ticket.release()
            DEFAULT_SERVICES.finish_tool_call(call_id)
            try:
                await _write_socket_error(
                    writer,
                    "server_shutting_down",
                    "The service is shutting down; retry after it restarts.",
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return

        try:
            # Read and validate one complete framed payload before decoding it.
            read_limit = (
                engine.max_buffer_bytes * SOCKET_JSON_MAX_EXPANSION
                + SOCKET_PAYLOAD_OVERHEAD
            )
            try:
                data = await asyncio.wait_for(
                    _read_socket_payload(reader, read_limit),
                    timeout=SOCKET_CLIENT_READ_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log_event(LOGGER, logging.WARNING, "socket_client_read_timeout")
                await _write_socket_error(
                    writer,
                    "socket_read_timeout",
                    "Socket request frame was not completed before the read timeout.",
                )
                return
            if not data:
                return
            # Frame v1 peers predating the JSON contract omitted ``text``;
            # the original server treated that as an empty capture.
            request = decode_socket_request(data, allow_legacy_missing_text=True)
            label = request["label"]
            text = request["text"]
            content_type = request["content_type"]
            truncated = request["truncated"]
            original_byte_size = request["original_byte_size"]
            command_exit_code = request["command_exit_code"]
            timed_out = request["timed_out"]
            duration_ms = request["duration_ms"]
            try:
                structured_metrics = normalize_structured_metrics(request["structured_metrics"])
            except (TypeError, ValueError) as exc:
                raise SocketRequestError(str(exc)) from exc

            cap, summary = await _run_admitted_thread(
                ticket,
                engine.ingest_with_summary,
                text,
                label=label,
                content_type=content_type,
                truncated=truncated,
                original_byte_size=original_byte_size,
                command_exit_code=command_exit_code,
                timed_out=timed_out,
                source="socket",
                duration_ms=duration_ms,
                structured_metrics=structured_metrics,
                include_previews=False,
                _service_context=DEFAULT_SERVICES,
                _call_id=call_id,
            )
            resp = {
                "status": "ok",
                "capture_id": cap.capture_id,
                "label": _capture_handlers._bounded_summary_text(
                    cap.label,
                    SUMMARY_LABEL_MAX_BYTES,
                    SUMMARY_LABEL_TRUNCATION_MARKER,
                )[0],
                "line_count": cap.line_count,
                "byte_size": cap.byte_size
            }
            if summary.get("status") == "ok":
                resp["summary"] = _capture_handlers._summary_payload(summary)
            response_frame = encode_frame(encode_socket_response(resp))
            METRICS.record_bytes("socket_response_bytes", len(response_frame))
            writer.write(response_frame)
            await writer.drain()
        except Exception as e:
            log_event(LOGGER, logging.ERROR, "socket_client_failed", error_type=type(e).__name__)
            code = "invalid_socket_payload" if isinstance(e, SocketRequestError) else "socket_request_failed"
            err_resp = {"status": "error", "code": code, "message": str(e)}
            response_frame = encode_frame(encode_socket_response(err_resp))
            METRICS.record_bytes("socket_response_bytes", len(response_frame))
            writer.write(response_frame)
            await writer.drain()
        finally:
            ticket.release()
            DEFAULT_SERVICES.finish_tool_call(call_id)
            writer.close()
            await writer.wait_closed()

    # StreamReaderProtocol ignores the connection callback's return value.
    # Retain the task explicitly until it completes so the loop cannot collect
    # an in-flight socket request.
    task = asyncio.create_task(_handle())
    _SOCKET_CLIENT_TASKS.add(task)
    task.add_done_callback(_SOCKET_CLIENT_TASKS.discard)
    return task


def run_socket_server():
    """Runs a Unix domain socket server in a separate thread so CLI tools can pipe to it."""
    global _SOCKET_STARTUP_GATED
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _set_socket_state("starting")
    bound_socket_identity = None
    socket_path_lock = None
    lock_acquired = False

    try:
        if socket_isolation_required() and not socket_isolation_configured():
            raise RuntimeError(
                "Socket isolation is required; set EPHEMERAL_SESSION_ID or EPHEMERAL_SOCKET_PATH"
            )
        socket_path_lock = _socket_path_lock(SOCKET_PATH)
        socket_path_lock.__enter__()
        lock_acquired = True
        if os.path.lexists(SOCKET_PATH):
            initial_path_stat = os.lstat(SOCKET_PATH)
            if not stat.S_ISSOCK(initial_path_stat.st_mode):
                raise RuntimeError(
                    f"Socket path exists but is not a Unix socket: {SOCKET_PATH}"
                )
            initial_socket_identity = (
                initial_path_stat.st_dev,
                initial_path_stat.st_ino,
            )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(SOCKET_PATH)
            except ConnectionRefusedError:
                try:
                    current_path_stat = os.lstat(SOCKET_PATH)
                except FileNotFoundError:
                    current_path_stat = None
                if current_path_stat is not None:
                    current_socket_identity = (
                        current_path_stat.st_dev,
                        current_path_stat.st_ino,
                    )
                    if not stat.S_ISSOCK(current_path_stat.st_mode):
                        raise RuntimeError(
                            f"Socket path changed to a non-socket path: {SOCKET_PATH}"
                        )
                    if current_socket_identity != initial_socket_identity:
                        raise RuntimeError(
                            f"Socket path changed while probing: {SOCKET_PATH}"
                        )
                    # No listener accepted the connection, so this is a stale
                    # socket. Claim it before removal so a replacement cannot
                    # be deleted by a pathname-only unlink.
                    if not _unlink_socket_if_identity(
                        SOCKET_PATH, current_socket_identity
                    ) and os.path.lexists(SOCKET_PATH):
                        raise RuntimeError(
                            f"Socket path changed while cleaning stale socket: {SOCKET_PATH}"
                        )
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
            nonlocal bound_socket_identity
            global _SOCKET_STARTUP_GATED
            socket_backlog = (
                MAX_ACTIVE_SOCKET_CLIENTS + MAX_QUEUED_SOCKET_CLIENTS
            )
            _SOCKET_STARTUP_GATED = True
            server = await asyncio.start_unix_server(
                handle_socket_client,
                path=SOCKET_PATH,
                backlog=socket_backlog,
            )
            async with server:
                bound_sockets = server.sockets or ()
                if not bound_sockets:
                    raise RuntimeError("Unix socket listener has no bound socket descriptor")
                bound_socket_identity = await _verify_bound_socket_path(
                    bound_sockets[0], SOCKET_PATH
                )
                os.chmod(SOCKET_PATH, 0o600)
                _set_socket_state("ready")
                _SOCKET_STARTUP_GATED = False
                await server.serve_forever()

        loop.run_until_complete(_main())
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
        _set_socket_state("failed", failure)
        log_event(LOGGER, logging.ERROR, "socket_server_failed", error_type=type(e).__name__)
        LOGGER.exception("socket_server_exception")
        # Preserve the established stderr diagnostic for callers and tests
        # that capture the server's direct error stream.
        print(f"Socket server error: {e}", file=sys.stderr)
    finally:
        _SOCKET_STARTUP_GATED = False
        try:
            if bound_socket_identity is not None:
                try:
                    current_path_stat = os.lstat(SOCKET_PATH)
                except FileNotFoundError:
                    pass
                else:
                    current_socket_identity = (
                        current_path_stat.st_dev,
                        current_path_stat.st_ino,
                    )
                    if (
                        stat.S_ISSOCK(current_path_stat.st_mode)
                        and current_socket_identity == bound_socket_identity
                    ):
                        try:
                            _unlink_socket_if_identity(
                                SOCKET_PATH, bound_socket_identity
                            )
                        except FileNotFoundError:
                            pass
        finally:
            if lock_acquired and socket_path_lock is not None:
                socket_path_lock.__exit__(None, None, None)
            loop.close()


def start_socket_server():
    """Start the IPC listener explicitly and return its background thread."""
    if SETTINGS.disable_socket_server.value:
        _set_socket_state("disabled")
        return None
    _set_socket_state("starting")
    socket_thread = threading.Thread(target=run_socket_server, daemon=True)
    socket_thread.start()
    return socket_thread
