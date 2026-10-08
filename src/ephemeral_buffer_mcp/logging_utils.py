"""Privacy-safe structured logging helpers for runtime operations."""

import errno
import json
import logging
import os
import re
import stat
import sys
import time
import traceback
import uuid
from typing import Any


LOGGER_NAME = "ephemeral_buffer"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_MAX_SAFE_FRAMES = 64


class _JsonFormatter(logging.Formatter):
    """Format operational events without exposing exception values by default."""

    def __init__(self, *, include_protected_traceback: bool = False) -> None:
        super().__init__()
        self.include_protected_traceback = include_protected_traceback

    @staticmethod
    def _safe_identifier(value: Any, fallback: str) -> str:
        candidate = value if isinstance(value, str) else ""
        return candidate if _SAFE_IDENTIFIER.fullmatch(candidate) else fallback

    @classmethod
    def _safe_exception(cls, record: logging.LogRecord) -> dict[str, Any]:
        exc_type, _exc_value, exc_tb = record.exc_info
        class_name = cls._safe_identifier(getattr(exc_type, "__name__", ""), "Exception")
        frames = []
        for frame in traceback.extract_tb(exc_tb)[-_MAX_SAFE_FRAMES:]:
            frames.append(
                {
                    "file": cls._safe_identifier(os.path.basename(frame.filename), "<unknown>"),
                    "function": cls._safe_identifier(frame.name, "<unknown>"),
                    "line": frame.lineno,
                }
            )
        correlation_id = getattr(record, "_ephemeral_correlation_id", None)
        if not correlation_id:
            correlation_id = uuid.uuid4().hex
            # Both handlers receive the same LogRecord, so the sanitized line
            # and its protected counterpart share an identifier.
            record._ephemeral_correlation_id = correlation_id
        return {
            "class": class_name,
            "frames": frames,
            "correlation_id": correlation_id,
        }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            # Exception messages and interpolation arguments can contain
            # capture-derived input. The paired structured event carries the
            # safe operation name where available.
            "event": "exception" if record.exc_info else record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self._safe_exception(record)
            if self.include_protected_traceback:
                payload["protected_traceback"] = "".join(
                    traceback.format_exception(*record.exc_info)
                )
        else:
            fields = getattr(record, "structured_fields", {})
            payload.update(fields)
        return json.dumps(payload, sort_keys=True, default=str)


class _PrivateFileHandler(logging.FileHandler):
    """Append to an owner-only regular file without following symlinks."""

    def _open(self):
        secure_open_supported = (
            hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "geteuid")
            and hasattr(os, "fchmod")
        )
        if not secure_open_supported:
            raise OSError(
                errno.ENOTSUP,
                "secure log file creation is unavailable",
                self.baseFilename,
            )

        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(self.baseFilename, flags, 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(errno.EINVAL, "log path is not a regular file", self.baseFilename)
            if file_stat.st_uid != os.geteuid():
                raise OSError(errno.EPERM, "log file is not owned by the current user", self.baseFilename)
            if file_stat.st_nlink != 1:
                raise OSError(errno.EMLINK, "log file has unexpected hard links", self.baseFilename)
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(
                descriptor,
                self.mode,
                encoding=self.encoding,
                errors=self.errors,
            )
            descriptor = -1
            return stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def get_logger(component: str) -> logging.Logger:
    """Return a configured component logger with JSON output to stderr."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{component}")
    if not logger.handlers:
        formatter = _JsonFormatter()
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        log_path = os.environ.get("EPHEMERAL_LOG_FILE", "").strip()
        if log_path:
            try:
                file_handler = _PrivateFileHandler(log_path, encoding="utf-8")
            except OSError:
                file_handler = None
            if file_handler is not None:
                diagnostics_mode = os.environ.get("EPHEMERAL_LOG_DIAGNOSTICS", "").strip().lower()
                file_handler.setFormatter(
                    _JsonFormatter(include_protected_traceback=diagnostics_mode == "traceback")
                )
                logger.addHandler(file_handler)
        logger.propagate = False
        level_name = os.environ.get("EPHEMERAL_LOG_LEVEL", "WARNING").upper()
        logger.setLevel(getattr(logging, level_name, logging.WARNING))
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event without including captured content."""
    logger.log(level, event, extra={"structured_fields": fields})
