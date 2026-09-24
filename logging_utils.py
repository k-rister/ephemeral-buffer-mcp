"""Privacy-safe structured logging helpers for runtime operations."""

import errno
import json
import logging
import os
import stat
import sys
import time
from typing import Any


LOGGER_NAME = "ephemeral_buffer"


class _JsonFormatter(logging.Formatter):
    """Format operational events as one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "structured_fields", {})
        payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
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
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        logger.propagate = False
        level_name = os.environ.get("EPHEMERAL_LOG_LEVEL", "WARNING").upper()
        logger.setLevel(getattr(logging, level_name, logging.WARNING))
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event without including captured content."""
    logger.log(level, event, extra={"structured_fields": fields})
