"""Privacy-safe structured logging helpers for runtime operations."""

import json
import logging
import os
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


def get_logger(component: str) -> logging.Logger:
    """Return a configured component logger with JSON output to stderr."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{component}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        level_name = os.environ.get("EPHEMERAL_LOG_LEVEL", "WARNING").upper()
        logger.setLevel(getattr(logging, level_name, logging.WARNING))
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event without including captured content."""
    logger.log(level, event, extra={"structured_fields": fields})
