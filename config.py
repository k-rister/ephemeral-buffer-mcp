"""Environment-backed server configuration helpers."""

import os
import sys


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
