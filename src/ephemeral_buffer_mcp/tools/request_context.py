"""Request-local, content-free MCP argument metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any


_SUPPLIED_ARGUMENTS: ContextVar[frozenset[str] | None] = ContextVar(
    "ephemeral_mcp_supplied_arguments", default=None
)
_REQUEST_SHAPE: ContextVar[dict[str, str] | None] = ContextVar(
    "ephemeral_mcp_request_shape", default=None
)
_CaptureTokens = tuple[Token[frozenset[str] | None], Token[dict[str, str] | None]]
_CAPTURE_STACK: ContextVar[tuple[_CaptureTokens, ...]] = ContextVar(
    "ephemeral_mcp_request_context_stack", default=()
)


def _safe_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other"


def capture_mcp_supplied_arguments(arguments: Any) -> None:
    if not isinstance(arguments, Mapping):
        return
    safe_items = {
        name: _safe_type_name(value)
        for name, value in arguments.items()
        if isinstance(name, str) and name.isidentifier() and not name.startswith("_")
    }
    tokens = (_SUPPLIED_ARGUMENTS.set(frozenset(safe_items)), _REQUEST_SHAPE.set(safe_items))
    _CAPTURE_STACK.set((*_CAPTURE_STACK.get(), tokens))


def reset_mcp_supplied_arguments() -> None:
    stack = _CAPTURE_STACK.get()
    if not stack:
        return
    supplied_token, shape_token = stack[-1]
    _SUPPLIED_ARGUMENTS.reset(supplied_token)
    _REQUEST_SHAPE.reset(shape_token)
    _CAPTURE_STACK.set(stack[:-1])


def current_mcp_supplied_arguments() -> frozenset[str] | None:
    return _SUPPLIED_ARGUMENTS.get()


def current_mcp_request_shape() -> dict[str, str] | None:
    return _REQUEST_SHAPE.get()
