"""Request-local, content-free MCP argument metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any


_SUPPLIED_ARGUMENTS: ContextVar[frozenset[str] | None] = ContextVar(
    "ephemeral_mcp_supplied_arguments", default=None
)
_CAPTURE_STACK: ContextVar[tuple[Token[frozenset[str] | None], ...]] = ContextVar(
    "ephemeral_mcp_supplied_arguments_stack", default=()
)


def capture_mcp_supplied_arguments(arguments: Any) -> None:
    if not isinstance(arguments, Mapping):
        return
    names = frozenset(
        name
        for name in arguments
        if isinstance(name, str) and name.isidentifier() and not name.startswith("_")
    )
    token = _SUPPLIED_ARGUMENTS.set(names)
    _CAPTURE_STACK.set((*_CAPTURE_STACK.get(), token))


def reset_mcp_supplied_arguments() -> None:
    stack = _CAPTURE_STACK.get()
    if not stack:
        return
    _SUPPLIED_ARGUMENTS.reset(stack[-1])
    _CAPTURE_STACK.set(stack[:-1])


def current_mcp_supplied_arguments() -> frozenset[str] | None:
    return _SUPPLIED_ARGUMENTS.get()
