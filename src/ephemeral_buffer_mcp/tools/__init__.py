"""Domain tool handlers and their transport-neutral registration metadata."""

from __future__ import annotations

from importlib import import_module

from .registry import (
    TOOL_CATEGORIES,
    ToolRegistration,
    get_tool_categories,
    get_tool_names,
    get_tool_registration,
    get_tool_registrations,
    get_socket_tool_registration,
    register_tool,
)


_DOMAIN_MODULES = (
    "execution",
    "capture",
    "search",
    "retrieval",
    "diagnostics",
    "configuration",
)


def load_tool_handlers() -> tuple[ToolRegistration, ...]:
    """Import each domain once and return registrations in stable source order."""
    for module_name in _DOMAIN_MODULES:
        import_module(f".{module_name}", __name__)
    return get_tool_registrations()


__all__ = [
    "TOOL_CATEGORIES",
    "ToolRegistration",
    "get_tool_categories",
    "get_tool_names",
    "get_tool_registration",
    "get_tool_registrations",
    "get_socket_tool_registration",
    "load_tool_handlers",
    "register_tool",
]
