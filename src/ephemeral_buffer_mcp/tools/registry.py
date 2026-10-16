"""Transport-neutral registry for the service's callable tool handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


TOOL_CATEGORIES = (
    "capture",
    "configuration",
    "diagnostics",
    "execution",
    "lifecycle",
    "retrieval",
    "search",
)


@dataclass(frozen=True)
class ToolRegistration:
    """Metadata needed by transports to expose one domain-level tool."""

    function: Callable[..., Any]
    name: str
    category: str
    structured_result_factory: Callable[..., Any] | None = None


_REGISTRATIONS: list[ToolRegistration] = []
_REGISTRATIONS_BY_NAME: dict[str, ToolRegistration] = {}


def register_tool(
    name: str,
    category: str,
    *,
    structured_result_factory: Callable[..., Any] | None = None,
):
    """Register a transport-neutral handler and retain its public metadata."""
    if category not in TOOL_CATEGORIES:
        raise ValueError(f"unknown tool category: {category}")

    def decorator(function):
        current = _REGISTRATIONS_BY_NAME.get(name)
        registration = ToolRegistration(
            function=function,
            name=name,
            category=category,
            structured_result_factory=structured_result_factory,
        )
        if current is not None:
            if current != registration:
                raise ValueError(f"duplicate tool registration: {name}")
            return function
        _REGISTRATIONS.append(registration)
        _REGISTRATIONS_BY_NAME[name] = registration
        return function

    return decorator


def get_tool_registration(name: str) -> ToolRegistration:
    return _REGISTRATIONS_BY_NAME[name]


def get_tool_registrations() -> tuple[ToolRegistration, ...]:
    return tuple(_REGISTRATIONS)


def get_tool_names() -> list[str]:
    return [registration.name for registration in _REGISTRATIONS]


def get_tool_categories() -> dict[str, str]:
    return {
        registration.name: registration.category
        for registration in _REGISTRATIONS
    }
