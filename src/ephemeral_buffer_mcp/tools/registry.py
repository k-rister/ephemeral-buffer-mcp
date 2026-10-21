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
    mcp_admission_lane: str = "mcp"
    socket_admission_lane: str = "socket"
    background_execution: bool = False
    socket_request_adapter: Callable[..., Any] | None = None
    socket_response_adapter: Callable[..., Any] | None = None
    legacy_error_code: str | None = None


_REGISTRATIONS: list[ToolRegistration] = []
_REGISTRATIONS_BY_NAME: dict[str, ToolRegistration] = {}


def register_tool(
    name: str,
    category: str,
    *,
    structured_result_factory: Callable[..., Any] | None = None,
    mcp_admission_lane: str = "mcp",
    socket_admission_lane: str = "socket",
    background_execution: bool = False,
    socket_request_adapter: Callable[..., Any] | None = None,
    socket_response_adapter: Callable[..., Any] | None = None,
    legacy_error_code: str | None = None,
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
            mcp_admission_lane=mcp_admission_lane,
            socket_admission_lane=socket_admission_lane,
            background_execution=background_execution,
            socket_request_adapter=socket_request_adapter,
            socket_response_adapter=socket_response_adapter,
            legacy_error_code=legacy_error_code,
        )
        if (socket_request_adapter is None) != (socket_response_adapter is None):
            raise ValueError(
                "socket request and response adapters must be registered together"
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


def get_socket_tool_registration() -> ToolRegistration:
    """Return the unique tool registration exposed by the socket transport."""
    registrations = tuple(
        item for item in _REGISTRATIONS if item.socket_request_adapter is not None
    )
    if len(registrations) != 1:
        raise RuntimeError(
            "the socket transport requires exactly one registered socket entrypoint"
        )
    return registrations[0]


def get_tool_names() -> list[str]:
    return [registration.name for registration in _REGISTRATIONS]


def get_tool_categories() -> dict[str, str]:
    return {
        registration.name: registration.category
        for registration in _REGISTRATIONS
    }
