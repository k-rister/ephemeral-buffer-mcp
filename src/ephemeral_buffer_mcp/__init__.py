"""Runtime implementation for the Ephemeral Buffer MCP server.

The supported interfaces are the installed commands and MCP protocol.
Modules in this package are implementation details unless documented otherwise.
"""
"""Public Python interface for embedding the Ephemeral Buffer service."""

from importlib import import_module

__all__ = ["create_mcp_server", "create_service_context"]


def __getattr__(name: str):
    if name in __all__:
        server = import_module(".server", __name__)
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
