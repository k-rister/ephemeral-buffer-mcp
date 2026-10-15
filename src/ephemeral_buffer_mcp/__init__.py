"""Public package interfaces for the Ephemeral Buffer MCP service.

The supported interfaces are the installed commands, the MCP protocol, and the
Python embedding API exported here: ``create_mcp_server`` and
``create_service_context``. Other package modules are implementation details.
"""

from importlib import import_module

__all__ = ["create_mcp_server", "create_service_context"]


def __getattr__(name: str):
    if name in __all__:
        server = import_module(".server", __name__)
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
