"""Configuration domain handlers."""

import inspect

from ..server import *  # noqa: F403
from .registry import register_tool


@register_tool("set_semantic_index_budget", "configuration")
@_instrument_tool("set_semantic_index_budget")
def set_semantic_index_budget(
    max_indexed_chunks: int,
    __invocation_mode: str = "direct",
) -> str:
    """Adjust this session's semantic-index chunk budget when explicitly enabled."""
    if not runtime_index_budget_adjustment_enabled():
        message = (
            "runtime semantic-index budget adjustment is disabled; set "
            "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1 at server startup to enable it."
        )
        if __invocation_mode == "mcp":
            return _execution_error_json("index_budget_disabled", message)
        return f"Error: {message}"
    try:
        return json.dumps(_active_engine().set_max_indexed_chunks(max_indexed_chunks), sort_keys=True)
    except (TypeError, ValueError) as exc:
        if __invocation_mode == "mcp":
            return _execution_error_json("invalid_index_budget", str(exc))
        return f"Error adjusting semantic-index budget: {exc}"


_set_semantic_index_budget_signature = inspect.signature(set_semantic_index_budget)
set_semantic_index_budget.__signature__ = _set_semantic_index_budget_signature.replace(
    parameters=tuple(
        parameter
        for parameter in _set_semantic_index_budget_signature.parameters.values()
        if parameter.name != "__invocation_mode"
    )
)


# --- Unix Domain Socket IPC for CLI piping (ephbuf) ---
