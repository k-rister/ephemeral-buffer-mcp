"""Configuration domain handlers."""

from ..server import *  # noqa: F403
from .registry import register_tool


@register_tool("set_semantic_index_budget", "configuration")
@_instrument_tool("set_semantic_index_budget")
def set_semantic_index_budget(max_indexed_chunks: int) -> str:
    """Adjust this session's semantic-index chunk budget when explicitly enabled."""
    if not runtime_index_budget_adjustment_enabled():
        return (
            "Error: runtime semantic-index budget adjustment is disabled; set "
            "EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1 at server startup to enable it."
        )
    try:
        return json.dumps(_active_engine().set_max_indexed_chunks(max_indexed_chunks), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return f"Error adjusting semantic-index budget: {exc}"


# --- Unix Domain Socket IPC for CLI piping (ephbuf) ---
