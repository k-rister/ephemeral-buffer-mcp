"""Search domain handlers."""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..engine import MAX_SEARCH_CONTEXT_LINES, MAX_SEARCH_TOP_K
from ..mcp_response import (
    ToolErrorEnvelope,
    ToolResponseEnvelope,
    call_tool_result_bytes as _call_tool_result_bytes,
)
from ..response_limits import (
    SEARCH_QUERY_MAX_BYTES,
    SEARCH_QUERY_TRUNCATION_MARKER,
    SEARCH_RESPONSE_MAX_BYTES,
    SEARCH_SNIPPET_TRUNCATION_MARKER,
    SEARCH_STRUCTURED_CONTEXT_MAX_BYTES,
    SEARCH_STRUCTURED_MATCH_MAX_BYTES,
    SUMMARY_LABEL_MAX_BYTES,
    SUMMARY_LABEL_TRUNCATION_MARKER,
)
from ._context import (
    active_engine as _active_engine,
    active_metrics as _active_metrics,
    compact_results_enabled as _compact_tool_results_enabled_for_mcp,
)
from .formatting import bounded_summary_text
from .instrumentation import _instrument_tool
from .registry import register_tool
from .payloads import PAYLOAD_CONTRACTS, PAYLOAD_SCHEMA_VERSIONS


def _render_search_text(
    result: Dict[str, Any],
    matches: List[Dict[str, Any]],
    *,
    omitted: bool,
    query: Optional[str] = None,
) -> str:
    query = str(result.get("query", "") if query is None else query)
    display_query, _ = bounded_summary_text(
        query,
        SEARCH_QUERY_MAX_BYTES,
        SEARCH_QUERY_TRUNCATION_MARKER,
    )
    label, _ = bounded_summary_text(
        str(result.get("label", "")),
        SUMMARY_LABEL_MAX_BYTES,
        SUMMARY_LABEL_TRUNCATION_MARKER,
    )
    fallback_note = (
        f"Semantic fallback active ({result['semantic_fallback']})."
        if result.get("semantic_fallback")
        else None
    )
    pending_note = (
        "Semantic index still building; results are lexical (BM25) only. Repeat the search for hybrid ranking."
        if result.get("semantic_coverage") == "pending"
        else None
    )
    if not matches:
        if omitted:
            return (
                "Matches were found but omitted to stay within the structured search response budget. "
                "Use search_capture with a smaller top_k or get_capture_slice for full content."
            )
        return (
            f"No matches found for '{display_query}' in capture '{result.get('capture_id')}' ({label})."
            + (f" {fallback_note}" if fallback_note else "")
            + (f" {pending_note}" if pending_note else "")
        )

    mode_label = str(result.get("mode", "unknown"))
    if result.get("semantic_fallback"):
        mode_label += f"; lexical fallback ({result['semantic_fallback']})"
    elif result.get("semantic_coverage") == "pending":
        mode_label += "; semantic pending (lexical only)"
    out = [
        f'Search Results for: "{display_query}" [Mode: {mode_label}]',
        f"Capture: `{result.get('capture_id')}` ({label}, {result.get('total_lines', 0)} total lines)",
        f"Found {len(matches)} relevant section(s):\n",
    ]
    if fallback_note:
        out.insert(1, fallback_note)
    elif pending_note:
        out.insert(1, pending_note)

    for index, match in enumerate(matches, start=1):
        out.extend([
            f"### Match #{index} (Score: {match.get('score', 0)}, Range: {match.get('matched_range', 'unknown')}, Context: {match.get('context_range', 'unknown')})",
            "```text",
            str(match.get("snippet", "")),
            "```\n",
        ])
    if omitted:
        out.append(
            "Additional matches omitted because the structured search response reached "
            f"its {SEARCH_RESPONSE_MAX_BYTES:,}-byte budget. Use get_capture_slice for full content."
        )
    return "\n".join(out)


@_instrument_tool("search_capture")
def _search_capture_response(
    query: str,
    mode: str = "hybrid",
    capture_id: str = "latest",
    top_k: int = 5,
    context_lines: int = 3,
    __invocation_mode: str = "direct",
) -> ToolResponseEnvelope:
    if not isinstance(query, str):
        message = "Search Error: query must be a string"
        return ToolResponseEnvelope(
            status="error", error=ToolErrorEnvelope(code="invalid_query", message=message), text=message
        )
    if len(query.encode("utf-8")) > SEARCH_QUERY_MAX_BYTES:
        message = f"Search Error: query exceeds the {SEARCH_QUERY_MAX_BYTES:,}-byte limit"
        return ToolResponseEnvelope(
            status="error", error=ToolErrorEnvelope(code="query_too_large", message=message), text=message
        )
    response_budget = SEARCH_RESPONSE_MAX_BYTES
    res = _active_engine().search(
        query=query,
        mode=mode,
        capture_id=capture_id,
        top_k=top_k,
        context_lines=context_lines,
        response_budget_bytes=response_budget,
    )
    if res.get("status") == "error":
        message, _ = bounded_summary_text(
            str(res.get("message", "search failed")),
            SUMMARY_LABEL_MAX_BYTES,
            "... [error truncated]",
        )
        text = f"Search Error: {message}"
        return ToolResponseEnvelope(
            status="error",
            error=ToolErrorEnvelope(
                code=str(res.get("error_code") or "search_failed"),
                message=message,
            ),
            text=text,
        )

    engine_matches = res.get("matches", [])
    construction_omitted = bool(res.get("response_matches_omitted", False))
    _active_metrics().record_result_count("search_capture", len(engine_matches))
    metadata = {
        key: res[key]
        for key in (
            "capture_id", "label", "total_lines", "mode", "query", "semantic_coverage",
            "semantic_fallback", "semantic_index_state", "semantic_wait_seconds", "message",
        )
        if key in res
    }
    if "label" in res:
        metadata["label"], metadata["label_truncated"] = bounded_summary_text(
            str(res["label"]),
            SUMMARY_LABEL_MAX_BYTES,
            SUMMARY_LABEL_TRUNCATION_MARKER,
        )
    metadata.setdefault("query", query)
    text_result = dict(res)
    text_result.setdefault("query", query)
    structured_matches = []
    for match in engine_matches:
        snippet, snippet_truncated = bounded_summary_text(
            str(match.get("snippet", "")),
            SEARCH_STRUCTURED_MATCH_MAX_BYTES,
            SEARCH_SNIPPET_TRUNCATION_MARKER,
        )
        context, context_truncated = bounded_summary_text(
            str(match.get("context", "")),
            SEARCH_STRUCTURED_CONTEXT_MAX_BYTES,
            "... [raw context truncated; use get_capture_slice for full content] ...",
        )
        structured_matches.append({
            key: match[key]
            for key in (
                "chunk_id", "chunk_index", "score", "matched_range", "context_range",
                "context_start_line", "context_end_line",
            )
            if key in match
        } | {
            "snippet": snippet,
            "snippet_truncated": snippet_truncated or bool(match.get("snippet_truncated")),
            "context": context,
            "context_truncated": context_truncated or bool(match.get("context_truncated")),
        })

    selected: List[Dict[str, Any]] = []
    best: Optional[ToolResponseEnvelope] = None
    for index, match in enumerate(structured_matches):
        candidate_matches = selected + [match]
        omitted = construction_omitted or index < len(structured_matches) - 1
        data = {
            **metadata,
            "matches": candidate_matches,
            "match_count": len(candidate_matches),
            "available_match_count": len(structured_matches),
        }
        candidate = ToolResponseEnvelope(
            status="ok",
            data=data,
            text=_render_search_text(text_result, candidate_matches, omitted=omitted, query=query),
            truncated=omitted,
        )
        if _call_tool_result_bytes(
            candidate,
            compact_text=_compact_tool_results_enabled_for_mcp(__invocation_mode),
        ) > response_budget:
            break
        selected.append(match)
        best = candidate

    if best is None and engine_matches:
        # The normal per-match caps leave room for the first match, but retain a
        # stable bounded response if a future metadata field changes that.
        data = {**metadata, "matches": [], "match_count": 0, "available_match_count": len(engine_matches)}
        text = _render_search_text(text_result, [], omitted=True, query=query)
        best = ToolResponseEnvelope(status="ok", data=data, text=text, truncated=True)
    if best is None:
        text = _render_search_text(
            text_result,
            [],
            omitted=construction_omitted,
            query=query,
        )
        best = ToolResponseEnvelope(
            status="ok",
            data={**metadata, "matches": [], "match_count": 0},
            text=text,
            truncated=construction_omitted,
        )
    return best


@register_tool(
    "search_capture",
    "search",
    result_policy="structured_envelope",
    structured_result_factory=_search_capture_response,
    payload_model=PAYLOAD_CONTRACTS["search_capture"],
    payload_schema_version=PAYLOAD_SCHEMA_VERSIONS["search_capture"],
)
def search_capture(
    query: str,
    mode: Literal["hybrid", "bm25", "semantic"] = "hybrid",
    capture_id: str = "latest",
    top_k: Annotated[int, Field(ge=1, le=MAX_SEARCH_TOP_K)] = 5,
    context_lines: Annotated[int, Field(ge=0, le=MAX_SEARCH_CONTEXT_LINES)] = 3,
) -> str:
    """Search a capture by query text and return bounded readable and structured matches.

    mode is hybrid (default, lexical and semantic), bm25 (lexical), or semantic
    (embedding based). capture_id defaults to latest; top_k defaults to 5
    (maximum 20), and context_lines defaults to 3 (maximum 100). Use
    get_capture_slice for complete or omitted content. Hybrid results may be
    marked semantic pending or unavailable. The lexical-only distribution
    returns a semantic_backend_unavailable error for hybrid and semantic modes.
    """
    return search_capture_result(query, mode, capture_id, top_k, context_lines).text or ""


def search_capture_result(
    query: str,
    mode: Literal["hybrid", "bm25", "semantic"] = "hybrid",
    capture_id: str = "latest",
    top_k: Annotated[int, Field(ge=1, le=MAX_SEARCH_TOP_K)] = 5,
    context_lines: Annotated[int, Field(ge=0, le=MAX_SEARCH_CONTEXT_LINES)] = 3,
) -> ToolResponseEnvelope:
    """Return a typed search result while preserving ``search_capture`` text callers."""
    return _search_capture_response(query, mode, capture_id, top_k, context_lines)
