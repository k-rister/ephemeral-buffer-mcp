"""Shared response-size and metadata limits for tools and transports."""

EXECUTION_OUTPUT_RESPONSE_MAX_BYTES = 64 * 1024
MCP_TOOL_RESPONSE_MAX_BYTES = 64 * 1024
MCP_JSONRPC_ENVELOPE_RESERVE_BYTES = 1024
SEARCH_RESPONSE_MAX_BYTES = MCP_TOOL_RESPONSE_MAX_BYTES - MCP_JSONRPC_ENVELOPE_RESERVE_BYTES
SEARCH_QUERY_MAX_BYTES = 1024
SEARCH_QUERY_TRUNCATION_MARKER = "... [query truncated]"
SEARCH_STRUCTURED_MATCH_MAX_BYTES = 2 * 1024
SEARCH_STRUCTURED_CONTEXT_MAX_BYTES = 1024
SEARCH_SNIPPET_TRUNCATION_MARKER = (
    "... [snippet truncated; use the Context range above with get_capture_slice for full content]"
)
SEARCH_MATCHES_OMITTED_MARKER = (
    "... [Additional matches omitted; the search response reached its byte budget]"
)
SEARCH_SNIPPET_AND_MATCHES_TRUNCATION_MARKER = (
    "... [snippet truncated; Additional matches omitted. "
    "Use the Context range above with get_capture_slice for full content]"
)
SUMMARY_COMMAND_MAX_BYTES = 1024
SUMMARY_COMMAND_TRUNCATION_MARKER = "... [command truncated]"
SUMMARY_LABEL_MAX_BYTES = 1024
SUMMARY_LABEL_TRUNCATION_MARKER = "... [label truncated]"
SUMMARY_DIFF_FILE_MAP_MAX_BYTES = 8 * 1024
SUMMARY_DIFF_FILE_MAP_MAX_ENTRIES = 100

CONSOLIDATION_MAX_CAPTURES = 25
CONSOLIDATION_CAPTURE_ID_MAX_BYTES = 256
