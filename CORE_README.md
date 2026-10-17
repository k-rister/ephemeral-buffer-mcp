# Ephemeral Buffer MCP Core

This package contains the shared runtime used by the full-featured and
lexical-only Ephemeral Buffer MCP distributions. Most users should install
`ephemeral-buffer-mcp`, which includes semantic and hybrid search. To request a
reduced install explicitly, install `ephemeral-buffer-mcp-lexical`.

The core distribution provides capture, retrieval, BM25 search, the server,
and CLI entry points. Semantic search dependencies are available through the
`semantic` extra and are selected by the default full-featured distribution.
