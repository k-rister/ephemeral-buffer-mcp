# Ephemeral Buffer MCP lexical-only distribution

This distribution installs the shared runtime without NumPy or FastEmbed.
Capture, slice, and BM25 search remain available. Hybrid and semantic search
return an error directing users to install the full `ephemeral-buffer-mcp`
distribution.

Install with:

```bash
python -m pip install ephemeral-buffer-mcp-lexical
```
