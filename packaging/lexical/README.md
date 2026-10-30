# Ephemeral Buffer MCP lexical-only distribution

Requires Python 3.11 or newer.

This distribution installs the shared runtime without NumPy or the ONNX
embedding dependencies.
Capture, slice, and BM25 search remain available. Hybrid and semantic search
return an error directing users to install the full `ephemeral-buffer-mcp`
distribution.

Install with:

```bash
python -m pip install ephemeral-buffer-mcp-lexical
```
