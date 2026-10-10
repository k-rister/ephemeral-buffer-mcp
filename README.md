# Ephemeral Buffer MCP Server (`ephemeral-buffer`)

[![CI](https://github.com/k-rister/ephemeral-buffer-mcp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/k-rister/ephemeral-buffer-mcp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/k-rister/ephemeral-buffer-mcp)](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ephemeral-buffer-mcp)](https://pypi.org/project/ephemeral-buffer-mcp/)
[![codecov](https://codecov.io/gh/k-rister/ephemeral-buffer-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/k-rister/ephemeral-buffer-mcp)

Ephemeral Buffer is an MCP server and command-line client for capturing command
output in a bounded, searchable buffer. Coding agents can search captures and
retrieve relevant portions instead of carrying entire build and test logs in
their conversation context.

## Quick start

Requires Python 3.10 or newer. Install the package in an isolated environment.

**macOS and Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ephemeral-buffer-mcp
```

Configure your MCP client using the [user guide](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/user-guide.md). For an isolated
server, the server and `ephbuf` must use the same unique session ID. On
macOS/Linux, export the ID before starting Codex with the launcher installed in
the virtual environment:

```bash
export EPHEMERAL_SESSION_ID=quick-start-session
.venv/bin/codex-ephemeral
```

Set the same value in another MCP client's server environment. Then capture
command output with `ephbuf`, passing the same ID.

**macOS and Linux**

```bash
.venv/bin/python -c 'print("Hello from a captured command")' 2>&1 |
  EPHEMERAL_SESSION_ID=quick-start-session .venv/bin/ephbuf --label "quick start"
```

Windows installation steps are in the [user guide's Windows support section](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/user-guide.md#windows-support). Windows runtime behavior isn't covered by CI; that section describes the unverified CLI capture and process-control paths.

The agent can then search the capture and retrieve bounded slices through the MCP tools.

## Context reduction in one agent comparison

In a five-repetition `gpt-6-luna` comparison run on 2026-10-09, mean MCP
tool-response bytes fell from 6,681 to 1,501 (77.5%); the Codex context proxy
fell from 23,910 to 17,643 bytes (26.2%). This is a result from one workload,
not a general savings guarantee. The context figure is a prompt/event-envelope
proxy, not the model's internal context size. See the [benchmark methodology
and comparison details](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/benchmarks.md#context-reduction-comparison-2026-10-09).

## Documentation

| Topic | Document |
|---|---|
| Installation, client setup, and privacy | [User guide](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/user-guide.md) |
| MCP tools and advanced workflows | [Tool reference](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/tool-reference.md) |
| Data flow | [Architecture](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/architecture.md) |
| Benchmark commands, metrics, and comparisons | [Benchmarks](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/docs/benchmarks.md) |
| Runtime configuration and operations | [Operations guide](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/OPERATIONS.md) |
| Development setup and contribution workflow | [Contributing](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/CONTRIBUTING.md) |
| Release history | [Changelog](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/CHANGELOG.md) |
