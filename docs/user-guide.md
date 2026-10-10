# User guide

This guide covers installation, client setup, command capture, and privacy controls.

## 📦 Installation

### Requirements

- Python 3.10 or newer
- MCP Python SDK (`mcp`) 2.3.0 or newer within the 2.x series
- A supported MCP client if you want to use the server from an AI coding assistant
- Network access on first use if FastEmbed needs to download its embedding model

The package is installed from PyPI as `ephemeral-buffer-mcp`. It provides both
the MCP server and the `ephbuf` command-line client.

### Recommended: install from PyPI

Create an isolated virtual environment and install the latest published
package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ephemeral-buffer-mcp
```

Verify that the CLI is available:

```bash
.venv/bin/ephbuf --help
```

#### Windows support

Use the equivalent commands from the virtual environment's `Scripts`
directory:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install ephemeral-buffer-mcp
.venv\Scripts\ephbuf.exe --help
```

These commands document package and CLI entry-point installation; they do not
establish full runtime support. The project CI currently runs on Ubuntu only.
The MCP server and `ephbuf` exchange captures over an `AF_UNIX` socket, and
command timeout and cancellation use Unix process-group APIs, so those paths
are unverified on Windows. Durable execution leases explicitly require Linux.

#### Upgrade or remove the package

On macOS and Linux:

```bash
.venv/bin/python -m pip install --upgrade ephemeral-buffer-mcp
.venv/bin/python -m pip uninstall ephemeral-buffer-mcp
```

On Windows (PowerShell):

```powershell
.venv\Scripts\python.exe -m pip install --upgrade ephemeral-buffer-mcp
.venv\Scripts\python.exe -m pip uninstall ephemeral-buffer-mcp
```

### Install from a source checkout

Use an editable install when developing or testing local changes:

```bash
git clone https://github.com/k-rister/ephemeral-buffer-mcp.git
cd ephemeral-buffer-mcp
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The editable install exposes the same `ephbuf` command and MCP server as the
PyPI installation. The reproducible, locked contributor environment is
documented in [contributor setup and checks](../CONTRIBUTING.md).

### Python embedding API

The supported Python embedding interface is exported by the
`ephemeral_buffer_mcp` package:

```python
from ephemeral_buffer_mcp import create_mcp_server, create_service_context
```

These package-level exports are the maintained Python callable API.
`python -m ephemeral_buffer_mcp.server` is the documented server process entry
point. The remaining package modules implement service internals. See the
[contributor guide](../CONTRIBUTING.md) for the source layout and the
[benchmark guide](benchmarks.md) for benchmark commands.

When launching from a source checkout with `run.sh`, the launcher prefers
`.venv/bin/python`, then supports the legacy `venv/bin/python` layout, before
using an explicit `PYTHON` override or `python3`. An invalid `PYTHON` override
fails with an actionable error.

### Start the MCP server manually

The MCP server uses stdio for communication with the MCP host. Start it with
the Python interpreter from the environment where the package was installed:

```bash
.venv/bin/python -m ephemeral_buffer_mcp.server
```

Normally you should let your MCP client start this process automatically. Do
not start a separate server for every shell command: the `ephbuf` CLI sends
captures to the running server over its local Unix socket.
The private CLI socket uses versioned length-prefixed request and response
frames, so fragmented reads do not depend on half-closing the connection.

### Embed an isolated service

Python callers can create an owned service context and bind an MCP application
to it. The context owns its engine, metrics, lazy execution manager, and metrics
snapshot destination; call `close()` when the host is done with the app.

```python
from ephemeral_buffer_mcp import create_mcp_server, create_service_context

context = create_service_context(metrics_file=None)
app = create_mcp_server(context)
try:
    # Mount or run `app` with the embedding host.
    ...
finally:
    context.close()
```

`context.start()` starts optional embedding warm-up. With the default
process-private execution identity, creating a context does not create the
execution-state directory; durable execution creates it on first use. Pass a
distinct `state_dir` when contexts need separate durable namespaces. Contexts
created with `create_service_context()` keep metrics in memory by default; pass
a context-specific `metrics_file` to persist a snapshot. The standard
`python -m ephemeral_buffer_mcp.server` entrypoint continues to use the
default context.

### Start an isolated Codex session

When using the global Codex MCP configuration, start Codex through the
`codex-ephemeral` launcher installed in the package environment. From a source
checkout, use `./codex-ephemeral`. The launcher creates a unique
`EPHEMERAL_SESSION_ID` when no session ID or explicit socket path was supplied.
It forwards the
session or explicit socket and state paths, along with supported EB settings,
through Codex's MCP configuration so they remain available when Codex
sanitizes the child environment.
The global configuration requires this identity, so starting Codex directly
will fail closed instead of attaching to another session's socket.

#### Update Codex's MCP server command after an upgrade

Codex stores its user configuration in `~/.codex/config.toml`, or in
`$CODEX_HOME/config.toml` when `CODEX_HOME` is set ([Codex configuration
reference](https://learn.chatgpt.com/docs/config-file/config-reference)). The
`codex-ephemeral` launcher supplies session environment values; it does not
change the configured server command or arguments. After upgrading from a
release that used the old top-level `server` module, update the existing
`ephemeral-buffer` entry to the package entry point:

```toml
[mcp_servers.ephemeral-buffer]
command = "/absolute/path/to/.venv/bin/python"
args = ["-m", "ephemeral_buffer_mcp.server"]
```

Use the Python executable from the environment where `ephemeral-buffer-mcp`
is installed. Change the `command` and `args` in the existing table, keeping
any other settings your setup needs. For a source checkout, the equivalent is
the checkout's `run.sh` with `args = []`. Save the config and restart Codex.
Confirm that the MCP tools appear, then call `get_runtime_diagnostics` and
check that the package version and socket lifecycle are correct. A package
upgrade does not rewrite Codex's config file.

The launcher also enables `EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET=1` because some
Codex execution environments deny Unix-socket creation. In that case MCP over
stdio remains available and runtime diagnostics report the socket failure;
`ephbuf` CLI support remains available when the environment permits socket
creation. Because the session is private, the launcher also defaults
`EPHEMERAL_METRICS=1` and passes that setting explicitly to the MCP server. Set
`EPHEMERAL_METRICS=0` before invoking the launcher to disable local metrics for
that session. The launcher also sets `EPHEMERAL_LOG_FILE` to a JSONL file
beside the session socket and defaults `EPHEMERAL_LOG_LEVEL` to `INFO`, so MCP
tool start and completion events are recorded. Set `EPHEMERAL_LOG_LEVEL` before
launching to choose another level, such as `WARNING` or `DEBUG`, or pass
`--log-level LEVEL` directly to either launcher:

```bash
./codex-ephemeral --log-level INFO
./ephemeral-agent --log-level INFO agy
```

Set `EPHEMERAL_LOG_FILE` to override the log path.

Socket and durable-state identity follow one precedence rule. An explicit
`EPHEMERAL_EXECUTION_STATE_DIR` selects the state directory. Otherwise an
explicit `EPHEMERAL_SOCKET_PATH` identifies both the socket and the derived
state directory. Without an explicit socket, `EPHEMERAL_SESSION_ID` derives
both paths. With neither identity value, EB keeps the legacy shared socket and
uses a private process state directory that is removed on normal shutdown
where secure cleanup is available. The session helper does not create an
unrelated session ID when an explicit socket path is supplied.
Use absolute paths for explicit socket and state directories in manually
configured MCP clients; the supplied shell launchers normalize them before
starting child processes.

EB parses its supported startup settings once per process. Invalid numeric and
boolean values are reported to stderr and fall back to the documented default;
empty optional values remain unset. `get_runtime_diagnostics()` includes the
effective paths, identity sources, durability mode, and a typed settings
snapshot with value, origin, validation status, and invalid-value behavior. It
reports a short session fingerprint instead of the raw session ID. The runtime
semantic-index budget permission is sampled for each adjustment request; other
settings use their startup snapshot.

### Configure other coding agents

The server is not Codex-specific. Any MCP client that launches a local stdio
server can use the same command and environment settings:

```json
{
  "mcpServers": {
    "ephemeral-buffer": {
      "command": "/absolute/path/to/ephemeral-buffer-mcp/run.sh",
      "args": [],
      "env": {
        "EPHEMERAL_REQUIRE_ISOLATION": "1",
        "EPHEMERAL_SESSION_ID": "agent-session-unique-value",
        "EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET": "1"
      }
    }
  }
}
```

This shape applies to clients such as Claude-style `.mcp.json` and
Gemini/Antigravity-style `mcp_config.json`. Use a different
`EPHEMERAL_SESSION_ID` for every concurrent agent session. If the client
supports environment interpolation, use its per-session identifier; otherwise
generate an ID when starting the agent and place that same value in the
agent's shell environment:

```bash
export EPHEMERAL_SESSION_ID="agent-$(date +%s)-$$"
export EPHEMERAL_REQUIRE_ISOLATION=1
export EPHEMERAL_ALLOW_STDIO_WITHOUT_SOCKET=1
```

The MCP server and `ephbuf` CLI must inherit the same session ID or explicit
socket path. An MCP client that does not pass environment variables to child
processes can still use the MCP tools, but its separate `ephbuf` shell
commands will need the same identity configured independently.

For other shell-launched agents, use the installed `ephemeral-agent` command
when the package environment is active, or `./ephemeral-agent` from a source
checkout:

```bash
ephemeral-agent claude
ephemeral-agent gemini
```

Or source the environment into an existing shell before launching the agent:

```bash
source ephemeral-session-env
claude   # or another coding-agent command
```

The session environment helper defaults `EPHEMERAL_METRICS=1` for the private
agent session; set `EPHEMERAL_METRICS=0` before sourcing it when metrics are not
wanted.

GUI-launched agents still need their MCP configuration to provide a unique
session ID per conversation; these scripts cannot modify an already-running
GUI process.

### Configure an MCP client

MCP clients generally need the command, arguments, and environment used to
start a stdio server. A portable configuration looks like this:

```json
{
  "mcpServers": {
    "ephemeral-buffer": {
      "command": "/absolute/path/to/ephemeral-buffer-mcp/.venv/bin/python",
      "args": ["-m", "ephemeral_buffer_mcp.server"]
    }
  }
}
```

Replace the command with the absolute path to the virtual-environment Python
interpreter. Using an absolute path avoids differences between GUI-launched
clients and interactive shells. On Windows, use a path such as
`C:\\path\\to\\ephemeral-buffer-mcp\\.venv\\Scripts\\python.exe`.

For clients that provide a command-line registration command, register the
same executable and arguments conceptually as:

```text
command: /absolute/path/to/.venv/bin/python
arguments: -m ephemeral_buffer_mcp.server
```

After saving the configuration, restart or reload the MCP client and confirm
that the `ephemeral-buffer` tools appear. The server exposes capture, search,
summary, slice, consolidation, diagnostics, and cleanup tools; the `ephbuf`
CLI is a separate convenience client for shell output.

### Isolate concurrent agent sessions

For a coding agent that may run alongside another agent, configure the same
session ID or explicit socket path for the MCP server and the `ephbuf` CLI:

```json
"env": {
  "EPHEMERAL_REQUIRE_ISOLATION": "1",
  "EPHEMERAL_SESSION_ID": "agent-session-1"
}
```

The server and CLI resolve the same socket and state identity. A missing
session ID and explicit socket path fails closed instead of falling back to
the shared legacy socket. An explicit state directory overrides derived state;
otherwise an explicit socket path derives state before a session ID does. The
MCP initialization instructions describe this policy to the client, and the
environment checks enforce isolation independently.

The CLI bounds each socket connect, send, and receive operation to 10 seconds
by default. The server uses the same setting to bound how long an admitted
client can take to send a complete request frame. Set
`EPHEMERAL_SOCKET_TIMEOUT_SECONDS` to a positive number of seconds when a
different limit is appropriate; CLI timeout failures return a nonzero result,
and stalled server-side reads return `socket_read_timeout` and release their
admission slot.

### Background model warm-up

After socket startup succeeds, the server loads FastEmbed and runs a small
deterministic embedding in a background thread. MCP startup and BM25 search do
not wait for this work. A semantic request arriving during warm-up waits on the
same model lock instead of starting a duplicate load. Subsequent operations use
the local model cache. To disable warm-up for lexical-only or memory-constrained
deployments, or to select a compatible model and cache location, set:

```bash
export EPHEMERAL_EMBEDDING_WARMUP=0
export EPHEMERAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5-fp32"
export EPHEMERAL_FASTEMBED_CACHE_DIR="$HOME/.cache/ephemeral-buffer"
```

The default model is `BAAI/bge-small-en-v1.5-fp32`, the upstream fp32 ONNX
export of bge-small-en-v1.5 (about 130 MB), which the server registers with
FastEmbed itself. FastEmbed's own catalogue entry, selectable as
`BAAI/bge-small-en-v1.5`, is a smaller reduced-precision export that produces
identical vectors but whose matrix kernels do not parallelize on common CPU
hosts. In a release-preparation comparison on a 16-vCPU Linux x86_64 host,
the fp32 export reached 69.1 semantic chunks per second with 1.85 seconds
median indexing at 1,024 lines, versus 3.1 chunks per second and 41.15 seconds
for the catalogue file. That roughly 22x difference is host-specific evidence,
not a universal performance guarantee. Embedding inference uses ONNX Runtime's
default thread count; bound it with
`EPHEMERAL_EMBEDDING_THREADS` when the server shares a host with an
interactive coding agent, and check both settings on the deployment host with
`benchmark_semantic_index.py`. `get_buffer_stats` reports the active thread
setting.

```bash
export EPHEMERAL_EMBEDDING_THREADS=4
```

Warm-up failures do not stop the server. BM25 remains available, and hybrid
search returns lexical results with a `semantic_fallback` error-class field
when semantic initialization fails. `get_buffer_stats` and
`get_runtime_diagnostics` report warm-up as `not-started`, `loading`, `ready`,
`failed`, or `disabled`; failures expose only the exception class.

Semantic indexing uses bounded inference by default: at most 16 chunks and
4,096 padded token slots per model call. Chunks are grouped by tokenizer length,
and the retained embedding matrix is assembled incrementally. Tune these with
`EPHEMERAL_EMBEDDING_BATCH_SIZE` and
`EPHEMERAL_EMBEDDING_MAX_BATCH_TOKENS`. ONNX Runtime's CPU memory arena is
disabled by default; set `EPHEMERAL_EMBEDDING_CPU_MEM_ARENA=1` to enable it
after measuring its retained memory on your host.

Dense indexing also has a 4 MiB per-capture semantic-input budget, configurable
with `EPHEMERAL_SEMANTIC_MAX_INDEX_INPUT_BYTES`. It counts UTF-8 bytes across
semantic windows, including overlap. Above the limit, EB skips embedding
inference for that capture and returns BM25 results for semantic and hybrid
requests with `semantic_coverage: unavailable` and
`semantic_fallback: SemanticIndexBudgetExceeded`. The capture remains fully
available to lexical search and exact slices. These are inference-work bounds,
not a process RSS limit; effective values and over-budget capture counts appear
in buffer stats and runtime diagnostics.

Lexical and semantic search index different chunk grids. BM25 keeps four-line
sliding windows with two-line overlap so exact line ranges stay tight. Semantic
embeddings use separate windows packed from consecutive lines, closed at eight
lines or 1,024 UTF-8 bytes, whichever comes first, with no overlap. That keeps
every window inside the model's token limit and, because embedding cost tracks
total tokens, roughly halves the work by not embedding the overlap twice. Hybrid
ranking fuses the two grids in line space: each semantic window boosts lexical
hits whose first line it contains. If semantic windows overlap, a lexical hit
receives only the boost from the highest-ranked matching window. A semantic
window appears on its own only when no lexical hit belongs to it. Each match
reports `chunk_index` as `lexical` or `semantic` alongside its line range. Tune
the windows when output has very long or very short lines:

```bash
export EPHEMERAL_SEMANTIC_CHUNK_LINES=8
export EPHEMERAL_SEMANTIC_CHUNK_BYTES=1024
export EPHEMERAL_SEMANTIC_CHUNK_OVERLAP=0
```

Larger windows embed fewer chunks but dilute single-line signals and cost more
per token; overlap improves recall across window boundaries at proportional
embedding cost. Compare strategies on the deployment host with
`benchmark_semantic_index.py --semantic-chunk-lines ... --semantic-chunk-overlap ...`.

Semantic indexing is prefetched after ingestion by default, so a search that
arrives after the background job finishes pays no indexing cost. Disable it for
lexical-only or CPU-constrained hosts, or bound the worker pool:

```bash
export EPHEMERAL_SEMANTIC_PREFETCH=0
export EPHEMERAL_SEMANTIC_PREFETCH_WORKERS=1
```

Every eligible capture is queued at ingestion and a bounded worker pool drains
the queue newest-first, since the latest capture is the most likely search
target; a burst of captures is never silently skipped. A semantic or hybrid
search waits for a job that is already running, and pulls a still-queued
capture out of the queue to index it on a separate on-demand pool so it does
not wait behind older work. That pool is bounded by the same worker count, so
searches that give up waiting on captures that are then evicted cannot
accumulate indexing threads. Failed jobs retry through the normal lazy path,
and evicted or cleared captures drop their queued prefetch and on-demand work.
Embedding inference is serialized by one model lock, so extra workers only
overlap bookkeeping; tune `EPHEMERAL_EMBEDDING_THREADS` instead of the worker
count for throughput. `get_buffer_stats` and `get_runtime_diagnostics` expose
only aggregate pending, queued, running, and failed counts.

Indexing cost grows linearly with capture size, so a hybrid search that arrives
before a large capture's index is ready waits at most a configurable budget:

```bash
export EPHEMERAL_SEMANTIC_WAIT_SECONDS=10
```

When the budget expires, hybrid search returns the BM25 results immediately
with `semantic_coverage` set to `pending` (the tool response says
`semantic pending (lexical only)`), and indexing continues in the background
so repeating the search returns full hybrid ranking. Every other hybrid
response reports `semantic_coverage` as `complete`, or `unavailable` when the
semantic backend failed or the semantic-input budget was exceeded and
`semantic_fallback` names the exception class;
BM25 results and exact line ranges are identical either way. The default of
`10` seconds kept a 2,048-line capture fully hybrid in the
release-preparation Linux run (about 4.0 seconds to index), while 8,192- and
16,384-line captures took about 17.0 and 32.9 seconds to index and therefore
answered lexical-first within the budget. Measured first-search p95 was about
4.0 seconds at 2,048 lines (complete) and 10.002 seconds at the larger sizes
(pending); every needle in the benchmark fixture still ranked first. These
figures are host-specific: Apple silicon and other Linux hosts can differ
materially, so run `benchmark_semantic_index.py` on the deployment host before
choosing a wait budget.
Set `0` to always answer lexical-first while the index builds, or `inf` to
wait for the index unconditionally. Semantic mode has no lexical result to
fall back on, so it always waits for the index. An empty capture has nothing
to index, so it reports `complete` coverage for semantic and hybrid searches.
`get_buffer_stats` reports the budget and the number of on-demand index jobs
running and queued.

The exact model and cache location can also be supplied in the MCP client's
`env` configuration. Keep the model cache writable by the user running the
MCP client.

### Installation choices at a glance

| Use case | Recommended installation |
| :--- | :--- |
| Normal user | PyPI install in a virtual environment |
| MCP host | PyPI install, then configure the installed server command |
| Shell/CLI use | PyPI install, then run `ephbuf` |
| CLI coding agent | PyPI install in a virtual environment, then run `ephemeral-agent` or `codex-ephemeral` |
| Contributor | Source checkout with `pip install -e .` |
| Release validation | Follow the procedures in [OPERATIONS.md](../OPERATIONS.md) |

If the MCP client cannot find the server, check the absolute interpreter path,
the selected Python environment, and the client's server logs. For socket,
capture-limit, logging, and deployment troubleshooting, see
[OPERATIONS.md](../OPERATIONS.md).

---

## Capture command output

### Capture from the terminal with `ephbuf`
You can pipe command output directly into the running MCP server:

```bash
# Pipe any command output into the buffer
pytest -v 2>&1 | ephbuf --label "pytest run"

# Pipe git diffs directly
git diff HEAD~3 | ephbuf --label "feature diff" --type diff

# Or wrap command execution
ephbuf --label "backend build" -- cargo build --verbose
```

The optional `--type`/`-t` hint accepts `auto` (the default), `diff`, `log`, or
`text`. Use `diff` for unified patches when automatic detection is ambiguous;
otherwise `auto` classifies diffs, build/test logs, and plain text from the
content and label.

`ephbuf` also bounds wrapped-command and piped-stdin capture with
`--max-output-bytes`; it defaults to `EPHEMERAL_MAX_BUFFER_BYTES` or 50 MiB and
retains the beginning and end of oversized output. Wrapped command output stays
on stdout, while the `ephbuf` execution banner and status messages go to stderr.
Use `--timeout-seconds` to stop a wrapped command after a bounded runtime; timed
out commands retain the output collected so far and exit with status 124.
Requested `max_output_bytes` and `capture_file` `max_bytes` values may not
exceed the configured buffer byte limit; the tools return a validation error
instead of silently clamping them.

For agent-side MCP operations, see the [MCP tool reference](tool-reference.md).

## Capture hygiene

Keep captures focused so search results remain useful and the agent receives
only the context it needs:

- Capture one command or related output stream at a time, using a descriptive
  label.
- Start with `get_capture_summary`, then use `search_capture` or
  `get_capture_slice` for targeted retrieval instead of repeatedly recapturing
  the same output.
- Use `clear_captures(capture_id)` when a capture is no longer needed; use
  `clear_captures("all")` between unrelated investigations.

The buffer is intentionally transient and bounded by the LRU capture limit,
but explicit cleanup prevents recent investigations from obscuring the active
one before automatic eviction occurs. Its memory metrics separate captured
content and embedding bytes from process RSS; the unaccounted RSS value includes
model, index, and Python object overhead and is approximate.

The server retains up to 256 captures by default. Override this limit with
`EPHEMERAL_MAX_CAPTURES`; the byte budget can be set with
`EPHEMERAL_MAX_BUFFER_BYTES`. Session-aware launchers can set
`EPHEMERAL_SESSION_ID` so each server/CLI pair automatically derives a unique
socket path; `EPHEMERAL_SOCKET_PATH` remains an explicit override. The byte
limit accounts for captured UTF-8 content plus its label; a capture is rejected
when their combined size exceeds the limit. `get_buffer_stats` also reports
embedding model readiness, embedding/cache settings, and process memory
separately.

Foreground MCP work and CLI socket clients have bounded active and queued
capacity. Defaults are 8 active and 16 queued MCP calls, plus 4 active and 8
queued socket clients. When a lane is full, the server returns a `server_busy`
error that can be retried. Configure these limits with
`EPHEMERAL_MAX_ACTIVE_TOOL_WORK`, `EPHEMERAL_MAX_QUEUED_TOOL_WORK`,
`EPHEMERAL_MAX_ACTIVE_SOCKET_CLIENTS`, and
`EPHEMERAL_MAX_QUEUED_SOCKET_CLIENTS`. `get_buffer_stats` reports active,
queued, and rejected work and reader-pinned storage that remains after capture
eviction, and has a reserved worker slot so it remains available under load.
The CLI can receive a busy response while uploading a large socket frame. These
concurrency limits make foreground work predictable; they do not cap process
RSS.

Shutdown stops admitting new work, requests cancellation of active command and
durable execution work, and drains for at most
`EPHEMERAL_SHUTDOWN_GRACE_SECONDS` (default `10`). If work remains, the
shutdown report and structured log identify unfinished work by type and
execution count. Each service context waits only for its own admitted calls;
work owned by another context in the same process does not hold up its cleanup.
Native embedding inference already running in a thread cannot
be forcibly stopped by cancelling its await; it is reported as unfinished and
may outlive the grace period. Python may keep the process alive until that
worker returns.

Indexed chunks are bounded separately by `EPHEMERAL_MAX_INDEXED_CHUNKS`, which
defaults to 32,768 total chunks across retained captures. LRU eviction makes
room for a new capture when possible. A capture that exceeds the entire index
budget is rejected rather than partially indexed, so accepted captures remain
fully searchable and cannot produce silent semantic false negatives. The
optional `set_semantic_index_budget` tool changes this limit for the current
session only and is disabled unless `EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1` is
set at startup. Decreases use the same deterministic LRU order and may evict
multiple captures; deployment defaults are not changed or persisted.
`execute_and_capture` retains the beginning and end of oversized command
output and marks the capture with its original byte count. Each returned head
or tail preview is independently capped at 4 KiB of UTF-8 data; a truncation
marker directs the agent to `get_capture_slice` for the complete content.

Call `get_runtime_diagnostics()` when reporting a field observation. It is
explicitly opt-in and returns operational metadata only; captured content,
labels, command arguments, and session ID values are excluded. Sanitize any
additional output before sharing it.

Operational events are written as privacy-safe JSON lines to stderr and, when
`EPHEMERAL_LOG_FILE` is set, to that file. Direct server launches default to
warnings and errors; the session launchers default to `INFO` and put the log
beside the session socket. Log files use owner-only `0600` permissions, and
symlink paths are rejected. If file logging cannot be opened safely, events
remain available on stderr and runtime diagnostics report the file as
unavailable. Tool lifecycle events contain only a local call ID, tool name,
duration, success state, and error class. Logs never include captured content,
labels, query text, or command text. A start event without a matching
completion or failure event identifies a stalled tool/session boundary.
The Codex A/B adapter can persist these events per MCP run with
`--diagnostic-log-dir /path/to/logs`; keep that directory outside the
repository. The files contain lifecycle metadata only.

## Optional local usage metrics

Set `EPHEMERAL_METRICS=1` to collect content-free, in-process usage metrics.
The isolated coding-agent launchers enable this setting by default; direct
server launches remain opt-in.
The metrics include per-tool call counts, success/failure counts, duration
totals, capture/search/retrieval, empty-search, eviction, and cleanup events,
plus interface coverage showing how many of the 22 exposed MCP tools were
called and the complete list of unused tools. MCP snapshots are scoped to the
active transport session and include a server-generated opaque
`attribution.id`; the persisted metrics file remains an aggregate process
snapshot. Interface coverage also reports descriptive coverage by primary
capability category; it is not a mandate for a client to use every category or
tool. The categories are:

| Category | Exposed tools |
| :--- | :--- |
| `capture` | `capture_text`, `capture_file`, `execute_and_capture`, `consolidate_captures` |
| `configuration` | `set_semantic_index_budget` |
| `diagnostics` | `preflight_command`, `get_buffer_stats`, `get_runtime_diagnostics`, `get_usage_metrics` |
| `execution` | `start_execution`, `resume_execution`, `cancel_execution`, `get_execution`, `get_execution_output`, `list_executions`, `get_execution_capacity`, `retire_executions` |
| `lifecycle` | `clear_captures` |
| `retrieval` | `get_capture_slice`, `get_capture_summary`, `list_captures` |
| `search` | `search_capture` |

They also include session-scoped data-path byte counters. The byte counters
cover input, retained, and original capture bytes; tool/search/retrieval
response bytes; and framed socket request/response bytes. They are disabled by
default, are never sent anywhere, and do
not retain captured content, labels, commands, or query text. When enabled,
`get_usage_metrics()` returns the versioned JSON form directly. Each enabled
response includes a `snapshot_token`; pass that token as `since` on a later
call to obtain a non-resetting task-window delta. A valid window reports
`window.status` as `ok`, including when all activity counters are zero. An
invalid, expired, or pre-restart token reports `window.status` as
`unavailable` instead of being interpreted as zero activity. `window.kind` is
`cumulative` for a full snapshot and `delta` for a snapshot based on a token.
Tokens are local to the active metrics scope and bounded in-memory token
history; a new isolated coding-agent session always starts a new measurement
scope. MCP
requests are additionally scoped to their transport session: the response
reports `scope: "mcp_session"` and an opaque `attribution.id`, and coverage,
funnel, byte, and per-tool counters are isolated between clients sharing one
server process. The ID is generated by the server and never contains the
configured session ID or socket path. Direct, unbound calls and the persisted
`EPHEMERAL_METRICS_FILE` snapshot use `scope: "process"` as the aggregate
fallback. Both `get_runtime_diagnostics()` and `get_buffer_stats()` include
the metrics snapshot for their active scope. Coverage uses the live MCP
registration inventory, so its available-tool count tracks the exposed API.
To keep long-running shared servers bounded, at most 128 process/client scope
states are retained; an inactive session whose state is evicted starts a fresh
metrics window if it later returns.
The `workflow_effectiveness` section derives operational signals from the
same process or task-window state. Funnel rates use explicit operation
denominators: capture-to-search uses `searches`, search-to-retrieval uses
`retrievals`, empty-search uses `searches`, and successful-call uses completed
calls (`successes + failures`). Response-byte ratios and reductions use
`capture_input_bytes` as the captured-byte baseline and require at least one
corresponding search or retrieval operation. A zero denominator or absent
operation is reported as `status: "unavailable"` with a null value. These
signals describe workflow behavior and response volume; they do not measure
answer quality or require clients to use every tool.
Each per-tool record also includes a bounded `latency_ms` distribution with
the call count and nearest-rank `p50`, `p95`, and `p99` estimates. The fixed
histogram uses millisecond upper-bound buckets from 1 ms through 60 seconds
plus an overflow bucket; a percentile whose rank lands in the overflow bucket
is reported as `null`, alongside `overflow_count`. Latency distributions are
additive in task-window deltas. Failed calls include zero-filled,
content-free `failure_categories` for `validation`, `timeout`, `socket`,
`embedding`, `eviction`, and `other`. These categories contain no exception
messages, commands, paths, queries, or capture identifiers.
Schema-invalid MCP arguments are measured at the MCPServer validation boundary
as failed calls in the `validation` category, even when the tool function does
not run. If the installed SDK does not support that instrumentation, normal
SDK validation and tool dispatch continue; `get_runtime_diagnostics` reports
which tools lack validation metrics.
The `semantic_index` section reports content-free indexing operations separately
for `prefetch` and `on_demand`. Each source includes queued, completed, failed,
cancelled, evicted, and cleared job counts; indexed chunk totals; and bounded
`queue_wait_ms` and `indexing_duration_ms` distributions. The `search` subsection
counts hybrid responses that were pending and semantic searches that fell back
to lexical results. Queue wait ends when embedding begins, while indexing
duration ends when the job publishes or fails, so host waiting and embedding
work remain distinguishable. `throughput` derives indexed chunks per indexing
second and reports an explicit unavailable status when its denominator is zero.
These counters are additive in task-window deltas and contain no capture
identifiers or content.
The event keys are stable
and zero-filled when no event has occurred, as are the byte-counter keys. Wire counts include framing
headers and payload bytes actually consumed, including partial malformed
requests; payload bytes rejected from an oversized frame before reading are
not counted. MCP tool-response counts measure UTF-8 response content and
exclude transport-envelope overhead. Metrics are scope-lifetime state:
restarting the server clears them, while capture-associated correlation state
is released when a capture is evicted or explicitly cleared. The process scope
is aggregate; MCP transport sessions use separate scopes.

## Effectiveness metrics and privacy

The built-in metrics are local operational telemetry. Direct server launches
are opt-in, while isolated coding-agent launchers enable them by default.
Nothing is uploaded or shared by the server. The metrics contain
counts, durations, byte sizes, and bounded lifecycle outcomes, but do not
retain captured content, labels, command arguments, or query text. Runtime
logs follow the same privacy model. Treat any captured output or diagnostic
excerpt as potentially sensitive and sanitize it before sharing.

Interpret the measurements in two separate layers:

| Layer | What it answers | What it cannot establish |
| :--- | :--- | :--- |
| Operational health | Did the server accept, store, search, retrieve, evict, and clean up requests? Were calls successful and how much local time or memory did they use? | That a search result was relevant, that the agent saw the right context, or that the user's task was completed. |
| Task-level effectiveness | Did a representative agent workflow find the needed signal, retrieve the right context, and complete its task? | A universal result from synthetic fixtures or a server-only benchmark. |

The effectiveness harness measures server-side behavior with deterministic
fixtures and does not invoke a coding-agent model. A successful targeted
retrieval means only that the fixture's expected marker was found. It is not a
measure of answer quality, search relevance in a real repository, token cost,
or end-to-end task completion. Use representative, privacy-reviewed tasks for
those questions and report the fixture, seed, repetition count, success rate,
useful-search rate, byte measurements, and local timing separately.

For a reproducible local diagnostic, start the server with
`EPHEMERAL_METRICS=1`, exercise the workflow, then request
`get_runtime_diagnostics()` and `get_buffer_stats()`. A safe bug report
includes the version/commit, Python/platform, configuration limits, operation
name, reproduction steps, and sanitized metric output; it excludes captures,
credentials, tokens, private paths, source code, user data, and raw queries.

See [OPERATIONS.md](../OPERATIONS.md) for deployment settings, troubleshooting,
release verification, and repository maintenance procedures.

---
