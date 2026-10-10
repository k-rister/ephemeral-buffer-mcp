# MCP tool reference

Use this reference after configuring a client with the [user guide](user-guide.md).


The agent has access to the following tools:

The always-listed MCP tool descriptions stay concise. This guide retains the
extended routing, argument, search, safety, and recovery details. See the
[execution path guidance](#choose-the-execution-path-based-on-expected-duration-and-output),
[resumable phase execution](#resumable-phase-execution), and
[repository-sensitive paths and file capture](#repository-sensitive-paths-and-file-capture)
sections for full workflows.

| Tool | Purpose |
| :--- | :--- |
| `execute_and_capture(command, cwd, label, content_type='auto', max_output_bytes=None, timeout_seconds=None, structured_metrics=None)` | Synchronously executes a shell command with bounded capture and returns a compact versioned JSON summary containing status, duration, sizes, approximate token counts, truncation, warnings/errors, and optional structured metrics. Use when expected to finish within the client's tool-call window and noisy output benefits from later search; cancellation or caller disconnection requests subprocess cleanup. |
| `preflight_command(command, cwd=None)` | Performs content-free path, symlink, local Git-root, and executable-resolution diagnostics without executing the requested command. |
| `start_execution(phases, execution_id=None, label='', resume_policy='safe', cwd=None, timeout_seconds=None, max_output_bytes=None)` | Persists and starts a sequential, durably checkpointed execution, then returns its ID and current status promptly. Use when expected duration may approach or exceed the client's tool-call window, even for one command as one phase; follow with `get_execution` and `get_execution_output`. Requires Linux recovery support and has a combined capacity of eight active or queued executions. |
| `resume_execution(execution_id, retry_failed=False, confirm_unsafe=False)` | Resumes from the first incomplete phase and returns promptly; use `get_execution` to follow progress. Retries and unsafe side effects require explicit controls. |
| `cancel_execution(execution_id)` | Requests cancellation of active durable work by ID. Completed phases remain checkpointed; process cleanup uses the existing interruption and fencing path. |
| `get_execution(execution_id, include_output=False)` | Retrieves persisted phase metadata, event history, retry requirements, and human/machine-readable completion status. |
| `get_execution_output(execution_id, phase_name=None, offset=0, max_bytes=8192)` | Retrieves bounded persisted output for all phases or one phase, including after a server restart. The all-phase response shares its byte budget across phases; set `phase_name` to page through a phase with `offset`. |
| `list_executions(limit=20, offset=0)` | Lists a bounded page of durable executions and their partial/completed summaries; oversized pages return compact IDs with pagination metadata. |
| `get_execution_capacity()` | Reports durable record storage and quota diagnostics plus active and queued background execution counts; it does not attempt recovery. |
| `retire_executions(execution_ids, archive_path=None, dry_run=True)` | Previews eligibility and projected reclaimed space; actual retirement writes a private tar archive outside the state directory before removing selected record pairs. |
| `capture_text(content, label, content_type='auto', structured_metrics=None)` | Ingests text directly into the buffer and returns the same compact summary schema. |
| `capture_file(file_path, label, content_type='auto', max_bytes=None, structured_metrics=None)` | Ingests a bounded regular file from disk and returns the same compact summary schema; symlinks are followed, but pipes and devices are rejected. |
| `consolidate_captures(capture_ids, label, max_captures=25, max_bytes=None)` | Creates one bounded, searchable JSON capture from multiple captures while preserving source IDs and source line numbers. |
| `search_capture(query, mode, capture_id='latest', top_k, context_lines)` | Hybrid/BM25/Semantic search over the selected capture. BM25 splits underscores and punctuation—including regex-like characters—into alphanumeric terms, then combines those terms with OR. For example, `database_connection` searches for `database` or `connection`, not one underscore-containing term. Hybrid ranking gives lexical matches priority over semantic-only matches. Returns bounded match snippets, exact numeric context boundaries, bounded raw-context previews, line numbers, and whether the match came from the lexical or semantic chunk grid. `top_k` is limited to 20 and `context_lines` to 100. The complete structured MCP result is capped at 64 KiB; use `get_capture_slice` for omitted content. Hybrid search waits at most `EPHEMERAL_SEMANTIC_WAIT_SECONDS` for a large capture's semantic index and otherwise returns lexical results marked `semantic pending`; repeat the search for hybrid ranking. Captures beyond the semantic-input budget return BM25 results with semantic coverage `unavailable`. |
| `get_capture_slice(start_line, end_line, capture_id='latest', max_bytes=65536, cursor=None)` | Retrieves one byte-bounded page from a 1-indexed line range. A long line can continue across pages; repeat the original range and pass back `next_cursor` until it is null. `max_bytes` bounds the serialized MCP result and must be between 4 KiB and 64 KiB. The cursor is opaque; each segment reports a zero-based UTF-8 byte offset within its line, and pages never split a Unicode character. Joining `structuredContent.data.content` from successive pages reconstructs the retained newline-joined text exactly. |
| `get_capture_summary(capture_id, include_previews=False)` | Returns the compact JSON summary; opt into bounded head/tail previews only when needed. |
| `get_buffer_stats()` | Reports aggregate capture count, content bytes, lines, chunks, embedding model readiness, embedding bytes, semantic memory limits, accounted bytes, and process RSS. When local metrics are enabled, it also includes the content-free metrics snapshot for the active MCP session (or the aggregate process scope for direct calls). |
| `get_runtime_diagnostics()` | Opt-in, content-free report of runtime version, platform, uptime, socket and durable-state identity, durability, startup setting values and origins, active log file and level, buffer limits, embedding readiness, and process memory. |
| `get_usage_metrics(since=None)` | Returns a versioned, content-free JSON snapshot of local usage metrics, including interface coverage, per-tool counters, workflow events, byte counters, and scope- or task-window measurement timestamps. Pass a prior `snapshot_token` as `since` for a task-window delta. |
| `set_semantic_index_budget(max_indexed_chunks)` | Adjusts the session's semantic-index chunk budget when `EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1`; decreases evict least-recently-used captures as needed. |
| `list_captures()` | Lists active captures in the ring buffer. |
| `clear_captures(capture_id)` | Clears buffer. |

Capture tools return a compact JSON summary so an agent can decide whether it
needs the full output before spending context on retrieval. The summary uses
`schema_version: 1` and includes `status` (`captured`, `success`, `failed`, or
`timed_out`), `duration_ms`, retained and original byte sizes, approximate
token counts, truncation and partial-execution flags, typed warning/error
signals, and bounded caller-provided `structured_metrics`. The token values
are deterministic planning estimates based on four UTF-8 bytes per token;
they are not provider billing counts. Use `get_capture_summary` for the
compact form, set `include_previews=True` only when a head/tail sample is
useful, and use `get_capture_slice` or `search_capture` for complete or
targeted content. Diff file maps are also bounded and report omitted entries;
use `get_capture_slice` for the complete diff. Raw retained captures are
unchanged by summary generation.

Capture, search, retrieval, and clear tools return a versioned MCP result
envelope with `schema_version`, `status`, `data`, optional `error` (`code` and
`message`), and truncation or continuation metadata. MCP clients continue to
receive readable text in `content[0].text`; the same result data is also
available in `structuredContent`. The envelope's `status` is `ok` or `error`.
Stable retrieval and search error codes include `capture_not_found`,
`invalid_range`, `invalid_cursor`, `invalid_byte_budget`,
`response_budget_too_small`, `invalid_query`, `query_too_large`, and
`unsupported_search_mode`.

The Python text helpers `search_capture` and `get_capture_slice` retain their
string return values. Python callers that need the structured form can use
`search_capture_result` and `get_capture_slice_result`; these return the same
typed envelope and readable `.text` value. For retrieval, concatenate each
page's `data["content"]` value to reconstruct the retained content. Cursor
offsets count UTF-8 bytes from the start of the current line; treat the cursor
itself as opaque and pass it back unchanged.

MCP text content is compact by default: it reports status, useful IDs and
counts, and follow-up guidance without repeating full capture content or
search snippets already present in `structuredContent`. This policy is the
same for every model and client. Set `EPHEMERAL_COMPACT_TOOL_RESULTS=0` on the
server to restore legacy text output, including JSON text from capture tools,
for clients that consume only `TextContent`. The setting does not change
Python helper return values or the existing serialized response limits.

Python engine callers receive a frozen `CaptureView` from `ingest`,
`get_capture`, and `get_capture_view`. The historical `get_capture` method
remains as a compatibility wrapper; these views expose supported metadata but
not raw lines, chunks, embeddings, SQLite connections, or reader state. Retrieve
content through the bounded slice and search APIs. Semantic benchmarks should
use `load_embedding_model`, `index_capture`, `wait_for_capture_index`, and
`get_capture_diagnostics`; the underscored model and indexing methods are
engine internals.

The deterministic summary benchmark measures the initial agent-prompt
reduction for representative successful, failed, noisy, truncated, and
timed-out captures. It exercises the public capture API, compares a
parent-contract formatted-text response with the current compact
summary-first decision prompt, and verifies through the public slice API that
retained output can still be retrieved unchanged. The parent response is
reconstructed from the prior `execute_and_capture` contract; the current path
uses the public compact response. The proxy divides UTF-8 prompt bytes by four
and is intended for regression comparison, not provider billing or exact
token accounting. Execution summaries also bound command and label metadata
so unusually long shell commands cannot re-expand the initial response.

To generate the machine-readable benchmark record:

```bash
.venv/bin/python -m benchmarks.benchmark_effectiveness --summary --output /tmp/capture-summary.json
```

## Choose the execution path based on expected duration and output

- Use direct command execution for a small, targeted inspection where the
  output is already bounded and immediate terminal feedback is sufficient.
- Use `execute_and_capture` when a command is expected to finish within the
  current MCP client's tool-call window and noisy, large, or uncertain output
  benefits from bounded capture and later search. A noisy test or build can use
  this path when its expected runtime fits that window.
- Use `start_execution` when a command may approach or exceed the current
  tool-call window, regardless of output size. A single long-running command
  can be submitted as one phase; use `get_execution` to follow progress and
  `get_execution_output` to retrieve bounded output. This is an advisory
  heuristic, not a fixed duration threshold.
- Use `capture_text` when output is already in hand, or `capture_file` for a
  file that has been checked and intentionally selected for ingestion.

Use `preflight_command` when repository identity or path resolution is
uncertain before a sensitive command. It reports resolved facts and explicit
unavailable states without running the requested command or exposing command
output. It cannot predict shell expansion, aliases, pipelines, redirections,
environment changes, or arbitrary shell logic, so normal command validation
and user intent checks remain necessary.

For `execute_and_capture`, `content_type` accepts `auto` (default), `diff`, `log`, or `text`.
An omitted `label` is derived from the command.
`max_output_bytes` defaults to the configured buffer limit; values above it are rejected.
`timeout_seconds` bounds runtime; timeout retains collected output and returns exit code 124.
`structured_metrics` accepts JSON-compatible named metrics.

## Resumable phase execution

Durable phase execution and its process-group recovery currently require Linux
with file-locking support and `/proc` process identities. The package's
Windows installations remain usable for MCP stdio and text/file capture.
Bounded subprocess capture requires POSIX pipe and process-group support, and
`start_execution` additionally requires Linux leases, `/proc` process
identities, pidfd signaling, and selector support; startup rejects the request
with a clear platform error when those recovery backends are unavailable.

Use `start_execution` when a command may approach or exceed the MCP client's
tool-call window, or when a longer workflow benefits from meaningful phase
checkpoints. Expected runtime determines whether work should run in the
background; output size is a separate choice. A single command is one phase:

```text
start_execution(
  execution_id="long-tests",
  phases=[{"name": "tests", "command": "python -m unittest", "cwd": "/path/to/repository"}],
)
```

The call returns promptly with the execution ID. Follow its status with
`get_execution(execution_id="long-tests")`, then retrieve that phase's
persisted output with
`get_execution_output(execution_id="long-tests", phase_name="tests")`.
Use `cancel_execution` to request termination. Disconnecting the MCP caller
detaches it from the durable execution and does not cancel it.

For workflows with multiple checkpoints, pass multiple phases:

```text
start_execution(
  execution_id="release-checks",
  phases=[
    {"name": "tests", "command": "python -m unittest", "timeout_seconds": 900},
    {"name": "publish", "command": "./publish.sh", "side_effects": "unsafe"},
  ],
)
```

`start_execution` returns the durable execution ID and current status without
waiting for all phases to finish. Use `get_execution` and
`get_execution_output` to follow it. Disconnecting from the MCP request only
detaches the caller; call `cancel_execution(execution_id)` to request
termination. Cancellation records the current phase as `interrupted`, keeps
completed checkpoints, and fences the subprocess if cleanup cannot be
confirmed. `resume_execution` also returns promptly after scheduling work.
Durable execution requires Linux leases, `/proc` process identities, pidfd
signaling, and selector support; startup reports a platform error when these
recovery backends are unavailable. The manager accepts up to eight active or
queued background executions; `get_execution_capacity` reports the current
counts. The default state directory is process-local; configure
`EPHEMERAL_SESSION_ID`, `EPHEMERAL_SOCKET_PATH`, or
`EPHEMERAL_EXECUTION_STATE_DIR` when state must survive a server restart. If a
background task fails outside normal phase handling, `get_execution` reports a
`background_error` with the recorded execution state.

Each phase is persisted as `pending`, `started`, `completed`, `failed`,
`interrupted`, or `timed_out`. Output, exit status, duration, truncation, and
caller metrics are written after every finished phase. If the server restarts
while a phase is `started`, the next inspection records it as `interrupted`.
`resume_execution` skips every completed phase and continues at the first
incomplete phase. Failed and timed-out phases require `retry_failed=True`;
safe phases recovered as `interrupted` resume automatically. A timed-out phase
whose process-group cleanup is not confirmed remains fence-pending and blocks
retry. An interrupted
phase marked `side_effects: "unsafe"` additionally requires
`confirm_unsafe=True` unless the execution was created with the explicit
`resume_policy="allow-unsafe"`. The persisted `idempotency_key` is
an audit boundary for an external operation; it does not replace confirmation
or provide an external deduplication guarantee. The compatibility alias
`unsafe_side_effects: true` is normalized to `side_effects: "unsafe"`; if both
fields are supplied, they must agree.
The command is run beneath a Linux subreaper supervisor that adopts and
terminates descendants which escape the original process group. The supervisor
identity and a durable phase launch marker are checkpointed while a phase runs.
During restart recovery, the supervisor is signalled through a pidfd, then the
remaining processes carrying that phase marker are found and terminated before
the original process group is checked for absence. Resume stays blocked if
identity lookup, marker scanning, supervisor termination, descendant cleanup,
or group absence cannot be confirmed. Linux process start and boot identities
are revalidated while the pidfd is pinned, protecting recovery from signalling
a reused process ID. A phase marks its launch fence before spawning the
command, so a crash before process identity is persisted also fails closed.
An unreadable process environment makes the marker scan inconclusive and keeps
recovery blocked, including when the process belongs to another user.
Older in-progress records without these identity fields are recovered as
fence-pending and must be inspected or retired rather than being retried
automatically.

Execution responses include a human-readable `summary`, machine-readable
`execution_status` and `partial` fields, per-phase event history, and the
next resumable phase. If detailed metadata would exceed the 64 KiB tool
response budget, the server returns a compact response with the durable
`execution_id` and `response_truncated: true`; call `get_execution` or the
bounded output tool to retrieve details. Durable JSON state defaults to a
temporary local process-private directory created securely with owner-only
permissions. An explicit `EPHEMERAL_EXECUTION_STATE_DIR` takes precedence for
storage; otherwise an explicit `EPHEMERAL_SOCKET_PATH` identifies the derived
state directory, followed by `EPHEMERAL_SESSION_ID` when no socket path is
explicit. Identity-derived state is retained across normal server shutdowns;
the private process directory is removed on normal shutdown where secure
cleanup is available. State contains commands and bounded output, so protect
explicitly configured directories when those contents are sensitive.

When both an explicit socket and session ID are set, the socket path now
selects the derived state namespace. If an existing session-derived directory
is detected, startup reports its location. Set
`EPHEMERAL_EXECUTION_STATE_DIR` to that exact directory to continue using its
records. EB never moves state automatically.

Execution metadata is bounded to 64 MiB per record, 64 phases, 32 attempts per
phase, 16 KiB of structured metrics, and 1,000 records per state directory;
list results are paginated with a maximum page size of 100. The aggregate state
quota defaults to 4 GiB (`EPHEMERAL_EXECUTION_STATE_QUOTA_BYTES`) with a 128 MiB
checkpoint reserve (`EPHEMERAL_EXECUTION_CHECKPOINT_RESERVE_BYTES`). The reserve
must be at least the per-record limit and smaller than the aggregate quota;
servers sharing a state directory must use the same values.
Record and summary JSON bytes, atomic-write temporary files (including crash
leftovers), and active phase reservations use the quota;
the configured reserve is excluded from committed data and provides allowance
for atomic checkpoint writes. Before a phase starts, the server checks current
filesystem free space for the bounded checkpoint reservation plus that reserve.
Other processes can still consume filesystem space after this check. There is no
automatic expiry. Use `get_execution_capacity()` to inspect usage, then call
`retire_executions(execution_ids)` to preview which records can be retired and
how many bytes their record/summary pairs occupy. Actual retirement requires
`dry_run=False` and a new `archive_path` outside the state directory. Active,
started, or fence-pending records are refused. State records and execution
leases use the same namespace precedence documented above: explicit state
directory, explicit socket path, then session ID. Without an identity, each
server process receives a fresh private state directory that is removed during
normal shutdown on POSIX platforms. Windows may retain that temporary
directory because secure owner-identity cleanup is not available there.

## Repository-sensitive paths and file capture

Before any repository-sensitive command or file capture, verify the intended
working directory and target path. Prefer an explicit `cwd`, confirm the
repository identity, and resolve symlinks when path identity matters. Shell
expansion, inherited working directories, and symlinks can target a different
location than the spelling suggests. Capture limits protect context size; they
do not validate command intent, path identity, or filesystem safety.

For diff captures, `get_capture_summary` reports the detected file map,
addition/deletion statistics, line ranges, and merge-conflict signals. Use
`get_capture_slice` with those ranges to retrieve the complete file context.

## Consolidating multi-result workflows

When a workflow produces several captures—for example, one command per
repository—use `consolidate_captures` to give the agent one bounded overview:

```text
consolidate_captures(
  capture_ids=["cap_1", "cap_2", "cap_3"],
  label="organization activity",
  max_captures=25,
  max_bytes=20000
)
```

The resulting capture is JSON with source metadata and records containing the
original `capture_id` and `source_line`. It can be searched normally with
`search_capture`, and exact consolidated context can be retrieved with
`get_capture_slice`. The response reports omitted records, missing IDs, and
the original source IDs; use those original IDs to retrieve complete detail
when the consolidated byte budget is reached. Calling the tool without
`capture_ids` consolidates the currently active captures, up to
`max_captures`.

This workflow keeps the server responsible for bounded execution, storage, and
retrieval while leaving prioritization and interpretation to the coding agent.
