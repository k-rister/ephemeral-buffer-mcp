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
| `execute_and_capture(command, cwd=None, label='', content_type='auto', max_output_bytes=None, timeout_seconds=None, structured_metrics=None)` | Synchronously executes a shell command with bounded capture and returns a compact versioned JSON summary containing status, duration, sizes, approximate token counts, truncation, warnings/errors, and optional structured metrics. `content_type` accepts `auto`, `diff`, `log`, or `text`; other values are rejected before the command runs. Use when expected to finish within the client's tool-call window and noisy output benefits from later search; cancellation or caller disconnection requests subprocess cleanup. |
| `preflight_command(command, cwd=None)` | Performs content-free path, symlink, local Git-root, and executable-resolution diagnostics without executing the requested command. |
| `start_execution(phases, execution_id=None, label='', resume_policy='safe', cwd=None, timeout_seconds=None, max_output_bytes=None)` | Persists and starts a sequential, durably checkpointed execution, then returns its ID and current status promptly. Use when expected duration may approach or exceed the client's tool-call window, even for one command as one phase; follow with `get_execution` and `get_execution_output`. Requires Linux recovery support and has a combined capacity of eight active or queued executions. |
| `resume_execution(execution_id, retry_failed=False, confirm_unsafe=False)` | Resumes from the first incomplete phase and returns promptly; use `get_execution` to follow progress. Retries and unsafe side effects require explicit controls. |
| `cancel_execution(execution_id)` | Requests cancellation of active durable work by ID. Completed phases remain checkpointed; process cleanup uses the existing interruption and fencing path. |
| `get_execution(execution_id, include_output=False)` | Retrieves persisted phase metadata, event history, retry requirements, and human/machine-readable completion status. |
| `get_execution_output(execution_id, phase_name=None, offset=0, max_bytes=8192)` | Retrieves bounded persisted output for all phases or one phase, including after a server restart. The all-phase response shares its byte budget across phases; set `phase_name` to page through a phase with a UTF-8 byte `offset`. Returned `offset` and `next_offset` values are byte positions. An input offset inside a multibyte character advances to the next character boundary, and the returned `offset` reports that effective position. Offsets beyond the output return an empty, non-truncated page. |
| `list_executions(limit=20, offset=0)` | Lists a bounded page of durable executions and their partial/completed summaries; oversized pages return compact IDs with pagination metadata. |
| `get_execution_capacity()` | Reports durable record storage and quota diagnostics plus active and queued background execution counts; it does not attempt recovery. |
| `retire_executions(execution_ids, archive_path=None, dry_run=True)` | Previews eligibility and projected reclaimed space; accepts at most 20 execution IDs per request. Actual retirement writes a private tar archive outside the state directory before removing selected record pairs. |
| `capture_text(content, label='', content_type='auto', structured_metrics=None)` | Ingests text directly into the buffer and returns the same compact summary schema. `content_type` accepts `auto`, `diff`, `log`, or `text`; other values are rejected. |
| `capture_file(file_path, label='', content_type='auto', max_bytes=None, structured_metrics=None)` | Ingests a bounded regular file from disk and returns the same compact summary schema. `content_type` accepts `auto`, `diff`, `log`, or `text`; other values are rejected before the file is read. Symlinks are followed, but pipes and devices are rejected. |
| `consolidate_captures(capture_ids=None, label='consolidated captures', max_captures=25, max_bytes=None)` | Creates one bounded, searchable JSON capture from multiple captures while preserving source IDs and source line numbers. `max_captures` limits source captures selected for this operation; the global capture-count default is 256. |
| `search_capture(query, mode='hybrid', capture_id='latest', top_k=5, context_lines=3)` | Hybrid/BM25/Semantic search over the selected capture. `mode` is `hybrid`, `bm25`, or `semantic`; `top_k` must be 1–20 and `context_lines` 0–100. BM25 splits underscores and punctuation—including regex-like characters—into alphanumeric terms, then combines those terms with OR. For example, `database_connection` searches for `database` or `connection`, not one underscore-containing term. SQLite FTS5 operational errors are logged and fall back to complete-token lexical matching. Hybrid ranking gives lexical matches priority over semantic-only matches. Returns bounded match snippets, exact numeric context boundaries, bounded raw-context previews, line numbers, and whether the match came from the lexical or semantic chunk grid. The complete structured MCP result is capped at 64 KiB; use `get_capture_slice` for omitted content. Hybrid search waits at most `EPHEMERAL_SEMANTIC_WAIT_SECONDS` for a large capture's semantic index and otherwise returns lexical results marked `semantic pending`; repeat the search for hybrid ranking. Captures beyond the semantic-input budget return BM25 results with semantic coverage `unavailable`. The lexical-only distribution supports `mode='bm25'`; `hybrid` and `semantic` return `semantic_backend_unavailable` with an install instruction. |
| `get_capture_slice(start_line, end_line, capture_id='latest', max_bytes=65536, cursor=None)` | Retrieves one byte-bounded page from a 1-indexed line range; both line numbers must be at least 1. A long line can continue across pages; repeat the original range and pass back `next_cursor` until it is null. `max_bytes` bounds the serialized MCP result and must be between 4 KiB and 64 KiB. The cursor is opaque; each segment reports a zero-based UTF-8 byte offset within its line, and pages never split a Unicode character. Joining `structuredContent.data.content` from successive pages reconstructs the retained newline-joined text exactly. |
| `get_capture_summary(capture_id='latest', include_previews=False)` | Returns the compact JSON summary; opt into bounded head/tail previews only when needed. |
| `get_buffer_stats()` | Reports aggregate capture count, content bytes, lines, chunks, semantic-backend availability, embedding model readiness, embedding bytes, semantic memory limits, accounted bytes, and process RSS. After a model-load failure it shows the failure class and remaining retry delay. Model loading is retried after a 60-second per-engine backoff, while hybrid search continues with lexical results. It always includes fixed `capture_removals_by_reason` counts for `count`, `bytes`, `chunks`, `clear`, and `budget`; overlapping limit reasons increment each matching count. It also reports fixed `capture_rejections_by_reason` counts for `bytes` and `chunks`, incremented when those capacity limits reject an attempted capture. When local metrics are enabled, it also includes the content-free metrics snapshot for the active MCP session (or the aggregate process scope for direct calls). |
| `get_runtime_diagnostics()` | Opt-in, content-free report of runtime version, platform, uptime, socket and durable-state identity, durability, startup setting values and origins, active log file and level, buffer limits, semantic-backend availability, embedding readiness, and process memory. |
| `get_usage_metrics(since=None)` | Returns a versioned, content-free JSON snapshot of local usage metrics, including interface coverage, per-tool counters, workflow events, byte counters, fixed `capture_removals_by_reason` and `capture_rejections_by_reason` counts, and scope- or task-window measurement timestamps. The rejection map always has `bytes` and `chunks` keys, including in task-window deltas. Pass a prior `snapshot_token` as `since` for a task-window delta. |
| `set_semantic_index_budget(max_indexed_chunks)` | Adjusts the session's semantic-index chunk budget when `EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1`; decreases evict least-recently-used captures as needed. The result includes the total count, up to 25 ID/reason entries in `evicted_capture_details`, and an `evicted_capture_details_omitted` count. |
| `list_captures()` | Lists active captures in the ring buffer. |
| `clear_captures(capture_id='all')` | Clears buffer. |

## Private CLI socket payload

The `ephbuf` CLI and MCP server exchange length-prefixed JSON frames over a
private Unix socket. Frame version 1 also identifies the request and response
schemas; incompatible changes to either schema must increment the frame
version. The server accepts these capture request fields; omitted fields use the defaults shown:

| Field | Type | Required | Default |
| :--- | :--- | :--- | :--- |
| `text` | string | yes for current clients; earlier v1 peers may omit it | empty string |
| `label` | string | no | `CLI pipe` |
| `content_type` | `auto`, `diff`, `log`, or `text` | no | `auto` |
| `truncated` | boolean | no | `false` |
| `original_byte_size` | non-negative integer or null | no | `null` |
| `command_exit_code` | integer or null | no | `null` |
| `timed_out` | boolean | no | `false` |
| `duration_ms` | finite number or null | no | `null` |
| `structured_metrics` | JSON object or null | no | `null` |

The server ignores unknown fields, rejects invalid values for known fields,
and applies the defaults above. It accepts a missing `text` from earlier v1
peers as an empty string for compatibility; current CLI requests include the
field. The CLI does not send `structured_metrics`, but the server accepts it
for callers that need to attach named metrics. Frames whose payload is not
valid UTF-8 JSON are kept as plain-text captures using the `CLI pipe` label;
valid JSON that is not an object returns an error.

Successful responses contain `status: "ok"`, `capture_id`, `label`,
`line_count`, and `byte_size`; `summary` is included when available. New error
responses contain `status: "error"`, a stable `code`, and `message`. When the
CLI receives an error from an earlier v1 server that omits `code`, it assigns
`socket_request_failed` while preserving the server's message.
Known socket error codes include `server_starting`, `server_shutting_down`,
`server_busy`, `socket_admission_failed`, `socket_read_timeout`,
`invalid_socket_payload`, and the fallback `socket_request_failed`. Unknown
response fields are ignored, while required fields and their types are
validated.

Capture tools return a compact JSON summary so an agent can decide whether it
needs the full output before spending context on retrieval. The summary uses
`schema_version: 1` and includes `status` (`captured`, `success`, `failed`, or
`timed_out`), `duration_ms`, retained and original byte sizes, approximate
token counts, truncation and partial-execution flags, typed warning/error
signals, bounded caller-provided `structured_metrics`, and `evicted_captures`.
That array contains up to 25 IDs and all applicable cap reasons for captures
removed to admit the new capture. `evicted_captures_omitted` reports any
additional removals; the array is empty and the count is zero when none occur.
The token values are deterministic planning estimates based on four UTF-8 bytes
per token; they are not provider billing counts. Use `get_capture_summary` for the
compact form, set `include_previews=True` only when a head/tail sample is
useful, and use `get_capture_slice` or `search_capture` for complete or
targeted content. Diff file maps are also bounded and report omitted entries;
use `get_capture_slice` for the complete diff. Raw retained captures are
unchanged by summary generation.

Capture, search, retrieval, and clear tools return a versioned MCP result
envelope with `schema_version`, `status`, `data`, optional `error` (`code` and
`message`, and optional `details`), and truncation or continuation metadata.
Returned MCP tool errors include `error.details.call_id`, including admission
rejections such as `server_busy` and `server_shutting_down`, so callers can
correlate each response with its lifecycle or rejection log. Unexpected MCP
tool exceptions return the generic `operation_failed` error and the same call
ID; exception text is not returned. Direct Python exceptions and MCP SDK
argument-validation responses keep their existing behavior. MCP clients
continue to receive readable text in `content[0].text`; the same result data is
available in `structuredContent`. The envelope's `status` is `ok` or `error`.
Stable capture error codes include `capture_failed` and
`consolidate_captures_failed`. Stable retrieval and search error codes include
`capture_not_found`, `retrieval_failed`, `invalid_range`, `invalid_cursor`,
`invalid_byte_budget`, `response_budget_too_small`, `invalid_query`,
`query_too_large`, and `unsupported_search_mode`. Search argument errors also
include `invalid_top_k`, `top_k_limit_exceeded`, `invalid_context_lines`,
`context_limit_exceeded`, and `invalid_response_budget`. Clear failures use
`capture_not_found`. Legacy text handlers without a more specific registration
code use `operation_failed`; a structured error's explicit code takes
precedence. Messages are for display and may change, so use `error.code` for
handling.

`set_semantic_index_budget` and `preflight_command` keep their successful JSON
responses. Their MCP failures use JSON text shaped as
`{"status":"error","error":{"code":"...","message":"..."}}`.
Configuration failures use `index_budget_disabled` when the setting is off and
`invalid_index_budget` for rejected values. Unexpected preflight failures use
`preflight_failed`. Direct Python calls keep their existing error strings or
preflight error JSON, and MCP argument validation remains the SDK's response.

Execution tools preserve their successful JSON text responses and Python
`str` return values. A failure returns JSON text with `status: "error"` and
`error.code` / `error.message`; read-only record diagnostics retain their
existing status and fields and add the same `error` object. When available,
`error.details` preserves a record reason code and schema version. Stable
execution error codes include `execution_busy`, `execution_not_found`,
`phase_not_found`, `malformed_record`,
`unsupported_schema_version`, `quota_exceeded`, `disk_full`,
`execution_storage_error`, `response_too_large`, and `execution_failed`.
`execution_not_found` covers IDs that never existed and IDs that have been
retired. `phase_not_found` means the execution exists but has no phase with the
requested name. Use `error.code` for handling; the message is for display and
may change.

MCP argument validation occurs in the SDK before an execution tool runs.
Those validation failures follow the SDK's validation response and do not carry
an execution `error.code`.

When `capture_not_found` is returned, the message says the buffer is empty
only if no captures remain; otherwise it says that other captures remain.

### Response schema versioning

Capture summaries, consolidated-capture JSON, and structured MCP tool result
envelopes have independently versioned schemas. Adding fields while preserving
existing field types and meanings does not change a schema version. Removing
or renaming fields, changing their types or meanings, or otherwise changing
how existing data must be interpreted increments that surface's integer
version by one. The private Unix-socket frame has its own version rule,
described above.

Clients should interpret structured data only when they recognize its
`schema_version`. For an unknown version, treat the structured data as
unsupported, preserve any readable text in `content[0].text`, and report the
unsupported version instead of guessing at compatibility.

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

For all capture tools, `content_type` accepts `auto` (default), `diff`, `log`, or `text`; other values are rejected.
An omitted `label` is derived from the command.
`max_output_bytes` defaults to the configured buffer limit; values above it are rejected.
`timeout_seconds` bounds runtime; timeout retains output collected through the
deadline and returns exit code 124. Process termination and cleanup happen after
that cutoff, so output produced during cleanup is not captured.
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
recovery backends are unavailable. Each server process's manager accepts up to
eight active or queued background executions; this limit is not shared across
processes that use the same state directory. `get_execution_capacity` reports
the current process's counts. The default state directory is process-local;
configure `EPHEMERAL_SESSION_ID`, `EPHEMERAL_SOCKET_PATH`, or
`EPHEMERAL_EXECUTION_STATE_DIR` when state must survive a server restart. If a
background task fails outside normal phase handling, `get_execution` reports a
`background_error` with the recorded execution state.

Execution IDs are limited to 256 UTF-8 bytes, labels to 1 KiB, working
directories to 4 KiB, and each phase command to 16 KiB. Phase names are limited
to 256 UTF-8 bytes and must be unique within an execution.

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
fence-pending. Resume and `retire_executions` refuse them. If process absence
can be confirmed, follow the [manual recovery procedure in OPERATIONS.md](../OPERATIONS.md#manual-recovery-for-fence-pending-records-and-unreadable-reservations);
otherwise leave the record fenced.

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
An explicit state directory must resolve to an absolute filesystem path;
tilde-prefixed paths are expanded, while relative paths and `:memory:` are
rejected during startup.

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
the server refuses to start when this relationship is invalid, and the error
identifies both configured values.
Servers sharing a state directory must use the same values.
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
started, or fence-pending records are refused. For a fence-pending record or
unreadable checkpoint reservation, follow the [manual recovery procedure in
OPERATIONS.md](../OPERATIONS.md#manual-recovery-for-fence-pending-records-and-unreadable-reservations).
State records and execution
leases use the same namespace precedence documented above: explicit state
directory, explicit socket path, then session ID. Without an identity, each
server process receives a fresh private state directory that is removed during
normal shutdown on POSIX platforms. Windows may retain that temporary
directory because secure owner-identity cleanup is not available there.

### Durable output and recovery limits

Capture content is held in a bounded in-memory buffer. When a capture is
evicted by an LRU limit or explicitly cleared, EB tools cannot retrieve it
again. Output omitted by a `max_output_bytes` limit is also not retained; the
response's truncation flag and byte-size metadata report when that happened.

For durable executions, retrying a failed or timed-out phase starts a new
attempt and clears that phase's previous output. Save any partial output needed
for diagnosis before retrying. Retiring an execution removes its managed record
and summary only after writing the requested archive; the archive is then the
remaining copy, and EB does not restore retired records automatically. Keep it
if those records may be needed later.

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

Consolidated captures count against the global capture-count, retained-byte,
and indexed-chunk limits. Admission may evict unrelated least-recently-used
captures to make room; the returned `evicted_captures` array identifies them
and lists every applicable reason. `get_buffer_stats` also reports aggregate
removal counts by reason.

This workflow keeps the server responsible for bounded execution, storage, and
retrieval while leaving prioritization and interpretation to the coding agent.
