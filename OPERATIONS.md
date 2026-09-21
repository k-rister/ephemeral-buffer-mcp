# Operations Guide

This guide covers the settings and maintenance procedures that matter when
running `ephemeral-buffer` outside of a local development session.

## Socket configuration

The CLI communicates with the server over a Unix domain socket. If the session
launcher sets `EPHEMERAL_SESSION_ID`, the server and CLI deterministically use
a separate socket for that session:

```bash
export EPHEMERAL_SESSION_ID="agent-session-1"
```

The session ID is hashed before it is included in the socket filename. This
keeps concurrent sessions isolated while allowing both processes to derive the
same path. An explicit path takes precedence when needed:

```bash
export EPHEMERAL_SOCKET_PATH=/run/user/1000/ephemeral-buffer.sock
```

The server probes an existing socket before startup: a live socket is preserved
and the new instance exits with an error, while a verified stale socket is
removed. Regular files, directories, and symlinks at the configured path are
never removed; the server exits with an error instead. The new socket is
created with owner-only permissions (`0600`). The parent directory must
already exist and be writable by the account running the server.

If neither variable is set, the legacy shared default is used. When running
multiple sessions without a session-aware launcher, give each server/CLI pair
its own explicit path rather than relying on that default:

```bash
export EPHEMERAL_SOCKET_PATH="${XDG_RUNTIME_DIR}/ephbuf-${SESSION_ID}.sock"
```

Keep the socket in a private directory when multiple users share a host.

For coding-agent launchers, require isolation so an incomplete environment
cannot accidentally attach to another session's shared socket:

```bash
export EPHEMERAL_REQUIRE_ISOLATION=1
export EPHEMERAL_SESSION_ID="agent-session-1"
```

The same variables must be inherited by the MCP server and the `ephbuf` CLI.
The server advertises the policy in its MCP initialization instructions and
also enforces it at startup; the CLI refuses to send when strict mode is
enabled without an explicit session ID or socket path.

This policy is client-neutral. For stdio MCP clients, configure the server's
`env` map with `EPHEMERAL_REQUIRE_ISOLATION=1` and a unique
`EPHEMERAL_SESSION_ID`, and arrange for shell commands in that agent session
to inherit the same values. If the client supports per-session environment
interpolation, use that facility; otherwise generate the ID in the launcher
that starts both the agent and its shell environment.

The package installs `ephemeral-agent`, `codex-ephemeral`, and
`ephemeral-session-env` alongside `ephbuf` in the virtual environment's
`bin` directory. Use the first launcher for generic CLI agents, the Codex
launcher for Codex's explicit MCP configuration override, and source the
environment helper when the agent is started separately:

```bash
ephemeral-agent claude
ephemeral-agent gemini
source ephemeral-session-env
```

These private, session-aware launchers default `EPHEMERAL_METRICS=1` so the
session's content-free usage data is available for diagnostics. Preserve the
normal opt-in behavior for direct or shared server launches, or disable metrics
for a launcher-created session with `EPHEMERAL_METRICS=0` before starting it.

CLI socket operations use a 10-second timeout by default. Override it with a
positive value when needed:

```bash
export EPHEMERAL_SOCKET_TIMEOUT_SECONDS=30
```

The timeout applies to connecting, sending the capture, and receiving the
server response. A timeout produces a nonzero CLI result rather than leaving
a shell pipeline blocked indefinitely.

## Capture limits and eviction

The buffer is intentionally transient. The defaults are:

- 25 captures (`EPHEMERAL_MAX_CAPTURES`)
- 50 MiB of captured UTF-8 content (`EPHEMERAL_MAX_BUFFER_BYTES`)
- least-recently-used (LRU) eviction when either limit is reached

Semantic and BM25 indexes have a separate total limit of 32,768 indexed
chunks, configurable with `EPHEMERAL_MAX_INDEXED_CHUNKS`. LRU eviction also
makes room under this limit. If one capture requires more indexed chunks than
the entire configured index budget, ingestion is rejected instead of creating
a partial index; this preserves complete search coverage for retained
captures.
For an explicitly authorized session, set `EPHEMERAL_ALLOW_RUNTIME_INDEX_BUDGET=1`
to enable `set_semantic_index_budget(max_indexed_chunks)`. The adjustment is
session-scoped and not persisted; decreases evict the least-recently-used
captures until the effective budget is satisfied. The result and latest
adjustment are included in `get_buffer_stats` and runtime diagnostics.

The byte budget covers retained capture content. Embedding storage, the search
index, Python objects, and process RSS are reported separately by
`get_buffer_stats`; process RSS is an approximate operational metric rather
than an allocation limit.

FastEmbed is warmed in a background thread after socket startup succeeds, so
the MCP handshake and BM25 search remain available while model loading and one
small deterministic embedding complete. Set `EPHEMERAL_EMBEDDING_WARMUP=0` for
lexical-only or memory-constrained deployments. Set `EPHEMERAL_EMBEDDING_MODEL`
to select a compatible model and `EPHEMERAL_FASTEMBED_CACHE_DIR` to place its
downloaded model files in a controlled cache directory. Set
`EPHEMERAL_EMBEDDING_THREADS` to a small positive integer to bound the ONNX
Runtime threads used for embedding inference; leave it unset for the runtime
default. The default model, `BAAI/bge-small-en-v1.5-fp32`, is the upstream fp32
ONNX export of bge-small-en-v1.5 registered by the engine; it downloads about
130 MB on first use. FastEmbed's catalogue file for the same model, selectable
as `EPHEMERAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, is 67 MB and produces
identical vectors, but its reduced-precision matrix kernels do not parallelize
on common CPU hosts. A release-preparation comparison on a 16-vCPU Linux
x86_64 host measured 3.1 chunks per second and 41.15 seconds median indexing
for the catalogue file versus 69.1 chunks per second and 1.85 seconds for the
fp32 export at 1,024 lines—about a 22x difference. Prefer bounding threads
over switching files when host impact is the concern. These are host-specific
diagnostic measurements; rerun `benchmark_semantic_index.py` on the deployment
host before generalizing them. Warm-up failure does
not block startup: BM25 remains available and hybrid search degrades to lexical
results. `get_buffer_stats` and `get_runtime_diagnostics` report warm-up state
and a content-free exception class on failure.

Lexical search uses SQLite FTS5 when the host SQLite library provides it. If
FTS5 is unavailable, captures remain searchable through a complete token-based
Python fallback with case- and diacritic-insensitive terms; this preserves
matching coverage but may be slower and does not provide FTS5 BM25 ranking.
`get_buffer_stats` reports the active lexical backend. FTS5 is therefore an
optional capability rather than a package-level platform prerequisite.

Semantic embeddings are computed over windows packed separately from the
BM25 grid: up to `EPHEMERAL_SEMANTIC_CHUNK_LINES` lines (default `8`) or
`EPHEMERAL_SEMANTIC_CHUNK_BYTES` UTF-8 bytes (default `1024`) per window, with
`EPHEMERAL_SEMANTIC_CHUNK_OVERLAP` shared lines (default `0`). The byte cap
keeps windows under the model's 512-token limit for ordinary text, and a single
oversized line becomes its own window. Embedding cost scales with total tokens,
so overlap and window count are the main levers; windows beyond about eight
short lines cost more per token without reducing total work. The index budget
counts lexical chunks; semantic windows are never more numerous than lexical
ones unless overlap is raised above the lexical grid's. `get_buffer_stats`
reports the active window settings and the semantic window count.

Semantic indexing prefetch is enabled by default. Set
`EPHEMERAL_SEMANTIC_PREFETCH=0` for lexical-only or CPU-constrained hosts, and
optionally set `EPHEMERAL_SEMANTIC_PREFETCH_WORKERS` (default `1`) to a small
positive value. Each capture then costs background embedding work right after
ingestion, on the same host as the coding agent; bound it with
`EPHEMERAL_EMBEDDING_THREADS` rather than disabling prefetch when only host
impact is the concern. Ingestion queues every eligible capture and the bounded worker pool drains the
queue newest-first, so bursts are never silently dropped; the queue is bounded
by the capture limit because eviction removes queued work. Semantic and hybrid
search wait for an active job, index a still-queued capture on a dedicated
thread instead of waiting behind older work, and retry failed jobs the same
way, preserving the lazy path as the correctness fallback. Explicit cleanup and
process shutdown drop queued work and let running jobs finish. Embedding
inference is serialized by one model lock, so raise
`EPHEMERAL_EMBEDDING_THREADS` rather than the worker count for throughput. Use
the content-free pending, queued, running, and failed counts in runtime
diagnostics when checking host impact.

Hybrid search blocks on a capture's semantic index for at most
`EPHEMERAL_SEMANTIC_WAIT_SECONDS` (default `10`). Past the budget it returns
BM25 results with `semantic_coverage` set to `pending` and a message telling
the caller to repeat the search; indexing continues in the background, so the
repeated search is fully hybrid. `complete` and `unavailable` (semantic backend
failure, with `semantic_fallback`) are the other values. The tradeoff is that
identical searches issued before and after indexing finishes can rank
differently on a very large capture; the marker is the contract for that.
Lower the budget on interactive hosts where a fast lexical answer beats a
delayed hybrid one, set it to `0` to never wait, or `inf` to restore the
previous wait-for-index behavior. Semantic mode always waits because an empty
result would only cost the caller a retry. `get_buffer_stats` reports the
budget and the number of on-demand index jobs; a persistently nonzero count
means searches keep arriving before prefetch finishes, so consider
`EPHEMERAL_EMBEDDING_THREADS` or a smaller capture size. On-demand jobs run on
a pool bounded by `EPHEMERAL_SEMANTIC_PREFETCH_WORKERS` (default `1`), and a
job whose capture is evicted or cleared is cancelled while queued and skipped
once it runs, so timed-out searches over churning captures never accumulate
threads beyond that bound.

Override the limits before starting the server:

```bash
export EPHEMERAL_MAX_CAPTURES=50
export EPHEMERAL_MAX_BUFFER_BYTES=$((100 * 1024 * 1024))
```

`execute_and_capture` and `capture_file` reject per-request limits above the
configured byte budget. Oversized command output is retained as a bounded
head/tail sample and marked as truncated. Capture tools return a compact
versioned JSON summary with status, duration, retained and original sizes,
deterministic approximate token counts, truncation and partial-execution
flags, typed warning/error signals, and optional structured metrics. The
default summary omits previews; set `include_previews=True` on
`get_capture_summary` when a bounded sample is needed. Each head or tail
preview is capped at 4 KiB of UTF-8 data. Use `search_capture` or
`get_capture_slice` for complete content. Diff file maps are bounded in the
summary and report omitted entries; retrieve the underlying diff slices for a
complete file map.

Use `timeout_seconds` with `execute_and_capture` or `--timeout-seconds` with
`ephbuf` when a command might block or run indefinitely. A timed-out command is
terminated as a process group, including descendants that outlive the shell
leader; pipe draining and final reaping remain bounded. Its output collected so
far is retained, and it returns exit status 124. Closing stdout does not end
the command deadline: the process is still awaited until it exits or the
requested deadline expires. Without a timeout, completion is awaited normally.
If timeout cleanup cannot confirm that the process group is gone, the durable
execution keeps its fence pending and blocks a retry until a later recovery
proves cleanup.

### Durable resumable executions

Durable phase execution is currently Linux-only: it relies on Linux file
leases, `/proc` process identities, pidfds, a subreaper supervisor for bounded
subprocess containment, and bounded pipe handling for restart fencing. Windows
remains supported for MCP stdio and text/file capture; bounded subprocess
capture requires POSIX pipe and process-group support. Startup validates the
pidfd, `/proc`, and selector recovery backends before launching a built-in
durable phase, and rejects unsupported hosts rather than leaving a phase
fence-pending. The execution tools return a platform error where the required
restart-safe backends are unavailable.

Long workflows can be represented as sequential phases with
`start_execution`. The server persists each phase transition and its bounded
output in `EPHEMERAL_EXECUTION_STATE_DIR` (by default, a local temporary
process-local directory created securely with owner-only permissions; use
`EPHEMERAL_SESSION_ID`, `EPHEMERAL_SOCKET_PATH`, or
`EPHEMERAL_EXECUTION_STATE_DIR` when state must survive a server restart; state files are also
owner-readable. `get_execution` exposes a human-readable summary plus structured
status, event history, metrics, and the first incomplete phase;
`get_execution_output` retrieves bounded chunks of persisted phase output after
a restart; pass `phase_name`, `offset`, and `max_bytes` to page through a large
phase without creating an oversized MCP response.
If detailed execution metadata would exceed the 64 KiB tool-response budget,
the server returns a compact response that preserves the durable execution ID
and sets `response_truncated: true`.
The state directory is checked for current-user ownership before use, and
directory metadata is synchronized after atomic record replacement so a
completed phase checkpoint survives normal host-crash recovery.

On restart, a phase left in `started` is recovered as `interrupted`. Resume
signals the persisted supervisor through a pidfd; that supervisor adopts and
terminates descendants that escaped the original process group before recovery
is permitted, then skips `completed` phases. Linux process start and boot
identities are revalidated while the pidfd is pinned so a reused process ID is
not signalled; if identity lookup, supervisor termination, or group absence
cannot be confirmed, resume remains blocked. The launch fence is written before
spawning, and older in-progress records without process identities are held
fence-pending until they are inspected or retired. A `failed` or `timed_out` phase is only retried with
`retry_failed=True`; a timed-out phase whose process-group cleanup is not
confirmed remains fence-pending and blocks retry. A safe phase recovered as `interrupted` resumes on the
normal resume call. Mark operations that can write, deploy, publish, or make
external requests with `side_effects: "unsafe"`; retrying an interrupted
unsafe phase also requires `confirm_unsafe=True`, or the explicit
`resume_policy: "allow-unsafe"` chosen when the execution was created. The
boolean compatibility alias `unsafe_side_effects: true` is normalized to
`side_effects: "unsafe"`; if both fields are supplied, they must agree. The
optional `idempotency_key` is persisted as an audit identifier, not as a
claim that the external system deduplicates requests. Keep the state directory
access-controlled because it contains the stored commands and bounded output.
Execution records are capped at 64 MiB, 64 phases, 32 attempts per phase, and
1,000 records per state directory; `list_executions` is paginated. There is no
automatic expiry. If the record cap is reached, stop the server and archive or
rotate the state directory, or remove completed records and their matching
summary files before restarting. When no explicit state directory is set,
state is isolated by `EPHEMERAL_SESSION_ID` or by the explicit socket path;
otherwise each server process receives a fresh private directory that is
removed during normal shutdown on POSIX platforms. Windows may retain that
temporary directory because secure owner-identity cleanup is not available
there.

Signal summaries recognize successful test-run markers and avoid treating
example error text inside a passing test run as an active failure while still
reporting explicit warnings. The complete captured output remains available
through search and slices. Approximate token counts are planning metrics based
on four UTF-8 bytes per token, not provider-reported usage.

Use `clear_captures("all")` between unrelated investigations when the active
buffer should be released immediately instead of waiting for LRU eviction.

## Agent-visible command and path guidance

MCP clients receive tool names, argument schemas, and tool descriptions; they
do not automatically read this document. The corresponding routing and path
validation reminders are therefore also kept in the `server.py` tool
docstrings.

Use direct command execution for small, bounded inspections. Use
`execute_and_capture` when output may be noisy, large, or uncertain—especially
tests, builds, and logs—where bounded output, search, and follow-up retrieval
are useful. Use `capture_text` for output
already held by the caller and `capture_file` only after intentionally selecting
and checking the file.

This routing rule is advisory. The synthetic `benchmark_routing.py` harness
compares direct and captured execution for 16-line targeted output, 256-line
test-like output, and 2048-line build/log-like output. Use its median and p95
measurements to calibrate expectations for a deployment; do not treat them as
portable hard thresholds. Direct execution usually minimizes latency for small
bounded output, while capture trades some overhead for bounded context,
searchability, and exact follow-up retrieval.

Before repository-sensitive work, confirm the intended repository and working
directory, pass an explicit `cwd`, and resolve symlinks when path identity
matters. Remember that an omitted `cwd` inherits the server process directory.
Shell expansion and symlinks can target a different location than expected.
The MCP server bounds captured output but does not validate command intent,
filesystem safety, or path identity; the calling agent remains responsible for
those checks.

When those facts need a content-free diagnostic before execution, call
`preflight_command(command, cwd)`. It reports the resolved working directory,
symlink status and target, detectable local Git root, and first executable-token
resolution. It never runs the requested command and does not expose captured
output or environment data. A result marked unavailable means the check could
not be established; it is not a safety approval. Shell expansion, aliases,
pipelines, redirections, environment changes, and arbitrary shell logic remain
outside preflight’s scope.

### Socket startup lifecycle

The server treats the Unix socket as part of the normal server lifecycle. On
startup, the socket moves from `starting` to `ready` only after the listener
has bound successfully. If probing, cleanup, binding, or permission checks
fail, the socket enters `failed` and the module entrypoint exits instead of
starting an apparently healthy stdio MCP session. The failure includes only an
error class and message; captured content and command arguments are never
included.

Importing the `server` module does not start or probe the socket. The normal
module entrypoint calls `start_socket_server()` explicitly before waiting for
readiness. Embedders that need the CLI socket can call that function during
their own initialization; imports and package smoke tests can safely use the
MCP tools without creating a Unix-socket side effect.
CLI socket requests and responses use a versioned fixed-width length prefix;
the server rejects unsupported versions, truncated frames, and payloads above
the configured limit before decoding or ingesting them.

Set `EPHEMERAL_SOCKET_STARTUP_TIMEOUT_SECONDS` to control how long the
entrypoint waits for the background listener to report readiness. The default
is five seconds. Tests and embedders that intentionally disable the listener
with `EPHEMERAL_DISABLE_SOCKET_SERVER=1` report `disabled` and do not apply
the startup gate. `get_runtime_diagnostics()` reports the current socket
lifecycle and any startup failure.

## Field-observation checklist

When investigating behavior from a real MCP session, collect operational
metadata rather than captured command content. This keeps reports useful while
avoiding accidental disclosure of source code, logs, credentials, or other
sensitive data.

The privacy model is local and opt-in: `EPHEMERAL_METRICS=1` enables aggregate
measurements in the current process, but the server does not transmit them.
Metrics and runtime logs are metadata-only by default. They may include counts,
durations, sizes, IDs, limits, and error classes, but must not include captured
content, command arguments, labels, query text, credentials, or session ID
values. Keep this boundary when adding integrations or preparing a report.

When enabled, the aggregate snapshot also includes `interface_coverage`: the
number and percentage of the 19 exposed MCP tools called during the process
lifetime and the complete list of tools not called. This is process-level
coverage; it does not identify which client made a call when multiple clients
share one server. The available-tool inventory comes from the same registration
path used to expose the MCP tools. The snapshot also includes `by_category`,
which groups the same `used`, `available`, `percentage`, and `unused_tools`
fields by each tool's primary capability category. Category coverage is
descriptive and is not a requirement that clients use every category.

Use `get_usage_metrics()` when a client needs the same content-free metrics as
versioned JSON rather than embedded JSON inside diagnostic text. The endpoint
returns schema version 2 and a `snapshot_token`. Pass that token as the
optional `since` argument to request a non-resetting task-window delta. The
response's `window.status` is `ok` for valid data, including a valid window
with zero activity, and `unavailable` for an invalid, expired, or pre-restart
token. The token history is bounded in memory, so callers should retain the
most recent token they intend to use. A new server process or isolated coding-
agent session starts a new measurement scope and cannot resolve tokens from a
previous process. Non-additive `max_duration_ms` remains the process-lifetime
maximum in delta tool records; additive counters, durations, events, bytes,
and interface coverage are window-scoped.
Calls that are still in flight at a task-window boundary are attributed
wholly to the following window. Therefore the `get_usage_metrics()` request
that produces a delta is excluded from that returned delta and appears in the
next one. This keeps tool counters, response bytes, and coverage coherent
without changing cumulative diagnostic snapshots.

Record the following before changing configuration:

- ephbuf version or commit, Python version, operating system, and installation
  method
- whether the server is running in a single session or alongside other
  sessions
- effective socket configuration, including whether
  `EPHEMERAL_SESSION_ID` or `EPHEMERAL_SOCKET_PATH` is set
- embedding model and cache configuration, without including cache contents
- buffer configuration: maximum captures, byte limit, and observed eviction
  behavior

For the behavior being investigated, record:

- the operation involved (`execute_and_capture`, `capture_text`, search, or
  another tool)
- approximate startup, first-capture, and subsequent-operation latency when
  relevant
- whether the issue is reproducible, and the smallest safe reproduction
- expected behavior versus observed behavior
- timeout, readiness, socket, embedding, cleanup, or eviction symptoms

Use `get_runtime_diagnostics()` for a content-free report of the running
version, Python/platform details, uptime, socket mode, effective socket path,
buffer limits, embedding readiness, and process memory. Use
`get_buffer_stats` and `get_capture_summary` for more focused aggregate
diagnostics. The runtime report is opt-in and does not include captured text,
labels, command arguments, or the session ID value.
For local workflow measurement, set `EPHEMERAL_METRICS=1` before starting the
server. This keeps aggregate per-tool counters and timers in process memory and
adds capture-to-search, search-to-retrieval, empty-search, eviction, and
cleanup counts to `get_runtime_diagnostics()` and `get_buffer_stats()`. Metrics
are disabled by default, never leave the process, and contain no captured
content, labels, commands, or queries. Event keys are stable and zero-filled;
capture correlation state is bounded by the active capture lifecycle and is
released on eviction or cleanup. Treat the counters as operational signals
rather than measures of task success; task completion and agent usefulness
require an external, privacy-reviewed evaluation.
When the server is launched directly from a checkout, its reported version is
read from that checkout's `pyproject.toml`; installed distributions use their
package metadata. Restarting a checkout-launched server therefore picks up a
version change without reinstalling the package.
When command output is needed, provide only a sanitized excerpt or line range;
do not attach an entire capture by default. Remove credentials, tokens,
private paths, source code, and user data before sharing diagnostics. A useful
report should be actionable without requiring access to the original capture.

## Interpreting effectiveness

Operational metrics describe service behavior: whether calls succeeded, how
many captures/searches/retrievals occurred, whether searches were empty, and
how much local time or memory the server used. They do not prove that a search
was relevant or that an agent completed its task.

Task-level effectiveness requires an evaluation that defines the task's
expected signal and completion criteria. The repository's effectiveness
benchmarks are server-side evaluations over deterministic synthetic fixtures;
they do not invoke a model, measure answer quality, or establish real-agent
performance. In particular:

- `success` means the expected fixture marker was found in the final retrieved
  result.
- `search_useful` means a search result contained that marker.
- byte reductions describe the MCP data path, not tokens saved or end-to-end
  latency.
- local timing covers capture, indexing, search, and retrieval only.

The separate `benchmark_relevance.py` harness evaluates retrieval quality for
BM25, semantic, and hybrid search against explicit synthetic markers. It
reports hit@1, hit@k, and mean reciprocal rank (MRR). These are retrieval
metrics only: they do not measure agent answer quality, token usage, or
performance on arbitrary repositories. Run it with `EPHEMERAL_TEST_EMBEDDINGS=1`
for deterministic results and compare it with the checked-in baseline:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_relevance.py \
  --baseline benchmark_relevance_baseline.json \
  --fail-on-regression --output search-relevance.json
```

The baseline is compact and versioned: it stores aggregate scores, query
counts, fixture/schema versions, embedding mode, and per-metric tolerances. A
baseline update is a deliberate repository change and should explain the
fixture or intended-behavior change in review. Never add captures, raw command
output, or user queries to the baseline. CI retains relevance, latency, and
effectiveness JSON outputs as machine-readable artifacts; timing artifacts are
diagnostic and are not exact cross-run gates.

The separate `benchmark_agent_ab.py` harness defines the protocol for a real
agent-level A/B evaluation. Generate its counterbalanced schedule, run matched
`control` and `mcp` tasks through an external adapter, then analyze a
metadata-only records envelope:

```bash
.venv/bin/python benchmark_agent_ab.py \
  --schedule-output agent-ab-schedule.json --repetitions 5 --seed 20260909
.venv/bin/python benchmark_agent_ab.py \
  --records agent-ab-records.json --output agent-ab-summary.json
```

The protocol requires non-secret identifiers for model configuration,
repository fixture, environment, reset policy, and adapter. Each run records
only task outcome and resource metadata. The analyzer validates pairing and
reports aggregate completion/retrieval outcomes, cost metrics, paired deltas,
and uncertainty. It does not invoke a model or collect telemetry. Keep records
and captures out of the repository and perform a privacy review before sharing
the aggregate summary.

For Codex CLI experiments, use `run_codex_agent_ab.py` as the external
adapter. It accepts a private task manifest, creates a fresh fixture copy per
scheduled run, invokes `codex exec` with the selected model, and configures the
ephemeral-buffer MCP server only for `mcp` runs. Control runs use an isolated
Codex configuration with no MCP server. The adapter records metadata only and
does not write transcripts or raw command output to the records file. Its
`context_bytes_proxy` field is an observable prompt/event-envelope proxy, not a
provider-reported model-context measurement.
If the fixture does not contain an importable `server` module, provide the
absolute server path with `--mcp-server-script`.
Records schema version 4 also includes exit code, failure reason, MCP-specific
tool-call counts, optional provider-reported input/output token counts, and
the complete provider usage samples observed in the Codex JSONL stream.
It separates the observable context proxy into prompt and output byte
components. Version-1 through version-3 records remain readable by the
analyzer; unavailable provider metrics are distinct from zero values.
The aggregate summary reports usage sample counts, monotonicity observations,
and first-to-last deltas. A monotonic sequence is not treated as proof of
cumulative accounting; use a controlled calibration matrix to establish the
provider semantics.

Issue #92's deterministic fixture generator is
`benchmark_agent_ab_fixtures.py`. Generate its synthetic fixture and private
run manifest before an experiment:

```bash
.venv/bin/python benchmark_agent_ab_fixtures.py \
  --fixture-output /tmp/agent-ab-fixture \
  --manifest-output /tmp/agent-ab-tasks.json
```

It emits large test/build output only when a task runs, includes a small
targeted-output control, and defines objective success criteria for each task.
Keep generated fixtures, manifests, records, and captures outside the
repository unless they have passed a separate privacy review.

Repeat the complete synthetic five-repetition Codex run with:

```bash
CODEX_HOME=/path/to/writable/authenticated-codex-home \
  ./run_agent_ab_experiment.sh
```

The script checks Codex authentication, creates a timestamped `/tmp` run
directory, executes paired control and MCP sessions, and writes metadata
records, an aggregate summary, the corresponding `records.result.json` and
`summary.result.json` workload result documents, and content-free lifecycle
logs there. Set `AGENT_AB_RUN_DIR` to choose another output directory, and
`AGENT_AB_EXPERIMENT` and `AGENT_AB_VARIANT` to tag the result documents with
an experiment group and variant so a series of runs can be listed and
compared by group. Use the lower-level runner commands for privacy-reviewed
repository fixtures.

The repeatable script enables `--require-mcp-calls`, so a run that completes
without any MCP tool call is recorded as `mcp_not_used` rather than being
accepted as MCP evidence.

For issue #93, create a checked-in aggregate baseline only from a reviewed
agent A/B summary:

```bash
.venv/bin/python benchmark_agent_ab_baseline.py \
  --summary /tmp/agent-ab-summary.json \
  --create-baseline \
  --output benchmark_agent_ab_baseline.json
```

Compare later summaries with `--baseline` and `--fail-on-regression`. The
baseline compares completion and retrieval as primary gated outcomes, latency
and context/token metrics with tolerances, and MCP/tool usage as reported
observations. Missing provider telemetry is unavailable rather than zero.
Do not make live Codex calls part of required pull-request CI; run this manual
workflow or an explicitly scheduled experiment after privacy review.

The accepted provisional routing guidance from the agent-level evaluation is to
prefer MCP for large noisy output and follow-up retrieval, while preferring
direct execution for small targeted inspections. This is guidance, not a
universal default or a set of hard numeric thresholds. The repository-shaped
evaluation profile is synthetic and should not be treated as representative of
every production repository. Provider-reported token deltas remain diagnostic
until a controlled calibration matrix distinguishes cumulative from per-turn
usage samples.

Example:

```bash
.venv/bin/python run_codex_agent_ab.py \
  --schedule agent-ab-schedule.json \
  --tasks /path/to/private-agent-tasks.json \
  --repository /path/to/privacy-reviewed-fixture \
  --model gpt-5.6-luna \
  --output /tmp/agent-ab-records.json
```

If the MCP arm is expected to call the configured server, add
`--allow-mcp-approvals`. The adapter then uses Codex automatic review and a
`workspace-write` sandbox for MCP runs so approval policy does not silently
force a shell fallback. Use this only with a disposable, privacy-reviewed
fixture; control runs retain the read-only sandbox without MCP approval routing.

Use a synthetic or privacy-reviewed fixture and do not commit the task
manifest, records, captures, or Codex transcripts.

For a reproducible local report, run the benchmark with a fixed seed and keep
the JSON output:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_effectiveness.py \
  --ab-runs 5 --seed 20260907 --output benchmark-effectiveness-ab.json
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_effectiveness.py \
  --consolidation-runs 5 --seed 20260907 \
  --output benchmark-effectiveness-consolidation.json
```

Report the evaluation name, seed, repetitions, scenario count, success rate,
useful-search rate, byte measurements, and timing scope. Pair those results
with `get_runtime_diagnostics()` or `get_buffer_stats()` when investigating a
real session, and sanitize all shared output. Never include an entire capture
or raw query merely to make a report reproducible; provide the smallest safe
excerpt or line range only when it is necessary.

## Consuming workload results

Benchmarks, evaluations, and the Codex A/B runner emit a common
`coding-agent-workload-result` document with `--result PATH` (see the README
section "Machine-readable workload results"). Comparison tooling, regression
checks, and dashboards should read that document instead of producer-specific
records:

- Validate first. `python workload_results.py result.json` rejects documents
  that do not match the format; refuse to compare an invalid or differently
  versioned document rather than guessing at its contents. The JSON Schema in
  `workload_result.schema.json` checks the same structure, including the unit
  pinned to each canonical measurement name, but it cannot enforce unique run
  ids beyond rejecting identical runs, so a schema-only consumer must check
  ids itself before matching runs. `format_version` changes only when a
  consumer would otherwise misread a document; new optional fields do not
  bump it.
- Compare like with like. Two results are comparable when `workload.name` and
  `format_version` match; then match runs by `id` and measurements by name and
  statistic (`median` against `median`, never `median` against `mean`).
  Report differences in `workload.parameters` and `environment` (host,
  CPU count, tool version, source revision) alongside any delta so the
  reader can see whether a change is a regression or a different setup.
- Treat `null` as missing. A statistic of `null`, a measurement with
  `samples: 0`, or a name present in only one document means the metric was
  unavailable; report it as incomparable instead of computing a delta. Tokens
  are the usual example: server-side harnesses report `estimated_tokens` as
  unavailable or as a labelled byte proxy, while agent runs report
  provider-counted `input_tokens` and `output_tokens`.
- Respect status. `status` is `success` only when every run succeeded;
  `partial`, `failure`, `timeout`, and `error` runs stay in the document with
  their `errors`. A regression check should fail on a non-success status
  before it looks at numbers, and a comparison should show success rates
  next to cost metrics rather than averaging over failed runs silently.
- Use phases for attribution. `phases` is an ordered timeline in seconds;
  compare it phase by phase to see where time moved, and use run-level
  `wall_time_seconds` for the headline number.
- Ignore `details` unless you are the producer. It is the native record and
  may change shape with `workload.producer_schema_version`.
- Organise by experiment, not by file name. The optional `experiment` block
  (`group` plus flat `metadata`) says which study a document belongs to and
  what varied; `list_workload_results.py` filters by `--group` and `--where
  KEY=VALUE`, and `compare_workload_results.py` accepts `DIR@GROUP` and
  `DIR@GROUP,KEY=VALUE` references (README section "Organizing
  experiments"). Consumers must accept documents without the block and
  metadata without a group, and must never match an absent metadata key,
  even against `null`. Order the documents of a group by `started_at`
  metadata (an ISO 8601 timestamp with a UTC offset, validated by the format)
  and fall back to `environment.recorded_at`, comparing instants rather than
  strings.
- Treat metadata as identifiers. Values are scalars of at most 256
  characters, keys that name credentials always hold `[redacted]`, and a
  producer's `--redact KEY` masks any other value; a consumer that publishes
  a listing should offer the same (`list_workload_results.py --redact KEY`).

`compare_workload_results.py` applies these rules (see the README section
"Comparing workload results"). It refuses invalid documents and mismatched
workload names, pairs runs by `id`, compares each statistic only with the same
statistic, lists missing and incompatible metrics instead of skipping them,
and prints the parameter and environment differences next to the deltas.

### Regression checks

Keep a baseline result per workload and host class (results from different
hosts are different setups, not regressions), regenerate it deliberately when
the workload or its parameters change, and gate on the headline metrics with a
tolerance wide enough for run-to-run noise on that host. Recording the
baseline and each candidate with `--experiment GROUP --metadata
environment=HOST_CLASS` lets `list_workload_results.py --group GROUP` show the
series and `DIR@GROUP,variant=baseline` name the baseline without a fixed
path:

```bash
.venv/bin/python compare_workload_results.py baseline.result.json candidate.result.json \
  --metric wall_time_seconds --statistic median --tolerance 25 --check \
  --output comparison.json
```

`--check` exits with status 2 when any compared metric regressed by more than
the tolerance or when either document has a non-success `status` (a document
narrowed with `PATH#RUN_ID` or `--select` is judged by its selected runs, so a
failure elsewhere in the file does not fail the check); invalid or
incomparable input exits with status 1. The JSON written by `--output` is a
`coding-agent-workload-comparison` document that records the options, every
document's workload block and environment, and each entry's outcome, so a
failed check can be reviewed without rerunning the workload. Missing metrics
and missing runs never fail the check on their own; inspect the `missing`
count in the summary when a producer stops reporting something.

Result documents contain workload parameters and host details but never
captured content, prompts, or user data; review any wrapper or attached
`details` block before sharing, as with every other benchmark output. A
comparison document embeds the same metadata and follows the same rule.

## Operational logging

Runtime events are emitted as one privacy-safe JSON object per stderr line.
Warnings and errors are enabled by default. Set `EPHEMERAL_LOG_LEVEL=INFO` to
include normal embedding readiness, capture eviction, and process lifecycle
events. MCP tool lifecycle events include only a local call ID, tool name,
duration, success state, and error class; captured content, labels, query text,
command text, and secrets are not logged. A start event without a matching
completion or failure event identifies a stalled tool/session boundary.
For the Codex A/B adapter, pass `--diagnostic-log-dir /path/to/logs` to write
one content-free lifecycle log per MCP run. Keep this directory outside the
repository and review it as diagnostic data.

Important events include command timeouts and termination, rejected capture or
socket payload limits, embedding load failures, capture eviction, storage
cleanup failures, and socket conflicts. Treat repeated cleanup, readiness, or
socket events as an investigation signal and pair them with the runtime
diagnostics report.

## Release verification

Releases are distributed as GitHub Actions artifacts and, once the trusted
publisher is registered, PyPI distributions.

1. Update the version in `pyproject.toml` and add release notes to
   `CHANGELOG.md`.
2. Run the focused and end-to-end test suites locally.
3. Replace the matching `Unreleased` changelog heading with release notes under
   a heading in the exact form `## X.Y.Z - YYYY-MM-DD`; the date must be a real
   calendar date. Then create and push an annotated `vX.Y.Z` tag from `main`. The tag
   must match the project version exactly and point to a commit contained in
   the default branch.
4. Wait for the tagged release workflow to finish.
5. Review the automatically created GitHub Release. It contains the wheel,
   source distribution, `SHA256SUMS`, release notes from `CHANGELOG.md`, and a
   link to the workflow run containing the provenance attestation.
6. Verify the checksums from the directory containing the distributions:

   ```bash
   sha256sum --check SHA256SUMS
   ```

7. Inspect the installed entry point from the wheel before distributing it:

   ```bash
   python -m venv /tmp/ephbuf-release-check
   /tmp/ephbuf-release-check/bin/python -m pip install --no-deps ephemeral_buffer_mcp-*.whl
   /tmp/ephbuf-release-check/bin/ephbuf --help
   ```

8. Confirm the package is available from PyPI and that its published metadata
   and files match the verified workflow artifacts.

The release workflow also checks tag format, package/changelog consistency
(including the exact dated heading format `## X.Y.Z - YYYY-MM-DD`),
clean source state, and tag ancestry before building. GitHub Release creation
must succeed before the PyPI publish job is allowed to run. Treat a failed
check, checksum mismatch, missing release asset, or missing attestation as a
release blocker.

### PyPI trusted publishing setup

The PyPI project owner must register a pending publisher for this GitHub
repository before the first publish:

- Owner: `k-rister`
- Repository: `ephemeral-buffer-mcp`
- Workflow: `release.yml`
- Environment: `pypi`

The workflow uses the `pypi` GitHub environment and OIDC trusted publishing;
no long-lived PyPI token is stored in GitHub. PyPI rejects reuse of an
already-published version, so each release must use a new version number.

## CI and branch protection

Changes to `main` must come through a pull request. The required checks are
`test (3.10)` and `test (3.12)`, and linear history is required. Reviews are
not currently required. The concurrency benchmark is optional and runs only
on its weekly schedule or through manual workflow dispatch. All GitHub Actions
are pinned to full commit SHAs; when updating an action, resolve the intended
release tag to its commit, retain the version comment beside the pin, and let
the required CI checks validate the change.

Pull-request CI uses a concurrency group keyed by the workflow, event, and
pull-request number. Pushing a newer commit to the same pull request cancels
the older in-progress run, including both Python test-matrix jobs. Runs for
different pull requests are independent; default-branch pushes, scheduled
runs, and manual dispatches use separate groups and are not canceled by PR
updates. When checking a pull request, use the completed or in-progress run
whose head SHA matches the current PR head; canceled runs for earlier commits
are no longer authoritative.

Repository administrators retain an explicit emergency bypass: an admin may
merge a pull request despite a failed required check when necessary. This is
an exception path, not the normal release process; record the reason in the
pull request before using it. Normal merges should wait for all required CI
checks to pass.

## Troubleshooting checklist

When a CLI capture fails:

1. Confirm the server is running.
2. Check that both processes resolve the same `EPHEMERAL_SOCKET_PATH`.
3. Check that the socket parent directory exists and is accessible.
4. Inspect `get_buffer_stats` for capture or byte-budget pressure.
5. Use `get_capture_summary` before retrieving larger slices or repeating a
   command.
