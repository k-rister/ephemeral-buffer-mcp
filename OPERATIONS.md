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
and the new instance exits with an error, while a stale socket is removed. The
new socket is created with owner-only permissions (`0600`). The parent
directory must already exist and be writable by the account running the server.

If neither variable is set, the legacy shared default is used. When running
multiple sessions without a session-aware launcher, give each server/CLI pair
its own explicit path rather than relying on that default:

```bash
export EPHEMERAL_SOCKET_PATH="${XDG_RUNTIME_DIR}/ephbuf-${SESSION_ID}.sock"
```

Keep the socket in a private directory when multiple users share a host.

## Capture limits and eviction

The buffer is intentionally transient. The defaults are:

- 25 captures (`EPHEMERAL_MAX_CAPTURES`)
- 50 MiB of captured UTF-8 content (`EPHEMERAL_MAX_BUFFER_BYTES`)
- least-recently-used (LRU) eviction when either limit is reached

The byte budget covers retained capture content. Embedding storage, the search
index, Python objects, and process RSS are reported separately by
`get_buffer_stats`; process RSS is an approximate operational metric rather
than an allocation limit.

FastEmbed is loaded lazily on the first capture or semantic search. Set
`EPHEMERAL_EMBEDDING_MODEL` to select a compatible model and
`EPHEMERAL_FASTEMBED_CACHE_DIR` to place its downloaded model files in a
controlled cache directory. Use `get_buffer_stats` to see the configured model
and whether it has been loaded yet.

Semantic indexing prefetch is disabled by default. To opt in, set
`EPHEMERAL_SEMANTIC_PREFETCH=1`; optionally set
`EPHEMERAL_SEMANTIC_PREFETCH_WORKERS` (default `1`) to a small positive value.
The bounded worker pool indexes captures after ingestion. Semantic and hybrid
search wait for an active job and retry failed jobs synchronously, preserving
the lazy path as the correctness fallback. Eviction and explicit cleanup cancel
queued work, and process shutdown waits for running jobs to finish. Use the
content-free prefetch counts in runtime diagnostics when checking host impact.

Override the limits before starting the server:

```bash
export EPHEMERAL_MAX_CAPTURES=50
export EPHEMERAL_MAX_BUFFER_BYTES=$((100 * 1024 * 1024))
```

`execute_and_capture` and `capture_file` reject per-request limits above the
configured byte budget. Oversized command output is retained as a bounded
head/tail sample and marked as truncated. Use `get_capture_summary` to inspect
the original size and truncation state.

Use `timeout_seconds` with `execute_and_capture` or `--timeout-seconds` with
`ephbuf` when a command might block or run indefinitely. A timed-out command is
terminated as a process group, its output collected so far is retained, and it
returns exit status 124.

Signal summaries recognize successful test-run markers and avoid treating
example error text inside a passing test run as an active failure. The complete
captured output remains available through search and slices.

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
Records schema version 3 also includes exit code, failure reason, MCP-specific
tool-call counts, optional provider-reported input/output token counts, and
the complete provider usage samples observed in the Codex JSONL stream.
Version-1 and version-2 records remain readable by the analyzer; unavailable
provider metrics are distinct from zero values.
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
records, an aggregate summary, and content-free lifecycle logs there. Set
`AGENT_AB_RUN_DIR` to choose another output directory. Use the lower-level
runner commands for privacy-reviewed repository fixtures.

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
3. Replace the matching `Unreleased` changelog heading with dated release
   notes, then create and push an annotated `vX.Y.Z` tag from `main`. The tag
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

The release workflow also checks tag format, package/changelog consistency,
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
