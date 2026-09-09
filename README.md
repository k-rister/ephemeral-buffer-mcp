# Ephemeral Buffer MCP Server (`ephemeral-buffer`)

[![CI](https://github.com/k-rister/ephemeral-buffer-mcp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/k-rister/ephemeral-buffer-mcp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/k-rister/ephemeral-buffer-mcp)](https://github.com/k-rister/ephemeral-buffer-mcp/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ephemeral-buffer-mcp)](https://pypi.org/project/ephemeral-buffer-mcp/)
[![codecov](https://codecov.io/gh/k-rister/ephemeral-buffer-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/k-rister/ephemeral-buffer-mcp)

An ephemeral in-memory command output capture and hybrid search engine (BM25 + Semantic Embeddings) for AI coding assistants (Claude Code, Antigravity, Cursor, etc.).

---

## 🎯 The Problem This Solves

When coding agents run commands that generate large outputs (thousands of lines of build logs, test runs, stack traces, JSON dumps), agents face two failure modes:
1. **Context Pollution:** Ingesting megabytes of raw text blows out token limits and degrades model reasoning.
2. **Blind Bash Filtering:** Agents waste multiple turns running `head`, `tail`, `grep`, and `awk` trying to guess error patterns.

## 💡 The Solution

`ephemeral-buffer` provides a transient in-memory ring buffer with **Dual Hybrid Indexing** and **Content-Aware Structure Parsing**:
- **BM25 Lexical Search (SQLite FTS5):** For exact matches on error codes (`NullPointerException`, `ECONNREFUSED`, `exit 137`, HTTP `502`).
- **Dense Semantic Vector Search (FastEmbed ONNX):** For fuzzy conceptual queries (*"Where did the DB connection pool fail?"* or *"Why did authentication fail?"*).
- **Unified Diff Structural Mapping:** Automatically detects git diffs and PR diffs (`gh pr diff`, `git show`, `git diff`), parses modified files, additions/deletions, and generates a line-indexed file map in the summary.
- **Smart Signal Filtering:** Scans command/build/test logs for diagnostic keywords, suppresses false positives in diffs and source code, and accurately captures test runner failures, unhandled exceptions, and merge conflicts. Use `content_type='log'` when a plain-text capture should be signal-scanned.
- Successful test-run summaries such as `OK` or `25 passed` suppress fixture-only error and failure keywords while retaining the original output for search.
- **Reciprocal Rank Fusion (RRF):** Blends lexical and semantic ranking for high precision retrieval.
- **LRU Capture Eviction:** Holds up to 25 captures and 50 MiB of captured content by default, evicting the least recently used captures when either limit is reached.
- **Thread-Safe Shared Engine:** Serializes ingestion, search, LRU updates, eviction, and cleanup across MCP requests and CLI socket clients.

---

## 📦 Installation

### Requirements

- Python 3.10 or newer
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

On Windows, use the equivalent commands from the virtual environment's
`Scripts` directory:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install ephemeral-buffer-mcp
.venv\Scripts\ephbuf.exe --help
```

To upgrade or remove the package:

```bash
.venv/bin/python -m pip install --upgrade ephemeral-buffer-mcp
.venv/bin/python -m pip uninstall ephemeral-buffer-mcp
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
documented in [Testing the Server](#-testing-the-server).

### Start the MCP server manually

The MCP server uses stdio for communication with the MCP host. Start it with
the Python interpreter from the environment where the package was installed:

```bash
.venv/bin/python -m server
```

Normally you should let your MCP client start this process automatically. Do
not start a separate server for every shell command: the `ephbuf` CLI sends
captures to the running server over its local Unix socket.

### Configure an MCP client

MCP clients generally need the command, arguments, and environment used to
start a stdio server. A portable configuration looks like this:

```json
{
  "mcpServers": {
    "ephemeral-buffer": {
      "command": "/absolute/path/to/ephemeral-buffer-mcp/.venv/bin/python",
      "args": ["-m", "server"]
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
arguments: -m server
```

After saving the configuration, restart or reload the MCP client and confirm
that the `ephemeral-buffer` tools appear. The server exposes capture, search,
summary, slice, consolidation, diagnostics, and cleanup tools; the `ephbuf`
CLI is a separate convenience client for shell output.

### First-use model initialization

FastEmbed loads the embedding model lazily on the first semantic or hybrid
search, rather than during capture or when the server process imports. The
first semantic operation may therefore take longer and may download model files. Subsequent operations use
the local model cache. To select a compatible model or move the cache, set:

```bash
export EPHEMERAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export EPHEMERAL_FASTEMBED_CACHE_DIR="$HOME/.cache/ephemeral-buffer"
```

Semantic indexing can optionally be prefetched after ingestion:

```bash
export EPHEMERAL_SEMANTIC_PREFETCH=1
export EPHEMERAL_SEMANTIC_PREFETCH_WORKERS=1
```

Prefetch is disabled by default. When enabled, the bounded worker pool indexes
captures in the background while ingestion returns; semantic and hybrid search
wait for a relevant in-progress job, and failed jobs retry through the normal
lazy path. Evicted or cleared captures do not retain queued work. Keep the
worker count small because embedding generation competes for host CPU and
memory; `get_buffer_stats` and `get_runtime_diagnostics` expose only aggregate
pending/failed counts.

The exact model and cache location can also be supplied in the MCP client's
`env` configuration. Keep the model cache writable by the user running the
MCP client.

### Installation choices at a glance

| Use case | Recommended installation |
| :--- | :--- |
| Normal user | PyPI install in a virtual environment |
| MCP host | PyPI install, then configure `python -m server` |
| Shell/CLI use | PyPI install, then run `ephbuf` |
| Contributor | Source checkout with `pip install -e .` |
| Release validation | Follow the procedures in [OPERATIONS.md](OPERATIONS.md) |

If the MCP client cannot find the server, check the absolute interpreter path,
the selected Python environment, and the client's server logs. For socket,
capture-limit, logging, and deployment troubleshooting, see
[OPERATIONS.md](OPERATIONS.md).

---

## 🏗 Architecture & Flow

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion Paths"]
        A["CLI Pipe: command 2>&1 | ephbuf"] --> D["Unix Socket (platform temp dir)"]
        B["Agent Tool: execute_and_capture(cmd)"] --> E["Ephemeral Ring Buffer Engine"]
        C["Agent Tool: capture_text / capture_file"] --> E
        D --> E
    end

    subgraph Indexing["2. Dual Hybrid Indexing & Classification"]
        E --> F["SQLite FTS5 (BM25 Lexical)"]
        E --> G["FastEmbed ONNX (Dense Vectors)"]
        E --> K["Diff & Signal Parser (File Maps & Conflict Detection)"]
    end

    subgraph Querying["3. Agent Query & Retrieval"]
        F & G --> H["Reciprocal Rank Fusion (RRF)"]
        H --> I["search_capture(query, mode='hybrid')"]
        K --> L["Diff File Map & get_capture_slice"]
        I --> J["Precise Context Chunk + Line Numbers"]
    end
```

---

## 🚀 How to Use It

### 1. From the Terminal (CLI Pipe via `ephbuf`)
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
retains the beginning and end of oversized output.
Use `--timeout-seconds` to stop a wrapped command after a bounded runtime; timed
out commands retain the output collected so far and exit with status 124.
Requested `max_output_bytes` and `capture_file` `max_bytes` values may not
exceed the configured buffer byte limit; the tools return a validation error
instead of silently clamping them.

### 2. From the AI Agent via MCP Tools

The agent has access to the following tools:

| Tool | Purpose |
| :--- | :--- |
| `execute_and_capture(command, cwd, label, content_type='auto', max_output_bytes=None, timeout_seconds=None)` | Executes a shell command with bounded head/tail capture, optional timeout, and a compact diagnostic summary (exit code, diff file map, error signals, and truncation status) to the agent context. |
| `preflight_command(command, cwd=None)` | Performs content-free path, symlink, local Git-root, and executable-resolution diagnostics without executing the requested command. |
| `capture_text(content, label, content_type='auto')` | Ingests text directly into the buffer. |
| `capture_file(file_path, label, content_type='auto', max_bytes=None)` | Ingests a bounded log/output file from disk; defaults to the configured buffer byte limit. |
| `consolidate_captures(capture_ids, label, max_captures=25, max_bytes=None)` | Creates one bounded, searchable JSON capture from multiple captures while preserving source IDs and source line numbers. |
| `search_capture(query, mode, top_k, context_lines)` | Hybrid/BM25/Semantic search over the captured output. BM25 extracts alphanumeric and underscore terms, treats punctuation—including regex-like characters—as separators, and combines terms with OR. Hybrid ranking gives lexical matches priority over semantic-only matches. Returns matching chunks with surrounding context lines, exact numeric context boundaries, raw context, and line numbers. |
| `get_capture_slice(start_line, end_line)` | Retrieves exact line ranges to inspect full stack traces, logs, or specific diff files. |
| `get_capture_summary(capture_id)` | Diagnostic overview (line counts, diff file maps, error signals, preview). |
| `get_buffer_stats()` | Reports aggregate capture count, content bytes, lines, chunks, embedding model readiness, embedding bytes, accounted bytes, and process RSS. When local metrics are enabled, it also includes the content-free aggregate metrics snapshot. |
| `get_runtime_diagnostics()` | Opt-in, content-free report of runtime version, platform, uptime, socket mode, buffer limits, embedding readiness, and process memory. |
| `list_captures()` | Lists active captures in the ring buffer. |
| `clear_captures(capture_id)` | Clears buffer. |

Choose the execution path based on the output and inspection goal:

- Use direct command execution for a small, targeted inspection where the
  output is already bounded and immediate terminal feedback is sufficient.
- Use `execute_and_capture` once output may be noisy, large, or uncertain—such
  as tests, builds, and logs—because it bounds context and makes later search
  and exact retrieval available. This is an advisory heuristic, not a hard
  line-count policy.
- Use `capture_text` when output is already in hand, or `capture_file` for a
  file that has been checked and intentionally selected for ingestion.

Use `preflight_command` when repository identity or path resolution is
uncertain before a sensitive command. It reports resolved facts and explicit
unavailable states without running the requested command or exposing command
output. It cannot predict shell expansion, aliases, pipelines, redirections,
environment changes, or arbitrary shell logic, so normal command validation
and user intent checks remain necessary.

Before any repository-sensitive command or file capture, verify the intended
working directory and target path. Prefer an explicit `cwd`, confirm the
repository identity, and resolve symlinks when path identity matters. Shell
expansion, inherited working directories, and symlinks can target a different
location than the spelling suggests. Capture limits protect context size; they
do not validate command intent, path identity, or filesystem safety.

For diff captures, `get_capture_summary` reports the detected file map,
addition/deletion statistics, line ranges, and merge-conflict signals. Use
`get_capture_slice` with those ranges to retrieve the complete file context.

### Consolidating multi-result workflows

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

### 3. Capture Hygiene

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

The server defaults can be overridden with `EPHEMERAL_MAX_CAPTURES` and
`EPHEMERAL_MAX_BUFFER_BYTES`. Session-aware launchers can set
`EPHEMERAL_SESSION_ID` so each server/CLI pair automatically derives a unique
socket path; `EPHEMERAL_SOCKET_PATH` remains an explicit override. The byte
limit accounts for
captured UTF-8 content; `get_buffer_stats` also reports embedding model
readiness, embedding/cache settings, and process memory separately.
`execute_and_capture` retains the beginning and end of oversized command
output and marks the capture with its original byte count.

Call `get_runtime_diagnostics()` when reporting a field observation. It is
explicitly opt-in and returns operational metadata only; captured content,
labels, command arguments, and session ID values are excluded. Sanitize any
additional output before sharing it.

Operational events are written as privacy-safe JSON lines to stderr. Warnings
and errors are enabled by default; set `EPHEMERAL_LOG_LEVEL=INFO` to include
normal readiness, eviction, and process lifecycle events. Logs never include
captured content or command text.

### Optional local usage metrics

Set `EPHEMERAL_METRICS=1` to collect content-free, in-process usage metrics.
The metrics include per-tool call counts, success/failure counts, duration
totals, and aggregate capture/search/retrieval, empty-search, eviction, and
cleanup events. They are disabled by default, are never sent anywhere, and do
not retain captured content, labels, commands, or query text. When enabled,
both `get_runtime_diagnostics()` and `get_buffer_stats()` include the same
aggregate metrics snapshot. The event keys are stable and zero-filled when no
event has occurred. Metrics are process-lifetime state: restarting the server
clears them, while capture-associated correlation state is released when a
capture is evicted or explicitly cleared.

### Effectiveness metrics and privacy

The built-in metrics are local, opt-in operational telemetry. Set
`EPHEMERAL_METRICS=1` only when you want measurements for the current server
process; nothing is uploaded or shared by the server. The metrics contain
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

See [OPERATIONS.md](OPERATIONS.md) for deployment settings, troubleshooting,
release verification, and repository maintenance procedures.

---

## 🛠 Testing the Server

Set up a local development environment from a fresh checkout:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev-lock-py312.txt
```

The committed `requirements-dev-lock-py312.txt` file is the reproducible
Python 3.12 development and release environment. Python 3.10 remains
supported through the direct requirements and tested `constraints.txt` file;
the CI matrix exercises both paths. Keep `requirements.txt` and
`requirements-dev.txt` as the reviewable dependency inputs, and regenerate the
Python 3.12 locks with `pip-tools` after an intentional dependency update:

```bash
.venv/bin/python -m pip install pip-tools
.venv/bin/pip-compile --generate-hashes --output-file=requirements-lock-py312.txt requirements.txt
.venv/bin/pip-compile --generate-hashes --output-file=requirements-dev-lock-py312.txt requirements-dev.txt
```

Review the resulting changes, run the full test matrix, and run `pip-audit`
before merging. Downstream users install the package normally; its compatible
dependency ranges in `pyproject.toml` are intentionally not replaced by the
development locks.

Run the test suite:
```bash
.venv/bin/python -m unittest test_engine.py test_capture_utils.py test_config.py test_cli.py test_server.py
.venv/bin/python -m unittest test_e2e_pipe.py
```

Measure focused-test coverage locally:
```bash
.venv/bin/python -m coverage run --source=. --omit='test_*.py,benchmark_concurrency.py,benchmark_effectiveness.py,benchmark_latency.py,release_checks.py' -m unittest test_benchmark_concurrency.py test_benchmark_effectiveness.py test_release_checks.py test_engine.py test_capture_utils.py test_config.py test_cli.py test_server.py
.venv/bin/python -m coverage report
```
The current focused-test baseline is 91%; CI enforces a 91% minimum after
adding coverage for defensive command, limit, cleanup, embedding, and socket
handling paths. Coverage reports are uploaded for inspection, and future
threshold increases should follow similarly targeted test additions.
The release guardrail utility is measured separately because it is a workflow
utility rather than application runtime code:
```bash
COVERAGE_FILE=.coverage.release .venv/bin/python -m coverage run --source=. -m unittest test_release_checks.py
COVERAGE_FILE=.coverage.release .venv/bin/python -m coverage report --include='release_checks.py'
```

GitHub Actions runs the compile check, focused tests, and end-to-end test on
Python 3.10 and 3.12 for pushes to `main` and pull requests. The FastEmbed
model is loaded on the first capture or semantic search rather than during
server import. Set `EPHEMERAL_EMBEDDING_MODEL` to select a compatible model and
`EPHEMERAL_FASTEMBED_CACHE_DIR` to control its cache directory. The model cache
is retained between CI runs to reduce startup time. CI unit and end-to-end
tests set the internal `EPHEMERAL_TEST_EMBEDDINGS=1` flag, which uses a small
deterministic embedding substitute so test execution does not depend on a
model download; release and benchmark jobs continue to exercise FastEmbed.
It also builds the wheel and verifies the installed `ephbuf` entry point.
CI audits the declared dependencies with `pip-audit` and fails if known
vulnerabilities are found.
CI installs the hashed Python 3.12 development/runtime locks and uses the
tested `constraints.txt` path for Python 3.10. The direct requirements and
constraints are updated only after the full test matrix passes; lock updates
must be reviewed together with their resolver output and audit results.

Pushing a version tag such as `v0.1.1` runs the release workflow, which first
verifies that the tag is valid SemVer, points to a commit contained in the
default branch, and starts from a clean checkout. It also requires the tag,
`pyproject.toml`, and a dated matching `CHANGELOG.md` section to agree. The
workflow then builds wheel and source distributions, validates their metadata,
verifies the installed package, and uploads the artifacts for review. A failed
guardrail reports the mismatched value or source-state problem before building.
The workflow creates a GitHub Release using the matching changelog section,
attaches the wheel, source distribution, and `SHA256SUMS`, and links back to
the workflow run containing the build-provenance attestation. Verify a
downloaded artifact with `sha256sum --check SHA256SUMS` from the directory
containing the files. The same verified distributions are then published to
PyPI through trusted publishing.
After the repository's `pypi` environment is configured with a PyPI trusted
publisher, the workflow publishes the distributions to PyPI automatically.

Run the concurrency benchmark:
```bash
.venv/bin/python benchmark_concurrency.py --captures 32 --workers 8
```
The benchmark accepts `--min-ingest-per-second` and `--min-reads-per-second`
thresholds for direct checks. For repeatable regression checks, pass
`--baseline benchmark_baseline.json --output benchmark-concurrency.json`.
The checked-in baseline uses a 20% tolerance: a run fails only when ingest or
read throughput drops below 80% of its baseline. Each scheduled or manually
dispatched GitHub Actions run records the raw JSON result as an artifact and
adds the measurements and regression status to the workflow summary. This
benchmark remains optional and is not part of the required pull-request checks;
update the baseline deliberately when the runner or benchmark workload changes.

Measure command-capture latency by output size and pipeline phase:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_latency.py \
  --samples 5 --output benchmark-latency.json
```
The latency harness reports cold-start time plus warm median and p95 timings
for command execution, BM25 capture/indexing, deferred semantic indexing,
summary generation, and the combined pipeline. Semantic embeddings are now
materialized when semantic or hybrid search first needs them, so use the
separate semantic-index phase when evaluating end-to-end costs. The benchmark
is diagnostic and optional, not a required pull-request check.

Compare lazy semantic indexing with opt-in asynchronous prefetch:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_prefetch.py \
  --line-count 256 --samples 5 --output benchmark-prefetch.json
```
The prefetch harness reports ingestion, first semantic search, and subsequent
semantic search medians for both modes. Prefetch should reduce first-query
latency when work completes during ingestion, while adding bounded background
resource use; treat the output as deployment-specific diagnostic evidence.

Measure direct-versus-captured routing tradeoffs with synthetic output profiles:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_routing.py \
  --samples 5 --output benchmark-routing.json
```
The routing harness compares direct command completion with bounded capture and
summary generation for targeted (16 lines), test-like (256 lines), and
build/log-like (2048 lines) output. Use the medians and p95 values to keep the
heuristic honest: direct execution generally has lower latency for small,
bounded output, while capture adds searchable context and bounded response
size for noisy or uncertain output. These synthetic measurements are guidance,
not universal thresholds or a required CI gate.

Measure command-output handling effectiveness with deterministic synthetic data:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_effectiveness.py \
  --mode both --output benchmark-effectiveness.json
```
The effectiveness harness compares a full-output baseline with an
engine-backed MCP workflow across large output, failure logs, diffs, and
follow-up searches. It reports per-scenario success, search usefulness,
retrievals, bytes examined, resolution time, and an aggregate comparison as
machine-readable JSON. Token usage is explicitly marked unavailable because
this harness does not invoke a model. Fixtures contain no project content and
are generated in code, so runs are reproducible. This is a server-side smoke
evaluation, not a claim about any particular coding agent or model.

Evaluate search relevance across supported modes with deterministic synthetic
fixtures:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_relevance.py \
  --top-k 3 --baseline benchmark_relevance_baseline.json \
  --output search-relevance.json
```
The relevance benchmark covers exact errors, punctuation-heavy queries,
conceptual semantic queries, and lexical/semantic conflicts. It reports
hit@1, hit@k, and mean reciprocal rank (MRR) for BM25, semantic, and hybrid
search. The fixtures define expected retrieval markers, so this measures
retrieval relevance only—not agent answer quality, token usage, or universal
performance. CI compares the deterministic scores with the checked-in
`benchmark_relevance_baseline.json` and fails only when a metric drops by more
than the recorded tolerance. Baseline changes must be deliberate, reviewable,
and accompanied by a fixture or intended-behavior explanation. The evaluation
uploads its machine-readable JSON result with the other benchmark artifacts.

The baseline stores only schema and fixture versions, deterministic embedding
mode, aggregate per-mode scores, query counts, and explicit tolerances. It does
not store captures, raw command output, or user queries. Environment-specific
latency and effectiveness results remain CI artifacts and summaries rather than
exact cross-run gates; retention follows the workflow artifact policy.

#### Example benchmark results

The reproducible paired evaluation was run on 2026-09-07 with five
repetitions, seed `20260907`, and deterministic test embeddings. Across four
synthetic scenarios, both the direct-output baseline and the MCP workflow
completed all 20 tasks, and every MCP search was useful. The MCP workflow
examined 68–90% fewer bytes than the baseline per scenario (81% on average):

| Scenario | Bytes examined reduction | Completion |
| :--- | ---: | ---: |
| Large build output | 90% | 5/5 |
| Failure log | 83% | 5/5 |
| Review diff | 68% | 5/5 |
| Timeout log | 85% | 5/5 |

The consolidation evaluation, using the same seed and five repetitions,
reduced the initial multi-result overview from an average of 4,298 bytes to
213 bytes (95% fewer overview bytes), while preserving a 100% targeted
retrieval success rate. These measurements quantify the MCP data path: fewer
bytes need to be returned to the agent before it asks for targeted detail.

They are not universal performance guarantees. The fixtures are synthetic,
the harness does not invoke a model, and the byte reduction is not a direct
token-savings measurement. In this run, consolidated processing also took
about 10 times longer locally than sequential processing, while retrieving
similar detail. The benchmark therefore demonstrates context-size and
workflow-shaping benefits, not that every workload will be faster or that
search results will be relevant for arbitrary repositories. Re-run the
commands below with representative, privacy-reviewed tasks before making
project-specific claims.

Run a controlled local A/B evaluation with repeated paired measurements:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_effectiveness.py \
  --ab-runs 5 --seed 20260907 --output benchmark-effectiveness-ab.json
```
The A/B report uses the same deterministic fixtures in both modes, seeded task
and mode ordering, and local-only measurements. It reports completion rate,
mean/min/max time, standard deviation, repeated commands, search usefulness,
byte reduction, and local MCP processing overhead for each scenario. The timing
ratio covers only local capture, indexing, search, and retrieval; it is not an
agent-level performance measurement. Since no model is invoked, token usage is
unavailable. Treat the recommendations as synthetic benchmark guidance and
repeat the evaluation with representative agent tasks before generalizing the
results.

For an end-to-end coding-agent evaluation, generate a counterbalanced,
privacy-safe schedule:
```bash
.venv/bin/python benchmark_agent_ab.py \
  --schedule-output agent-ab-schedule.json --repetitions 5 --seed 20260909
```
Run each scheduled task with MCP disabled (`control`) and enabled (`mcp`) using
the same model configuration, repository fixture, environment, and reset policy.
Have an external agent adapter write a records envelope containing only the
schedule, non-secret protocol identifiers, and per-run fields: completion,
signal retrieval, duration, tool calls, repeated commands, context bytes, and
peak RSS bytes. Summarize it with:
```bash
.venv/bin/python benchmark_agent_ab.py \
  --records agent-ab-records.json --output agent-ab-summary.json
```
The analyzer validates balanced paired runs, reports per-mode aggregates and
MCP-minus-control deltas with uncertainty, and emits recommendations. It does
not invoke a model, require credentials, or accept raw prompts, transcripts,
commands, captures, or user content. Do not commit agent records or generated
captures; share only the aggregate summary after privacy review.

The repository includes a Codex CLI adapter for executing that protocol. It
uses the requested model explicitly, creates a fresh fixture copy for every
run, isolates control and MCP configuration, and writes metadata-only records.
Create a private task manifest (do not commit it) with one prompt and optional
signal marker for each scheduled task:

```json
{
  "tasks": {
    "targeted-inspection": {"prompt": "Inspect the fixture and report the marker.", "signal_marker": "MARKER"},
    "noisy-test-failure": {"prompt": "Run the fixture test and report the failure marker.", "signal_marker": "MARKER"},
    "build-log-search": {"prompt": "Inspect the build log and report the marker.", "signal_marker": "MARKER"},
    "follow-up-context": {"prompt": "Find the earlier marker and report it.", "signal_marker": "MARKER"}
  }
}
```

Run the adapter from the repository checkout:

```bash
.venv/bin/python run_codex_agent_ab.py \
  --schedule agent-ab-schedule.json \
  --tasks /path/to/private-agent-tasks.json \
  --repository /path/to/privacy-reviewed-fixture \
  --model gpt-5.6-luna \
  --output /tmp/agent-ab-records.json
.venv/bin/python benchmark_agent_ab.py \
  --records /tmp/agent-ab-records.json \
  --output /tmp/agent-ab-summary.json
```

The runner requires a locally authenticated `codex` CLI. It uses
`codex exec --json --ephemeral`, disables approval prompts, and defaults to a
read-only sandbox. `context_bytes_proxy` is an observable prompt/event-envelope
proxy because the CLI does not expose the model's internal context size.
When the fixture does not contain an importable `server` module, pass
`--mcp-server-script /absolute/path/to/server.py`.
For MCP experiments where the client must be allowed to call the configured
server, add `--allow-mcp-approvals`. This uses Codex automatic review with a
`workspace-write` sandbox, and should only be used with a disposable,
privacy-reviewed fixture. Control runs continue to use approval policy
`never`; the records protocol identifies the selected policy.
Review prompts, fixtures, and generated records for privacy before sharing;
the runner does not persist transcripts in its records output.

Runner records use version 2 and add exit code, failure reason, MCP-specific
tool-call count, and provider-reported input/output token counts when Codex
emits them. Older version-1 records remain readable; missing provider metrics
are reported as unavailable rather than zero.

Generate the reviewed synthetic EB-heavy fixture and its task manifest with:

```bash
.venv/bin/python benchmark_agent_ab_fixtures.py \
  --fixture-output /tmp/agent-ab-fixture \
  --manifest-output /tmp/agent-ab-tasks.json
```

The fixture generates large test and build output at runtime, plus a small
targeted-output task and a follow-up retrieval task. The manifest includes
output-size bands, expected signals, and objective success criteria. It is
synthetic and contains no user logs or captured output.

Create and compare an aggregate agent A/B baseline after a privacy review:

```bash
.venv/bin/python benchmark_agent_ab_baseline.py \
  --summary /tmp/agent-ab-summary.json \
  --create-baseline \
  --output benchmark_agent_ab_baseline.json
.venv/bin/python benchmark_agent_ab_baseline.py \
  --summary /tmp/agent-ab-summary.json \
  --baseline benchmark_agent_ab_baseline.json \
  --fail-on-regression \
  --output /tmp/agent-ab-comparison.json
```

The baseline stores model, fixture, seed, repetition, aggregate metrics, and
explicit tolerances only. It never stores prompts, transcripts, commands,
captures, or user content. This comparison is a documented manual workflow;
live model calls are not part of required pull-request CI. Update a checked-in
baseline only when fixture or model changes are explained in review.

Compare sequential per-capture retrieval with the consolidated workflow:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python benchmark_effectiveness.py \
  --consolidation-runs 5 --seed 20260907 \
  --output benchmark-effectiveness-consolidation.json
```
This report measures overview and retrieval response bytes, targeted retrieval
success, search/retrieval counts, local processing time, and omitted records.
It models each synthetic scenario as a repository result and does not invoke a
coding-agent model; response-byte reductions therefore describe the MCP data
path, not end-to-end agent performance.

When sharing a result, include the command, seed, repetitions, benchmark
evaluation name, success rate, useful-search rate, byte reduction, and timing
scope. Do not attach generated captures or paste raw command output. Check any
surrounding report or wrapper for repository-specific content before sharing
the benchmark JSON.
