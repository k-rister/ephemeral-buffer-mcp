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

## Field-observation checklist

When investigating behavior from a real MCP session, collect operational
metadata rather than captured command content. This keeps reports useful while
avoiding accidental disclosure of source code, logs, credentials, or other
sensitive data.

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
When command output is needed, provide only a sanitized excerpt or line range;
do not attach an entire capture by default. Remove credentials, tokens,
private paths, source code, and user data before sharing diagnostics. A useful
report should be actionable without requiring access to the original capture.

## Operational logging

Runtime events are emitted as one privacy-safe JSON object per stderr line.
Warnings and errors are enabled by default. Set `EPHEMERAL_LOG_LEVEL=INFO` to
include normal embedding readiness, capture eviction, and process lifecycle
events. Logged fields describe event types, IDs, sizes, limits, signal names,
and error classes; captured content, labels, command text, and secrets are not
logged.

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
