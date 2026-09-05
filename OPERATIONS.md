# Operations Guide

This guide covers the settings and maintenance procedures that matter when
running `ephemeral-buffer` outside of a local development session.

## Socket configuration

The CLI communicates with the server over a Unix domain socket. By default,
the socket is created as `ephemeral_buffer.sock` in the platform temporary
directory. Override that location when the default temporary directory is not
shared by the server and CLI processes:

```bash
export EPHEMERAL_SOCKET_PATH=/run/user/1000/ephemeral-buffer.sock
```

The server removes a stale socket at startup and creates the new socket with
owner-only permissions (`0600`). The parent directory must already exist and
be writable by the account running the server. Keep the socket in a private
directory when multiple users share a host.

## Capture limits and eviction

The buffer is intentionally transient. The defaults are:

- 25 captures (`EPHEMERAL_MAX_CAPTURES`)
- 50 MiB of captured UTF-8 content (`EPHEMERAL_MAX_BUFFER_BYTES`)
- least-recently-used (LRU) eviction when either limit is reached

The byte budget covers retained capture content. Embedding storage, the search
index, Python objects, and process RSS are reported separately by
`get_buffer_stats`; process RSS is an approximate operational metric rather
than an allocation limit.

Override the limits before starting the server:

```bash
export EPHEMERAL_MAX_CAPTURES=50
export EPHEMERAL_MAX_BUFFER_BYTES=$((100 * 1024 * 1024))
```

`execute_and_capture` and `capture_file` reject per-request limits above the
configured byte budget. Oversized command output is retained as a bounded
head/tail sample and marked as truncated. Use `get_capture_summary` to inspect
the original size and truncation state.

Use `clear_captures("all")` between unrelated investigations when the active
buffer should be released immediately instead of waiting for LRU eviction.

## Release verification

Releases are currently distributed as GitHub Actions artifacts; PyPI
publishing is intentionally deferred.

1. Update the version in `pyproject.toml`.
2. Run the focused and end-to-end test suites locally.
3. Create and push an annotated `vX.Y.Z` tag. The tag must match the project
   version exactly.
4. Wait for the tagged release workflow to finish.
5. Download the wheel, source distribution, `SHA256SUMS`, and provenance
   attestation from the workflow run.
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

The release workflow also checks the tag/version match and generates a GitHub
build-provenance attestation. Treat a failed check, checksum mismatch, or
missing attestation as a release blocker.

## CI and branch protection

Changes to `main` must come through a pull request. The required checks are
`test (3.10)` and `test (3.12)`, and linear history is required. Reviews are
not currently required. The concurrency benchmark is optional and runs only
on its weekly schedule or through manual workflow dispatch.

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
