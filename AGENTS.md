# Agent and Contributor Guide

This file contains project-level guidance for any coding agent or contributor.
It is intentionally independent of a particular editor, coding-agent product,
MCP client, or operating environment.

## Project workflow

- Keep `main` protected. Make changes on a development branch and open a pull
  request; do not push directly to `main`.
- Ensure the pull request CI checks pass before merging. The repository permits
  maintainer overrides for exceptional cases, but those are exceptions.
- Preserve the linear history policy. When merging, use a rebase-style merge
  without squashing unless the repository owner explicitly chooses otherwise.
- Remove merged development branches locally and remotely, then verify that the
  local checkout is clean and synchronized with `origin/main`.
- Keep commits focused and explain behavior changes in the commit message.

## Local validation

Create the supported development environment with the committed Python 3.12
development lock:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev-lock-py312.txt
```

Run the focused suite and application coverage with deterministic test
embeddings:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m coverage run \
  --source=. --omit='test_*.py,benchmark_concurrency.py,release_checks.py' \
  -m unittest test_benchmark_concurrency.py test_release_checks.py \
  test_engine.py test_capture_utils.py test_config.py test_cli.py test_server.py
.venv/bin/python -m coverage report --fail-under=91
.venv/bin/python -m unittest test_e2e_pipe.py
```

Track release guardrail coverage separately from application coverage:

```bash
COVERAGE_FILE=.coverage.release .venv/bin/python -m coverage run \
  --source=. -m unittest test_release_checks.py
COVERAGE_FILE=.coverage.release .venv/bin/python -m coverage report \
  --include='release_checks.py' --fail-under=90
```

The concurrency benchmark is optional and may use the real FastEmbed model:

```bash
.venv/bin/python benchmark_concurrency.py --captures 32 --workers 8
```

Do not add the benchmark to required pull-request checks; it is intended for
scheduled or manually dispatched workflow runs.

## Documentation and release expectations

- Update `README.md`, `OPERATIONS.md`, or `CHANGELOG.md` when behavior,
  configuration, operational procedures, or release content changes.
- Keep user-facing names and examples environment-neutral. Examples may
  describe environment-specific quirks when that helps users troubleshoot.
- Follow `OPERATIONS.md` for release preparation and verification. Releases
  require matching package metadata, a dated changelog section, a clean source
  state, and a tag contained in the default branch.
- Tagged releases create a GitHub Release with changelog notes, distributions,
  checksums, and a provenance-workflow link before PyPI publication proceeds.

## External artifacts and AI attribution

When an AI tool creates or modifies an externally visible artifact—such as a
commit, tag, issue, pull request, review, or release note—identify the actual
tool, model, and effort level used. For commits, use trailers such as:

```text
AI-Tool: <tool name>
AI-Model: <exact model identifier>
AI-Effort: <effort or not specified>
```

For GitHub artifacts, add a concise signed footer. Preserve human authorship
and co-author information; AI attribution supplements it and does not grant
authorization for external actions.
