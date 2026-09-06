# Contributing to ephemeral-buffer-mcp

Thanks for helping improve ephemeral-buffer-mcp. Contributions should be
focused, documented when user-visible behavior changes, and accompanied by
tests where practical.

## Before you start

- For a bug, check the existing issues and include a minimal reproduction.
- For a feature or behavior change, open an issue first when the scope is not
  obvious. This helps avoid duplicated work.
- Do not include captured command output, credentials, tokens, private paths,
  source code, or user data in issues, pull requests, or test fixtures.

## Development setup

The supported development environment uses Python 3.12 and the committed
development lock file:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev-lock-py312.txt
```

Run the focused suite and coverage checks described in `AGENTS.md` before
opening a pull request. At minimum, run the tests relevant to your change.
Use deterministic test embeddings with `EPHEMERAL_TEST_EMBEDDINGS=1` when
running the application test suite.

## Pull requests

1. Create a development branch from `main`.
2. Keep each commit focused and explain behavior changes in its commit
   message.
3. Update `README.md`, `OPERATIONS.md`, or `CHANGELOG.md` when behavior,
   configuration, operations, or release content changes.
4. Describe the problem, the solution, and the validation performed in the
   pull request.
5. Wait for the required CI checks to pass before merging.

Changes to `main` go through a pull request. The repository uses linear
history, so merge using the repository's rebase-style workflow without
squashing unless the maintainer explicitly chooses otherwise.

If an AI tool contributed to a pull request, identify the tool, exact model,
and effort level in the description, following the attribution policy in
`AGENTS.md`.

## Code style and tests

Keep changes compatible with the supported Python versions declared in
`pyproject.toml`. Prefer the standard library and existing project patterns.
Add or update tests for changed behavior, including error paths and privacy
boundaries where relevant.

## Questions

Open an issue with enough context for someone else to reproduce or evaluate
the question. For sensitive security concerns, follow `SECURITY.md` instead.
