---
name: eb-activity-summary
description: Summarize activity for the Ephemeral Buffer MCP repository, including commits, pull requests, issues, releases, and CI. Use for EB weekly or custom-period activity reports; do not use for implementing repository changes.
metadata:
  short-description: Summarize Ephemeral Buffer activity
---

# Ephemeral Buffer activity summary

Produce a read-only activity report for the repository in the current
workspace. The reconstructed skill is intentionally repository-specific:
`k-rister/ephemeral-buffer-mcp` (project name: Ephemeral Buffer MCP, or EB).

## Scope defaults

When the request does not specify them, use:

- audience: developer, including implementation, tests, documentation, CI,
  and release work;
- authors: all contributors to the EB repository;
- period: the previous seven days ending now, expressed with explicit local
  dates and UTC timestamps.

If the host supports interactive questions and the user has not already
provided the scope, ask for:

1. **Audience:** User-facing changes or Developer/all changes.
2. **Authors:** All authors, the authenticated user, or a named GitHub user.
3. **Date range:** Past week, past two weeks, past month, or a custom
   `YYYY-MM-DD..YYYY-MM-DD` range.

Do not ask again when the user has already supplied an unambiguous scope.
The phrase “all EB activity” means developer/all authors unless the user says
otherwise.

## Collect the activity

First verify the repository identity from `git remote -v` or the GitHub API.
Do not silently summarize a different repository. Use read-only Git and
GitHub operations:

- commits in the interval, from the local history and, when available, the
  repository commits API;
- pull requests created, merged, closed, or updated in the interval, plus
  currently open pull requests;
- issues created, closed, or updated in the interval, plus currently open
  issues;
- releases published in the interval;
- GitHub Actions runs in the interval, grouped by workflow and conclusion;
- the current branch/worktree state when the report is generated.

For GitHub searches, use `YYYY-MM-DD` dates. For commits and workflow APIs,
use ISO 8601 timestamps with an explicit UTC offset. Prefer bounded JSON
queries and `--jq` projections over unbounded pages. When output could be
large, route it through the repository's `ephemeral-buffer` capture/search
tools and retrieve only the relevant slices.

Useful single-repository query shapes are:

```text
gh api "search/issues?q=repo:k-rister/ephemeral-buffer-mcp+type:pr+created:START..END&per_page=100"
gh api "search/issues?q=repo:k-rister/ephemeral-buffer-mcp+type:pr+merged:START..END&per_page=100"
gh api "search/issues?q=repo:k-rister/ephemeral-buffer-mcp+type:issue+created:START..END&per_page=100"
gh api "repos/k-rister/ephemeral-buffer-mcp/commits?since=START_ISO&until=END_ISO&per_page=100"
gh run list --repo k-rister/ephemeral-buffer-mcp --created ">=START"
gh release list --repo k-rister/ephemeral-buffer-mcp --limit 100
```

Use `gh pr list`, `gh issue list`, or additional API queries when created,
merged, closed, and updated events need separate counts. Do not treat a failed
GitHub request as zero activity: report the affected section as unavailable
and use local evidence where possible.

If a Jira MCP server is available, include PERFNFV tickets created or updated
in the same interval and link them to `https://issues.redhat.com/browse/KEY`.
Otherwise omit Jira entirely; do not warn merely because it is unavailable.

## Interpret and filter

For developer/all reports, retain all relevant activity and identify the
author for each linked PR or issue. For user-facing reports, include changes
that affect users or operators: new features, fixes, new tools or benchmarks,
metrics/data/API/schema changes, user-facing CLI changes, and user-facing
documentation. Exclude internal refactors, CI-only work, test-only work,
chore/maintenance changes, developer-only documentation, and internal tooling
unless they have a clear user impact.

Group related work into a small number of themes. Each theme should explain:

1. what changed;
2. why it matters or what impact it has; and
3. optionally, how it fits a larger EB initiative.

Link every referenced pull request, issue, release, and Jira ticket. Use
nested bullets when a theme contains distinct subtopics. Do not infer intent
from a terse commit title when the linked PR or issue provides better evidence;
say when a theme is an inference.

## Report format

Return the report in Markdown and also write `/tmp/activity-summary.html` for
Google Docs copy/paste. The Markdown report should contain:

- a headline with repository and date range;
- a compact stats line covering commits, PRs, issues, releases, and CI;
- key themes with links and concise impact descriptions;
- releases and notable operational/release events;
- commits by day or major workstream when useful;
- currently open work and repository state;
- a short limitations note when any source was unavailable.

The HTML report should:

- begin with `<meta charset="UTF-8">`;
- use only simple `<p>`, `<ul>`, `<li>`, `<b>`, `<br>`, and `<a>` elements;
- use HTML entities such as `&mdash;` and `&ndash;` instead of raw special
  punctuation;
- make all GitHub and Jira references clickable;
- avoid headings, code tags, captured command output, and secrets.

The report is read-only with one expected local output artifact:
`/tmp/activity-summary.html`. Do not create GitHub issues, comments, releases,
commits, tags, or other external artifacts as part of summarization.
