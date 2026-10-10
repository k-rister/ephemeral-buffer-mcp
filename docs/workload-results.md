# Workload result documents

This reference describes the machine-readable benchmark result format, comparison rules, and experiment metadata. Operational guidance for consuming these results is in [OPERATIONS.md](../OPERATIONS.md).

## Machine-readable workload results

Every benchmark, evaluation, and the Codex A/B runner can emit one common,
versioned JSON document in addition to its own report and `--output` record.
Pass `--result PATH` to write it to a file, or `--result -` to print it on
stdout; the human-readable report then moves to stderr so stdout stays valid
JSON:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_latency \
  --samples 3 --result benchmark-latency.result.json
.venv/bin/python -m benchmarks.benchmark_semantic_index --samples 3 --result - > semantic-index.result.json
.venv/bin/python -m benchmarks.benchmark_semantic_memory --threads 1 --result semantic-memory.result.json
.venv/bin/python -m benchmarks.workload_results benchmark-latency.result.json semantic-index.result.json semantic-memory.result.json
```

The document is a `coding-agent-workload-result` (format version 1). It is
tool- and task-agnostic: it records *what* was measured, never how, so a
consumer does not need to know about embeddings, BM25, or any other producer
mechanism. The reference validator is `benchmarks/workload_results.py` (also
usable as a CLI, as above) and the same contract is published as JSON Schema
in `benchmarks/schemas/workload_result.schema.json`.

The fixed result structure is closed. Adding a field to the result envelope,
workload descriptor, run, measurement, phase, or experiment record requires a
`format_version` change because strict readers reject unknown fields. Changes to
the type or meaning of declared fields also require a version change. Readers
reject versions they do not support. Additive entries are compatible within the
existing version only in the format's open data maps: `workload.parameters`,
`environment` (including `environment.tool`), workload and run measurement
maps, run `labels`, `experiment.metadata`, and producer-owned `details`.
Consumers should treat unrecognized entries in those maps as opaque data.

```json
{
  "format": "coding-agent-workload-result",
  "format_version": 1,
  "workload": {"name": "capture-latency", "kind": "benchmark",
               "producer": "benchmark_latency.py",
               "parameters": {"line_counts": [16, 256, 2048], "samples": 3}},
  "environment": {"python_version": "3.12.14", "platform": "...",
                  "cpu_count": 8, "recorded_at": "2026-09-18T15:00:00+00:00",
                  "tool": {"name": "ephemeral-buffer-mcp", "version": "0.4.0"},
                  "source_revision": "..."},
  "status": "success",
  "errors": [],
  "measurements": {},
  "runs": [
    {"id": "lines-256",
     "labels": {"cache_state": "warm", "line_count": 256},
     "status": "success",
     "measurements": {
       "output_bytes": {"unit": "bytes", "value": 3840},
       "wall_time_seconds": {"unit": "seconds", "median": 0.012, "p95": 0.015, "samples": 3}},
     "phases": [
       {"name": "command", "unit": "seconds", "median": 0.004, "p95": 0.005, "samples": 3},
       {"name": "ingest", "unit": "seconds", "median": 0.006, "p95": 0.008, "samples": 3}],
     "errors": []}
  ],
  "experiment": {"group": "chunk-sweep",
                 "metadata": {"variant": "chunk-8", "model": "bge-small-fp32"}},
  "details": {"...": "the producer's own record, for humans"}
}
```

- `workload.name` identifies the measurement and `workload.parameters` holds
  everything needed to repeat it (sizes, seeds, modes, option overrides);
  `environment` explains why two results may legitimately differ.
- Each **run** is one comparable unit of work with a stable `id`, descriptive
  `labels` (`cache_state`, `mode`, `task_id`, `repetition`, `line_count`,
  `profile`, ...), a `status` of `success`, `failure`, `timeout`, `error`, or
  `partial`, and its own `errors`. Failed, timed-out, and partial runs are kept
  with their status rather than dropped.
- A **measurement** has a `unit` (`seconds`, `bytes`, `count`, `tokens`,
  `ratio`, `per_second`, or `score`) and one or more statistics (`value`,
  `sum`, `mean`, `median`, `min`, `max`, `p95`, `stdev`), optionally with the
  number of `samples` and a `note` explaining a proxy. `null` means the
  statistic is unavailable, never zero. `phases` is an ordered timeline of
  `seconds` measurements.
- Canonical names such as `wall_time_seconds`, `queue_wait_seconds`,
  `tool_calls`, `output_bytes`, `context_bytes`, `estimated_tokens`,
  `retained_summary_tokens`, `input_tokens`, `output_tokens`,
  `peak_rss_bytes`, `rss_delta_bytes`, `success_rate`, and
  `throughput_per_second` pin their unit so results from different producers
  line up; producers add their own names beside them. The JSON Schema pins
  those units too.
- Run `id`s are unique within a document. JSON Schema cannot express that
  rule (its `uniqueItems` only rejects fully identical runs), so a consumer
  that validates with the schema alone must check ids itself or run
  `workload_results.py` on the document first.
- `details` carries the producer's native record for people who need it; its
  shape is producer-specific and versioned separately by
  `workload.producer_schema_version`.
- The optional **experiment** block assigns the document to a named `group`
  of related runs and carries flat `metadata` (scalar values only) describing
  what varied; see "Organizing experiments" below. Every producer accepts
  `--experiment GROUP`, `--metadata KEY=VALUE` (repeatable; values parse as
  JSON when possible), and `--redact KEY`.

[OPERATIONS.md](../OPERATIONS.md) describes how comparison tooling and
regression checks should consume these documents. The shared format is not a
privacy exemption: apply
the same review to a result file as to any other benchmark output before
sharing it.

## Comparing workload results

`compare_workload_results.py` compares two or more result documents without
rerunning anything. The first reference is the baseline and every later one is
compared against it. For each run and measurement the documents share it
prints the absolute delta, the percentage change, and an outcome; runs pair by
`id`, measurements and phases by name, and every statistic is compared only
with the same statistic (`median` against `median`, never against `mean`).
The example below records a baseline, halves the semantic chunk size, and
compares the two:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_latency \
  --samples 3 --line-counts 256 2048 --result before.result.json
EPHEMERAL_TEST_EMBEDDINGS=1 EPHEMERAL_SEMANTIC_CHUNK_LINES=2 .venv/bin/python -m benchmarks.benchmark_latency \
  --samples 3 --line-counts 256 2048 --result after.result.json
.venv/bin/python -m benchmarks.compare_workload_results before.result.json after.result.json \
  --tolerance 5 --statistic median --metric wall_time_seconds --metric ingest --metric semantic_index
```

```text
workload: capture-latency
  baseline: before.result.json  producer=benchmark_latency.py  kind=benchmark  status=success  runs=3  recorded=2026-09-18T18:16:11+00:00  python=3.12.14  tool=ephemeral-buffer-mcp 0.4.0  revision=2b58f2490571
  candidate: after.result.json  producer=benchmark_latency.py  kind=benchmark  status=success  runs=3  recorded=2026-09-18T18:16:28+00:00  python=3.12.14  tool=ephemeral-buffer-mcp 0.4.0  revision=2b58f2490571
  statistics=median tolerance=5% metrics=ingest,semantic_index,wall_time_seconds

before.result.json -> after.result.json
  run         metric                stat    baseline     candidate    delta         change   outcome
  lines-256   wall_time_seconds     median  0.009854 s   0.01054 s    +0.0006824 s  +6.9%    regressed
  lines-256   phase:ingest          median  0.0007052 s  0.0007507 s  +4.546e-05 s  +6.4%    regressed
  lines-256   phase:semantic_index  median  0.0008512 s  0.002113 s   +0.001262 s   +148.3%  regressed
  lines-2048  wall_time_seconds     median  0.02084 s    0.02982 s    +0.008981 s   +43.1%   regressed
  lines-2048  phase:ingest          median  0.002724 s   0.002792 s   +6.775e-05 s  +2.5%    unchanged
  lines-2048  phase:semantic_index  median  0.006279 s   0.01706 s    +0.01078 s    +171.6%  regressed
  summary: 0 improved, 5 regressed, 0 changed, 1 unchanged, 0 missing, 0 incompatible; runs compared=3 missing=0; status success -> success
```

The phase rows attribute the wall-time regression to semantic indexing rather
than ingestion. Outcomes are:

- `improved` or `regressed`: the value moved beyond `--tolerance PERCENT`
  (default 0) in the metric's better or worse direction. Canonical
  measurements know their direction (time, bytes, tokens, tool calls, and
  memory are better lower; `success_rate` and throughput are better higher);
  other `seconds`, `bytes`, `tokens`, and `per_second` metrics follow their
  unit, and `--direction NAME=lower|higher` declares the rest.
- `changed`: the value moved but the metric has no known direction, such as a
  producer-specific `count`, `ratio`, or `score`. A change from a zero baseline
  has no percentage and counts as beyond any tolerance.
- `unchanged`: within the tolerance.
- `missing`: the run, measurement, or statistic is absent, `null`, or has
  `samples: 0` on at least one side. These are listed with the side that lacks
  them, never silently skipped.
- `incompatible`: both sides report the metric with different units.

Documents must describe the same `workload.name` unless
`--allow-workload-mismatch` is given, and each comparison lists the
`workload.parameters` and `environment` fields that differ (CPU count, tool
version, source revision) so a different setup is not mistaken for a
regression. Non-success runs appear with their status and errors.

`PATH#RUN_ID` selects one run from a document. When the baseline and a
candidate each select a single run, those two runs pair even though their ids
differ, which compares two configurations recorded in the same document:

For semantic-prefetch timing, run `benchmark_prefetch.py` once. It measures
prefetch off and on with the same synthetic 256-line workload and five fresh
engines per mode, then reports median ingest, first-search, and subsequent
search times. First-search time includes any wait for background indexing.

```bash
# Agent configurations: control versus MCP in one A/B summary, or two
# agent-run documents recorded under different models or policies.
.venv/bin/python -m benchmarks.benchmark_agent_ab --records runs.json --result summary.result.json
.venv/bin/python -m benchmarks.compare_workload_results summary.result.json#control summary.result.json#mcp --statistic mean
.venv/bin/python -m benchmarks.compare_workload_results codex-gpt5.result.json codex-candidate.result.json \
  --select mode=mcp --metric input_tokens --metric output_tokens --metric wall_time_seconds --metric tool_calls

# Semantic-prefetch policies: compare phase medians for the same synthetic workload.
.venv/bin/python -m benchmarks.benchmark_prefetch --line-count 256 --samples 5 --result semantic-prefetch.result.json
.venv/bin/python -m benchmarks.compare_workload_results semantic-prefetch.result.json#prefetch-off semantic-prefetch.result.json#prefetch-on \
  --metric ingest --metric first_search --metric subsequent_search --statistic median

# Summarization strategies: retained-summary size and prompt-token proxies per
# task; the reduction ratios need an explicit direction.
.venv/bin/python -m benchmarks.benchmark_effectiveness --summary --result summary-a.result.json
.venv/bin/python -m benchmarks.compare_workload_results summary-a.result.json summary-b.result.json \
  --metric retained_summary_tokens --metric estimated_tokens --metric summary_token_reduction \
  --direction summary_token_reduction=higher
```

`--select KEY=VALUE` keeps only runs whose label matches (values parse as JSON
when possible, so `line_count=256` compares a number), `--metric NAME` limits
the report to named measurements or phases, and `--statistic NAME` chooses the
statistics (default `value`, `median`, `mean`, and `p95`; `all` adds `sum`,
`min`, `max`, and `stdev`; `stdev` describes spread rather than level, so its
changes are reported as `changed` and never judged). `--format json` prints,
and `--output PATH` writes, a `coding-agent-workload-comparison` document
(format version 1) with the same entries plus each document's workload and
environment blocks. `--check` exits with status 2 when any metric regressed or
any document has a non-success status, which [OPERATIONS.md](../OPERATIONS.md)
uses for regression checks. A document narrowed with `PATH#RUN_ID` or `--select` is
judged by its selected runs, and the report shows the whole file's
`document_status` beside it when the two differ. In check mode, `--select` and
`--metric` must each select at least one run or measurement; empty selections
fail instead of passing without a comparison.

## Organizing experiments

A performance or token-efficiency study is rarely one comparison: a chunk-size
sweep, a model change, or a prompt-policy A/B produces several result
documents whose relationship is otherwise only in their file names. Every
producer therefore accepts `--experiment GROUP` to assign its result document
to an experiment or run group, and `--metadata KEY=VALUE` to record what
varied. The workflow is: tag each run when it is recorded, list the group to
see what exists and which runs failed, and compare documents by group and
metadata instead of by path.

**Record.** Give every run of one study the same group and describe the
variable under test in metadata. Conventional keys are `task_type`,
`repository_revision`, `agent_configuration`, `model`, `tool_version`,
`variant`, `environment`, `workload_size`, and `started_at`; any other
`snake_case` key is allowed. Values are identifiers (at most 256 characters),
never prompts or captured content, and `started_at` must be an ISO 8601
timestamp with a UTC offset (`2026-09-18T10:00:00+00:00` or a trailing `Z`)
so documents order by instant. A value that breaks these rules is rejected
when the arguments are parsed, before the workload runs:

```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_latency --samples 3 --line-counts 256 \
  --result results/chunk-8.result.json --experiment chunk-sweep \
  --metadata variant=chunk-8 --metadata model=bge-small-fp32
EPHEMERAL_TEST_EMBEDDINGS=1 EPHEMERAL_SEMANTIC_CHUNK_LINES=4 .venv/bin/python -m benchmarks.benchmark_latency --samples 3 --line-counts 256 \
  --result results/chunk-4.result.json --experiment chunk-sweep \
  --metadata variant=chunk-4 --metadata model=bge-small-fp32
EPHEMERAL_TEST_EMBEDDINGS=1 EPHEMERAL_SEMANTIC_CHUNK_LINES=16 .venv/bin/python -m benchmarks.benchmark_latency --samples 3 --line-counts 256 \
  --result results/chunk-16.result.json --experiment chunk-sweep \
  --metadata variant=chunk-16 --metadata model=bge-small-fp32
```

For coding-agent runs, `run_agent_ab_experiment.sh` tags both of its result
documents (`records.result.json` and `summary.result.json`) with the model,
fixture profile, repetition count, embedding environment, and start time, and
takes the group from `AGENT_AB_EXPERIMENT` and the variant from
`AGENT_AB_VARIANT`:

```bash
AGENT_AB_EXPERIMENT=policy-ab AGENT_AB_VARIANT=summarize-first \
  CODEX_HOME=/path/to/writable/authenticated-codex-home ./benchmarks/run_agent_ab_experiment.sh
```

**List.** `list_workload_results.py` searches files and directories
(recursively) for result documents, ignores other JSON files such as records
and schedules, and prints one row per document with its group, status, run
count, time, and metadata. `--group NAME` and `--where KEY=VALUE` filter by
group and metadata, `--workload NAME` and `--status STATUS` narrow further,
`--field KEY` shows chosen metadata keys as columns, and `--runs` lists every
run with its labels and status so failed, timed-out, and partial runs inside a
document are visible:

```bash
.venv/bin/python -m benchmarks.list_workload_results results --field variant --field model
```

```text
path                          group        workload           status   runs  time                       variant   model           errors
results/chunk-8.result.json   chunk-sweep  capture-latency    success  2     2026-09-18T18:52:02+00:00  chunk-8   bge-small-fp32
results/chunk-16.result.json  chunk-sweep  capture-latency    success  2     2026-09-18T18:52:03+00:00  chunk-16  bge-small-fp32
results/chunk-4.result.json   chunk-sweep  capture-latency    success  2     2026-09-18T18:52:03+00:00  chunk-4   bge-small-fp32
results/prefetch.result.json  -            semantic-prefetch  success  2     2026-09-18T18:52:04+00:00  -         bge-small-fp32
```

Rows are ordered by group, then by `started_at` metadata (or the recording
time when it is absent, both normalised to UTC), then by path. Documents that do not satisfy the
format are listed as `invalid` with the reason and make the command exit with
status 1, so a broken file is never mistaken for an absent one. `--format
json` prints a `coding-agent-workload-listing` document whose entries carry
each document's group, metadata, status, errors, and per-run status, and
`--format paths` prints only the selected paths for shell substitution.

**Compare.** `compare_workload_results.py` accepts `DIR@GROUP` references
beside file references: the reference expands to every document under the
directory that belongs to the group, in the same order as the listing, and
`DIR@GROUP,KEY=VALUE` keeps only documents whose metadata matches. A lone
`results@chunk-sweep` compares every later document of the sweep against the
earliest; naming two selectors picks the baseline explicitly, and `#RUN_ID`
still selects one run from each document:

```bash
.venv/bin/python -m benchmarks.compare_workload_results results@chunk-sweep,variant=chunk-8 results@chunk-sweep,variant=chunk-4 \
  --statistic median --metric wall_time_seconds --metric semantic_index
```

```text
workload: capture-latency
  baseline: results/chunk-8.result.json  group=chunk-sweep  producer=benchmark_latency.py  kind=benchmark  status=success  runs=2  recorded=2026-09-18T18:52:02+00:00  python=3.12.14  tool=ephemeral-buffer-mcp 0.4.0  revision=0a9e8da36b46
  candidate: results/chunk-4.result.json  group=chunk-sweep  producer=benchmark_latency.py  kind=benchmark  status=success  runs=2  recorded=2026-09-18T18:52:03+00:00  python=3.12.14  tool=ephemeral-buffer-mcp 0.4.0  revision=0a9e8da36b46
  statistics=median tolerance=0% metrics=semantic_index,wall_time_seconds

results/chunk-8.result.json -> results/chunk-4.result.json
  experiment differences: variant: "chunk-8" -> "chunk-4"
  run        metric                stat    baseline    candidate   delta         change  outcome
  lines-256  wall_time_seconds     median  0.0159 s    0.01414 s   -0.00176 s    -11.1%  improved
  lines-256  phase:semantic_index  median  0.001104 s  0.001814 s  +0.0007096 s  +64.3%  regressed
  summary: 1 improved, 1 regressed, 0 changed, 0 unchanged, 0 missing, 0 incompatible; runs compared=2 missing=0; status success -> success
```

The report and the JSON comparison show the group beside each document and
list the metadata that differs (`experiment differences`) next to the
parameter and environment differences, so a delta can be read together with
the variable that caused it. Selector values may not contain commas; a value
containing `@` is fine because only the first `@` separates the directory from
the group. A group reference fails on an invalid document only when that
document claims the requested group; stale or broken files in other groups do
not block the comparison (the listing still reports them). Several group
references into one directory scan it once.

**Sensitive metadata.** Metadata keys that name credentials (`token`, `key`,
`password`, `secret`, `credentials`, `authorization`, `bearer`, or any
`*_token` or `*_key`) are stored as `[redacted]` by every producer, the
validator rejects a document that carries a real value under such a key, and
`--redact KEY` stores `[redacted]` for any other key whose value should not
leave the machine (the key stays visible so readers know it was set). The
listing tool's `--redact KEY` masks a value in its output without changing the
file. Metadata that should not be recorded at all is simply not passed; the
result format is still not a privacy exemption, so review documents before
sharing them as with any other benchmark output.
