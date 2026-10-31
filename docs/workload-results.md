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

## FastEmbed-to-ONNX BGE comparison

The FastEmbed 0.9.0 BGE fp32 path and the text-only ONNX adapter were compared
on the same pinned `BAAI/bge-small-en-v1.5` snapshot
(`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`). Both used Python 3.12.12,
ONNX Runtime 1.29.0, one embedding thread, hybrid search, an unbounded semantic
wait, and the same 1,024-line benchmark fixture. The machine was Linux x86_64
with 16 logical CPUs. Each path ran once with five fresh captures and once with
three; run order was FastEmbed, ONNX, ONNX, FastEmbed. The reported times below
are the per-run medians, not pooled samples.

| Measure | FastEmbed runs | ONNX runs | Paired change |
| --- | ---: | ---: | ---: |
| Semantic indexing, 128 chunks | 15.30s, 15.71s | 14.38s, 15.35s | 6.0% and 2.3% faster |
| First hybrid search, including indexing | 15.317s, 15.725s | 14.393s, 15.361s | 6.0% and 2.3% faster |
| Subsequent search | 11.50ms, 11.12ms | 13.44ms, 11.68ms | 1.95ms and 0.57ms slower |
| Needle hit at rank 1 / MRR | 1.00 / 1.00 in both runs | 1.00 / 1.00 in both runs | unchanged |

A separate parity run embedded 135 benchmark chunks, queries, and edge-case
strings with both implementations. The vectors were bit-for-bit equal
(maximum absolute difference 0, minimum cosine similarity 1.0), and all
top-10 rankings matched. Together, these runs show no material indexing or
retrieval-quality regression on this host. The small subsequent-search timing
differences are in the 0.6–2.0ms range and varied between repeats.

A separate warm-cache startup check used the normal loader paths and a shared
Hugging Face cache root whose `main` ref pointed to the same pinned snapshot.
It measured model loading plus the benchmark's tiny warm-up capture in two new
processes per backend: FastEmbed took 0.730s and 0.703s; ONNX took 0.843s and
0.566s. Their medians were 0.717s and 0.705s, respectively. This small sample
shows no consistent startup difference. The larger indexing comparison used a
direct snapshot path for FastEmbed to guarantee identical model files, so its
model-load timings are excluded from that table. These measurements are
host-specific and do not cover every CPU architecture.

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

## EB effectiveness comparisons (2026-10-30)

The fixture-version-3 results below (and earlier v2 runs) are historical.
Fixture version 4 binds source lines to their expected files. Historical results
have not been rescored and must not be pooled with v4 measurements.

The agent-level comparison uses four synthetic tasks, `gpt-6-luna`, the
`synthetic-eb-heavy-v2` fixture, a fresh fixture copy per invocation, and five
repetitions per task and mode. Fixture version 3 scored the targeted inspection
answer's line number as a typed criterion, accepting formats such as `line 32`,
`line number: 32`, and `L32`.

`control` runs without the EB MCP server; the `mcp` arm requires the agent to
use the configured EB tools for the capture and search workflow. Both arms use
the same Codex automatic-review policy and `workspace-write` sandbox, and the
deterministic test embedding backend. Records contain task-scoped pass/fail
metadata, not prompts, criterion text, tool responses, or transcripts. The
fixture is synthetic and small; these results do not establish performance on
other models or representative agent work.

### Corrected fixture-v4 batch (seed 20261046)

This run used five repetitions for each of four tasks and both modes, for 20
paired runs (40 invocations). It used the same model, fixture profile,
deterministic test embeddings, and matched sandbox/approval setup as the v3
comparison, with the v4 file-bound source criteria.

| Task | Control | EB available | Difference |
| --- | ---: | ---: | ---: |
| Build log search | 5/5 (100%) | 5/5 (100%) | 0 pp |
| Follow-up context | 5/5 (100%) | 5/5 (100%) | 0 pp |
| Noisy test failure | 4/5 (80%) | 5/5 (100%) | +20 pp |
| Targeted inspection | 4/5 (80%) | 2/5 (40%) | -40 pp |
| **Overall** | **18/20 (90%)** | **17/20 (85%)** | **-5 pp** |

The paired 95% t interval for the overall difference is approximately -22.3 to
+12.3 pp. There were 1 pair where EB succeeded and control failed, 17 ties, and
2 pairs where control succeeded and EB failed. This single batch does not show
a reliable overall difference and does not reproduce the v3 estimate; its
sample is too small to establish that EB reduces effectiveness.

| Task criteria | Control | EB available |
| --- | ---: | ---: |
| Build: marker, source file, file-bound line | 5/5 each | 5/5 each |
| Follow-up: test name, marker | 5/5 each | 5/5 each |
| Noisy test: failed test, marker, assertion | 5/5, 5/5, 4/5 | 5/5 each |
| Targeted inspection: marker, output line 32 | 5/5, 4/5 | 5/5, 2/5 |

Search responses exposed every criterion in all five MCP runs for every task.
`get_capture_slice` returned successful evidence in 2/5 noisy-test runs and 3/5
targeted-inspection runs, and none of the build or follow-up runs. The targeted
line criterion passed in only 2/5 MCP answers despite appearing in search
responses in all five runs; records retain no answers, so the reason for those
misses cannot be diagnosed from this aggregate data.

| Measure | Control | EB available | Paired mean difference (approx. 95% CI) |
| --- | ---: | ---: | ---: |
| Wall time | 16.44 s | 25.24 s | +8.80 s [+1.58, +16.02] |
| Total tool calls | 2.75 | 3.50 | +0.75 [-0.35, +1.85] |
| MCP tool calls | 0 | 3.50 | +3.50 |
| Context-bytes proxy | 3.48 KB | 14.68 KB | +11.20 KB [+5.70, +16.71] |
| Provider-reported input tokens | 94.4K | 171.3K | +76.9K (per-run means) |
| Provider-reported output tokens | 1,041 | 880 | -161 (per-run means) |
| Peak RSS | 424.0 MB | 432.2 MB | +8.2 MB [+0.6, +15.8] |

Both arms had 100% invocation completion and signal retrieval. EB averaged
30.9 KB of captured command input per run. Token totals are provider-reported
usage snapshots, not verified sums.

### Matched-sandbox fixture-v3 results

The response guidance was adjusted in stages after earlier batches showed that
log evidence was retrieved but not always carried into the final answer. Each
seed has 20 paired runs per mode.

| Seed and EB response behavior | Control | EB available | Difference |
| --- | ---: | ---: | ---: |
| 20261039, before log previews | 17/20 (85%) | 15/20 (75%) | -10 pp |
| 20261040–41, previews only | 36/40 (90%) | 35/40 (87.5%) | -2.5 pp |
| 20261042, previews plus test/assertion quoting guidance | 19/20 (95%) | 18/20 (90%) | -5 pp |
| **20261043–45, previews plus guidance to preserve requested evidence verbatim** | **48/60 (80%)** | **56/60 (93.3%)** | **+13.3 pp** |

For the latest guidance, the paired 95% t interval is approximately +2.2 to
+24.5 pp. There were 10 pairs where EB succeeded and control failed, 48 ties,
and 2 pairs where control succeeded and EB failed; the exact two-sided paired
sign test gives `p ≈ 0.039`. This supports an objective success gain on this
fixture and model. It does not establish generalization beyond this synthetic
task set.

| Task | Control, seeds 43–45 | EB available, seeds 43–45 | Difference |
| --- | ---: | ---: | ---: |
| Build log search | 15/15 (100%) | 15/15 (100%) | 0 pp |
| Follow-up context | 15/15 (100%) | 15/15 (100%) | 0 pp |
| Noisy test failure | 10/15 (66.7%) | 14/15 (93.3%) | +26.7 pp |
| Targeted inspection | 8/15 (53.3%) | 12/15 (80%) | +26.7 pp |
| **Overall** | **48/60 (80%)** | **56/60 (93.3%)** | **+13.3 pp** |

Criterion-level outcomes for the same runs were:

| Task criteria | Control | EB available |
| --- | ---: | ---: |
| Build: marker, source file, line | 15/15 each | 15/15 each |
| Follow-up: test name, marker | 15/15 each | 15/15 each |
| Noisy test: failed test, marker, assertion | 14/15, 15/15, 11/15 | 15/15, 15/15, 14/15 |
| Targeted inspection: marker, correct line | 14/15, 9/15 | 15/15, 12/15 |

The runner recorded 100% invocation completion and signal retrieval in both
arms. Secondary per-run means for seeds 43–45 were:

| Measure | Control | EB available | Paired mean difference (approx. 95% CI) |
| --- | ---: | ---: | ---: |
| Wall time | 16.25 s | 18.81 s | +2.55 s [+0.64, +4.47] |
| Total tool calls | 2.93 | 3.27 | +0.33 [-0.09, +0.76] |
| MCP tool calls | 0 | 2.88 | +2.88 |
| Context-bytes proxy | 4.43 KB | 12.87 KB | +8.44 KB [+5.90, +10.98] |
| Provider-reported input tokens | 96.8K | 146.2K | +49.4K [+37.3K, +61.5K] |
| Provider-reported output tokens | 1,171 | 800 | -371 [-506, -236] |
| Peak RSS | 440.8 MB | 455.7 MB | +14.9 MB [+9.5, +20.3] |

The context-bytes measure is an observable prompt/event proxy, not the model's
internal context size. Codex emitted one usage sample per run, so token totals
are provider-reported snapshots rather than verified sums. EB averaged 31.8KB
of captured command input per run. The fixture-v3 matched-sandbox batch improved
objective task success on all four tasks, with higher context, input-token,
wall-time, and memory costs. The corrected fixture-v4 result is reported
separately above.

### Earlier v3 runs with unmatched sandbox policy

Seeds `20261037` and `20261038` used the same v3 rubric but gave the control
arm a read-only sandbox and the MCP arm automatic review plus `workspace-write`.
They are retained as historical data and excluded from the matched estimate.

| Seed | Control | EB available | Difference |
| ---: | ---: | ---: | ---: |
| 20261037 | 16/20 (80%) | 15/20 (75%) | -5 pp |
| 20261038 | 18/20 (90%) | 15/20 (75%) | -15 pp |

### Earlier runs with literal line-phrase scoring

Seeds `20261033` through `20261036` used fixture version 2 and required the
literal phrase `line 32`. Keep these historical scores separate from the v3
matched-sandbox runs above; raw answers were not retained, so they cannot be
rescored under the typed line-number rubric.

| Seed | Control | EB available | Difference |
| ---: | ---: | ---: | ---: |
| 20261033 | 17/20 (85%) | 20/20 (100%) | +15 pp |
| 20261034 | 16/20 (80%) | 15/20 (75%) | -5 pp |
| 20261035 | 20/20 (100%) | 16/20 (80%) | -20 pp |
| 20261036 | 19/20 (95%) | 16/20 (80%) | -15 pp |
| **Pooled** | **72/80 (90%)** | **67/80 (83.8%)** | **-6.3 pp** |

The pooled paired difference for those v2 batches was -6.3 pp, with an
approximate 95% interval from -15.8 to +3.3 pp. Their task totals were:

| Task, fixture v2 rubric | Control | EB available |
| --- | ---: | ---: |
| Build log search | 20/20 | 20/20 |
| Follow-up context | 20/20 | 20/20 |
| Noisy test failure | 14/20 | 15/20 |
| Targeted inspection | 18/20 | 12/20 |

The matched v3 estimate is 13.3 pp above control, with a paired 95% interval
from +2.2 to +24.5 pp. The separate v4 batch above has a -5 pp estimate with a
wide interval that includes zero. Neither synthetic result generalizes to
representative agent work; collect more v4 repetitions and run privacy-reviewed
representative tasks before treating either result as a performance guarantee.

An earlier seed-`20261031` attempt is excluded because it omitted
`--allow-mcp-approvals` and used a different approval/sandbox policy.

Other effectiveness checks provide narrower evidence and are not substitutes
for the agent-level no-EB comparison:

| Check | Result |
| --- | --- |
| Direct output workflow, five repetitions per synthetic scenario | All four scenarios succeeded 5/5 with and without EB. Mean bytes examined changed from 11,047 to 3,664 for large build output (-66.8%); 6,909 to 3,655 for failure logs (-47.1%); 4,525 to 5,275 for review diffs (+16.6%); and 7,734 to 3,652 for timeout logs (-52.8%). Mean reduction across scenarios was 37.5%. |
| Search relevance, four queries | BM25 Hit@1, Hit@K, and MRR were each 0.75; hybrid and semantic were each 1.00. All matched their fixture baseline within the 5-point tolerance. |
| Summary-first capture, five tasks | Mean byte reduction 67.7%; prompt-byte reduction 58.2%; prompt-token-proxy reduction 58.1%; overall token-proxy reduction 67.6%. |
| Consolidated vs sequential retrieval, five repetitions | Both succeeded 5/5. Consolidation reduced overview bytes 96.1% and retrieval bytes 6.6%, while mean local time rose from 20.34ms to 53.92ms (2.65x). |

The first row's local MCP call cost is measured against microsecond-scale
in-process baseline operations, so its relative overhead is large even though
the absolute MCP timings were about 2.8-5.2ms. These synthetic checks measure
retrieval and output handling; only the agent comparison measures objective
task success with EB available versus disabled. Detailed run records and
captures remain local to the benchmark host; this section records aggregate
results only.
