# Benchmarks and measured comparisons

This guide is for contributors evaluating performance and effectiveness. It
documents benchmark commands and selected results. The machine-readable result
format is described in [Workload result documents](workload-results.md).

## Context-reduction comparison (2026-10-09)

The exhaustive collection compared the revision immediately before the context-reduction work with the revision after it:

- Baseline: `6c716cc931439bbe16c6fdd8af6b2e661b997e9d`
- Candidate: `be18d05b52d4ed3924f899924552f6961b335558`, including `bf1cc26` and `be18d05`
- Model: `gpt-6-luna`; five repetitions per task; seed `20261006`
- All 48 collection steps completed successfully.

The comparison evaluates the context-reduction changes. It does not estimate the effect of changing MCP SDK versions.

| Mean measurement in the MCP condition | Before | After | Change |
|---|---:|---:|---:|
| MCP tool response bytes | 6,681.45 | 1,500.55 | −77.5% |
| Codex context proxy bytes | 23,909.85 | 17,642.60 | −26.2% |
| Context overhead over the control condition | 16,650.65 | 10,404.30 | −37.5% |
| Input tokens | 130,008.70 | 95,917.05 | −26.2% |

Agent task success was 16/20 (80%) before and 17/20 (85%) after; signal retrieval was 100% in both runs. The agent comparison also flagged regressions in some duration, output-token, and peak-memory measures. With five repetitions per task and wide task-level intervals, these results are indicative for this workload rather than a general performance guarantee.

`tool_response_bytes` counts bytes returned by MCP tools. `context_bytes` is a Codex CLI prompt/event-envelope proxy; it does not expose the model's internal context size. The checked-in summary records the comparison; detailed run logs remain local to the benchmark environment.

## Benchmark command reference

Run the concurrency benchmark:
```bash
.venv/bin/python -m benchmarks.benchmark_concurrency --captures 32 --workers 8
```
The benchmark accepts `--min-ingest-per-second` and `--min-reads-per-second`
thresholds for direct checks. For repeatable regression checks, pass
`--baseline benchmarks/data/benchmark_baseline.json --output benchmark-concurrency.json`.
The checked-in baseline uses a 20% tolerance: a run fails only when ingest or
read throughput drops below 80% of its baseline. Each scheduled or manually
dispatched GitHub Actions run records the raw JSON result as an artifact and
adds the measurements and regression status to the workflow summary. This
benchmark remains optional and is not part of the required pull-request checks;
update the baseline deliberately when the runner or benchmark workload changes.

Interpret summary-read results in light of the workload. The built-in benchmark
uses 20 identical plain-text lines per capture and `benchmark-N` labels, so it
does not exercise the log-signal scan in `detect_signals`. Treat its read rate
as a microbenchmark; use representative logs and mixed read/ingest load when
assessing user impact.

For [issue #370](https://github.com/k-rister/ephemeral-buffer-mcp/issues/370),
local reruns compared `v0.6.3` with `f70c14c` on Python 3.12.12. With 32 plain-text
captures and 8 workers, the median paired read rate fell 25.7%, while the median
read batch grew by about 1.4 ms. With 2,000-line log captures, the median read
rate fell 3.5% and the 32-summary batch grew by 28 ms. In a mixed run with 8
concurrent log summaries and an ingest, median ingest latency fell from 150.4 ms
to 26.0 ms, while summary-batch latency stayed roughly flat (209.6 ms to
205.8 ms). We accepted this isolated read-throughput cost to keep summary
construction outside the engine-wide lock, which lets ingestion proceed during
large log scans. These local measurements are workload- and machine-specific.

A 20-pair follow-up on the same Linux host and Python 3.12.12 compared
`6d2dc21` with `80ecfc6`, alternating revision order between runs. Median
paired summary-read throughput change was +41.7% (95% paired bootstrap
interval: +36.5% to +50.9%); ingest throughput's paired median change was +2.5%
(interval: -2.6% to +4.2%).
Across the four deterministic MCP capture/search/slice scenarios, paired median
latency decreased 5.7% for `large-build` (95% interval: 3.2% to 10.8%), 10.0%
for `failure-log` (6.4% to 12.4%), 19.0% for `review-diff` (15.1% to 22.6%),
and 8.2% for `timeout-log` (2.8% to 12.0%). The intervals are percentile
bootstrap intervals over paired run changes, using 20,000 resamples. The MCP
timings cover capture, search, and slice retrieval together; they do not isolate
snippet-formatting time. These results are specific to this host and workload.

Measure MCP and Unix-socket admission under controlled saturation:

```bash
.venv/bin/python -m benchmarks.benchmark_admission \
  --repetitions 3 \
  --output /tmp/benchmark-admission.json \
  --result /tmp/benchmark-admission.result.json
```

The benchmark uses the configured active and queued limits, adds four overflow
requests by default, and reports successful throughput, request latency, and
`server_busy` rejection rates for each transport. MCPServer tool dispatch runs a
short local sleep command. Socket clients hold idle connections until the
active slots and queue are full, then send complete requests; this lets the
benchmark read overflow responses without unread request bytes. The MCP timings
cover the server tool adapter without external client transport framing.
Semantic prefetch is disabled so model work does not mask admission behavior.
It fails if either transport does not reach its configured capacity, reject
the overflow requests, or return to idle. The scheduled and manually dispatched
benchmark workflow uploads both the detailed report and a common workload-result
document.

Measure command-capture latency by output size and pipeline phase:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_latency \
  --samples 5 --output benchmark-latency.json
```
The latency harness reports cold-start time plus warm median and p95 timings
for command execution, BM25 capture/indexing, deferred semantic indexing,
summary generation, and the combined pipeline. Semantic embeddings are now
materialized when semantic or hybrid search first needs them, so the warmup
explicitly materializes the warm engine's embeddings before timed samples begin.
Use the separate semantic-index phase when evaluating end-to-end costs; cold
start includes the first engine's model and embedding setup, while warm samples
reuse the configured model cache and engine. The benchmark is diagnostic and
optional, not a required pull-request check.

Compare lazy semantic indexing with the default asynchronous prefetch:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_prefetch \
  --line-count 256 --samples 5 --output benchmark-prefetch.json
```
The prefetch harness reports ingestion, first semantic search, and subsequent
semantic search medians for both modes. Prefetch can reduce first-query latency
when work completes during ingestion, while adding bounded background resource
use. In the release-preparation 256-line Linux run, first-search median changed
from 0.435 seconds without prefetch to 0.425 seconds with it, while ingestion
changed from 1.19 ms to 1.63 ms; treat this as deployment-specific diagnostic
evidence rather than a guaranteed improvement.

Measure semantic indexing cost and first hybrid-search latency by capture size:
```bash
.venv/bin/python -m benchmarks.benchmark_semantic_index \
  --samples 3 --output benchmark-semantic-index.json
```
The semantic-index harness ingests a fresh deterministic log-like capture per
sample, then reports median and p95 timings for ingestion, lazy embedding
materialization, the first hybrid (or `--mode semantic`) search that triggers
it, and a subsequent search over the complete index, plus chunk count and
indexing throughput. The first search runs against the semantic wait budget
(`--semantic-wait-seconds`, default `EPHEMERAL_SEMANTIC_WAIT_SECONDS` or 10),
so the report shows how often it answered with pending semantic coverage
(`first_search_pending_rate`) and the needle rank it achieved
(`first_search_needle_mrr`) next to the rank over the complete index
(`needle_mrr`); pass `--semantic-wait-seconds inf` to time the full lazy
indexing cost inside the first search instead. Default sizes are 16, 256,
2,048, and 8,192 lines. Run it without `EPHEMERAL_TEST_EMBEDDINGS=1` to measure
the configured FastEmbed model; with deterministic test embeddings it only
validates the harness. Prefetch and startup warm-up are disabled inside the
harness so the lazy cost is visible. Results are host-specific diagnostic
evidence, not a required CI gate.

A five-sample release comparison on Linux with Python 3.12.12, the
`BAAI/bge-small-en-v1.5-fp32` model, one embedding thread, hybrid mode, an
unbounded semantic wait, and the CPU memory arena disabled measured:

| Release | Median semantic indexing | Throughput | Process peak RSS |
| --- | ---: | ---: | ---: |
| 0.6.1 | 154.03 s | 6.65 chunks/s | 4,320,944 KiB |
| 0.6.2 | 108.34 s | 9.45 chunks/s | 508,476 KiB |

Needle MRR and hit-at-1 were 1.00 for both releases, and subsequent search
remained about 12.85 ms. Peak RSS came from `/usr/bin/time -v` over the whole
five-sample benchmark process. The 8,192-line fixture was 566,736 bytes, below
the default 4 MiB semantic-input budget, so this comparison measures normal
indexing rather than the over-budget fallback. Treat these results as
single-host diagnostic evidence, not a general performance guarantee.

A separate three-run boundary probe with deterministic test embeddings checked
the fallback behavior:

| Fixture | 0.6.1 | 0.6.2 |
| --- | --- | --- |
| 56,000 lines; 3,914,474 semantic-input bytes | Complete coverage; 10,752,000 embedding bytes; target rank 1 in both modes | Same |
| 64,000 lines; 4,474,505 semantic-input bytes | Complete coverage; 12,288,000 embedding bytes; hybrid rank 1, semantic rank 2 | Budget fallback; zero embedding bytes; BM25 target rank 1 in both modes |

This is a deterministic behavior comparison, not a real-model latency or RSS
comparison. A separate real-model 0.6.2 memory run above the limit also reported
zero retained embeddings and passed both BM25 fallback searches.

Measure process RSS for model loading and bounded semantic indexing with the
memory harness:

```bash
.venv/bin/python -m benchmarks.benchmark_semantic_memory --threads 1
```

It reports the model-load and indexing sampled peaks, whether semantic indexing
completed or exceeded its work budget, retained embedding bytes, and RSS after
clearing the capture and collecting Python objects. When the budget is exceeded,
it also checks that semantic and hybrid searches report the fallback and retrieve
a fixed BM25 sentinel. The defaults reproduce a 256 KiB synthetic capture using
the configured model. Run it on the deployment host with the same model, thread,
arena, and batch settings; it is a diagnostic measurement, not a CI gate.
It also accepts `--result PATH` or `--result -` to emit the shared versioned
workload format. The export includes separate model-load and semantic-index
runs with timing and RSS measurements, and preserves the native record under
`details`. With `--result PATH`, the common document is written to that path and
the existing native JSON stays on stdout. With `--result -`, stdout contains
the workload JSON and a concise human-readable report moves to stderr.

For a release comparison, record the same workload under each model and compare
the selected run with the versioned result tool:

```bash
.venv/bin/python -m benchmarks.benchmark_semantic_index \
  --embedding-model BAAI/bge-small-en-v1.5-fp32 --line-counts 1024 --samples 3 \
  --result results/fp32.result.json --experiment embedding-model \
  --metadata variant=fp32 --metadata host_class=linux-x86_64
.venv/bin/python -m benchmarks.benchmark_semantic_index \
  --embedding-model BAAI/bge-small-en-v1.5 --line-counts 1024 --samples 3 \
  --result results/catalogue.result.json --experiment embedding-model \
  --metadata variant=catalogue --metadata host_class=linux-x86_64
.venv/bin/python -m benchmarks.compare_workload_results \
  results/fp32.result.json#lines-1024 results/catalogue.result.json#lines-1024 \
  --statistic median --metric semantic_index --metric throughput_per_second
```

Keep result documents together with the release benchmark artifacts. Compare
results only within the same host class and record model, thread, chunking,
prefetch, warm-up, and wait-budget settings; do not treat a different host as
a regression.

Compare lazy model loading with background startup warm-up:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_warmup \
  --samples 5 --output benchmark-warmup.json
```
The warm-up harness reports engine initialization, time to embedding readiness,
first semantic-search latency, and process RSS change for both policies. For the
lazy policy, embedding readiness is measured through completion of the first
semantic search rather than reported as instantaneous. Run it without
deterministic test embeddings to measure the configured FastEmbed model and host
cache; each policy is measured in a fresh worker process so model pages retained
by the allocator do not contaminate the other policy. In the release-preparation
Linux run, first-search median was 0.335 seconds without warm-up versus 0.0122
seconds with warm-up, with approximately 186 MB RSS increase in both modes.
Results are deployment-specific and the benchmark is optional.

Measure direct-versus-captured routing tradeoffs with synthetic output profiles:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_routing \
  --samples 5 --output benchmark-routing.json
```
The routing harness compares direct command completion with bounded capture and
summary generation for targeted (16 lines), test-like (256 lines), and
build/log-like (2048 lines) output. Use the medians and p95 values to keep the
heuristic honest: direct execution generally has lower latency for small,
bounded output, while capture adds searchable context and bounded response
size for noisy or uncertain output. These synthetic measurements are guidance,
not universal thresholds or a required CI gate. Both benchmark summaries use
the nearest-rank p95 convention: for `n` samples, p95 is the value at sorted
rank `ceil(0.95 * n)`, with ranks starting at one.

Measure command-output handling effectiveness with deterministic synthetic data:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_effectiveness \
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
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_relevance \
  --top-k 3 --baseline benchmarks/data/benchmark_relevance_baseline.json \
  --output search-relevance.json
```
The relevance benchmark covers exact errors, punctuation-heavy queries,
conceptual semantic queries, and lexical/semantic conflicts. It reports
hit@1, hit@k, and mean reciprocal rank (MRR) for BM25, semantic, and hybrid
search. Each query has four competing semantic windows, and fixtures define
expected evidence ranges scored against each result's matched lines, not its
surrounding context. This measures retrieval relevance only—not agent answer
quality, token usage, or universal performance. CI compares deterministic
scores with `benchmark_relevance_baseline.json` and fails only when a metric
drops by more than the recorded tolerance. Baseline changes must be deliberate,
reviewable, and accompanied by a fixture or intended-behavior explanation. The
evaluation uploads its machine-readable JSON result with the other benchmark
artifacts.

The baseline stores only schema and fixture versions, deterministic embedding
mode, aggregate per-mode scores, query counts, and explicit tolerances. It does
not store captures, raw command output, or user queries. Environment-specific
latency and effectiveness results remain CI artifacts and summaries rather than
exact cross-run gates; retention follows the workflow artifact policy.

## Synthetic effectiveness example (2026-09-11)

The reproducible paired evaluation was run on 2026-09-11 with five
repetitions, seed `20260907`, and deterministic test embeddings. Across four
synthetic scenarios, both the direct-output baseline and the MCP workflow
completed all 20 tasks, and every MCP search was useful. The MCP workflow
examined 30–85% fewer bytes than the baseline per scenario (68% on average):

| Scenario | Bytes examined reduction | Completion |
| :--- | ---: | ---: |
| Large build output | 85% | 5/5 |
| Failure log | 77% | 5/5 |
| Review diff | 30% | 5/5 |
| Timeout log | 79% | 5/5 |

The consolidation evaluation, using the same seed and five repetitions,
reduced the initial multi-result overview from an average of 4,298 bytes to
213 bytes (95% fewer overview bytes), while preserving a 100% targeted
retrieval success rate. The consolidated workflow retrieved 3.6% fewer bytes
overall and took 2.2 times as long locally as sequential processing in this
run. These measurements quantify the MCP data path: fewer bytes need to be
returned to the agent before it asks for targeted detail.

They are not universal performance guarantees. The fixtures are synthetic,
the harness does not invoke a model, and the byte reduction is not a direct
token-savings measurement. In this run, consolidated processing took about 2.2
times longer locally than sequential processing, while retrieving
similar detail. The benchmark therefore demonstrates context-size and
workflow-shaping benefits, not that every workload will be faster or that
search results will be relevant for arbitrary repositories. Re-run the
commands below with representative, privacy-reviewed tasks before making
project-specific claims.

Run a controlled local A/B evaluation with repeated paired measurements:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_effectiveness \
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
.venv/bin/python -m benchmarks.benchmark_agent_ab \
  --schedule-output agent-ab-schedule.json --repetitions 5 --seed 20260909
```
Run each scheduled task with MCP disabled (`control`) and enabled (`mcp`) using
the same model configuration, repository fixture, environment, and reset policy.
Have an external agent adapter write a records envelope containing only the
schedule, non-secret protocol identifiers, and per-run fields: completion,
task success, signal retrieval, task-scoped criterion pass booleans, duration,
tool calls, repeated commands, context proxy bytes (total plus prompt/output
components), provider usage samples, and peak RSS bytes sampled from that
invocation's process. On hosts without a supported
per-process RSS interface, peak RSS is recorded as zero and should be treated
as unavailable. Summarize it with:
```bash
.venv/bin/python -m benchmarks.benchmark_agent_ab \
  --records agent-ab-records.json --output agent-ab-summary.json
```
The analyzer validates balanced paired runs, reports per-mode and per-criterion
aggregates plus MCP-minus-control deltas with uncertainty, and emits
recommendations. It does not invoke a model, require credentials, or accept raw
prompts, transcripts, commands, captures, or user content. Do not commit agent
records or generated captures; share only the aggregate summary after privacy
review.

The repository includes a Codex CLI adapter for executing that protocol. It
uses the requested model explicitly, creates a fresh fixture copy for every
run, isolates control and MCP configuration, and writes metadata-only records.
Create a private task manifest (do not commit it) with fixture version 2, a
prompt, signal marker, and objective success criteria for each scheduled task:

```json
{
  "fixture_version": 2,
  "tasks": {
    "targeted-inspection": {
      "prompt": "Inspect the fixture and report the marker and line number.",
      "signal_marker": "TARGETED_SIGNAL",
      "success_criteria": {
        "description": "Report the signal and its correct output line.",
        "required_phrases": ["TARGETED_SIGNAL", "line 32"]
      }
    },
    "noisy-test-failure": {
      "prompt": "Run the fixture test and report the failed test, marker, and assertion.",
      "signal_marker": "TEST_FAILURE_SIGNAL",
      "success_criteria": {
        "description": "Identify the failed test, marker, and assertion.",
        "required_phrases": [
          "test_case_1379",
          "TEST_FAILURE_SIGNAL",
          "expected status=ready, got status=stalled"
        ]
      }
    },
    "build-log-search": {
      "prompt": "Inspect the build log and report the marker, source file, and line.",
      "signal_marker": "BUILD_FAILURE_SIGNAL",
      "success_criteria": {
        "description": "Report the marker and its source location.",
        "required_phrases": ["BUILD_FAILURE_SIGNAL", "src/parser.c:917"]
      }
    },
    "follow-up-context": {
      "prompt": "Find the earlier marker and report which test failed.",
      "signal_marker": "TEST_FAILURE_SIGNAL",
      "success_criteria": {
        "description": "Identify the earlier failed test and marker.",
        "required_phrases": ["test_case_1379", "TEST_FAILURE_SIGNAL"]
      }
    }
  }
}
```

Run the adapter from the repository checkout:

```bash
.venv/bin/python -m benchmarks.run_codex_agent_ab \
  --schedule agent-ab-schedule.json \
  --tasks /path/to/private-agent-tasks.json \
  --repository /path/to/privacy-reviewed-fixture \
  --model gpt-5.6-luna \
  --output /tmp/agent-ab-records.json
.venv/bin/python -m benchmarks.benchmark_agent_ab \
  --records /tmp/agent-ab-records.json \
  --output /tmp/agent-ab-summary.json
```

The runner requires a locally authenticated `codex` CLI. It uses
`codex exec --json --ephemeral`, uses a read-only sandbox by default. `context_bytes_proxy` is an observable prompt/event-envelope
proxy because the CLI does not expose the model's internal context size.
When the fixture does not contain the application package, select the source
checkout's server module and add its `src` directory, for example
`--mcp-module ephemeral_buffer_mcp.server --mcp-python-path /path/to/checkout/src`.
For MCP experiments where the client must be allowed to call the configured
server, add `--allow-mcp-approvals`. This uses Codex automatic review with a
`workspace-write` sandbox, and should only be used with a disposable,
privacy-reviewed fixture. Control runs continue to use the read-only sandbox
without MCP approval routing; the records protocol identifies the selected
policy. Add `--require-mcp-calls` when the MCP arm must exercise at least one
MCP tool; runs that bypass MCP are marked with reason `mcp_not_used`. A zero
exit records invocation completion, while objective task success is scored
separately from all required answer phrases. Explicit refusals do not pass
task scoring, even if they repeat a signal marker.
Review prompts, fixtures, and generated records for privacy before sharing;
the runner does not persist transcripts in its records output.

Runner records use version 8. Version 7 added `criterion_passes`, ordered
booleans for the task's required answer phrases. Summary indexes are one-based
and scoped to the task ID and fixture version. Phrase text and final answers
are not stored. Empty or refused answers fail every phrase; objective `task_success`
also requires an eligible invocation and all criteria to pass. The `completed`
invocation flag remains separate. Records include exit code, failure reason,
MCP-specific tool-call count, provider-reported input/output token counts, and
every provider usage sample when Codex emits them. They break the observable
context proxy into prompt and output byte components. Version-1 through
version-7 records remain readable. Criterion scores are unavailable for
versions 1 through 6; task-success and affirmative retrieval scores are
unavailable for versions 1 through 5. Missing values are not inferred from
invocation status or marker-only scoring. Missing provider metrics are
reported as unavailable rather than zero. Version-5 and later MCP records
include content-free session data-path byte counters for capture
input/retention, tool/search/retrieval responses, and framed socket traffic.
The adapter enables local metrics for MCP runs and collects the server
snapshot after each run; missing snapshots are represented as zero counters
and should be treated as unavailable when diagnosing a failed run. Summaries
include per-criterion pass counts/rates by mode and paired MCP-minus-control
deltas, without exposing criterion or answer text. Version 8 adds ordered
`criterion_search_response_hits` and `criterion_slice_response_hits` vectors
for MCP runs, plus counts of successful `search_capture` and
`get_capture_slice` responses. Each boolean says whether that criterion's
phrase appeared in any response from the named tool during that run; control
vectors are null because no MCP response exists. Summary exposure rates are
scoped by task and criterion index. Versions 1 through 7 report these exposure
metrics as unavailable. The runner reads response text in memory to score it,
then retains only booleans and counts; it does not write tool-response text to
the records or summary. Aggregate summary schema version 2 includes source-
specific successful-response counts and per-task exposure rates.
Summary schema version 3 adds per-task objective task-success rates by mode and
paired MCP-minus-control deltas with 95% confidence intervals. It cross-tabulates
criterion answer pass/fail against phrase-hit, successful-response-without-hit,
and no-successful-response states for each MCP source. Phrase matching uses
case-folded text, normalized whitespace, and phrase boundaries. No successful
response does not prove that a tool was not called, and no phrase match does not
prove that semantically useful evidence was absent. Older summaries show these
new fields as unavailable.
Summaries also report usage sample counts, monotonicity observations, and
first-to-last deltas. Monotonic samples are explicitly inconclusive: they may
be cumulative or per-turn values and require a controlled calibration matrix.

From a clean candidate checkout, collect only the paired agent A/B runs and
use a separate repetition count from local benchmarks:

```bash
CODEX_HOME=/path/to/writable/authenticated-codex-home \
  ./scripts/collect-release-benchmarks.sh \
  --baseline-ref v0.6.3 \
  --model gpt-6-luna \
  --agent-ab-repetitions 20 \
  --agent-ab-only \
  --seed 20261006
```

The wrapper creates one counterbalanced schedule and reuses it for control and
MCP runs on both revisions, passing the same model to every run. The bundle
report includes per-task success counts and paired MCP-minus-control deltas,
per-criterion final-answer counts and paired deltas, and source-specific MCP
response exposure cross-tabs against answer pass/fail. `phrase_hit` means a
case-folded, whitespace-normalized criterion phrase matched with phrase
boundaries; phrase absence does not establish semantic irrelevance.
`--agent-ab-only` skips the other benchmark sections;
`--agent-ab-repetitions` defaults to `--repetitions` when omitted.

Generate the reviewed synthetic EB-heavy fixture and its task manifest with:

```bash
.venv/bin/python -m benchmarks.benchmark_agent_ab_fixtures \
  --fixture-output /tmp/agent-ab-fixture \
  --manifest-output /tmp/agent-ab-tasks.json
```

The fixture generates large test and build output at runtime, plus a small
targeted-output task and a follow-up retrieval task. The manifest includes
output-size bands, expected signals, and objective success criteria. It is
synthetic and contains no user logs or captured output.

For a repository-shaped evaluation, set `AGENT_AB_FIXTURE_PROFILE`:

```bash
AGENT_AB_FIXTURE_PROFILE=repository-shaped-v1 \
  CODEX_HOME=/path/to/writable/authenticated-codex-home \
  ./benchmarks/run_agent_ab_experiment.sh
```

This profile contains source, tests, configuration, and repository workflow
tooling. Its test and build commands inspect that layout while producing
deterministic noisy signals. It is still synthetic and privacy-safe; it does
not contain a production repository or user data.

Repeat the complete five-repetition Codex A/B run with the repository script.
Set `CODEX_HOME` to a writable, authenticated Codex home; generated fixtures,
records, and lifecycle logs remain under a unique temporary directory (normally
`/tmp`) by default:

```bash
CODEX_HOME=/path/to/writable/authenticated-codex-home \
  ./benchmarks/run_agent_ab_experiment.sh
```

Override `AGENT_AB_RUN_DIR`, `AGENT_AB_MODEL`, `AGENT_AB_REPETITIONS`,
`AGENT_AB_SEED`, `AGENT_AB_TIMEOUT_SECONDS`, or `AGENT_AB_TEST_EMBEDDINGS` to
repeat a different experiment. The script also writes
`records.result.json` and `summary.result.json` in the common workload result
format; set `AGENT_AB_EXPERIMENT` and `AGENT_AB_VARIANT` to assign them to an
experiment group (see "Organizing experiments"). The test embedding setting defaults to `1` for
deterministic, offline runs; set it to `0` to exercise the configured FastEmbed
model and measure real model startup and first-query behavior. Keep the model,
embedding cache, and other environment settings identical across paired runs.
The script uses the synthetic fixture by default; set
`AGENT_AB_FIXTURE_PROFILE=repository-shaped-v1` for the repository-shaped
synthetic profile. Use the lower-level runner commands above for a
privacy-reviewed production-repository fixture and private task manifest.

Create and compare an aggregate agent A/B baseline after a privacy review:

```bash
.venv/bin/python -m benchmarks.benchmark_agent_ab_baseline \
  --summary /tmp/agent-ab-summary.json \
  --create-baseline \
  --output benchmarks/data/benchmark_agent_ab_baseline.json
.venv/bin/python -m benchmarks.benchmark_agent_ab_baseline \
  --summary /tmp/agent-ab-summary.json \
  --baseline benchmarks/data/benchmark_agent_ab_baseline.json \
  --fail-on-regression \
  --output /tmp/agent-ab-comparison.json
```

The baseline stores agent and embedding model configuration, embedding mode and
cache, fixture, seed, repetition, aggregate metrics, and explicit tolerances
only. Objective task success and affirmative signal retrieval are primary
gated outcomes; invocation completion is reported separately. The checked-in
fixture v1 baseline predates objective scoring, so its task-success and
retrieval values are unavailable. Regenerate it from a reviewed fixture v2
experiment before using those gates. It never stores prompts, transcripts, commands,
captures, or user content. This comparison is a documented manual workflow;
live model calls are not part of required pull-request CI. Update a checked-in
baseline only when fixture or model changes are explained in review.

The current agent-level evaluation supports a provisional routing policy:
prefer MCP for large noisy output and follow-up retrieval, and prefer direct
execution for small targeted inspections. Do not treat this as a universal
default or convert it into hard numeric thresholds yet. The repository-shaped
profile is synthetic, so production-repository behavior should be validated
separately before generalizing the result. Provider-reported token deltas are
diagnostic until a calibration matrix establishes whether usage samples are
cumulative or per-turn.

Compare sequential per-capture retrieval with the consolidated workflow:
```bash
EPHEMERAL_TEST_EMBEDDINGS=1 .venv/bin/python -m benchmarks.benchmark_effectiveness \
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
