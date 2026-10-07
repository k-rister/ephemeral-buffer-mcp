#!/usr/bin/env bash
# Collect release-to-candidate benchmark results and matched agent A/B runs.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
baseline_ref=""
model=""
output_dir=""
samples=5
semantic_samples=3
repetitions=5
seed=20261006
agent_timeout=900
benchmark_timeout=3600
failures=0
cleanup_failure=0

usage() {
    cat <<'USAGE'
Usage: scripts/collect-release-benchmarks.sh --baseline-ref TAG --model MODEL [options]

Options:
  --output-dir PATH          New output directory (default: timestamped path under /tmp)
  --samples N                Latency, prefetch, routing, and warm-up samples (default: 5)
  --semantic-samples N       Semantic-index samples per size (default: 3)
  --repetitions N            Effectiveness and agent A/B repetitions (default: 5)
  --seed N                   Shared synthetic workload seed (default: 20261006)
  --agent-timeout SECONDS    Per-task Codex timeout (default: 900)
  --benchmark-timeout SECONDS Timeout for each local benchmark (default: 3600)
  -h, --help                 Show this help

Requires a clean checkout, .venv (or PYTHON_BIN), CODEX_HOME, an authenticated
Codex CLI, and a real model identifier. Agent A/B runs make live Codex calls.
USAGE
}

while (($#)); do
    case "$1" in
        --baseline-ref) baseline_ref="${2:?missing value for --baseline-ref}"; shift 2 ;;
        --model) model="${2:?missing value for --model}"; shift 2 ;;
        --output-dir) output_dir="${2:?missing value for --output-dir}"; shift 2 ;;
        --samples) samples="${2:?missing value for --samples}"; shift 2 ;;
        --semantic-samples) semantic_samples="${2:?missing value for --semantic-samples}"; shift 2 ;;
        --repetitions) repetitions="${2:?missing value for --repetitions}"; shift 2 ;;
        --seed) seed="${2:?missing value for --seed}"; shift 2 ;;
        --agent-timeout) agent_timeout="${2:?missing value for --agent-timeout}"; shift 2 ;;
        --benchmark-timeout) benchmark_timeout="${2:?missing value for --benchmark-timeout}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$baseline_ref" || -z "$model" || "$model" == "your-fixed-model" ]]; then
    echo "--baseline-ref and a real --model identifier are required" >&2
    usage >&2
    exit 2
fi
for count in "$samples" "$semantic_samples" "$repetitions" "$agent_timeout" "$benchmark_timeout"; do
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "sample counts, repetitions, and timeouts must be positive integers" >&2
        exit 2
    fi
done
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi
codex_home="${CODEX_HOME:-}"
if [[ -z "$codex_home" || ! -d "$codex_home" || ! -w "$codex_home" ]]; then
    echo "CODEX_HOME must point to a writable, authenticated Codex home" >&2
    exit 2
fi
codex_home="$(cd "$codex_home" && pwd)"
export CODEX_HOME="$codex_home"
if ! command -v codex >/dev/null 2>&1; then
    echo "Codex CLI is not on PATH" >&2
    exit 2
fi
if ! CODEX_HOME="$CODEX_HOME" codex login status >/dev/null 2>&1; then
    echo "Codex CLI is not authenticated in CODEX_HOME" >&2
    exit 2
fi
if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=all)" ]]; then
    echo "working tree must be clean so the candidate revision is reproducible" >&2
    exit 2
fi
if ! baseline_revision="$(git -C "$repo_dir" rev-parse --verify --end-of-options "${baseline_ref}^{commit}")"; then
    echo "baseline ref is not available locally: $baseline_ref (fetch tags first)" >&2
    exit 2
fi
candidate_revision="$(git -C "$repo_dir" rev-parse HEAD)"

if [[ -z "$output_dir" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="${TMPDIR:-/tmp}/ephemeral-buffer-release-benchmarks-$stamp-$$"
fi
output_dir="$("$python_bin" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$output_dir")"
case "$output_dir" in
    "$repo_dir"|"$repo_dir"/*)
        echo "output directory must be outside the repository" >&2
        exit 2
        ;;
esac
if [[ -e "$output_dir" ]]; then
    echo "output directory already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/comparisons"
printf 'status\tname\texit_code\tlog_or_reason\n' > "$output_dir/STATUS.tsv"

record_status() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$output_dir/STATUS.tsv"
}

skip() {
    record_status skipped "$1" - "$2"
    printf 'SKIP %s: %s\n' "$1" "$2"
}

run() {
    local name="$1" cwd="$2" test_embeddings="$3" log="$output_dir/logs/$1.log" code status
    shift 3
    mkdir -p "$(dirname "$log")"
    printf 'RUN  %s\n' "$name"
    if (cd "$cwd" && EPHEMERAL_TEST_EMBEDDINGS="$test_embeddings" "$@") > "$log" 2>&1; then
        code=0
        status=success
    else
        code=$?
        status=failed
        failures=$((failures + 1))
    fi
    record_status "$status" "$name" "$code" "logs/$(basename "$log")"
    printf '%s %s (exit %s)\n' "${status^^}" "$name" "$code"
    if [[ "$status" == failed ]]; then
        printf '  log: %s\n' "$log"
    fi
}

run_benchmark() {
    local revision="$1" source_dir="$2" name="$3" embeddings="$4" script="$5"
    shift 5
    if [[ ! -f "$source_dir/$script" ]]; then
        skip "$revision-$name" "$script is absent at this revision"
        return
    fi
    run "$revision-$name" "$source_dir" "$embeddings" "$python_bin" "$source_dir/$script" "$@"
}

worktree="$output_dir/baseline-worktree"
cleanup_worktree() {
    if [[ -d "$worktree" ]]; then
        if ! git -C "$repo_dir" worktree remove --force "$worktree" >/dev/null 2>&1; then
            cleanup_failure=1
            failures=$((failures + 1))
            record_status failed cleanup-baseline-worktree - "remove failed: $worktree"
            echo "failed to remove temporary worktree: $worktree" >&2
        fi
    fi
}
trap cleanup_worktree EXIT
if ! git -C "$repo_dir" worktree add --detach "$worktree" "$baseline_revision" > "$output_dir/logs/worktree.log" 2>&1; then
    echo "failed to create baseline worktree; see $output_dir/logs/worktree.log" >&2
    exit 1
fi

run_revision_suite() {
    local revision="$1" source_dir="$2" native="$output_dir/$1/native" results="$output_dir/$1/results"
    local -a memory_args
    mkdir -p "$native" "$results"

    run_benchmark "$revision" "$source_dir" concurrency 0 benchmark_concurrency.py \
        --captures 32 --workers 8 --baseline "$source_dir/benchmark_baseline.json" \
        --output "$native/concurrency.json" --result "$results/concurrency.result.json"
    run_benchmark "$revision" "$source_dir" admission 0 benchmark_admission.py \
        --repetitions 3 --output "$native/admission.json" --result "$results/admission.result.json"
    run_benchmark "$revision" "$source_dir" latency 0 benchmark_latency.py \
        --samples "$samples" --output "$native/latency.json" --result "$results/latency.result.json"
    run_benchmark "$revision" "$source_dir" prefetch 0 benchmark_prefetch.py \
        --line-count 256 --samples "$samples" --output "$native/prefetch.json" --result "$results/prefetch.result.json"
    run_benchmark "$revision" "$source_dir" routing 1 benchmark_routing.py \
        --samples "$samples" --output "$native/routing.json" --result "$results/routing.result.json"
    run_benchmark "$revision" "$source_dir" warmup 0 benchmark_warmup.py \
        --samples "$samples" --output "$native/warmup.json" --result "$results/warmup.result.json"
    run_benchmark "$revision" "$source_dir" semantic-index 0 benchmark_semantic_index.py \
        --samples "$semantic_samples" --output "$native/semantic-index.json" --result "$results/semantic-index.result.json"

    if [[ -f "$source_dir/benchmark_semantic_memory.py" ]]; then
        memory_args=(--threads 1)
        if grep -q 'add_result_argument(parser)' "$source_dir/benchmark_semantic_memory.py"; then
            memory_args+=(--result "$results/semantic-memory.result.json")
        fi
        run_benchmark "$revision" "$source_dir" semantic-memory 0 benchmark_semantic_memory.py "${memory_args[@]}"
    else
        skip "$revision-semantic-memory" "benchmark_semantic_memory.py is absent at this revision"
    fi

    run_benchmark "$revision" "$source_dir" effectiveness-modes 1 benchmark_effectiveness.py \
        --mode both --output "$native/effectiveness-modes.json" --result "$results/effectiveness-modes.result.json"
    run_benchmark "$revision" "$source_dir" effectiveness-ab 1 benchmark_effectiveness.py \
        --ab-runs "$repetitions" --seed "$seed" --output "$native/effectiveness-ab.json" --result "$results/effectiveness-ab.result.json"
    run_benchmark "$revision" "$source_dir" effectiveness-consolidation 1 benchmark_effectiveness.py \
        --consolidation-runs "$repetitions" --seed "$seed" \
        --output "$native/effectiveness-consolidation.json" --result "$results/effectiveness-consolidation.result.json"
    run_benchmark "$revision" "$source_dir" effectiveness-summary 1 benchmark_effectiveness.py \
        --summary --output "$native/effectiveness-summary.json" --result "$results/effectiveness-summary.result.json"
    run_benchmark "$revision" "$source_dir" relevance 1 benchmark_relevance.py \
        --top-k 3 --baseline "$source_dir/benchmark_relevance_baseline.json" --fail-on-regression \
        --output "$native/relevance.json" --result "$results/relevance.result.json"
}

run_revision_suite baseline "$worktree"
run_revision_suite candidate "$repo_dir"

compare_result() {
    local name="$1" baseline_result="$2" candidate_result="$3" destination="$4"
    run "compare-$name" "$repo_dir" 0 "$python_bin" "$repo_dir/compare_workload_results.py" \
        "$baseline_result" "$candidate_result" --statistic all --output "$destination"
}

for candidate_result in "$output_dir"/candidate/results/*.result.json; do
    [[ -f "$candidate_result" ]] || continue
    result_name="$(basename "$candidate_result")"
    baseline_result="$output_dir/baseline/results/$result_name"
    comparison_name="${result_name%.result.json}"
    if [[ -f "$baseline_result" ]]; then
        compare_result "$comparison_name" "$baseline_result" "$candidate_result" \
            "$output_dir/comparisons/$comparison_name.comparison.json"
    else
        skip "compare-$comparison_name" "baseline workload-result document is unavailable"
    fi
done
for baseline_result in "$output_dir"/baseline/results/*.result.json; do
    [[ -f "$baseline_result" ]] || continue
    result_name="$(basename "$baseline_result")"
    if [[ ! -f "$output_dir/candidate/results/$result_name" ]]; then
        skip "compare-${result_name%.result.json}" "candidate workload-result document is unavailable"
    fi
done

for revision in baseline candidate; do
    prefetch_result="$output_dir/$revision/results/prefetch.result.json"
    if [[ -f "$prefetch_result" ]]; then
        run "compare-$revision-prefetch-policies" "$repo_dir" 0 "$python_bin" \
            "$repo_dir/compare_workload_results.py" \
            "$prefetch_result#prefetch-off" "$prefetch_result#prefetch-on" \
            --statistic median --metric ingest --metric first_search --metric subsequent_search \
            --output "$output_dir/comparisons/$revision-prefetch.comparison.json"
    fi
done

fixture_dir="$output_dir/agent-ab/fixture"
tasks="$output_dir/agent-ab/tasks.json"
schedule="$output_dir/agent-ab/schedule.json"
mkdir -p "$output_dir/agent-ab"
run create-agent-ab-v2-fixture "$repo_dir" 0 "$python_bin" "$repo_dir/benchmark_agent_ab_fixtures.py" \
    --fixture-output "$fixture_dir" --manifest-output "$tasks"
run create-agent-ab-v2-schedule "$repo_dir" 0 "$python_bin" "$repo_dir/benchmark_agent_ab.py" \
    --schedule-output "$schedule" --repetitions "$repetitions" --seed "$seed"

for revision in baseline candidate; do
    if [[ "$revision" == baseline ]]; then
        server_dir="$worktree"
        server_revision="$baseline_revision"
    else
        server_dir="$repo_dir"
        server_revision="$candidate_revision"
    fi
    ab_dir="$output_dir/agent-ab/$revision"
    mkdir -p "$ab_dir"
    agent_metadata=(--metadata "variant=$revision" --metadata "server_revision=$server_revision")
    run "agent-ab-$revision" "$repo_dir" 1 "$python_bin" "$repo_dir/run_codex_agent_ab.py" \
        --schedule "$schedule" --tasks "$tasks" --repository "$fixture_dir" \
        --repository-fixture synthetic-eb-heavy-v2 --model "$model" \
        --mcp-server-script "$server_dir/server.py" --mcp-python "$python_bin" \
        --allow-mcp-approvals --require-mcp-calls --sandbox read-only --timeout "$agent_timeout" \
        --diagnostic-log-dir "$ab_dir/lifecycle" --output "$ab_dir/records.json" \
        --result "$ab_dir/records.result.json" "${agent_metadata[@]}"
    if [[ -f "$ab_dir/records.json" ]]; then
        run "summarize-agent-ab-$revision" "$repo_dir" 0 "$python_bin" "$repo_dir/benchmark_agent_ab.py" \
            --records "$ab_dir/records.json" --output "$ab_dir/summary.json" \
            --result "$ab_dir/summary.result.json" "${agent_metadata[@]}"
    else
        skip "summarize-agent-ab-$revision" "agent runner did not create records"
    fi
done

if [[ -f "$output_dir/agent-ab/baseline/summary.result.json" && -f "$output_dir/agent-ab/candidate/summary.result.json" ]]; then
    compare_result agent-ab \
        "$output_dir/agent-ab/baseline/summary.result.json" \
        "$output_dir/agent-ab/candidate/summary.result.json" \
        "$output_dir/comparisons/agent-ab.comparison.json"
else
    skip compare-agent-ab "both aggregate result documents are required"
fi

{
    printf '# Release benchmark bundle\n\n'
    printf '%s\n' "- Baseline: \`$baseline_ref\` (\`${baseline_revision:0:12}\`)"
    printf '%s\n' "- Candidate: \`${candidate_revision:0:12}\`"
    printf '%s\n' "- Model: \`$model\`"
    printf '%s\n' "- Repetitions: $repetitions; seed: $seed; samples: $samples; semantic-index samples: $semantic_samples"
    printf '%s\n\n' "- Results: \`$output_dir\`"
    printf '## Command status\n\n| Status | Name | Exit | Log or reason |\n|---|---|---:|---|\n'
    while IFS=$'\t' read -r status name code log_or_reason; do
        [[ "$status" == status ]] && continue
        printf '| %s | %s | %s | %s |\n' "$status" "$name" "$code" "$log_or_reason"
    done < "$output_dir/STATUS.tsv"
    printf '\nGenerated fixture and records remain in this local bundle. The fixture is synthetic; keep the bundle out of the repository.\n'
    printf 'Native metrics are under `baseline/native/` and `candidate/native/`; common result JSON is under each revision’s `results/`.\n'
    printf 'Compare JSON is under `comparisons/`; command logs are under `logs/`. Missing or incompatible results are recorded above or in each comparison.\n'
} > "$output_dir/REPORT.md"

cleanup_worktree
trap - EXIT
if ((cleanup_failure)); then
    printf 'Temporary baseline worktree cleanup failed.\n' >&2
fi
printf '\nBenchmark bundle: %s\nReport: %s/REPORT.md\n' "$output_dir" "$output_dir"
if ((failures)); then
    printf 'Completed with %s failed command(s); inspect STATUS.tsv and logs.\n' "$failures" >&2
    exit 1
fi
