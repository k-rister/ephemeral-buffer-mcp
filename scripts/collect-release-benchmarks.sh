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
agent_ab_repetitions=""
agent_ab_only=0
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
  --repetitions N            Effectiveness repetitions (default: 5)
  --agent-ab-repetitions N   Agent A/B repetitions (default: --repetitions)
  --agent-ab-only            Skip non-agent benchmarks and comparisons
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
        --agent-ab-repetitions) agent_ab_repetitions="${2:?missing value for --agent-ab-repetitions}"; shift 2 ;;
        --agent-ab-only) agent_ab_only=1; shift ;;
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
if [[ -z "$agent_ab_repetitions" ]]; then
    agent_ab_repetitions="$repetitions"
fi
for count in "$samples" "$semantic_samples" "$repetitions" "$agent_ab_repetitions" "$agent_timeout" "$benchmark_timeout"; do
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
    local source_path=""
    shift 3
    mkdir -p "$(dirname "$log")"
    if [[ -d "$cwd/src/ephemeral_buffer_mcp" ]]; then
        source_path="$cwd/src:$cwd"
    else
        source_path="$cwd"
    fi
    printf 'RUN  %s\n' "$name"
    if (cd "$cwd" && PYTHONPATH="${source_path}${PYTHONPATH:+:$PYTHONPATH}" \
        EPHEMERAL_TEST_EMBEDDINGS="$test_embeddings" "$@") > "$log" 2>&1; then
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

tool_command=()
tool_path=""

resolve_tool() {
    local source_dir="$1" script="$2" module="${2%.py}"
    if [[ -f "$source_dir/benchmarks/$script" ]]; then
        tool_path="$source_dir/benchmarks/$script"
        tool_command=("$python_bin" -m "benchmarks.$module")
    elif [[ -f "$source_dir/$script" ]]; then
        tool_path="$source_dir/$script"
        tool_command=("$python_bin" "$tool_path")
    else
        tool_path=""
        tool_command=()
    fi
}

run_tool() {
    local name="$1" source_dir="$2" embeddings="$3" script="$4"
    shift 4
    resolve_tool "$source_dir" "$script"
    if [[ -z "$tool_path" ]]; then
        skip "$name" "$script is absent at this revision"
        return
    fi
    run "$name" "$source_dir" "$embeddings" "${tool_command[@]}" "$@"
}

run_benchmark() {
    local revision="$1" source_dir="$2" name="$3" embeddings="$4" script="$5"
    shift 5
    run_tool "$revision-$name" "$source_dir" "$embeddings" "$script" "$@"
}

resolve_baseline_asset() {
    local source_dir="$1" name="$2"
    if [[ -f "$source_dir/benchmarks/data/$name" ]]; then
        printf '%s\n' "$source_dir/benchmarks/data/$name"
    elif [[ -f "$source_dir/$name" ]]; then
        printf '%s\n' "$source_dir/$name"
    else
        printf '%s\n' "$source_dir/benchmarks/data/$name"
    fi
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
    local revision="$1" source_dir="$2" source_revision="$3"
    local native="$output_dir/$1/native" results="$output_dir/$1/results"
    local concurrency_baseline relevance_baseline semantic_memory_script
    mkdir -p "$native" "$results"
    concurrency_baseline="$(resolve_baseline_asset "$source_dir" benchmark_baseline.json)"
    relevance_baseline="$(resolve_baseline_asset "$source_dir" benchmark_relevance_baseline.json)"

    run_benchmark "$revision" "$source_dir" concurrency 0 benchmark_concurrency.py \
        --captures 32 --workers 8 --baseline "$concurrency_baseline" \
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

    resolve_tool "$source_dir" benchmark_semantic_memory.py
    semantic_memory_script="$tool_path"
    if [[ -n "$semantic_memory_script" ]]; then
        if grep -q 'add_result_argument(parser)' "$semantic_memory_script"; then
            run_benchmark "$revision" "$source_dir" semantic-memory 0 benchmark_semantic_memory.py \
                --threads 1 --result "$results/semantic-memory.result.json"
        else
            run_benchmark "$revision" "$source_dir" semantic-memory 0 benchmark_semantic_memory.py --threads 1
            run "normalize-$revision-semantic-memory" "$repo_dir" 0 "$python_bin" \
                "$repo_dir/scripts/normalize-semantic-memory-result.py" \
                --log "$output_dir/logs/$revision-semantic-memory.log" \
                --revision-dir "$source_dir" --source-revision "$source_revision" \
                --native-output "$native/semantic-memory.json" \
                --result-output "$results/semantic-memory.result.json"
        fi
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
        --top-k 3 --baseline "$relevance_baseline" --fail-on-regression \
        --output "$native/relevance.json" --result "$results/relevance.result.json"
}

compare_result() {
    local name="$1" baseline_result="$2" candidate_result="$3" destination="$4"
    run_tool "compare-$name" "$repo_dir" 0 compare_workload_results.py \
        "$baseline_result" "$candidate_result" --statistic all --output "$destination"
}

if ((agent_ab_only == 0)); then
    run_revision_suite baseline "$worktree" "$baseline_revision"
    run_revision_suite candidate "$repo_dir" "$candidate_revision"

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
            run_tool "compare-$revision-prefetch-policies" "$repo_dir" 0 compare_workload_results.py \
                "$prefetch_result#prefetch-off" "$prefetch_result#prefetch-on" \
                --statistic median --metric ingest --metric first_search --metric subsequent_search \
                --output "$output_dir/comparisons/$revision-prefetch.comparison.json"
        fi
    done
else
    skip non-agent-benchmarks "--agent-ab-only was selected"
fi

fixture_dir="$output_dir/agent-ab/fixture"
tasks="$output_dir/agent-ab/tasks.json"
schedule="$output_dir/agent-ab/schedule.json"
mkdir -p "$output_dir/agent-ab"
run_tool create-agent-ab-v2-fixture "$repo_dir" 0 benchmark_agent_ab_fixtures.py \
    --fixture-output "$fixture_dir" --manifest-output "$tasks"
run_tool create-agent-ab-v2-schedule "$repo_dir" 0 benchmark_agent_ab.py \
    --schedule-output "$schedule" --repetitions "$agent_ab_repetitions" --seed "$seed"

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
    if [[ -f "$server_dir/src/ephemeral_buffer_mcp/server.py" ]]; then
        server_args=(--mcp-module ephemeral_buffer_mcp.server \
            --mcp-python-path "$server_dir/src" --mcp-python-path "$server_dir")
    else
        server_args=(--mcp-module server --mcp-server-script "$server_dir/server.py")
    fi
    run_tool "agent-ab-$revision" "$repo_dir" 1 run_codex_agent_ab.py \
        --schedule "$schedule" --tasks "$tasks" --repository "$fixture_dir" \
        --repository-fixture synthetic-eb-heavy-v2 --model "$model" \
        "${server_args[@]}" --mcp-python "$python_bin" \
        --allow-mcp-approvals --require-mcp-calls --sandbox read-only --timeout "$agent_timeout" \
        --diagnostic-log-dir "$ab_dir/lifecycle" --output "$ab_dir/records.json" \
        --result "$ab_dir/records.result.json" "${agent_metadata[@]}"
    if [[ -f "$ab_dir/records.json" ]]; then
        run_tool "summarize-agent-ab-$revision" "$repo_dir" 0 benchmark_agent_ab.py \
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

cleanup_worktree
trap - EXIT
if ((cleanup_failure)); then
    printf 'Temporary baseline worktree cleanup failed.\n' >&2
fi

if ! RELEASE_BENCHMARK_BASELINE_REF="$baseline_ref" \
    RELEASE_BENCHMARK_BASELINE_REVISION="$baseline_revision" \
    RELEASE_BENCHMARK_CANDIDATE_REVISION="$candidate_revision" \
    RELEASE_BENCHMARK_MODEL="$model" \
    RELEASE_BENCHMARK_SAMPLES="$samples" \
    RELEASE_BENCHMARK_SEMANTIC_SAMPLES="$semantic_samples" \
    RELEASE_BENCHMARK_REPETITIONS="$repetitions" \
    RELEASE_BENCHMARK_AGENT_AB_REPETITIONS="$agent_ab_repetitions" \
    RELEASE_BENCHMARK_AGENT_AB_ONLY="$agent_ab_only" \
    RELEASE_BENCHMARK_SEED="$seed" \
    RELEASE_BENCHMARK_AGENT_TIMEOUT="$agent_timeout" \
    "$python_bin" - "$output_dir/manifest.json" "$output_dir/STATUS.tsv" <<'PY'
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
status_counts = {}
with status_path.open(encoding="utf-8", newline="") as status_file:
    for row in csv.DictReader(status_file, delimiter="\t"):
        status = row.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

manifest = {
    "schema_version": 1,
    "benchmark": "release-benchmarks",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "baseline": {
        "ref": os.environ["RELEASE_BENCHMARK_BASELINE_REF"],
        "revision": os.environ["RELEASE_BENCHMARK_BASELINE_REVISION"],
    },
    "candidate": {
        "revision": os.environ["RELEASE_BENCHMARK_CANDIDATE_REVISION"],
    },
    "configuration": {
        "model": os.environ["RELEASE_BENCHMARK_MODEL"],
        "samples": int(os.environ["RELEASE_BENCHMARK_SAMPLES"]),
        "semantic_index_samples": int(os.environ["RELEASE_BENCHMARK_SEMANTIC_SAMPLES"]),
        "repetitions": int(os.environ["RELEASE_BENCHMARK_REPETITIONS"]),
        "agent_ab_repetitions": int(os.environ["RELEASE_BENCHMARK_AGENT_AB_REPETITIONS"]),
        "agent_ab_only": os.environ["RELEASE_BENCHMARK_AGENT_AB_ONLY"] == "1",
        "seed": int(os.environ["RELEASE_BENCHMARK_SEED"]),
        "agent_timeout_seconds": int(os.environ["RELEASE_BENCHMARK_AGENT_TIMEOUT"]),
        "python_executable": sys.executable,
    },
    "environment": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
    },
    "status_counts": status_counts,
    "artifacts": {
        "report": "REPORT.md",
        "status": "STATUS.tsv",
        "baseline_native": "baseline/native/",
        "candidate_native": "candidate/native/",
        "baseline_results": "baseline/results/",
        "candidate_results": "candidate/results/",
        "comparisons": "comparisons/",
        "agent_ab": "agent-ab/",
        "logs": "logs/",
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
then
    failures=$((failures + 1))
    record_status failed write-manifest - "manifest generation failed"
    printf 'Failed to write manifest.json.\n' >&2
fi

{
    printf '# Release benchmark bundle\n\n'
    printf '%s\n' "- Baseline: \`$baseline_ref\` (\`${baseline_revision:0:12}\`)"
    printf '%s\n' "- Candidate: \`${candidate_revision:0:12}\`"
    printf '%s\n' "- Model: \`$model\`"
    printf '%s\n' "- Repetitions: effectiveness=$repetitions; agent A/B=$agent_ab_repetitions; seed: $seed; samples: $samples; semantic-index samples: $semantic_samples"
    if ((agent_ab_only)); then
        printf '%s\n' '- Scope: agent A/B only; other local benchmarks and comparisons were skipped.'
    fi
    printf '%s\n' '- Manifest: `manifest.json`'
    printf '%s\n\n' "- Results: \`$output_dir\`"
    printf '## Command status\n\n| Status | Name | Exit | Log or reason |\n|---|---|---:|---|\n'
    while IFS=$'\t' read -r status name code log_or_reason; do
        [[ "$status" == status ]] && continue
        printf '| %s | %s | %s | %s |\n' "$status" "$name" "$code" "$log_or_reason"
    done < "$output_dir/STATUS.tsv"
    printf '\nGenerated fixture and records remain in this local bundle. The fixture is synthetic; keep the bundle out of the repository.\n'
    if ((agent_ab_only)); then
        printf 'Agent A/B records and summaries are under `agent-ab/`; no native local-benchmark outputs were requested.\n'
    else
        printf 'Native JSON is under `baseline/native/` and `candidate/native/` where supported; common result JSON is under each revision’s `results/`.\n'
    fi
    printf 'Compare JSON is under `comparisons/`; command logs are under `logs/`. Missing or incompatible results are recorded above or in each comparison.\n'
} > "$output_dir/REPORT.md"

if [[ -f "$output_dir/agent-ab/baseline/summary.json" || -f "$output_dir/agent-ab/candidate/summary.json" ]]; then
    if ! "$python_bin" "$repo_dir/benchmarks/render_agent_ab_report.py" \
        "$output_dir/agent-ab/baseline/summary.json" \
        "$output_dir/agent-ab/candidate/summary.json" \
        "$output_dir/REPORT.md"; then
        failures=$((failures + 1))
        record_status failed render-agent-ab-report - "task and criterion report rendering failed"
        printf 'Failed to render agent A/B report sections.\n' >&2
    fi
fi

printf '\nBenchmark bundle: %s\nReport: %s/REPORT.md\n' "$output_dir" "$output_dir"
if ((failures)); then
    printf 'Completed with %s failed command(s); inspect STATUS.tsv and logs.\n' "$failures" >&2
    exit 1
fi
