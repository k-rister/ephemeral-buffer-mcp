#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$benchmark_dir/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$project_dir/.venv/bin/python}"
model="${AGENT_AB_MODEL:-gpt-5.6-luna}"
repetitions="${AGENT_AB_REPETITIONS:-5}"
seed="${AGENT_AB_SEED:-20260909}"
timeout_seconds="${AGENT_AB_TIMEOUT_SECONDS:-900}"
fixture_profile="${AGENT_AB_FIXTURE_PROFILE:-synthetic-eb-heavy-v1}"
test_embeddings="${AGENT_AB_TEST_EMBEDDINGS:-1}"
run_dir="${AGENT_AB_RUN_DIR:-}"
experiment="${AGENT_AB_EXPERIMENT:-}"
variant="${AGENT_AB_VARIANT:-}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"

if [[ -z "${CODEX_HOME:-}" ]]; then
    echo "CODEX_HOME must point to a writable, authenticated Codex home" >&2
    exit 2
fi
if [[ ! -d "$CODEX_HOME" || ! -w "$CODEX_HOME" ]]; then
    echo "CODEX_HOME is missing or not writable: $CODEX_HOME" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi
if ! command -v codex >/dev/null 2>&1; then
    echo "codex CLI is not on PATH" >&2
    exit 2
fi
if [[ "$test_embeddings" != "0" && "$test_embeddings" != "1" ]]; then
    echo "AGENT_AB_TEST_EMBEDDINGS must be 0 or 1" >&2
    exit 2
fi

CODEX_HOME="$CODEX_HOME" codex login status >/dev/null
if [[ -z "$run_dir" ]]; then
    run_dir="$(mktemp -d "${TMPDIR:-/tmp}/agent-ab-run.XXXXXX")"
fi
mkdir -p "$run_dir"

# Experiment group and metadata recorded in both workload result documents so
# list_workload_results.py and compare_workload_results.py can organise runs.
experiment_args=(
    --metadata "model=$model"
    --metadata "task_type=$fixture_profile"
    --metadata "workload_size=$repetitions"
    --metadata "environment=$([[ "$test_embeddings" == "1" ]] && echo test-embeddings || echo fastembed)"
    --metadata "started_at=$started_at"
)
if [[ -n "$experiment" ]]; then
    experiment_args+=(--experiment "$experiment")
fi
if [[ -n "$variant" ]]; then
    experiment_args+=(--metadata "variant=$variant")
fi

if [[ "$fixture_profile" == "repository-shaped-v1" ]]; then
    "$python_bin" -m benchmarks.benchmark_agent_ab_repository_fixture \
        --fixture-output "$run_dir/fixture" \
        --manifest-output "$run_dir/tasks.json"
else
    "$python_bin" -m benchmarks.benchmark_agent_ab_fixtures \
        --fixture-output "$run_dir/fixture" \
        --manifest-output "$run_dir/tasks.json"
fi

"$python_bin" -m benchmarks.benchmark_agent_ab \
    --schedule-output "$run_dir/schedule.json" \
    --repetitions "$repetitions" \
    --seed "$seed"

EPHEMERAL_TEST_EMBEDDINGS="$test_embeddings" CODEX_HOME="$CODEX_HOME" \
"$python_bin" -m benchmarks.run_codex_agent_ab \
    --schedule "$run_dir/schedule.json" \
    --tasks "$run_dir/tasks.json" \
    --repository "$run_dir/fixture" \
    --repository-fixture "$fixture_profile" \
    --model "$model" \
    --mcp-module ephemeral_buffer_mcp.server \
    --mcp-python-path "$project_dir/src" \
    --mcp-python "$python_bin" \
    --allow-mcp-approvals \
    --require-mcp-calls \
    --sandbox read-only \
    --timeout "$timeout_seconds" \
    --diagnostic-log-dir "$run_dir/lifecycle" \
    --output "$run_dir/records.json" \
    --result "$run_dir/records.result.json" \
    "${experiment_args[@]}"

"$python_bin" -m benchmarks.benchmark_agent_ab \
    --records "$run_dir/records.json" \
    --output "$run_dir/summary.json" \
    --result "$run_dir/summary.result.json" \
    "${experiment_args[@]}"

echo "Run complete:"
echo "  records: $run_dir/records.json"
echo "  summary: $run_dir/summary.json"
echo "  workload results: $run_dir/records.result.json $run_dir/summary.result.json"
echo "  lifecycle: $run_dir/lifecycle/"
