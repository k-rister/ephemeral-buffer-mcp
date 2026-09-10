#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
MODEL="${AGENT_AB_MODEL:-gpt-5.6-luna}"
REPETITIONS="${AGENT_AB_REPETITIONS:-5}"
SEED="${AGENT_AB_SEED:-20260909}"
TIMEOUT_SECONDS="${AGENT_AB_TIMEOUT_SECONDS:-900}"
RUN_DIR="${AGENT_AB_RUN_DIR:-/tmp/agent-ab-run-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${CODEX_HOME:-}" ]]; then
    echo "CODEX_HOME must point to a writable, authenticated Codex home" >&2
    exit 2
fi
if [[ ! -d "$CODEX_HOME" || ! -w "$CODEX_HOME" ]]; then
    echo "CODEX_HOME is missing or not writable: $CODEX_HOME" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if ! command -v codex >/dev/null 2>&1; then
    echo "codex CLI is not on PATH" >&2
    exit 2
fi

CODEX_HOME="$CODEX_HOME" codex login status >/dev/null
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/benchmark_agent_ab_fixtures.py" \
    --fixture-output "$RUN_DIR/fixture" \
    --manifest-output "$RUN_DIR/tasks.json"

"$PYTHON_BIN" "$PROJECT_DIR/benchmark_agent_ab.py" \
    --schedule-output "$RUN_DIR/schedule.json" \
    --repetitions "$REPETITIONS" \
    --seed "$SEED"

EPHEMERAL_TEST_EMBEDDINGS=1 CODEX_HOME="$CODEX_HOME" \
"$PYTHON_BIN" "$PROJECT_DIR/run_codex_agent_ab.py" \
    --schedule "$RUN_DIR/schedule.json" \
    --tasks "$RUN_DIR/tasks.json" \
    --repository "$RUN_DIR/fixture" \
    --repository-fixture synthetic-eb-heavy-v1 \
    --model "$MODEL" \
    --mcp-server-script "$PROJECT_DIR/server.py" \
    --mcp-python "$PYTHON_BIN" \
    --allow-mcp-approvals \
    --require-mcp-calls \
    --sandbox read-only \
    --timeout "$TIMEOUT_SECONDS" \
    --diagnostic-log-dir "$RUN_DIR/lifecycle" \
    --output "$RUN_DIR/records.json"

"$PYTHON_BIN" "$PROJECT_DIR/benchmark_agent_ab.py" \
    --records "$RUN_DIR/records.json" \
    --output "$RUN_DIR/summary.json"

echo "Run complete:"
echo "  records: $RUN_DIR/records.json"
echo "  summary: $RUN_DIR/summary.json"
echo "  lifecycle: $RUN_DIR/lifecycle/"
