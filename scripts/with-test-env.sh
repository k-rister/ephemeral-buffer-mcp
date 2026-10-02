#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: $0 command [args...]" >&2
    exit 2
fi

test_runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/ephemeral-buffer-test.XXXXXX")"
cleanup() {
    rm -rf "$test_runtime_dir"
}
trap cleanup EXIT

# Test subprocesses inherit one private namespace even when the caller started
# inside an active EB session with explicit socket or execution-state paths.
export EPHEMERAL_SESSION_ID="test-$(basename "$test_runtime_dir")"
export EPHEMERAL_SOCKET_PATH="$test_runtime_dir/ephemeral-buffer.sock"
export EPHEMERAL_EXECUTION_STATE_DIR="$test_runtime_dir/executions"
export EPHEMERAL_METRICS_FILE="$test_runtime_dir/metrics.json"
export EPHEMERAL_LOG_FILE="$test_runtime_dir/logs.jsonl"
export EPHEMERAL_REQUIRE_ISOLATION=1
export EPHEMERAL_TEST_EMBEDDINGS="${EPHEMERAL_TEST_EMBEDDINGS:-1}"
mkdir -p "$EPHEMERAL_EXECUTION_STATE_DIR"

"$@"
