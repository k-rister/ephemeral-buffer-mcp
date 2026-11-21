import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.analyze_agent_failures import (
    DEFAULT_INACTIVITY_THRESHOLD_SECONDS,
    _metadata_summary,
    analysis_workload_result,
    build_analysis,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def records_payload(runs, schedule=None):
    if schedule is None:
        schedule = [
            {
                "sequence": index,
                "task_id": run["task_id"],
                "repetition": run["repetition"],
                "mode": run["mode"],
            }
            for index, run in enumerate(runs, 1)
        ]
    return {
        "schema_version": 14,
        "records_schema_version": 14,
        "runs": runs,
        "schedule": {"schedule": schedule},
    }


def timeout_run(completed=False, **extra):
    run = {
        "task_id": "noisy-test-failure",
        "repetition": 41,
        "mode": "mcp",
        "completed": completed,
        "task_success": False,
        "failure_reason": None if completed else "timeout",
        "duration_seconds": 900.0,
        "input_token_samples": [10, 20],
        "output_token_samples": [3],
        "mcp_tool_failures": [],
    }
    run.update(extra)
    return run


def write_lifecycle(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for event in events:
        if isinstance(event, str):
            lines.append(event)
        else:
            lines.append(json.dumps(event))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trace(root, *, metadata=None, items=(), raw_lines=()):
    directory = root / "0001-mcp"
    directory.mkdir(parents=True, exist_ok=True)
    lines = list(raw_lines)
    for started, item in items:
        event = {"type": "item.started" if started else "item.completed", "item": item}
        lines.append(json.dumps(event))
    (directory / "stdout.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metadata is not None:
        (directory / "metadata.json").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )


class AnalyzeAgentFailuresTests(unittest.TestCase):
    def test_original_timeout_completed_in_replay_is_not_reproduced(self):
        source = records_payload([timeout_run()])
        replay = records_payload([timeout_run(completed=True)])
        analysis = build_analysis(
            records_payload=replay,
            lifecycle_dir=None,
            failure_traces_dir=None,
            timeout_source_payload=source,
        )

        self.assertEqual(analysis["summary"]["timeout_categories"], {"not-reproduced": 1})
        self.assertEqual(
            analysis["timeout_classifications"][0]["task"], "noisy-test-failure"
        )
        self.assertEqual(analysis["timeout_classifications"][0]["repetition"], 41)

    def test_pending_tool_call_is_blocked(self):
        analysis = build_analysis(
            records_payload=records_payload([
                timeout_run(mcp_invocation_timeout={"pending_tool_calls": 1})
            ]),
            lifecycle_dir=None,
            failure_traces_dir=None,
            timeout_source_payload=None,
        )

        self.assertEqual(analysis["summary"]["timeout_categories"], {"blocked": 1})
        self.assertEqual(
            analysis["timeout_classifications"][0]["last_tool_state"],
            {
                "last_tool_event": None,
                "last_tool_name": None,
                "last_tool_call_id": None,
            },
        )
        self.assertEqual(analysis["timeout_classifications"][0]["pending_call_count"], 1)

    def test_three_unresolved_identical_actions_are_a_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_trace(
                root,
                metadata={
                    "schema_version": 2,
                    "timing": {
                        "started_at": "2026-11-21T00:00:00Z",
                        "ended_at": "2026-11-21T00:15:00Z",
                        "duration_seconds": 900,
                        "timeout_seconds": 900,
                    },
                },
                items=[
                    (True, {"type": "command_execution", "id": index, "command": "private command"})
                    for index in range(3)
                ],
            )
            analysis = build_analysis(
                records_payload=records_payload([timeout_run()]),
                lifecycle_dir=None,
                failure_traces_dir=root,
                timeout_source_payload=None,
            )

        self.assertEqual(analysis["summary"]["timeout_categories"], {"loop": 1})
        self.assertEqual(analysis["timeout_classifications"][0]["repeated_signature_count"], 1)
        self.assertEqual(analysis["timeout_classifications"][0]["distinct_action_count"], 1)

    def test_long_local_activity_gap_is_external_inactivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lifecycle = root / "001-mcp.jsonl"
            write_lifecycle(lifecycle, [
                {
                    "event": "embedding_model_ready",
                    "timestamp": "2026-11-21T00:00:00Z",
                },
                {
                    "event": "mcp_tool_started",
                    "tool": "search_capture",
                    "call_id": 1,
                    "timestamp": "2026-11-21T00:00:01Z",
                },
                {
                    "event": "mcp_tool_completed",
                    "tool": "search_capture",
                    "call_id": 1,
                    "timestamp": "2026-11-21T00:00:02Z",
                },
            ])
            write_trace(
                root,
                metadata={
                    "schema_version": 2,
                    "timing": {
                        "started_at": "2026-11-21T00:00:00Z",
                        "ended_at": "2026-11-21T00:10:00Z",
                        "duration_seconds": 600,
                        "timeout_seconds": 600,
                    },
                },
                items=[
                    (True, {"type": "mcp_tool_call", "id": 1, "tool": "one", "arguments": {}}),
                ],
            )
            analysis = build_analysis(
                records_payload=records_payload([timeout_run()]),
                lifecycle_dir=root,
                failure_traces_dir=root,
                timeout_source_payload=None,
                inactivity_threshold_seconds=300,
            )

        self.assertEqual(
            analysis["summary"]["timeout_categories"], {"external-inactivity-suspected": 1}
        )

    def test_distinct_recent_actions_are_budget_limited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lifecycle = root / "001-mcp.jsonl"
            write_lifecycle(lifecycle, [
                {
                    "event": "embedding_model_ready",
                    "timestamp": "2026-11-21T00:00:00Z",
                },
                {
                    "event": "mcp_tool_completed",
                    "tool": "search_capture",
                    "call_id": 1,
                    "timestamp": "2026-11-21T00:14:30Z",
                },
            ])
            write_trace(
                root,
                metadata={
                    "schema_version": 2,
                    "timing": {
                        "started_at": "2026-11-21T00:00:00Z",
                        "ended_at": "2026-11-21T00:15:00Z",
                        "duration_seconds": 900,
                        "timeout_seconds": 900,
                    },
                },
                items=[
                    item
                    for index in range(3)
                    for item in (
                        (True, {
                            "type": "mcp_tool_call",
                            "id": index,
                            "tool": f"tool-{index}",
                            "arguments": {"index": index},
                        }),
                        (False, {
                            "type": "mcp_tool_call",
                            "id": index,
                            "tool": f"tool-{index}",
                            "arguments": {"index": index},
                        }),
                    )
                ],
            )
            analysis = build_analysis(
                records_payload=records_payload([timeout_run()]),
                lifecycle_dir=root,
                failure_traces_dir=root,
                timeout_source_payload=None,
            )

        self.assertEqual(analysis["summary"]["timeout_categories"], {"budget-limited": 1})
        self.assertEqual(analysis["timeout_classifications"][0]["distinct_action_count"], 3)

    def test_timeout_without_discriminating_evidence_is_insufficient(self):
        analysis = build_analysis(
            records_payload=records_payload([timeout_run()]),
            lifecycle_dir=None,
            failure_traces_dir=None,
            timeout_source_payload=None,
        )

        self.assertEqual(analysis["summary"]["timeout_categories"], {"insufficient-evidence": 1})

    def test_validation_failures_are_grouped_without_values(self):
        run = timeout_run(completed=True, mcp_tool_failures=[
            {
                "tool": "get_capture_slice",
                "error_code": "invalid_range",
                "request_shape": {"start_line": "integer", "end_line": "integer"},
                "result_status": "error",
                "capture_id_supplied": True,
                "cursor_supplied": False,
                "duration_ms": 2,
            },
            {
                "tool": "get_capture_slice",
                "error_code": "invalid_range",
                "request_shape": {"start_line": "integer", "end_line": "integer"},
                "result_status": "error",
                "capture_id_supplied": True,
                "cursor_supplied": False,
                "duration_ms": 4,
            },
        ])
        analysis = build_analysis(
            records_payload=records_payload([run]),
            lifecycle_dir=None,
            failure_traces_dir=None,
            timeout_source_payload=None,
        )

        self.assertEqual(analysis["summary"]["validation_failure_events"], 2)
        self.assertEqual(analysis["validation_failure_groups"], [
            {
                "tool": "get_capture_slice",
                "error_code": "invalid_range",
                "request_shape": {"start_line": "integer", "end_line": "integer"},
                "result_status": "error",
                "capture_id_supplied": True,
                "cursor_supplied": False,
                "count": 2,
                "mean_duration_ms": 3,
            }
        ])

    def test_failure_metadata_versions_one_and_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_trace(root, metadata={"schema_version": 1})
            version_one = _metadata_summary(root / "0001-mcp" / "metadata.json")
            write_trace(
                root,
                metadata={
                    "schema_version": 2,
                    "timing": {
                        "started_at": "2026-11-21T00:00:00Z",
                        "ended_at": "2026-11-21T00:15:00Z",
                        "duration_seconds": 900,
                        "timeout_seconds": 900,
                    },
                },
            )
            version_two = _metadata_summary(root / "0001-mcp" / "metadata.json")

        self.assertEqual(version_one, {
            "schema_version": 1,
            "malformed": False,
            "started_at": None,
            "ended_at": None,
            "duration_seconds": None,
            "timeout_seconds": None,
        })
        self.assertEqual(version_two["schema_version"], 2)
        self.assertEqual(version_two["duration_seconds"], 900.0)
        self.assertEqual(version_two["timeout_seconds"], 900.0)

    def test_missing_and_malformed_inputs_are_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lifecycle = root / "001-mcp.jsonl"
            write_lifecycle(lifecycle, ["{not json"])
            write_trace(root, raw_lines=["{also not json"])
            (root / "0001-mcp" / "metadata.json").write_text("not json\n", encoding="utf-8")
            analysis = build_analysis(
                records_payload=records_payload([timeout_run()]),
                lifecycle_dir=root,
                failure_traces_dir=root,
                timeout_source_payload=None,
            )

        self.assertEqual(analysis["inputs"], {
            "lifecycle_files_read": 1,
            "failure_trace_files_read": 1,
            "malformed_lifecycle_lines": 1,
            "malformed_failure_trace_lines": 1,
            "malformed_failure_metadata_files": 1,
        })

    def test_output_is_privacy_safe_and_cli_writes_result(self):
        private_values = [
            "PRIVATE_PROMPT",
            "private command",
            "PRIVATE_TOOL_RESULT",
            "sha256-private-signature",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_trace(
                root,
                metadata={
                    "schema_version": 2,
                    "timing": {
                        "started_at": "2026-11-21T00:00:00Z",
                        "ended_at": "2026-11-21T00:15:00Z",
                        "duration_seconds": 900,
                        "timeout_seconds": 900,
                    },
                },
                items=[
                    (True, {"type": "command_execution", "id": 1, "command": private_values[1]}),
                    (False, {
                        "type": "command_execution",
                        "id": 1,
                        "command": private_values[1],
                        "aggregated_output": private_values[2],
                        "signature_digest": private_values[3],
                    }),
                ],
                raw_lines=[json.dumps({"prompt": private_values[0]})],
            )
            records = records_payload([timeout_run(completed=True)])
            records["runs"][0]["mcp_tool_failures"] = [{
                "tool": "search_capture",
                "error_code": "invalid_top_k",
                "request_shape": {"query": "string"},
                "result_status": "error",
                "capture_id_supplied": False,
                "cursor_supplied": False,
                "duration_ms": 1,
            }]
            records_path = root / "records.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")
            output = root / "analysis.json"
            result = root / "analysis.result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.analyze_agent_failures",
                    "--records", str(records_path),
                    "--lifecycle-dir", str(root),
                    "--failure-traces-dir", str(root),
                    "--output", str(output),
                    "--result", str(result),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            analysis = json.loads(output.read_text(encoding="utf-8"))
            result_payload = json.loads(result.read_text(encoding="utf-8"))
            serialized = json.dumps(analysis) + json.dumps(result_payload["details"])

        self.assertIn('"timeout_classifications": 0', completed.stdout)
        for private_value in private_values:
            self.assertNotIn(private_value, serialized)
        self.assertEqual(result_payload["workload"]["name"], "agent-failure-analysis")
        self.assertEqual(analysis_workload_result(analysis)["details"], analysis)


if __name__ == "__main__":
    unittest.main()
