"""Tests for the privacy-safe agent-level A/B evaluation harness."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_agent_ab
import workload_results as wr
from benchmark_agent_ab import (
    DATA_PATH_BYTE_FIELDS,
    RECORDS_SCHEMA_VERSION,
    build_schedule,
    records_workload_result,
    summarize_records,
    summary_workload_result,
    validate_records,
)


def _records_payload():
    schedule = build_schedule(repetitions=2, seed=17)
    runs = []
    for item in schedule["schedule"]:
        mcp = item["mode"] == "mcp"
        runs.append({
            "task_id": item["task_id"],
            "repetition": item["repetition"],
            "mode": item["mode"],
            "completed": mcp,
            "signal_retrieved": mcp,
            "duration_seconds": 2 if mcp else 3,
            "tool_calls": 2 if mcp else 4,
            "repeated_commands": 0 if mcp else 1,
            "context_bytes": 100 if mcp else 400,
            "peak_rss_bytes": 1000,
        })
    return schedule, {
        "schema_version": 1,
        "benchmark": "agent-ab",
        "task_fixture_version": schedule["task_fixture_version"],
        "protocol": {
            "model_config": "test-model",
            "repository_fixture": "synthetic-repo-v1",
            "environment": "test-environment",
            "reset_policy": "fresh-worktree-per-run",
            "agent_adapter": "test-adapter",
            "embedding_mode": "test",
            "embedding_model": "deterministic-test",
            "embedding_cache": "not-applicable",
        },
        "schedule": schedule,
        "runs": runs,
    }


class TestAgentAbBenchmark(unittest.TestCase):
    def test_schedule_is_reproducible_and_balanced(self):
        first = build_schedule(repetitions=3, seed=17)
        self.assertEqual(first, build_schedule(repetitions=3, seed=17))
        self.assertEqual(len(first["schedule"]), 24)
        self.assertEqual({item["mode"] for item in first["schedule"]}, {"control", "mcp"})
        serialized = json_text(first)
        self.assertNotIn('"prompt":', serialized)
        self.assertNotIn('"commands":', serialized)
        self.assertNotIn('"content":', serialized)

    def test_summary_reports_mode_and_paired_metrics(self):
        schedule, payload = _records_payload()
        summary = summarize_records(payload, schedule)
        self.assertEqual(summary["mode_summaries"]["mcp"]["runs"], 8)
        self.assertEqual(summary["mode_summaries"]["mcp"]["completion_rate"], 1.0)
        self.assertEqual(summary["paired_deltas_mcp_minus_control"]["completed"]["mean"], 1.0)
        self.assertTrue(summary["recommendations"])

    def test_summary_excludes_unavailable_rss_measurements(self):
        schedule, payload = _records_payload()
        for record in payload["runs"]:
            if record["mode"] == "control":
                record["peak_rss_bytes"] = 2000
            elif record["repetition"] == 1:
                record["peak_rss_bytes"] = 0
            else:
                record["peak_rss_bytes"] = 3000

        summary = summarize_records(payload, schedule)
        mcp_rss = summary["mode_summaries"]["mcp"]["peak_rss_bytes"]
        paired_rss = summary["paired_deltas_mcp_minus_control"]["peak_rss_bytes"]
        self.assertTrue(mcp_rss["available"])
        self.assertEqual(mcp_rss["count"], 4)
        self.assertEqual(mcp_rss["mean"], 3000.0)
        self.assertTrue(paired_rss["available"])
        self.assertEqual(paired_rss["count"], 4)
        self.assertEqual(paired_rss["mean"], 1000.0)

        for record in payload["runs"]:
            record["peak_rss_bytes"] = 0
        unavailable = summarize_records(payload, schedule)
        self.assertEqual(
            unavailable["mode_summaries"]["mcp"]["peak_rss_bytes"],
            {"available": False, "count": 0},
        )
        self.assertFalse(unavailable["paired_deltas_mcp_minus_control"]["peak_rss_bytes"]["available"])

    def test_validation_rejects_missing_runs_and_raw_fields(self):
        schedule, payload = _records_payload()
        payload["runs"] = payload["runs"][:-1]
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)

    def test_validation_accepts_legacy_protocol_without_embedding_metadata(self):
        schedule, payload = _records_payload()
        for field in ("embedding_mode", "embedding_model", "embedding_cache"):
            del payload["protocol"][field]
        self.assertEqual(len(validate_records(payload, schedule)), len(payload["runs"]))

    def test_validation_accepts_version_four_records_and_breaks_out_proxy(self):
        schedule, payload = _records_payload()
        payload["records_schema_version"] = 4
        payload["runs"] = [
            {
                "task_id": record["task_id"],
                "repetition": record["repetition"],
                "mode": record["mode"],
                "completed": record["completed"],
                "signal_retrieved": record["signal_retrieved"],
                "duration_seconds": record["duration_seconds"],
                "tool_calls": record["tool_calls"],
                "repeated_commands": record["repeated_commands"],
                "context_bytes_proxy": record["context_bytes"],
                "peak_rss_bytes": record["peak_rss_bytes"],
                "exit_code": 0,
                "failure_reason": None,
                "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_samples": [100] if record["mode"] == "control" else [100, 120],
                "output_token_samples": [20] if record["mode"] == "control" else [20, 24],
                "prompt_bytes_proxy": 40,
                "output_bytes_proxy": 60 if record["mode"] == "mcp" else 360,
            }
            for record in payload["runs"]
        ]
        summary = summarize_records(payload, schedule)
        self.assertIn("context_bytes_proxy", summary["mode_summaries"]["mcp"])
        self.assertEqual(summary["mode_summaries"]["mcp"]["prompt_bytes_proxy"]["mean"], 40.0)
        self.assertEqual(summary["mode_summaries"]["mcp"]["output_bytes_proxy"]["mean"], 60.0)
        self.assertIn("prompt_bytes_proxy", summary["paired_deltas_mcp_minus_control"])
        usage = summary["mode_summaries"]["mcp"]["usage_accounting"]["input_tokens"]
        self.assertEqual(usage["runs_with_multiple_samples"], 8)
        self.assertEqual(usage["observation"], "monotonic_samples_inconclusive")
        self.assertEqual(summary["records_schema_version"], 4)
        _, payload = _records_payload()
        payload["runs"][0]["raw_output"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)

    def test_summary_reports_session_data_path_bytes_for_version_five(self):
        schedule, payload = _records_payload()
        payload["records_schema_version"] = RECORDS_SCHEMA_VERSION
        for record in payload["runs"]:
            record.pop("context_bytes", None)
            record.update({
                "context_bytes_proxy": 100 if record["mode"] == "mcp" else 400,
                "exit_code": 0,
                "failure_reason": None,
                "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
                "input_tokens": None,
                "output_tokens": None,
                "input_token_samples": [],
                "output_token_samples": [],
                "prompt_bytes_proxy": 40,
                "output_bytes_proxy": 60 if record["mode"] == "mcp" else 360,
            })
            for field in DATA_PATH_BYTE_FIELDS:
                record[field] = 0
            if record["mode"] == "mcp":
                record["capture_input_bytes"] = 1000
                record["capture_retained_bytes"] = 600
                record["tool_response_bytes"] = 120
                record["search_response_bytes"] = 80
                record["retrieval_response_bytes"] = 40

        summary = summarize_records(payload, schedule)
        self.assertEqual(
            summary["mode_summaries"]["mcp"]["data_path_bytes"]["capture_input_bytes"]["mean"],
            1000.0,
        )
        self.assertEqual(
            summary["paired_deltas_mcp_minus_control"]["tool_response_bytes"]["mean"],
            120.0,
        )

    def test_build_schedule_rejects_invalid_repetitions(self):
        with self.assertRaises(ValueError):
            build_schedule(repetitions=0)


def json_text(value):
    """Serialize test metadata for a simple privacy assertion."""
    import json

    return json.dumps(value, sort_keys=True)


    def test_summary_workload_result_reports_modes_and_paired_deltas(self):
        schedule, payload = _records_payload()
        summary = summarize_records(payload, schedule)
        result = summary_workload_result(summary)
        self.assertEqual(result["workload"]["name"], "agent-ab")
        self.assertEqual(result["workload"]["kind"], "evaluation")
        self.assertEqual(result["workload"]["parameters"]["protocol"]["agent_adapter"], "test-adapter")
        self.assertEqual([item["id"] for item in result["runs"]], ["control", "mcp", "paired-delta"])
        control, mcp, paired = result["runs"]
        self.assertEqual(control["status"], "failure")
        self.assertEqual(mcp["status"], "success")
        self.assertEqual(mcp["measurements"]["success_rate"], {"unit": "ratio", "value": 1.0, "samples": 8})
        self.assertEqual(mcp["measurements"]["wall_time_seconds"]["mean"], 2.0)
        self.assertEqual(mcp["measurements"]["context_bytes"]["note"], benchmark_agent_ab.CONTEXT_PROXY_NOTE)
        # Version-one records carry no token or data-path fields: they are absent, not zero.
        self.assertEqual(mcp["measurements"]["input_tokens"]["value"], None)
        self.assertEqual(mcp["measurements"]["capture_input_bytes"]["value"], None)
        self.assertEqual(paired["labels"], {"comparison": "mcp_minus_control"})
        self.assertEqual(paired["measurements"]["success_rate"]["mean"], 1.0)
        self.assertEqual(paired["measurements"]["wall_time_seconds"]["mean"], -1.0)
        self.assertEqual(result["details"], summary)

    def test_records_workload_result_keeps_timeouts_and_unavailable_values(self):
        schedule = build_schedule(repetitions=1, seed=3)
        runs = []
        for item in schedule["schedule"]:
            mcp = item["mode"] == "mcp"
            runs.append({
                "task_id": item["task_id"],
                "repetition": item["repetition"],
                "mode": item["mode"],
                "completed": mcp,
                "signal_retrieved": mcp,
                "duration_seconds": 1.5,
                "tool_calls": 3,
                "repeated_commands": 0,
                "context_bytes_proxy": 250,
                "peak_rss_bytes": 4096 if mcp else 0,
                "exit_code": 0 if mcp else 124,
                "failure_reason": None if mcp else "timeout",
                "mcp_tool_calls": 1 if mcp else 0,
                "input_tokens": 100 if mcp else None,
                "output_tokens": 20 if mcp else None,
            })
        payload = {
            "schema_version": 1,
            "benchmark": "agent-ab",
            "records_schema_version": 2,
            "task_fixture_version": schedule["task_fixture_version"],
            "protocol": {
                "model_config": "m", "repository_fixture": "f", "environment": "e",
                "reset_policy": "r", "agent_adapter": "a",
            },
            "runs": runs,
        }
        result = records_workload_result(payload, schedule, producer="custom-runner")
        self.assertEqual(result["workload"]["kind"], "agent-run")
        self.assertEqual(result["workload"]["producer"], "custom-runner")
        self.assertEqual(result["workload"]["producer_schema_version"], 2)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["runs"]), 8)
        control = next(item for item in result["runs"] if item["labels"]["mode"] == "control")
        self.assertEqual(control["status"], "timeout")
        self.assertEqual(control["errors"], ["timeout"])
        self.assertIsNone(control["measurements"]["peak_rss_bytes"]["value"])
        self.assertIsNone(control["measurements"]["input_tokens"]["value"])
        mcp = next(item for item in result["runs"] if item["labels"]["mode"] == "mcp")
        self.assertEqual(mcp["id"], f"{mcp['labels']['task_id']}-r1-mcp")
        self.assertEqual(mcp["status"], "success")
        self.assertEqual(mcp["measurements"]["peak_rss_bytes"]["value"], 4096)
        self.assertEqual(mcp["measurements"]["input_tokens"]["value"], 100)
        self.assertEqual(mcp["measurements"]["wall_time_seconds"]["value"], 1.5)
        self.assertNotIn("details", result)
        self.assertNotIn("prompt_bytes", mcp["measurements"])

    def test_main_result_flag_requires_records_mode(self):
        schedule, payload = _records_payload()
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.json"
            records.write_text(json.dumps(payload), encoding="utf-8")
            argv = ["benchmark_agent_ab.py", "--records", str(records), "--result", "-"]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                benchmark_agent_ab.main()
            result = wr.validate_result(json.loads(stdout.getvalue()))
            self.assertEqual(result["workload"]["name"], "agent-ab")
            self.assertEqual(json.loads(stderr.getvalue())["benchmark"], "agent-ab")

            argv = ["benchmark_agent_ab.py", "--schedule-output", str(Path(directory) / "s.json"), "--result", "-"]
            with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                benchmark_agent_ab.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
