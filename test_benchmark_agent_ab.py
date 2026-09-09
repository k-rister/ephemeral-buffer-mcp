"""Tests for the privacy-safe agent-level A/B evaluation harness."""

import unittest

from benchmark_agent_ab import build_schedule, summarize_records, validate_records


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

    def test_validation_rejects_missing_runs_and_raw_fields(self):
        schedule, payload = _records_payload()
        payload["runs"] = payload["runs"][:-1]
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)
        _, payload = _records_payload()
        payload["runs"][0]["raw_output"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)

    def test_build_schedule_rejects_invalid_repetitions(self):
        with self.assertRaises(ValueError):
            build_schedule(repetitions=0)


def json_text(value):
    """Serialize test metadata for a simple privacy assertion."""
    import json

    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
