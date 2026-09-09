"""Tests for aggregate agent A/B baselines."""

import copy
import unittest

from benchmark_agent_ab import build_schedule, summarize_records
from benchmark_agent_ab_baseline import build_baseline, compare_summary, load_baseline_dict


def summary_fixture():
    schedule = build_schedule(repetitions=2, seed=17)
    runs = []
    for item in schedule["schedule"]:
        mcp = item["mode"] == "mcp"
        runs.append({
            "task_id": item["task_id"], "repetition": item["repetition"], "mode": item["mode"],
            "completed": True, "signal_retrieved": True, "duration_seconds": 2 if mcp else 1,
            "tool_calls": 3, "repeated_commands": 0, "context_bytes_proxy": 100 if mcp else 80,
            "peak_rss_bytes": 1000, "exit_code": 0, "failure_reason": None,
            "mcp_tool_calls": 2 if mcp else 0, "input_tokens": 100, "output_tokens": 20,
        })
    payload = {
        "schema_version": 1, "benchmark": "agent-ab", "records_schema_version": 2,
        "task_fixture_version": schedule["task_fixture_version"], "protocol": {
            "model_config": "gpt-5.6-luna", "repository_fixture": "fixture-v1",
            "environment": "test", "reset_policy": "fresh-copy-per-run", "agent_adapter": "codex-cli",
        }, "schedule": schedule, "runs": runs,
    }
    return summarize_records(payload, schedule)


class TestAgentAbBaseline(unittest.TestCase):
    def test_build_baseline_is_aggregate_only(self):
        baseline = build_baseline(summary_fixture())
        self.assertEqual(baseline["benchmark"], "agent-ab-baseline")
        self.assertIn("duration_seconds", baseline["metrics"]["mcp"])
        self.assertNotIn("runs", baseline)
        load_baseline_dict(baseline)

    def test_compare_passes_same_summary(self):
        summary = summary_fixture()
        result = compare_summary(summary, build_baseline(summary))
        self.assertTrue(result["passed"])
        self.assertFalse(result["regressions"])

    def test_compare_detects_gated_latency_regression(self):
        summary = summary_fixture()
        baseline = build_baseline(summary)
        changed = copy.deepcopy(summary)
        changed["mode_summaries"]["mcp"]["duration_seconds"]["mean"] = 4.0
        result = compare_summary(changed, baseline)
        self.assertFalse(result["passed"])
        self.assertIn("mcp.duration_seconds exceeded its tolerated change", result["regressions"])

    def test_unavailable_metrics_are_not_regressions(self):
        summary = summary_fixture()
        baseline = build_baseline(summary)
        summary["mode_summaries"]["mcp"]["input_tokens"] = {"available": False, "count": 0}
        baseline["metrics"]["mcp"]["input_tokens"] = None
        result = compare_summary(summary, baseline)
        self.assertTrue(result["passed"])
        self.assertFalse(result["comparisons"]["mcp"]["input_tokens"]["available"])


if __name__ == "__main__":
    unittest.main()
