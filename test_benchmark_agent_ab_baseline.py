"""Tests for aggregate agent A/B baselines."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from benchmark_agent_ab import DATA_PATH_BYTE_FIELDS, RECORDS_SCHEMA_VERSION, build_schedule, summarize_records
from benchmark_agent_ab_baseline import build_baseline, compare_summary, load_baseline, load_baseline_dict, main


def summary_fixture():
    schedule = build_schedule(repetitions=2, seed=17)
    runs = []
    for item in schedule["schedule"]:
        mcp = item["mode"] == "mcp"
        record = {
            "task_id": item["task_id"], "repetition": item["repetition"], "mode": item["mode"],
            "completed": True, "task_success": mcp, "signal_retrieved": mcp,
            "criterion_passes": [mcp, mcp],
            "duration_seconds": 2 if mcp else 1,
            "tool_calls": 3, "repeated_commands": 0, "context_bytes_proxy": 100 if mcp else 80,
            "peak_rss_bytes": 1000, "exit_code": 0, "failure_reason": None,
            "mcp_tool_calls": 2 if mcp else 0, "input_tokens": 100, "output_tokens": 20,
            "input_token_samples": [100], "output_token_samples": [20],
            "prompt_bytes_proxy": 40, "output_bytes_proxy": 60 if mcp else 40,
        }
        record.update({field: 0 for field in DATA_PATH_BYTE_FIELDS})
        runs.append(record)
    payload = {
        "schema_version": 1, "benchmark": "agent-ab", "records_schema_version": RECORDS_SCHEMA_VERSION,
        "task_fixture_version": schedule["task_fixture_version"], "protocol": {
            "model_config": "gpt-5.6-luna", "repository_fixture": "fixture-v1",
            "environment": "test", "reset_policy": "fresh-copy-per-run", "agent_adapter": "codex-cli",
            "embedding_mode": "test", "embedding_model": "deterministic-test",
            "embedding_cache": "not-applicable",
        }, "schedule": schedule, "runs": runs,
    }
    return summarize_records(payload, schedule)


class TestAgentAbBaseline(unittest.TestCase):
    def assert_threshold_rejected_by_both_loaders(self, summary, baseline, message):
        with self.assertRaisesRegex(ValueError, message):
            load_baseline_dict(copy.deepcopy(baseline))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_baseline(path)

        with self.assertRaisesRegex(ValueError, message):
            compare_summary(summary, copy.deepcopy(baseline))

    def test_build_baseline_is_aggregate_only(self):
        baseline = build_baseline(summary_fixture())
        self.assertEqual(baseline["benchmark"], "agent-ab-baseline")
        self.assertIn("duration_seconds", baseline["metrics"]["mcp"])
        self.assertEqual(baseline["metrics"]["mcp"]["task_success_rate"], 1.0)
        self.assertNotIn("runs", baseline)
        load_baseline_dict(baseline)

    def test_build_baseline_rejects_invalid_threshold_direction_and_gate(self):
        for override, message in (
            ({"direction": "higer"}, "completion_rate.direction must be one of"),
            ({"gate": "false"}, "completion_rate.gate must be a boolean"),
            ({"gate": 1}, "completion_rate.gate must be a boolean"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                build_baseline(summary_fixture(), thresholds={"completion_rate": override})

    def test_misspelled_threshold_direction_is_rejected_by_file_and_memory_paths(self):
        summary = summary_fixture()
        baseline = build_baseline(summary)
        baseline["thresholds"]["completion_rate"]["direction"] = "higer"
        changed = copy.deepcopy(summary)
        changed["mode_summaries"]["mcp"]["completion_rate"] = 0.0

        self.assert_threshold_rejected_by_both_loaders(
            changed,
            baseline,
            "completion_rate.direction must be one of higher, lower, informational",
        )

    def test_nonboolean_threshold_gate_is_rejected_by_file_and_memory_paths(self):
        summary = summary_fixture()
        for gate in ("false", 1):
            baseline = build_baseline(summary)
            baseline["thresholds"]["completion_rate"]["gate"] = gate
            with self.subTest(gate=gate):
                self.assert_threshold_rejected_by_both_loaders(
                    summary,
                    baseline,
                    "completion_rate.gate must be a boolean",
                )

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

    def test_compare_rejects_embedding_configuration_change(self):
        summary = summary_fixture()
        baseline = build_baseline(summary)
        changed = copy.deepcopy(summary)
        changed["protocol"]["embedding_mode"] = "fastembed"
        result = compare_summary(changed, baseline)
        self.assertFalse(result["passed"])
        self.assertIn("protocol.embedding_mode changed", result["regressions"][0])

    def test_build_baseline_accepts_legacy_summary_without_embedding_metadata(self):
        summary = summary_fixture()
        for field in ("embedding_mode", "embedding_model", "embedding_cache"):
            del summary["protocol"][field]
        baseline = build_baseline(summary)
        self.assertNotIn("embedding_mode", baseline["protocol"])

    def test_unavailable_metrics_are_not_regressions(self):
        summary = summary_fixture()
        baseline = build_baseline(summary)
        summary["mode_summaries"]["mcp"]["input_tokens"] = {"available": False, "count": 0}
        baseline["metrics"]["mcp"]["input_tokens"] = None
        result = compare_summary(summary, baseline)
        self.assertTrue(result["passed"])
        self.assertFalse(result["comparisons"]["mcp"]["input_tokens"]["available"])

    def test_legacy_summaries_keep_objective_outcomes_unavailable(self):
        legacy = summary_fixture()
        legacy["records_schema_version"] = 5

        legacy_baseline = build_baseline(legacy)
        for mode in ("control", "mcp"):
            self.assertIsNone(legacy_baseline["metrics"][mode]["task_success_rate"])
            self.assertIsNone(legacy_baseline["metrics"][mode]["signal_retrieval_rate"])

        current_baseline = build_baseline(summary_fixture())
        result = compare_summary(legacy, current_baseline)
        for mode in ("control", "mcp"):
            for metric in ("task_success_rate", "signal_retrieval_rate"):
                comparison = result["comparisons"][mode][metric]
                self.assertIsNone(comparison["current"])
                self.assertFalse(comparison["available"])

    def test_version_six_keeps_task_outcomes_available_without_phrase_scores(self):
        summary = summary_fixture()
        summary["records_schema_version"] = 6
        summary.pop("criterion_summaries_by_mode")
        summary.pop("paired_criterion_deltas_mcp_minus_control")

        baseline = build_baseline(summary)
        self.assertEqual(baseline["metrics"]["control"]["task_success_rate"], 0.0)
        self.assertEqual(baseline["metrics"]["mcp"]["task_success_rate"], 1.0)
        self.assertEqual(baseline["metrics"]["mcp"]["signal_retrieval_rate"], 1.0)

    def test_checked_in_legacy_baseline_does_not_claim_objective_scores(self):
        baseline = load_baseline(Path(__file__).with_name("benchmark_agent_ab_baseline.json"))
        for mode in ("control", "mcp"):
            self.assertIsNone(baseline["metrics"][mode]["task_success_rate"])
            self.assertIsNone(baseline["metrics"][mode]["signal_retrieval_rate"])

    def test_create_baseline_rejects_fail_on_regression(self):
        stderr = io.StringIO()
        argv = [
            "benchmark_agent_ab_baseline.py",
            "--summary", "summary.json",
            "--create-baseline",
            "--output", "baseline.json",
            "--fail-on-regression",
        ]
        with patch.object(sys, "argv", argv), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --baseline comparison", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
