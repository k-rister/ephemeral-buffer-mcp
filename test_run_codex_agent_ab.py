"""Tests for the Codex-specific agent A/B runner."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_agent_ab import build_schedule
from run_codex_agent_ab import _event_metrics, _load_manifest, _run_one, run_schedule


class TestCodexAgentRunner(unittest.TestCase):
    def test_event_metrics_counts_tools_and_duplicate_commands(self):
        output = "\n".join([
            json.dumps({"type": "response.output_item.done", "item": {"type": "function_call", "command": "pytest"}}),
            json.dumps({"type": "command_execution", "command": "pytest"}),
            json.dumps({"type": "command_execution", "command": "pytest"}),
        ])
        self.assertEqual(_event_metrics(output), (3, 0, 2, None, None))

    def test_event_metrics_extracts_mcp_calls_and_usage(self):
        output = "\n".join([
            json.dumps({"type": "mcp_tool_call", "usage": {"input_tokens": 120, "output_tokens": 9}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 140, "output_tokens": 12}}),
        ])
        self.assertEqual(_event_metrics(output), (1, 1, 0, 140, 12))

    def test_manifest_requires_all_schedule_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps({"tasks": {"targeted-inspection": {"prompt": "inspect"}}}))
            with self.assertRaises(ValueError):
                _load_manifest(path)

    def test_run_one_is_metadata_only_and_uses_requested_model(self):
        schedule = build_schedule(repetitions=1, seed=3)
        item = schedule["schedule"][0]
        task = {"prompt": "inspect the fixture", "signal_marker": "SUCCESS"}
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
        )
        result = subprocess_result(stdout=json.dumps({"type": "command_execution", "command": "find"}) + "\nSUCCESS\n")
        with tempfile.TemporaryDirectory() as directory, patch("run_codex_agent_ab.subprocess.run", return_value=result) as run:
            repository = Path(directory) / "repo"
            repository.mkdir()
            (repository / "fixture.txt").write_text("fixture")
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))
        self.assertTrue(output["completed"])
        self.assertTrue(output["signal_retrieved"])
        self.assertNotIn("SUCCESS", json.dumps(output))
        command = run.call_args.args[0]
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn("--ephemeral", command)
        self.assertEqual(output["mcp_tool_calls"], 0)
        self.assertIsNone(output["input_tokens"])

    def test_run_schedule_writes_balanced_records(self):
        schedule = build_schedule(repetitions=1, seed=3)
        manifest = {
            task["id"]: {"prompt": task["id"], "signal_marker": "ok"}
            for task in schedule["tasks"]
        }
        args = argparse.Namespace(
            repository=".", repository_fixture="fixture", environment="test", model="gpt-5.6-luna",
            codex="codex", mcp_python="python", mcp_module="server", mcp_server_script=None,
            sandbox="read-only", timeout=10,
            dry_run=False,
        )
        result = subprocess_result(stdout=json.dumps({"type": "completed"}) + "\nok\n")
        with patch("run_codex_agent_ab.subprocess.run", return_value=result):
            payload = run_schedule(schedule, manifest, args)
        self.assertEqual(len(payload["runs"]), 8)
        self.assertEqual(payload["protocol"]["agent_adapter"], "codex-cli")


def subprocess_result(**kwargs):
    return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": "", **kwargs})()


if __name__ == "__main__":
    unittest.main()
