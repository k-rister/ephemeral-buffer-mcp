"""Tests for the Codex-specific agent A/B runner."""

import argparse
import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_codex_agent_ab
import workload_results as wr
from benchmark_agent_ab import TASK_FIXTURE_VERSION, build_schedule, records_workload_result
from benchmark_agent_ab_fixtures import build_manifest
from run_codex_agent_ab import (
    _data_path_bytes,
    _event_metrics,
    _load_manifest,
    _run_codex_process,
    _signal_retrieved,
    _run_one,
    _status_peak_rss_bytes,
    _task_success,
    run_schedule,
)


class TestCodexAgentRunner(unittest.TestCase):
    def test_data_path_bytes_reads_content_free_snapshot(self):
        from benchmark_agent_ab import DATA_PATH_BYTE_FIELDS

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"bytes": {
                "capture_input_bytes": 120,
                "tool_response_bytes": 45,
            }}), encoding="utf-8")
            counters = _data_path_bytes(path)

        self.assertEqual(counters["capture_input_bytes"], 120)
        self.assertEqual(counters["tool_response_bytes"], 45)
        self.assertEqual(counters["socket_response_bytes"], 0)
        self.assertEqual(set(counters), set(DATA_PATH_BYTE_FIELDS))

    def test_codex_timeout_uses_bounded_pipe_drains_after_leader_exit(self):
        class Process:
            pid = 42
            returncode = -9
            stdout = None
            stderr = None

            def __init__(self):
                self.communicate_calls = 0

            def communicate(self, timeout=None):
                self.communicate_calls += 1
                raise subprocess.TimeoutExpired(
                    ["codex"], timeout, output=b"partial", stderr=b"timed out"
                )

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        process = Process()
        with patch("run_codex_agent_ab.subprocess.Popen", return_value=process), \
                patch("run_codex_agent_ab.os.getpgid", return_value=4242), \
                patch("run_codex_agent_ab.os.killpg") as killpg:
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                _run_codex_process(["codex"], cwd=Path("."), timeout=1)

        self.assertEqual(process.communicate_calls, 3)
        self.assertEqual(raised.exception.output, "partial")
        self.assertEqual(killpg.call_args_list[0].args, (4242, 15))
        self.assertEqual(killpg.call_args_list[1].args, (4242, 9))

    def test_codex_interruption_terminates_process_group_and_reaps_leader(self):
        class Process:
            pid = 42
            returncode = -15

            def __init__(self):
                self.wait_calls = []

            def communicate(self, timeout=None):
                raise KeyboardInterrupt

            def wait(self, timeout=None):
                self.wait_calls.append(timeout)
                return self.returncode

            def poll(self):
                return self.returncode

        process = Process()
        with patch("run_codex_agent_ab.subprocess.Popen", return_value=process), \
                patch("run_codex_agent_ab.os.getpgid", return_value=4242), \
                patch("run_codex_agent_ab.os.killpg") as killpg:
            with self.assertRaises(KeyboardInterrupt):
                _run_codex_process(["codex"], cwd=Path("."), timeout=1)

        self.assertEqual(killpg.call_args_list[0].args, (4242, 15))
        self.assertEqual(killpg.call_args_list[1].args, (4242, 9))
        self.assertEqual(process.wait_calls, [1, None])

    def test_timeout_emits_record_and_next_run_can_continue(self):
        item = {"sequence": 1, "repetition": 1, "task_id": "timeout", "mode": "control"}
        next_item = {"sequence": 2, "repetition": 1, "task_id": "after-timeout", "mode": "control"}
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "SUCCESS",
            "success_criteria": {"required_phrases": ["SUCCESS", "task found"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=1,
            allow_mcp_approvals=False,
        )
        timeout = subprocess.TimeoutExpired(["codex"], 1, output=b"partial", stderr=b"timed out")
        successful = subprocess_result(stdout="SUCCESS task found\n", peak_rss_bytes=4321)
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_codex_agent_ab._run_codex_process", side_effect=[timeout, successful]
        ):
            repository = Path(directory) / "repo"
            repository.mkdir()
            timed_out = _run_one(
                item, task, args=args, repository=repository, scratch=Path(directory)
            )
            continued = _run_one(
                next_item, task, args=args, repository=repository, scratch=Path(directory)
            )

        self.assertFalse(timed_out["completed"])
        self.assertEqual(timed_out["failure_reason"], "timeout")
        self.assertEqual(timed_out["exit_code"], None)
        self.assertGreater(timed_out["output_bytes_proxy"], 0)
        self.assertTrue(continued["completed"])
        self.assertTrue(continued["task_success"])
        self.assertEqual(continued["peak_rss_bytes"], 4321)

    def test_status_peak_rss_parser_converts_linux_kib(self):
        self.assertEqual(_status_peak_rss_bytes("Name:\tcodex\nVmHWM:\t1234 kB\n"), 1234 * 1024)
        self.assertEqual(_status_peak_rss_bytes("Name:\tcodex\n"), 0)

    def test_event_metrics_counts_tools_and_duplicate_commands(self):
        output = "\n".join([
            json.dumps({"type": "response.output_item.done", "item": {"type": "function_call", "command": "pytest"}}),
            json.dumps({"type": "command_execution", "command": "pytest"}),
            json.dumps({"type": "command_execution", "command": "pytest"}),
        ])
        self.assertEqual(_event_metrics(output), (3, 0, 2, None, None, [], []))

    def test_event_metrics_deduplicates_started_and_completed_snapshots(self):
        output = "\n".join([
            json.dumps({
                "type": "item.started",
                "item": {"id": "call-1", "type": "function_call", "command": "pytest"},
            }),
            json.dumps({
                "type": "item.completed",
                "item": {"item_id": "call-1", "type": "function_call", "command": "pytest"},
            }),
        ])

        self.assertEqual(_event_metrics(output), (1, 0, 0, None, None, [], []))

    def test_event_metrics_keeps_distinct_ids_for_repeated_commands(self):
        output = "\n".join([
            json.dumps({"type": "command_execution", "id": "call-1", "command": "pytest"}),
            json.dumps({"type": "command_execution", "id": "call-2", "command": "pytest"}),
        ])

        self.assertEqual(_event_metrics(output), (2, 0, 1, None, None, [], []))

    def test_event_metrics_merges_command_from_partial_lifecycle_snapshot(self):
        output = "\n".join([
            json.dumps({"type": "item.started", "item": {"id": "call-1", "type": "function_call"}}),
            json.dumps({"type": "item.completed", "item": {"id": "call-1", "type": "function_call", "command": "pytest"}}),
            json.dumps({"type": "command_execution", "id": "call-2", "command": "pytest"}),
        ])

        self.assertEqual(_event_metrics(output), (2, 0, 1, None, None, [], []))

    def test_event_metrics_deduplicates_mcp_snapshots_with_call_id(self):
        output = "\n".join([
            json.dumps({"type": "mcp_tool_call", "call_id": "mcp-1"}),
            json.dumps({"type": "mcp_tool_call.completed", "call_id": "mcp-1"}),
        ])

        self.assertEqual(_event_metrics(output), (1, 1, 0, None, None, [], []))

    def test_event_metrics_extracts_mcp_calls_and_usage(self):
        output = "\n".join([
            json.dumps({"type": "mcp_tool_call", "usage": {"input_tokens": 120, "output_tokens": 9}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 140, "output_tokens": 12}}),
        ])
        self.assertEqual(_event_metrics(output), (1, 1, 0, 140, 12, [120, 140], [9, 12]))

    def test_signal_retrieval_ignores_request_only_marker(self):
        output = json.dumps({
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "grep TARGETED_SIGNAL missing-file",
                "status": "failed",
                "exit_code": 2,
            },
        })

        self.assertFalse(_signal_retrieved(output, "TARGETED_SIGNAL"))

    def test_signal_retrieval_accepts_successful_command_result(self):
        output = json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "grep TARGETED_SIGNAL fixture.txt",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": "inspection result: TARGETED_SIGNAL found",
            },
        })

        self.assertTrue(_signal_retrieved(output, "TARGETED_SIGNAL"))

    def test_signal_retrieval_accepts_nested_mcp_result_content(self):
        output = json.dumps({
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "status": "completed",
                "result": {
                    "content": [
                        {"type": "text", "text": "successful lookup found TARGETED_SIGNAL"}
                    ]
                },
            },
        })

        self.assertTrue(_signal_retrieved(output, "TARGETED_SIGNAL"))

    def test_signal_retrieval_ignores_nested_mcp_error_result(self):
        output = json.dumps({
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "status": "completed",
                "result": {
                    "isError": True,
                    "content": [
                        {"type": "text", "text": "lookup failed: TARGETED_SIGNAL"}
                    ],
                },
            },
        })

        self.assertFalse(_signal_retrieved(output, "TARGETED_SIGNAL"))

    def test_signal_retrieval_ignores_marker_in_failed_result(self):
        output = json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "status": "failed",
                "exit_code": 1,
                "aggregated_output": "TARGETED_SIGNAL was not found",
            },
        })

        self.assertFalse(_signal_retrieved(output, "TARGETED_SIGNAL"))

    def test_signal_retrieval_accepts_agent_answer_and_plain_answer_lines(self):
        agent_answer = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "The result is TARGETED_SIGNAL."},
        })
        self.assertTrue(_signal_retrieved(agent_answer, "TARGETED_SIGNAL"))
        self.assertTrue(_signal_retrieved("TARGETED_SIGNAL\n", "TARGETED_SIGNAL"))

    def test_signal_retrieval_ignores_marker_in_refusal(self):
        for text in (
            "I could not find TEST_FAILURE_SIGNAL and cannot answer the task.",
            "I must decline to answer, but the marker is TEST_FAILURE_SIGNAL.",
        ):
            refusal = json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": text},
            })
            self.assertFalse(_signal_retrieved(refusal, "TEST_FAILURE_SIGNAL"))

    def test_manifest_requires_all_schedule_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps({
                "fixture_version": TASK_FIXTURE_VERSION,
                "tasks": {
                    "targeted-inspection": {
                        "prompt": "inspect",
                        "signal_marker": "SIGNAL",
                        "success_criteria": {
                            "description": "report the signal and location",
                            "required_phrases": ["SIGNAL", "line 32"],
                        },
                    },
                },
            }))
            with self.assertRaises(ValueError):
                _load_manifest(path)

    def test_manifest_rejects_criteria_that_only_repeat_the_marker(self):
        manifest = copy.deepcopy(build_manifest())
        manifest["tasks"]["noisy-test-failure"]["success_criteria"]["required_phrases"] = [
            "TEST_FAILURE_SIGNAL", " TEST_FAILURE_SIGNAL "
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate required_phrases"):
                _load_manifest(path)

    def test_run_one_is_metadata_only_and_uses_requested_model(self):
        schedule = build_schedule(repetitions=1, seed=3)
        item = schedule["schedule"][0]
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "SUCCESS",
            "success_criteria": {"required_phrases": ["SUCCESS", "line 12"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        result = subprocess_result(
            stdout="\n".join([
                json.dumps({"type": "command_execution", "command": "find"}),
                json.dumps({
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "The result is SUCCESS on line 12.",
                    },
                }),
            ]),
            peak_rss_bytes=1234,
        )
        with tempfile.TemporaryDirectory() as directory, patch("run_codex_agent_ab._run_codex_process", return_value=result) as run:
            repository = Path(directory) / "repo"
            repository.mkdir()
            (repository / "fixture.txt").write_text("fixture")
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))
        self.assertTrue(output["completed"])
        self.assertTrue(output["task_success"])
        self.assertEqual(output["criterion_passes"], [True, True])
        self.assertTrue(output["signal_retrieved"])
        self.assertNotIn("SUCCESS", json.dumps(output))
        command = run.call_args.args[0]
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn("--ephemeral", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertEqual(output["mcp_tool_calls"], 0)
        self.assertIsNone(output["input_tokens"])
        self.assertEqual(output["peak_rss_bytes"], 1234)

    def test_task_scoring_separates_positive_partial_incorrect_refused_and_timeout(self):
        item = {
            "sequence": 1,
            "repetition": 1,
            "task_id": "noisy-test-failure",
            "mode": "control",
        }
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "TEST_FAILURE_SIGNAL",
            "success_criteria": {
                "description": "report the failed test, signal, and assertion",
                "required_phrases": [
                    "test_case_1379",
                    "TEST_FAILURE_SIGNAL",
                    "expected status=ready, got status=stalled",
                ],
            },
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=1,
            allow_mcp_approvals=False,
        )

        def score(answer=None, error=None):
            with tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / "repo"
                repository.mkdir()
                result = subprocess_result(stdout=json.dumps({
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": answer or ""},
                }))
                patch_options = {"side_effect": error} if error is not None else {"return_value": result}
                with patch("run_codex_agent_ab._run_codex_process", **patch_options):
                    return _run_one(
                        item,
                        task,
                        args=args,
                        repository=repository,
                        scratch=Path(directory),
                    )

        positive = score(
            "test_case_1379 failed: TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        self.assertTrue(positive["completed"])
        self.assertTrue(positive["task_success"])
        self.assertEqual(positive["criterion_passes"], [True, True, True])
        self.assertNotIn("test_case_1379", json.dumps(positive))
        self.assertTrue(positive["signal_retrieved"])

        qualified = score(
            "I can't verify this independently, but the failed test is "
            "test_case_1379: TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        self.assertTrue(qualified["task_success"])
        self.assertTrue(qualified["signal_retrieved"])

        idiomatic = score(
            "I can't help but report test_case_1379: TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        self.assertTrue(idiomatic["task_success"])
        self.assertTrue(idiomatic["signal_retrieved"])

        limitation = score(
            "The test runner could not execute the full suite, but test_case_1379 "
            "reported TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        self.assertTrue(limitation["task_success"])
        self.assertTrue(limitation["signal_retrieved"])

        partial = score("The test failure is TEST_FAILURE_SIGNAL.")
        self.assertTrue(partial["completed"])
        self.assertFalse(partial["task_success"])
        self.assertEqual(partial["criterion_passes"], [False, True, False])
        self.assertTrue(partial["signal_retrieved"])

        incorrect = score(
            "test_case_1378 failed: TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        self.assertFalse(incorrect["task_success"])
        self.assertEqual(incorrect["criterion_passes"], [False, True, True])
        self.assertTrue(incorrect["signal_retrieved"])

        refused = score(
            "I could not find TEST_FAILURE_SIGNAL and cannot answer the task."
        )
        self.assertTrue(refused["completed"])
        self.assertFalse(refused["task_success"])
        self.assertEqual(refused["criterion_passes"], [False, False, False])
        self.assertFalse(refused["signal_retrieved"])
        self.assertEqual(refused["failure_reason"], "task_success_criteria_not_met")

        empty = score("")
        self.assertTrue(empty["completed"])
        self.assertFalse(empty["task_success"])
        self.assertEqual(empty["criterion_passes"], [False, False, False])

        complete_evidence = (
            "test_case_1379: TEST_FAILURE_SIGNAL; AssertionError: "
            "expected status=ready, got status=stalled"
        )
        for refusal_text in (
            "I can't do that.",
            "I won't do that.",
            "I will not be able to run the test.",
            "I cannot fulfill your request.",
            "I cannot fulfil your request.",
        ):
            refused_with_evidence = score(f"{refusal_text} {complete_evidence}")
            self.assertTrue(refused_with_evidence["completed"])
            self.assertFalse(refused_with_evidence["task_success"])
            self.assertEqual(refused_with_evidence["criterion_passes"], [False, False, False])
            self.assertFalse(refused_with_evidence["signal_retrieved"])

        timed_out = score(error=subprocess.TimeoutExpired("codex", 1, output="partial"))
        self.assertFalse(timed_out["completed"])
        self.assertFalse(timed_out["task_success"])
        self.assertEqual(timed_out["criterion_passes"], [False, False, False])
        self.assertEqual(timed_out["failure_reason"], "timeout")

    def test_task_scoring_requires_complete_phrase_boundaries(self):
        targeted_criteria = {
            "required_phrases": ["TARGETED_SIGNAL", "line 32"],
        }
        self.assertTrue(
            _task_success(
                "TARGETED_SIGNAL appears on line 32.",
                targeted_criteria,
                "TARGETED_SIGNAL",
            )
        )
        self.assertFalse(
            _task_success(
                "TARGETED_SIGNAL appears on line 320.",
                targeted_criteria,
                "TARGETED_SIGNAL",
            )
        )

        test_criteria = {
            "required_phrases": ["TEST_FAILURE_SIGNAL", "test_case_1379"],
        }
        self.assertFalse(
            _task_success(
                "TEST_FAILURE_SIGNAL is in test_case_13790.",
                test_criteria,
                "TEST_FAILURE_SIGNAL",
            )
        )

        build_criteria = {
            "required_phrases": ["BUILD_FAILURE_SIGNAL", "src/parser.c", "917"],
        }
        for answer in (
            "BUILD_FAILURE_SIGNAL at src/parser.c:917",
            "BUILD_FAILURE_SIGNAL at src/parser.c, line 917",
        ):
            self.assertTrue(_task_success(answer, build_criteria, "BUILD_FAILURE_SIGNAL"))
        self.assertFalse(
            _task_success(
                "BUILD_FAILURE_SIGNAL at src/parser.c, line 918",
                build_criteria,
                "BUILD_FAILURE_SIGNAL",
            )
        )

    def test_run_one_ignores_stderr_for_events_and_signal_markers(self):
        item = {"sequence": 1, "repetition": 1, "task_id": "stderr-marker", "mode": "control"}
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "TARGETED_SIGNAL",
            "success_criteria": {"required_phrases": ["TARGETED_SIGNAL", "line 32"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        stderr = json.dumps({
            "type": "command_execution",
            "command": "grep TARGETED_SIGNAL missing-file",
            "aggregated_output": "TARGETED_SIGNAL diagnostic",
        })
        completed = subprocess_result(stdout="normal response\n", stderr=stderr)
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_codex_agent_ab._run_codex_process", return_value=completed
        ):
            repository = Path(directory) / "repo"
            repository.mkdir()
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))

        self.assertFalse(output["signal_retrieved"])
        self.assertEqual(output["tool_calls"], 0)
        self.assertEqual(output["output_bytes_proxy"], len((completed.stdout + stderr).encode()))

    def test_mcp_approval_mode_uses_automatic_review_and_isolated_write_sandbox(self):
        from run_codex_agent_ab import _codex_command

        command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="mcp", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script="/tmp/server.py",
            allow_mcp_approvals=True, sandbox="read-only", timeout=10,
            mcp_env={"EPHEMERAL_TEST_EMBEDDINGS": "1"},
        )
        self.assertIn("--approve-for-me", command)
        self.assertNotIn("--ask-for-approval", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")

        env_config = next(item for item in command if item.startswith("mcp_servers.ephemeral-buffer.env="))
        self.assertIn('EPHEMERAL_TEST_EMBEDDINGS="1"', env_config)

    def test_mcp_forwards_embedding_model_and_cache_configuration(self):
        from run_codex_agent_ab import _codex_command

        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_TEST_EMBEDDINGS": "0",
                "EPHEMERAL_EMBEDDING_MODEL": "custom/model",
                "EPHEMERAL_FASTEMBED_CACHE_DIR": "/tmp/embedding-cache",
            },
            clear=False,
        ):
            command = _codex_command(
                codex="codex", model="gpt-5.6-luna", mode="mcp", fixture=Path("/tmp/fixture"),
                mcp_python="python", mcp_module="server", mcp_server_script="/tmp/server.py",
                allow_mcp_approvals=False, sandbox="read-only", timeout=10,
                mcp_env={
                    name: os.environ[name]
                    for name in (
                        "EPHEMERAL_TEST_EMBEDDINGS",
                        "EPHEMERAL_EMBEDDING_MODEL",
                        "EPHEMERAL_FASTEMBED_CACHE_DIR",
                    )
                },
            )
        env_config = next(item for item in command if item.startswith("mcp_servers.ephemeral-buffer.env="))
        self.assertIn('EPHEMERAL_EMBEDDING_MODEL="custom/model"', env_config)
        self.assertIn('EPHEMERAL_FASTEMBED_CACHE_DIR="/tmp/embedding-cache"', env_config)

    def test_codex_command_uses_current_exec_subcommand_syntax(self):
        from run_codex_agent_ab import _codex_command

        command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="control", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script=None,
            allow_mcp_approvals=False, sandbox="read-only", timeout=10,
        )

        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertNotIn("--ask-for-approval", command)

    def test_mcp_approval_flag_is_global_before_exec_subcommand(self):
        from run_codex_agent_ab import _codex_command

        command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="mcp", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script="/tmp/server.py",
            allow_mcp_approvals=True, sandbox="read-only", timeout=10,
        )

        self.assertEqual(command[:3], ["codex", "--approve-for-me", "exec"])

    def test_required_mcp_usage_marks_mcp_bypass_as_incomplete(self):
        item = {"sequence": 1, "repetition": 1, "task_id": "noisy-test-failure", "mode": "mcp"}
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "SUCCESS",
            "success_criteria": {"required_phrases": ["SUCCESS", "expected answer"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=True, require_mcp_calls=True,
        )
        result = subprocess_result(stdout="SUCCESS\n")
        with tempfile.TemporaryDirectory() as directory, patch("run_codex_agent_ab._run_codex_process", return_value=result) as run:
            repository = Path(directory) / "repo"
            repository.mkdir()
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))

        self.assertTrue(output["completed"])
        self.assertFalse(output["task_success"])
        self.assertEqual(output["failure_reason"], "mcp_not_used")
        self.assertIn("must use the configured ephemeral-buffer MCP tools", run.call_args.args[0][-1])

    def test_run_schedule_writes_balanced_records(self):
        schedule = build_schedule(repetitions=1, seed=3)
        manifest = {
            task["id"]: {
                "prompt": task["id"],
                "signal_marker": "ok",
                "success_criteria": {
                    "description": "report the signal and a deterministic result",
                    "required_phrases": ["ok", "expected answer"],
                },
            }
            for task in schedule["tasks"]
        }
        args = argparse.Namespace(
            repository=".", repository_fixture="fixture", environment="test", model="gpt-5.6-luna",
            codex="codex", mcp_python="python", mcp_module="server", mcp_server_script=None,
            sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
            dry_run=False,
        )
        result = subprocess_result(stdout=json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "ok is the expected answer"},
        }) + "\n")
        with patch("run_codex_agent_ab._run_codex_process", return_value=result), patch.dict(
            os.environ,
            {
                "EPHEMERAL_TEST_EMBEDDINGS": "",
                "EPHEMERAL_FASTEMBED_CACHE_DIR": "",
                "EPHEMERAL_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5-fp32",
            },
            clear=False,
        ):
            payload = run_schedule(schedule, manifest, args)
        self.assertEqual(len(payload["runs"]), 8)
        self.assertTrue(all(item["criterion_passes"] == [True, True] for item in payload["runs"]))
        self.assertEqual(payload["records_schema_version"], 8)
        self.assertEqual(payload["protocol"]["agent_adapter"], "codex-cli")
        result = records_workload_result(payload, producer="run_codex_agent_ab.py")
        self.assertEqual(result["workload"]["producer"], "run_codex_agent_ab.py")
        self.assertEqual(result["workload"]["parameters"]["protocol"]["agent_adapter"], "codex-cli")
        self.assertEqual(len(result["runs"]), 8)
        self.assertTrue(all(item["status"] == "success" for item in result["runs"]))
        self.assertIn("capture_input_bytes", result["runs"][0]["measurements"])
        self.assertEqual(payload["protocol"]["embedding_mode"], "fastembed")
        self.assertEqual(payload["protocol"]["embedding_model"], "BAAI/bge-small-en-v1.5-fp32")
        self.assertEqual(payload["protocol"]["embedding_cache"], "default")


def subprocess_result(**kwargs):
    return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": "", **kwargs})()


class TestCodexAgentWorkloadResult(unittest.TestCase):
    def test_main_emits_a_workload_result_next_to_the_records(self):
        schedule = build_schedule(repetitions=1, seed=3)
        manifest = build_manifest()
        completed = subprocess_result(stdout=json.dumps({"type": "completed"}) + "\nok\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
            (root / "tasks.json").write_text(json.dumps(manifest), encoding="utf-8")
            argv = [
                "run_codex_agent_ab.py", "--schedule", str(root / "schedule.json"), "--tasks", str(root / "tasks.json"),
                "--output", str(root / "records.json"), "--repository", ".", "--result", str(root / "result.json"),
            ]
            with patch("run_codex_agent_ab._run_codex_process", return_value=completed), patch(
                "sys.argv", argv
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                run_codex_agent_ab.main()
            result = wr.load_result(root / "result.json")
            self.assertEqual(json.loads((root / "records.json").read_text(encoding="utf-8"))["benchmark"], "agent-ab")
        self.assertEqual(result["workload"]["kind"], "agent-run")
        self.assertEqual(len(result["runs"]), 8)
        self.assertEqual(json.loads(stdout.getvalue())["runs"], 8)


if __name__ == "__main__":
    unittest.main()
