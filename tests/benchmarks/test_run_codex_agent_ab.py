"""Tests for the Codex-specific agent A/B runner."""

import argparse
import copy
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmarks.run_codex_agent_ab as run_codex_agent_ab
import benchmarks.workload_results as wr
from benchmarks.benchmark_agent_ab import TASK_FIXTURE_VERSION, build_schedule, records_workload_result
from benchmarks.benchmark_agent_ab_fixtures import build_manifest
from benchmarks.benchmark_agent_ab_repository_fixture import build_manifest as build_repository_manifest
from benchmarks.run_codex_agent_ab import (
    _data_path_bytes,
    _event_metrics,
    _criterion_text_hits,
    _line_reference_diagnostics,
    _load_manifest,
    _mcp_result_criterion_exposure,
    _run_codex_process,
    _signal_retrieved,
    _run_one,
    _status_peak_rss_bytes,
    _task_success,
    run_schedule,
)


class TestCodexAgentRunner(unittest.TestCase):
    def test_runner_rejects_zero_and_unplanned_repetition_caps_before_invocation(self):
        schedule = build_schedule(seed=47, powered=True, design={"selected_repetitions": 4})
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
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                repository=".", repository_fixture="fixture", environment="test", model="gpt-5.6-luna",
                codex="codex", mcp_python="python", mcp_module="server", mcp_server_script=None,
                sandbox="read-only", timeout=10, allow_mcp_approvals=False,
                dry_run=False, output=Path(directory) / "records.json", max_repetitions=0, resume=False,
            )
            with patch("benchmarks.run_codex_agent_ab._run_codex_process") as runner:
                with self.assertRaisesRegex(ValueError, "max_repetitions"):
                    run_schedule(schedule, manifest, args)
                runner.assert_not_called()

                args.max_repetitions = 1
                with self.assertRaisesRegex(ValueError, "planned stage look"):
                    run_schedule(schedule, manifest, args)
                runner.assert_not_called()

    def test_source_locations_bind_lines_to_files_in_answers_and_exposure(self):
        criteria = {
            "required_phrases": ["SIGNAL", "src/parser.c"],
            "required_line_numbers": [32],
            "required_source_locations": [{"path": "src/parser.c", "line": 917}],
        }
        valid = (
            "src/parser.c:917", "src/parser.c#L917", "src/parser.c:917:4",
            "src/parser.c:915-920", "src/parser.c#L915-L920",
            "`src/parser.c`, line 917", "src/parser.c at line 917",
            "src/parser.c on line number: 917", "line 917 of `src/parser.c`",
            "line 917 in src/parser.c", "line 917 at src/parser.c",
            "line 917 of src/parser.c.",
            "src/parser.c, lines 915–920", "src/parser.c, lines 920-915",
            "src/parser.c, L917", "src/parser.c: line **917**",
            "[parser.c](/tmp/fixture/src/parser.c:917)",
            "[parser.c](src/parser.c#L917)", "./src/parser.c:917",
            "C:\\fixture\\src\\parser.c:917", "SRC/PARSER.C:917.",
        )
        invalid = (
            "src/parser.c:918, README.md:917", "src/parser.c:99; line 917 of README.md",
            "src/parser.c, README.md at line 917", "src/parser.c. Line 917",
            "src/parser.c\nline 917", "src/parser.c; line 917",
            "parser.c:917", "other/src/parser.c:917", "src/parser.c.bak:917",
            "src/parser.cpp:917", "src/parser.c:1917", "src/parser.c:9170",
            "src/parser.c logged at 2026-10-30T16:32:00", "src/parser.c:917.5",
            "src/parser.c, lines 918-920",
            "line 917 of src/parser.c:99", "line 917 of src/parser.c#L99",
        )
        for expected, references in ((True, valid), (False, invalid)):
            for reference in references:
                with self.subTest(reference=reference):
                    answer = "SIGNAL line 32: " + reference
                    passes = run_codex_agent_ab._criterion_passes(answer, criteria, "SIGNAL")
                    self.assertEqual(passes[-1], expected)
                    output = json.dumps({"type": "item.completed", "item": {
                        "type": "mcp_tool_call", "tool": "search_capture", "status": "completed",
                        "result": {"content": [{"type": "text", "text": answer}]},
                    }})
                    hits, counts = _mcp_result_criterion_exposure(output, criteria)
                    self.assertEqual(hits["search_capture"], passes)
                    self.assertEqual(hits["get_capture_slice"], [False] * 4)
                    self.assertEqual(counts["search_capture"], 1)
        self.assertEqual(run_codex_agent_ab._criterion_passes("", criteria), [False] * 4)
        self.assertEqual(run_codex_agent_ab._criterion_passes("I cannot help", criteria), [False] * 4)

    def test_repository_location_regression_rejects_unrelated_correct_number(self):
        criteria = build_repository_manifest()["tasks"]["targeted-inspection"]["success_criteria"]
        answer = "TARGETED_SIGNAL appears in src/packet_parser/parser.py:99. See README.md:12 for background."
        self.assertEqual(
            run_codex_agent_ab._criterion_passes(answer, criteria),
            [True, True, False, False],
        )
        self.assertFalse(_task_success(answer, criteria))

    def test_source_location_manifest_validation_and_normalization(self):
        invalid = (None, {}, [None], [{}], [{"path": "p.c"}],
                   [{"path": "", "line": 1}], [{"path": "./", "line": 1}],
                   [{"path": 1, "line": 1}], [{"path": "p.c", "line": True}],
                   [{"path": "p.c", "line": 0}], [{"path": "p.c", "line": 1.5}],
                   [{"path": "p.c", "line": "1"}],
                   [{"path": "./SRC/parser.c", "line": 917}, {"path": "src\\parser.c", "line": 917}])
        for locations in invalid:
            manifest = copy.deepcopy(build_manifest())
            criteria = manifest["tasks"]["targeted-inspection"]["success_criteria"]
            criteria["required_source_locations"] = locations
            with self.subTest(locations=locations), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "tasks.json"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "required_source_locations"):
                    _load_manifest(path)
                self.assertIsNone(run_codex_agent_ab._criterion_passes("SIGNAL", criteria))
                with self.assertRaisesRegex(ValueError, "required_source_locations"):
                    _mcp_result_criterion_exposure("", criteria)
        criteria = {"required_phrases": ["SIGNAL"], "required_source_locations": [{"path": "./SRC\\parser.c", "line": 917}]}
        self.assertEqual(run_codex_agent_ab._criterion_passes("SIGNAL src/parser.c:917", criteria), [True, True])

    def test_current_fixture_manifests_are_accepted_by_runner(self):
        builders = (("synthetic", build_manifest), ("repository", build_repository_manifest))
        with tempfile.TemporaryDirectory() as directory:
            for name, builder in builders:
                with self.subTest(fixture=name):
                    payload = builder()
                    self.assertEqual(payload["fixture_version"], TASK_FIXTURE_VERSION)
                    path = Path(directory) / f"{name}-manifest.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")

                    manifest = _load_manifest(path)

                    self.assertEqual(set(manifest), set(payload["tasks"]))
                    criteria = manifest["targeted-inspection"]["success_criteria"]
                    if name == "synthetic":
                        self.assertEqual(criteria["required_line_numbers"], [32])
                    else:
                        self.assertEqual(criteria["required_line_numbers"], [32])
                        self.assertEqual(
                            [item["line"] for item in criteria["required_source_locations"]],
                            [12],
                        )
                        answer = (
                            "TARGETED_SIGNAL src/packet_parser/parser.py:12; "
                            "command-output line 32"
                        )
                        self.assertEqual(
                            _criterion_text_hits(
                                answer,
                                criteria["required_phrases"],
                                criteria["required_line_numbers"],
                                criteria["required_source_locations"],
                            ),
                            [True, True, True, True],
                        )

    def test_mcp_exposure_counts_model_readable_text_not_structured_payloads(self):
        output = "\n".join(json.dumps(event) for event in (
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "ephemeral-buffer.search_capture",
                    "status": "completed",
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "Matched line 32 — TARGETED_SIGNAL",
                        }],
                        "structuredContent": {
                            "data": {"matches": [{"snippet": "private assertion"}]},
                        },
                    },
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "ephemeral-buffer.get_capture_slice",
                    "status": "completed",
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "Capture slice, line 32: private assertion",
                        }],
                        "structuredContent": {
                            "data": {"content": "TARGETED_SIGNAL"},
                        },
                    },
                },
            },
        ))

        exposure, counts = _mcp_result_criterion_exposure(output, {
            "required_phrases": ["TARGETED_SIGNAL", "private assertion"],
            "required_line_numbers": [32],
        })

        self.assertEqual(exposure["search_capture"], [True, False, True])
        self.assertEqual(exposure["get_capture_slice"], [False, True, True])
        self.assertEqual(counts, {"search_capture": 1, "get_capture_slice": 1})

    def test_line_number_criteria_accept_paths_and_ranges(self):
        for reference in ("src/parser.c:917", "parser.c:917", "src/README:917"):
            with self.subTest(reference=reference):
                self.assertEqual(
                    _criterion_text_hits(
                        f"TARGETED_SIGNAL appears at {reference}",
                        ["TARGETED_SIGNAL"],
                        [917],
                    ),
                    [True, True],
                )
        self.assertEqual(
            _criterion_text_hits(
                "TARGETED_SIGNAL is covered in lines 32-35",
                ["TARGETED_SIGNAL"],
                [34],
            ),
            [True, True],
        )
        self.assertEqual(
            _criterion_text_hits(
                "TARGETED_SIGNAL is covered in lines 32-35 and src/parser.c:1917",
                ["TARGETED_SIGNAL"],
                [917],
            ),
            [True, False],
        )
        self.assertEqual(
            _criterion_text_hits(
                "TARGETED_SIGNAL appears in parser.c; logged at 2026-10-30T16:32:00",
                ["TARGETED_SIGNAL"],
                [32],
            ),
            [True, False],
        )

    def test_line_reference_diagnostics_use_private_indexes_and_fixed_categories(self):
        criteria = {
            "required_phrases": ["SIGNAL", "DETAIL"],
            "required_line_numbers": [32, 917, 12],
            "required_source_locations": [{"path": "private/source.py", "line": 12}],
        }
        diagnostics = _line_reference_diagnostics(
            "SIGNAL DETAIL at L32 and src/parser.c:917, plus lines 10-14",
            criteria,
        )
        self.assertEqual(diagnostics, [
            {"criterion_index": 3, "category": "recognized_reference"},
            {"criterion_index": 4, "category": "recognized_reference"},
            {"criterion_index": 5, "category": "recognized_reference"},
        ])
        self.assertNotIn("32", json.dumps(diagnostics))
        self.assertNotIn("private/source.py", json.dumps(diagnostics))
        self.assertEqual(
            _line_reference_diagnostics("SIGNAL src/parser.c:32", {
                "required_phrases": ["SIGNAL"],
                "required_source_locations": [{"path": "src/parser.c", "line": 32}],
            }),
            [],
        )

        cases = (
            ("SIGNAL 32", 32, "expected_number_present_unrecognized_format"),
            ("SIGNAL 32, please", 32, "expected_number_present_unrecognized_format"),
            ("SIGNAL 132", 32, "expected_number_absent"),
            ("SIGNAL line 32.5", 32, "expected_number_absent"),
            ("SIGNAL 1,232", 32, "expected_number_absent"),
            ("SIGNAL 1,232", 232, "expected_number_absent"),
            ("SIGNAL line 32,456", 32, "expected_number_absent"),
            ("SIGNAL src/parser.c:1,232", 232, "expected_number_absent"),
        )
        for answer, line_number, category in cases:
            with self.subTest(answer=answer, line_number=line_number):
                self.assertEqual(
                    _line_reference_diagnostics(answer, {
                        "required_phrases": ["SIGNAL"],
                        "required_line_numbers": [line_number],
                    }),
                    [{"criterion_index": 2, "category": category}],
                )
        self.assertEqual(
            _criterion_text_hits("SIGNAL line 32.5", ["SIGNAL"], [32]),
            [True, False],
        )
        self.assertEqual(
            _criterion_text_hits("SIGNAL line 32,456", ["SIGNAL"], [32]),
            [True, False],
        )
        self.assertEqual(
            _criterion_text_hits("SIGNAL src/parser.c:1,232", ["SIGNAL"], [232]),
            [True, False],
        )

    def test_line_diagnostic_is_computed_even_when_refusal_gates_scoring(self):
        item = {
            "sequence": 1,
            "repetition": 1,
            "task_id": "targeted-inspection",
            "mode": "control",
        }
        task = build_manifest()["tasks"]["targeted-inspection"]
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        answer = "I cannot fulfill your request. The result is TARGETED_SIGNAL at line 32."
        result = subprocess_result(stdout=json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": answer},
        }))
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            with patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result):
                record = _run_one(
                    item,
                    task,
                    args=args,
                    repository=repository,
                    scratch=Path(directory),
                )

        self.assertFalse(record["task_success"])
        self.assertEqual(record["criterion_passes"], [False, False])
        self.assertEqual(record["line_reference_diagnostics"], [
            {"criterion_index": 2, "category": "recognized_reference"},
        ])
        self.assertNotIn("line 32", json.dumps(record))

    def test_failed_task_saves_full_trace_outside_records(self):
        item = {
            "sequence": 7,
            "repetition": 2,
            "task_id": "targeted-inspection",
            "mode": "mcp",
        }
        task = build_manifest()["tasks"]["targeted-inspection"]
        tool_event = {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "ephemeral-buffer.search_capture",
                "status": "completed",
                "result": {
                    "content": [{"type": "text", "text": "TARGETED_SIGNAL at line 32"}],
                },
            },
        }
        answer_event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "The relevant output is on line 32."},
        }
        stdout = "\n".join(json.dumps(event) for event in (tool_event, answer_event)) + "\n"
        stderr = "diagnostic from the runner\n"
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            traces = root / "failure-traces"
            args.failure_artifacts_dir = traces
            with patch(
                "benchmarks.run_codex_agent_ab._run_codex_process",
                return_value=subprocess_result(stdout=stdout, stderr=stderr),
            ):
                record = _run_one(
                    item,
                    task,
                    args=args,
                    repository=repository,
                    scratch=root,
                )

            artifact = traces / "0007-mcp"
            self.assertEqual((artifact / "stdout.jsonl").read_text(encoding="utf-8"), stdout)
            self.assertEqual((artifact / "stderr.txt").read_text(encoding="utf-8"), stderr)
            metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))

        self.assertFalse(record["task_success"])
        self.assertEqual(record["criterion_passes"], [False, True])
        self.assertEqual(metadata["run"], {
            "sequence": 7,
            "task_id": "targeted-inspection",
            "repetition": 2,
            "mode": "mcp",
            "task_fixture_version": run_codex_agent_ab.TASK_FIXTURE_VERSION,
            "task_rubric_version": run_codex_agent_ab.TASK_RUBRIC_VERSION,
        })
        self.assertEqual(metadata["outcome"]["failure_reason"], "task_success_criteria_not_met")
        self.assertNotIn("TARGETED_SIGNAL", json.dumps(record))

    def test_failure_trace_write_error_does_not_change_task_result(self):
        item = {"sequence": 3, "repetition": 1, "task_id": "targeted-inspection", "mode": "control"}
        task = build_manifest()["tasks"]["targeted-inspection"]
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            trace_root = root / "not-a-directory"
            trace_root.write_text("block", encoding="utf-8")
            args.failure_artifacts_dir = trace_root
            result = subprocess_result(stdout="The result is on line 32.")
            warning = io.StringIO()
            with patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result), \
                    contextlib.redirect_stderr(warning):
                record = _run_one(
                    item,
                    task,
                    args=args,
                    repository=repository,
                    scratch=root,
                )

        self.assertFalse(record["task_success"])
        self.assertIn("unable to save failure trace for run 3", warning.getvalue())

    def test_successful_run_does_not_create_failure_trace_directory(self):
        item = {"sequence": 4, "repetition": 1, "task_id": "targeted-inspection", "mode": "control"}
        task = build_manifest()["tasks"]["targeted-inspection"]
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            traces = root / "failure-traces"
            args.failure_artifacts_dir = traces
            with patch(
                "benchmarks.run_codex_agent_ab._run_codex_process",
                return_value=subprocess_result(stdout="TARGETED_SIGNAL at line 32"),
            ):
                record = _run_one(
                    item,
                    task,
                    args=args,
                    repository=repository,
                    scratch=root,
                )

        self.assertTrue(record["task_success"])
        self.assertFalse(traces.exists())

    def test_mcp_exposure_does_not_count_timestamp_as_line_evidence(self):
        output = json.dumps({
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "ephemeral-buffer.search_capture",
                "status": "completed",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": (
                            "TARGETED_SIGNAL appears in parser.c; "
                            "logged at 2026-10-30T16:32:00"
                        ),
                    }],
                },
            },
        })

        exposure, counts = _mcp_result_criterion_exposure(output, {
            "required_phrases": ["TARGETED_SIGNAL"],
            "required_line_numbers": [32],
        })

        self.assertEqual(exposure["search_capture"], [True, False])
        self.assertEqual(exposure["get_capture_slice"], [False, False])
        self.assertEqual(counts, {"search_capture": 1, "get_capture_slice": 0})

    def test_data_path_bytes_reads_content_free_snapshot(self):
        from benchmarks.benchmark_agent_ab import DATA_PATH_BYTE_FIELDS

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ephemeral_buffer_mcp.metrics.json"
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
        with patch("benchmarks.run_codex_agent_ab.subprocess.Popen", return_value=process), \
                patch("benchmarks.run_codex_agent_ab.os.getpgid", return_value=4242), \
                patch("benchmarks.run_codex_agent_ab.os.killpg") as killpg:
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
        with patch("benchmarks.run_codex_agent_ab.subprocess.Popen", return_value=process), \
                patch("benchmarks.run_codex_agent_ab.os.getpgid", return_value=4242), \
                patch("benchmarks.run_codex_agent_ab.os.killpg") as killpg:
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
            "benchmarks.run_codex_agent_ab._run_codex_process", side_effect=[timeout, successful]
        ):
            repository = Path(directory) / "repo"
            repository.mkdir()
            traces = Path(directory) / "failure-traces"
            args.failure_artifacts_dir = traces
            timed_out = _run_one(
                item, task, args=args, repository=repository, scratch=Path(directory)
            )
            continued = _run_one(
                next_item, task, args=args, repository=repository, scratch=Path(directory)
            )
            self.assertEqual(
                (traces / "0001-control" / "stdout.jsonl").read_text(encoding="utf-8"),
                "partial",
            )
            self.assertEqual(
                (traces / "0001-control" / "stderr.txt").read_text(encoding="utf-8"),
                "timed out",
            )
            self.assertFalse((traces / "0002-control").exists())

        self.assertFalse(timed_out["completed"])
        self.assertEqual(timed_out["failure_reason"], "timeout")
        self.assertEqual(timed_out["exit_code"], None)
        self.assertGreater(timed_out["output_bytes_proxy"], 0)
        self.assertTrue(continued["completed"])
        self.assertTrue(continued["task_success"])
        self.assertEqual(continued["peak_rss_bytes"], 4321)

    def test_nonzero_exit_saves_full_trace(self):
        item = {"sequence": 5, "repetition": 1, "task_id": "nonzero", "mode": "control"}
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "SUCCESS",
            "success_criteria": {"required_phrases": ["SUCCESS", "task found"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            traces = root / "failure-traces"
            args.failure_artifacts_dir = traces
            with patch(
                "benchmarks.run_codex_agent_ab._run_codex_process",
                return_value=subprocess_result(returncode=2, stdout="partial event stream\n", stderr="failed\n"),
            ):
                record = _run_one(item, task, args=args, repository=repository, scratch=root)

            artifact = traces / "0005-control"
            self.assertEqual(record["failure_reason"], "codex_exit_nonzero")
            self.assertEqual((artifact / "stdout.jsonl").read_text(encoding="utf-8"), "partial event stream\n")
            self.assertEqual((artifact / "stderr.txt").read_text(encoding="utf-8"), "failed\n")

    def test_codex_spawn_error_saves_failure_trace(self):
        item = {"sequence": 6, "repetition": 1, "task_id": "spawn-error", "mode": "control"}
        task = {
            "prompt": "inspect the fixture",
            "signal_marker": "SUCCESS",
            "success_criteria": {"required_phrases": ["SUCCESS", "task found"]},
        }
        args = argparse.Namespace(
            codex="codex", model="gpt-5.6-luna", mcp_python="python", mcp_module="server",
            mcp_server_script=None, sandbox="read-only", timeout=10,
            allow_mcp_approvals=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            traces = root / "failure-traces"
            args.failure_artifacts_dir = traces
            with patch(
                "benchmarks.run_codex_agent_ab._run_codex_process",
                side_effect=FileNotFoundError("codex executable not found"),
            ):
                record = _run_one(item, task, args=args, repository=repository, scratch=root)

            artifact = traces / "0006-control"
            self.assertEqual(record["failure_reason"], "codex_spawn_error")
            self.assertEqual((artifact / "stdout.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(
                (artifact / "stderr.txt").read_text(encoding="utf-8"),
                "codex executable not found",
            )

    def test_resume_warns_when_old_failure_trace_cannot_be_recreated(self):
        item = {"sequence": 9, "repetition": 1, "task_id": "targeted-inspection", "mode": "mcp"}
        records = [{
            "repetition": 1,
            "task_id": "targeted-inspection",
            "mode": "mcp",
            "task_success": False,
        }]
        schedule = {"schedule": [item]}
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            warning = io.StringIO()
            with contextlib.redirect_stderr(warning):
                run_codex_agent_ab._warn_missing_resumed_failure_artifacts(
                    records,
                    schedule,
                    artifact_root,
                )
            self.assertIn("completed runs are not rerun to backfill", warning.getvalue())

            warning = io.StringIO()
            artifact = artifact_root / "0009-mcp"
            artifact.mkdir()
            (artifact / "metadata.json").write_text("{}\n", encoding="utf-8")
            with contextlib.redirect_stderr(warning):
                run_codex_agent_ab._warn_missing_resumed_failure_artifacts(
                    records,
                    schedule,
                    artifact_root,
                )
            self.assertEqual(warning.getvalue(), "")

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
            "success_criteria": {"required_phrases": ["SUCCESS"], "required_source_locations": [{"path": "private/source.py", "line": 12}]},
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
                        "text": "The result is SUCCESS in private/source.py, line 12.",
                    },
                }),
            ]),
            peak_rss_bytes=1234,
        )
        with tempfile.TemporaryDirectory() as directory, patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result) as run:
            repository = Path(directory) / "repo"
            repository.mkdir()
            (repository / "fixture.txt").write_text("fixture")
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))
        self.assertTrue(output["completed"])
        self.assertTrue(output["task_success"])
        self.assertEqual(output["criterion_passes"], [True, True])
        self.assertTrue(output["signal_retrieved"])
        self.assertNotIn("SUCCESS", json.dumps(output))
        self.assertNotIn("private/source.py", json.dumps(output))
        self.assertNotIn("required_source_locations", json.dumps(output))
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
                with patch("benchmarks.run_codex_agent_ab._run_codex_process", **patch_options):
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
            "benchmarks.run_codex_agent_ab._run_codex_process", return_value=completed
        ):
            repository = Path(directory) / "repo"
            repository.mkdir()
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))

        self.assertFalse(output["signal_retrieved"])
        self.assertEqual(output["tool_calls"], 0)
        self.assertEqual(output["output_bytes_proxy"], len((completed.stdout + stderr).encode()))

    def test_matched_approval_mode_uses_write_sandbox_for_both_arms(self):
        from benchmarks.run_codex_agent_ab import _codex_command

        mcp_command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="mcp", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script="/tmp/server.py",
            allow_mcp_approvals=True, sandbox="read-only", timeout=10,
            mcp_env={"EPHEMERAL_TEST_EMBEDDINGS": "1"},
        )
        control_command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="control", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script=None,
            allow_mcp_approvals=True, sandbox="read-only", timeout=10,
        )
        for command in (control_command, mcp_command):
            self.assertIn("--approve-for-me", command)
            self.assertNotIn("--ask-for-approval", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
        self.assertFalse(any(value.startswith("mcp_servers.") for value in control_command))

        env_config = next(item for item in mcp_command if item.startswith("mcp_servers.ephemeral-buffer.env="))
        self.assertIn('EPHEMERAL_TEST_EMBEDDINGS="1"', env_config)

    def test_mcp_forwards_embedding_model_and_cache_configuration(self):
        from benchmarks.run_codex_agent_ab import _codex_command

        with patch.dict(
            os.environ,
            {
                "EPHEMERAL_TEST_EMBEDDINGS": "0",
                "EPHEMERAL_EMBEDDING_MODEL": "custom/model",
                "EPHEMERAL_EMBEDDING_CACHE_DIR": "/tmp/embedding-cache",
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
                        "EPHEMERAL_EMBEDDING_CACHE_DIR",
                    )
                },
            )
        env_config = next(item for item in command if item.startswith("mcp_servers.ephemeral-buffer.env="))
        self.assertIn('EPHEMERAL_EMBEDDING_MODEL="custom/model"', env_config)
        self.assertIn('EPHEMERAL_EMBEDDING_CACHE_DIR="/tmp/embedding-cache"', env_config)

    def test_mcp_server_source_paths_are_forwarded_to_the_child(self):
        from benchmarks.run_codex_agent_ab import _codex_command

        command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="mcp", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="ephemeral_buffer_mcp.server", mcp_server_script=None,
            allow_mcp_approvals=False, sandbox="read-only", timeout=10,
            mcp_python_path=["/tmp/revision/src", "/tmp/revision"],
        )

        env_config = next(item for item in command if item.startswith("mcp_servers.ephemeral-buffer.env="))
        self.assertIn('PYTHONPATH="/tmp/revision/src:/tmp/revision', env_config)

    def test_module_fingerprint_uses_configured_interpreter_path_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            module_dir = Path(directory) / "mcp-source"
            module_dir.mkdir()
            module_file = module_dir / "agent_ab_fingerprint_probe.py"
            module_file.write_text("VALUE = 'first'\n", encoding="utf-8")
            args = argparse.Namespace(
                mcp_python=sys.executable,
                mcp_module="agent_ab_fingerprint_probe",
                mcp_server_script=None,
                mcp_python_path=[module_dir],
            )
            with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False):
                first_fingerprint = run_codex_agent_ab._mcp_launch_fingerprint(args)
                self.assertRegex(first_fingerprint, r"^sha256:[0-9a-f]{64}$")

                module_file.write_text("VALUE = 'second'\n", encoding="utf-8")
                second_fingerprint = run_codex_agent_ab._mcp_launch_fingerprint(args)

                self.assertNotEqual(first_fingerprint, second_fingerprint)
                self.assertNotIn(str(module_dir), first_fingerprint)

                args.mcp_module = "missing_agent_ab_fingerprint_probe"
                self.assertRegex(
                    run_codex_agent_ab._mcp_launch_fingerprint(args),
                    r"^sha256:[0-9a-f]{64}$",
                )
                with self.assertRaisesRegex(ValueError, "cannot resume.*MCP module source"):
                    run_codex_agent_ab._mcp_launch_fingerprint(
                        args,
                        require_module_source=True,
                    )

    def test_codex_command_uses_current_exec_subcommand_syntax(self):
        from benchmarks.run_codex_agent_ab import _codex_command

        command = _codex_command(
            codex="codex", model="gpt-5.6-luna", mode="control", fixture=Path("/tmp/fixture"),
            mcp_python="python", mcp_module="server", mcp_server_script=None,
            allow_mcp_approvals=False, sandbox="read-only", timeout=10,
        )

        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertNotIn("--ask-for-approval", command)

    def test_mcp_approval_flag_is_global_before_exec_subcommand(self):
        from benchmarks.run_codex_agent_ab import _codex_command

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
        with tempfile.TemporaryDirectory() as directory, patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result) as run:
            repository = Path(directory) / "repo"
            repository.mkdir()
            traces = Path(directory) / "failure-traces"
            args.failure_artifacts_dir = traces
            output = _run_one(item, task, args=args, repository=repository, scratch=Path(directory))
            artifact = traces / "0001-mcp"
            self.assertEqual((artifact / "stdout.jsonl").read_text(encoding="utf-8"), "SUCCESS\n")

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
        with patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result), patch.dict(
            os.environ,
            {
                "EPHEMERAL_TEST_EMBEDDINGS": "",
                "EPHEMERAL_EMBEDDING_CACHE_DIR": "/private/cache/agent-ab",
                "EPHEMERAL_FASTEMBED_CACHE_DIR": "",
                "EPHEMERAL_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5-fp32",
            },
            clear=False,
        ):
            payload = run_schedule(schedule, manifest, args)
        self.assertEqual(len(payload["runs"]), 8)
        self.assertTrue(all(item["criterion_passes"] == [True, True] for item in payload["runs"]))
        self.assertEqual(payload["records_schema_version"], 12)
        self.assertEqual(payload["protocol"]["agent_adapter"], "codex-cli")
        result = records_workload_result(payload, producer="run_codex_agent_ab.py")
        self.assertEqual(result["workload"]["producer"], "run_codex_agent_ab.py")
        self.assertEqual(result["workload"]["parameters"]["protocol"]["agent_adapter"], "codex-cli")
        self.assertEqual(len(result["runs"]), 8)
        self.assertTrue(all(item["status"] == "success" for item in result["runs"]))
        self.assertIn("capture_input_bytes", result["runs"][0]["measurements"])
        self.assertEqual(payload["protocol"]["embedding_mode"], "onnx")
        self.assertEqual(payload["protocol"]["embedding_model"], "BAAI/bge-small-en-v1.5-fp32")
        self.assertEqual(payload["protocol"]["embedding_cache"], "custom")
        self.assertEqual(payload["protocol"]["timeout_seconds"], 10)
        self.assertEqual(payload["protocol"]["effective_sandbox"], "read-only")
        fingerprint = payload["protocol"]["mcp_launch_fingerprint"]
        self.assertRegex(fingerprint, r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("/private/cache/agent-ab", json.dumps(payload["protocol"]))

    def test_resume_extends_staged_run_without_rerunning_matched_pairs(self):
        stage_design = {"selected_repetitions": 2}
        first_schedule = build_schedule(seed=53, powered=True, design=stage_design)
        extended_schedule = build_schedule(seed=53, powered=True, design=stage_design)
        manifest = {
            task["id"]: {
                "prompt": task["id"],
                "signal_marker": "ok",
                "success_criteria": {
                    "description": "report the signal and a deterministic result",
                    "required_phrases": ["ok", "expected answer"],
                },
            }
            for task in first_schedule["tasks"]
        }
        result = subprocess_result(stdout=json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "ok is the expected answer"},
        }) + "\n")
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.json"
            mcp_source_dir = Path(directory) / "mcp-source"
            mcp_source_dir.mkdir()
            mcp_source_file = mcp_source_dir / "resume_test_mcp.py"
            original_mcp_source = "VALUE = 'original'\n"
            mcp_source_file.write_text(original_mcp_source, encoding="utf-8")
            (mcp_source_dir / "different_server_module.py").write_text(
                "VALUE = 'different module'\n",
                encoding="utf-8",
            )
            alternate_mcp_source_dir = Path(directory) / "alternate-mcp-source"
            alternate_mcp_source_dir.mkdir()
            (alternate_mcp_source_dir / "resume_test_mcp.py").write_text(
                "VALUE = 'alternate source path'\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                repository=".", repository_fixture="fixture", environment="test", model="gpt-5.6-luna",
                codex="codex", mcp_python=sys.executable, mcp_module="resume_test_mcp", mcp_server_script=None,
                mcp_python_path=[mcp_source_dir],
                sandbox="read-only", timeout=10, allow_mcp_approvals=False,
                dry_run=False, output=records_path, max_repetitions=1, resume=False,
            )
            with patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=result) as runner, patch.dict(
                os.environ,
                {
                    "EPHEMERAL_TEST_EMBEDDINGS": "",
                    "EPHEMERAL_EMBEDDING_CACHE_DIR": "/private/cache/agent-ab",
                    "EPHEMERAL_FASTEMBED_CACHE_DIR": "",
                    "EPHEMERAL_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5-fp32",
                },
                clear=False,
            ):
                first_payload = run_schedule(first_schedule, manifest, args)
                self.assertEqual(first_payload["protocol"]["timeout_seconds"], 10)
                self.assertEqual(first_payload["protocol"]["effective_sandbox"], "read-only")
                self.assertEqual(first_payload["protocol"]["embedding_cache"], "custom")
                self.assertRegex(
                    first_payload["protocol"]["mcp_launch_fingerprint"],
                    r"^sha256:[0-9a-f]{64}$",
                )
                self.assertNotIn("/private/cache/agent-ab", json.dumps(first_payload["protocol"]))
                first_result = records_workload_result(
                    first_payload,
                    producer="run_codex_agent_ab.py",
                )
                self.assertEqual(len(first_result["runs"]), len(first_payload["runs"]))
                self.assertEqual(first_result["workload"]["parameters"]["repetitions"], 2)
                records_path.write_text(json.dumps(first_payload), encoding="utf-8")
                args.max_repetitions = 2
                args.resume = True

                mcp_source_file.write_text("VALUE = 'changed'\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                self.assertEqual(runner.call_count, 8)

                mcp_source_file.unlink()
                with self.assertRaisesRegex(ValueError, "unable to resolve or hash.*MCP module source"):
                    run_schedule(extended_schedule, manifest, args)
                self.assertEqual(runner.call_count, 8)

                mcp_source_file.write_text(original_mcp_source, encoding="utf-8")
                extended_payload = run_schedule(extended_schedule, manifest, args)
                extended_result = records_workload_result(
                    extended_payload,
                    producer="run_codex_agent_ab.py",
                )
                self.assertEqual(len(extended_result["runs"]), len(extended_payload["runs"]))
                records_path.write_text(json.dumps(extended_payload), encoding="utf-8")
                args.max_repetitions = 1
                with self.assertRaisesRegex(ValueError, "below completed records"):
                    run_schedule(extended_schedule, manifest, args)
                args.max_repetitions = 2
                args.model = "different-model"
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                args.model = "gpt-5.6-luna"
                args.timeout = 11
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                args.timeout = 10
                args.sandbox = "workspace-write"
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                args.sandbox = "read-only"
                args.mcp_python = "different-python"
                with self.assertRaisesRegex(ValueError, "cannot resume.*MCP module source"):
                    run_schedule(extended_schedule, manifest, args)
                args.mcp_python = sys.executable
                args.mcp_module = "different_server_module"
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                args.mcp_module = "resume_test_mcp"
                args.mcp_python = sys.executable
                args.mcp_python_path = [alternate_mcp_source_dir]
                with self.assertRaisesRegex(ValueError, "different model/configuration protocol"):
                    run_schedule(extended_schedule, manifest, args)
                args.mcp_python_path = [mcp_source_dir]
                legacy_payload = copy.deepcopy(extended_payload)
                legacy_payload["records_schema_version"] = 10
                records_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "different records schema"):
                    run_schedule(extended_schedule, manifest, args)

        self.assertEqual(len(first_payload["runs"]), 8)
        self.assertEqual(len(extended_payload["runs"]), 16)
        self.assertEqual(runner.call_count, 16)
        keys = {
            (item["repetition"], item["task_id"], item["mode"])
            for item in extended_payload["runs"]
        }
        self.assertEqual(len(keys), 16)

    def test_runner_rejects_model_or_fixture_different_from_bound_schedule(self):
        schedule = build_schedule(
            repetitions=1,
            seed=61,
            model_config="pilot-model",
            repository_fixture="pilot-fixture",
        )
        with self.assertRaisesRegex(ValueError, "schedule model_config"):
            run_schedule(
                schedule,
                {},
                argparse.Namespace(model="other-model", repository_fixture="pilot-fixture"),
            )
        with self.assertRaisesRegex(ValueError, "schedule repository_fixture"):
            run_schedule(
                schedule,
                {},
                argparse.Namespace(model="pilot-model", repository_fixture="other-fixture"),
            )


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
            trace_dir = root / "failure-traces"
            argv = [
                "run_codex_agent_ab.py", "--schedule", str(root / "schedule.json"), "--tasks", str(root / "tasks.json"),
                "--output", str(root / "records.json"), "--repository", ".", "--result", str(root / "result.json"),
                "--failure-artifacts-dir", str(trace_dir),
            ]
            with patch("benchmarks.run_codex_agent_ab._run_codex_process", return_value=completed), patch(
                "sys.argv", argv
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                run_codex_agent_ab.main()
            result = wr.load_result(root / "result.json")
            records = json.loads((root / "records.json").read_text(encoding="utf-8"))
            self.assertEqual(records["benchmark"], "agent-ab")
            self.assertTrue(any(trace_dir.glob("*/metadata.json")))
            self.assertNotIn("stdout.jsonl", json.dumps(records))
        self.assertEqual(result["workload"]["kind"], "agent-run")
        self.assertEqual(len(result["runs"]), 8)
        self.assertEqual(json.loads(stdout.getvalue())["runs"], 8)


if __name__ == "__main__":
    unittest.main()
