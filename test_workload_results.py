"""Tests for the versioned coding-agent workload result format."""

import argparse
import importlib.util
import io
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import workload_results as wr
from workload_results import (
    WorkloadResultError,
    build_result,
    environment,
    measurement,
    phase,
    run,
    summarize,
    unavailable,
    validate_result,
    write_result,
)


SCHEMA = json.loads(wr.SCHEMA_PATH.read_text(encoding="utf-8"))


def sample_result(**overrides):
    result = build_result(
        workload="sample",
        kind="benchmark",
        producer="test",
        parameters={"samples": 2},
        runs=[
            run(
                "lines-16",
                labels={"line_count": 16, "cache_state": "warm", "flag": True, "missing": None},
                measurements={
                    "wall_time_seconds": measurement("seconds", median=0.5, p95=0.7, samples=2),
                    "output_bytes": measurement("bytes", value=128),
                    "estimated_tokens": unavailable("tokens", "no model was invoked"),
                },
                phases=[phase("ingest", median=0.1, p95=0.2), phase("search", value=0.3)],
            ),
            run("lines-256", status="failure", errors=["needle missed"]),
        ],
        measurements={"cold_start_seconds": measurement("seconds", value=1.5)},
        details={"raw": True},
        producer_schema_version=3,
        fixture_version=2,
        description="sample result",
        privacy="no user content",
    )
    result.update(overrides)
    return result


class TestMeasurements(unittest.TestCase):
    def test_measurement_records_unit_statistics_samples_and_note(self):
        value = measurement("bytes", value=10, mean=2.5, samples=4, note="proxy")
        self.assertEqual(value, {"unit": "bytes", "value": 10, "mean": 2.5, "samples": 4, "note": "proxy"})
        self.assertEqual(unavailable("tokens"), {"unit": "tokens", "value": None, "samples": 0})
        self.assertEqual(measurement("ratio", value=None), {"unit": "ratio", "value": None})

    def test_measurement_rejects_bad_inputs(self):
        with self.assertRaisesRegex(WorkloadResultError, "unknown unit"):
            measurement("furlongs", value=1)
        with self.assertRaisesRegex(WorkloadResultError, "at least one statistic"):
            measurement("seconds")
        with self.assertRaisesRegex(WorkloadResultError, "unknown statistic"):
            measurement("seconds", average=1)
        with self.assertRaisesRegex(WorkloadResultError, "finite number or null"):
            measurement("seconds", value=True)
        with self.assertRaisesRegex(WorkloadResultError, "finite number or null"):
            measurement("seconds", value=float("nan"))
        with self.assertRaisesRegex(WorkloadResultError, "samples must be"):
            measurement("seconds", value=1, samples=-1)
        with self.assertRaisesRegex(WorkloadResultError, "samples must be"):
            measurement("seconds", value=1, samples=True)
        with self.assertRaisesRegex(WorkloadResultError, "note must be"):
            measurement("seconds", value=1, note="")

    def test_summarize_uses_nearest_rank_p95_and_marks_empty_unavailable(self):
        summary = summarize([1, 2, 100, None], "seconds", note="three samples")
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["p95"], 100)
        self.assertEqual(summary["median"], 2)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 100)
        self.assertEqual(summary["sum"], 103)
        self.assertAlmostEqual(summary["mean"], 103 / 3)
        self.assertGreater(summary["stdev"], 0)
        self.assertEqual(summary["note"], "three samples")
        single = summarize([4.0], "bytes")
        self.assertEqual(single["stdev"], 0.0)
        self.assertEqual(summarize([None], "tokens"), {"unit": "tokens", "value": None, "samples": 0})

    def test_nearest_rank_validates_inputs(self):
        self.assertEqual(wr.nearest_rank([3, 1, 2]), 3)
        self.assertEqual(wr.nearest_rank([3, 1, 2], percentile=0.5), 2)
        with self.assertRaises(ValueError):
            wr.nearest_rank([])
        with self.assertRaises(ValueError):
            wr.nearest_rank([1], percentile=0)

    def test_statistic_treats_absent_null_and_zero_samples_as_unavailable(self):
        self.assertEqual(wr.statistic(measurement("seconds", median=0.5, samples=3), "median"), 0.5)
        self.assertEqual(wr.statistic(measurement("count", value=0), "value"), 0)
        self.assertIsNone(wr.statistic(measurement("seconds", median=0.5), "p95"))
        self.assertIsNone(wr.statistic(measurement("seconds", median=None), "median"))
        self.assertIsNone(wr.statistic(measurement("seconds", median=0.5, samples=0), "median"))
        self.assertIsNone(wr.statistic(unavailable("tokens"), "value"))

    def test_canonical_directions_cover_every_canonical_measurement(self):
        self.assertEqual(set(wr.CANONICAL_DIRECTIONS), set(wr.CANONICAL_MEASUREMENTS))
        self.assertTrue(set(wr.CANONICAL_DIRECTIONS.values()) <= {"lower", "higher", None})
        self.assertEqual(wr.CANONICAL_DIRECTIONS["success_rate"], "higher")
        self.assertIsNone(wr.CANONICAL_DIRECTIONS["input_bytes"])

    def test_phase_is_a_named_seconds_measurement(self):
        self.assertEqual(phase("ingest", median=0.1), {"name": "ingest", "unit": "seconds", "median": 0.1})
        with self.assertRaisesRegex(WorkloadResultError, "phase name"):
            phase("Ingest Phase", value=1)


class TestBuildAndValidate(unittest.TestCase):
    def test_sample_result_is_valid_and_json_serializable(self):
        result = sample_result()
        self.assertEqual(result["format"], wr.FORMAT)
        self.assertEqual(result["format_version"], wr.FORMAT_VERSION)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["workload"]["producer_schema_version"], 3)
        self.assertEqual(result["workload"]["fixture_version"], 2)
        self.assertEqual(result["workload"]["privacy"], "no user content")
        self.assertEqual(result["details"], {"raw": True})
        self.assertEqual([entry["name"] for entry in result["runs"][0]["phases"]], ["ingest", "search"])
        json.dumps(result, allow_nan=False)

    def test_status_is_derived_unless_given(self):
        self.assertEqual(wr.derive_status([], []), "success")
        self.assertEqual(wr.derive_status([run("a")], []), "success")
        self.assertEqual(wr.derive_status([run("a", status="failure")], []), "failure")
        self.assertEqual(wr.derive_status([run("a"), run("b", status="timeout")], []), "partial")
        self.assertEqual(wr.derive_status([run("a")], ["boom"]), "error")
        explicit = build_result(workload="w", kind="evaluation", producer="p", status="timeout")
        self.assertEqual(explicit["status"], "timeout")
        self.assertNotIn("details", explicit)
        self.assertEqual(set(explicit["workload"]), {"name", "kind", "producer", "parameters"})

    def test_environment_describes_host_and_rejects_reserved_extras(self):
        described = environment(embedding_model="m")
        self.assertEqual(described["embedding_model"], "m")
        self.assertEqual(described["tool"]["name"], wr.PROJECT_NAME)
        self.assertRegex(described["tool"]["version"], r"^\d+\.\d+\.\d+")
        self.assertRegex(described["recorded_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")
        self.assertTrue(described["source_revision"] is None or len(described["source_revision"]) == 40)
        with self.assertRaisesRegex(WorkloadResultError, "reserved"):
            environment(platform="x")

    def test_project_version_falls_back_to_metadata_then_none(self):
        with patch.object(wr.Path, "read_text", side_effect=OSError("missing")):
            with patch.object(wr._metadata, "version", return_value="9.9.9"):
                self.assertEqual(wr._project_version(), "9.9.9")
            with patch.object(wr._metadata, "version", side_effect=wr._metadata.PackageNotFoundError("x")):
                self.assertIsNone(wr._project_version())
        with patch.object(wr.Path, "read_text", return_value="[project]\nname = 'x'\n"), patch.object(
            wr._metadata, "version", return_value="8.8.8"
        ):
            self.assertEqual(wr._project_version(), "8.8.8")

    def test_source_revision_handles_missing_git_and_failures(self):
        completed = subprocess.CompletedProcess(["git"], 0, stdout="abc123\n", stderr="")
        with patch.object(wr.subprocess, "run", return_value=completed):
            self.assertEqual(wr._source_revision(), "abc123")
        failed = subprocess.CompletedProcess(["git"], 128, stdout="", stderr="not a repo")
        with patch.object(wr.subprocess, "run", return_value=failed):
            self.assertIsNone(wr._source_revision())
        with patch.object(wr.subprocess, "run", side_effect=OSError("no git")):
            self.assertIsNone(wr._source_revision())
        with patch.object(wr.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 5)):
            self.assertIsNone(wr._source_revision())

    def assertInvalid(self, result, message):
        with self.assertRaisesRegex(WorkloadResultError, message):
            validate_result(result)

    def test_validation_rejects_envelope_problems(self):
        self.assertInvalid([], "must be an object")
        self.assertInvalid(sample_result(format="other"), "format must be")
        self.assertInvalid(sample_result(format_version=2), "format_version must be")
        broken = sample_result()
        del broken["runs"]
        self.assertInvalid(broken, "missing runs")
        self.assertInvalid(sample_result(extra=1), "unexpected fields extra")
        self.assertInvalid(sample_result(status="unknown"), "status must be one of")
        self.assertInvalid(sample_result(errors=[1]), "errors must be a list of strings")
        self.assertInvalid(sample_result(measurements=[]), "measurements must be an object")
        self.assertInvalid(sample_result(runs={}), "runs must be a list")
        self.assertInvalid(sample_result(details=[]), "details must be an object")
        self.assertInvalid(sample_result(environment=[]), "environment must be an object")
        self.assertInvalid(sample_result(environment={"python_version": "3", "platform": ""}), "environment.platform")
        self.assertInvalid(sample_result(details={"value": {1, 2}}), "not JSON-serializable")

    def test_validation_rejects_workload_problems(self):
        def with_workload(**changes):
            result = sample_result()
            result["workload"].update(changes)
            return result

        self.assertInvalid(sample_result(workload="x"), "workload must be an object")
        self.assertInvalid(with_workload(name=""), "workload.name")
        self.assertInvalid(with_workload(kind="thing"), "workload.kind")
        self.assertInvalid(with_workload(parameters=None), "workload.parameters")
        self.assertInvalid(with_workload(fixture_version=-1), "workload.fixture_version")
        self.assertInvalid(with_workload(producer_schema_version=True), "workload.producer_schema_version")
        self.assertInvalid(with_workload(description=""), "workload.description")
        self.assertInvalid(with_workload(colour="blue"), "unexpected fields colour")

    def test_validation_rejects_measurement_problems(self):
        self.assertInvalid(sample_result(measurements={"Bad Name": measurement("bytes", value=1)}), "must match")
        self.assertInvalid(sample_result(measurements={"x": "fast"}), "measurements.x must be an object")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "furlongs", "value": 1}}), "measurements.x.unit")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "bytes"}}), "at least one statistic")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "bytes", "value": "1"}}), "finite number or null")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "bytes", "value": 1, "samples": 1.5}}), "samples must be")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "bytes", "value": 1, "note": 3}}), "note must be")
        self.assertInvalid(sample_result(measurements={"x": {"unit": "bytes", "value": 1, "extra": 3}}), "not a recognised")
        # Canonical names pin their unit so consumers can compare across producers.
        self.assertInvalid(sample_result(measurements={"wall_time_seconds": measurement("bytes", value=1)}), "unit must be 'seconds'")

    def test_validation_rejects_run_problems(self):
        def with_run(**changes):
            item = run("r", measurements={"output_bytes": measurement("bytes", value=1)})
            item.update(changes)
            return sample_result(runs=[item])

        self.assertInvalid(sample_result(runs=["r"]), "runs\\[0\\] must be an object")
        self.assertInvalid(with_run(extra=1), "exactly id, labels")
        self.assertInvalid(with_run(id=""), "runs\\[0\\].id")
        self.assertInvalid(with_run(labels=[]), "labels must be an object")
        self.assertInvalid(with_run(labels={"Bad": 1}), "labels key")
        self.assertInvalid(with_run(labels={"nested": {}}), "labels.nested must be")
        self.assertInvalid(with_run(status="meh"), "runs\\[0\\].status")
        self.assertInvalid(with_run(measurements={"x": 1}), "runs\\[0\\].measurements.x")
        self.assertInvalid(with_run(phases={}), "phases must be a list")
        self.assertInvalid(with_run(phases=[{"unit": "seconds", "value": 1}]), "phases\\[0\\].name")
        self.assertInvalid(with_run(phases=[phase("a", value=1), phase("a", value=2)]), "duplicated")
        self.assertInvalid(with_run(phases=[{"name": "a", "unit": "bytes", "value": 1}]), "unit must be 'seconds'")
        self.assertInvalid(with_run(errors=[None]), "errors must be a list of strings")
        self.assertInvalid(sample_result(runs=[run("dup"), run("dup")]), "runs\\[1\\].id 'dup' is duplicated")

    def test_validation_rejects_experiment_problems(self):
        def with_experiment(block):
            return sample_result(experiment=block)

        self.assertInvalid(with_experiment([]), "experiment must be an object")
        self.assertInvalid(with_experiment({"group": "g"}), "exactly group and metadata")
        self.assertInvalid(with_experiment({"group": "", "metadata": {}}), "experiment.group must be a non-empty string or null")
        self.assertInvalid(with_experiment({"group": 5, "metadata": {}}), "experiment.group must be")
        self.assertInvalid(with_experiment({"group": None, "metadata": []}), "experiment.metadata must be an object")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"Bad": 1}}), "experiment.metadata key 'Bad' must match")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"nested": {}}}), "experiment.metadata.nested must be a string")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"prompt": "x" * 257}}), "experiment.metadata.prompt must be at most 256")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"api_key": "s3cret"}}), "experiment.metadata.api_key looks like a credential")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"ratio": float("inf")}}), "experiment.metadata.ratio must be a finite number")
        self.assertInvalid(with_experiment({"group": "g", "metadata": {"started_at": "2026-09-18"}}), "experiment.metadata.started_at must be an ISO 8601")
        accepted = with_experiment({"group": None, "metadata": {"api_key": wr.REDACTED, "model": "m", "size": 5, "flag": False, "none": None}})
        self.assertIs(validate_result(accepted), accepted)


class TestExperiment(unittest.TestCase):
    def test_sensitive_keys_match_parts_and_suffixes_not_substrings(self):
        for key in ("token", "key", "access_token", "api_key", "ssh_key", "client_secret", "secrets", "password", "passwd", "credentials", "credential_id", "apikey", "authorization", "bearer_value", "cache_key"):
            self.assertTrue(wr.is_sensitive_metadata_key(key), key)
        for key in ("max_tokens", "token_budget", "keyboard", "model", "secretary", "keys_pressed", "repository_revision"):
            self.assertFalse(wr.is_sensitive_metadata_key(key), key)

    def test_experiment_builder_redacts_and_validates(self):
        self.assertEqual(wr.experiment(), {"group": None, "metadata": {}})
        block = wr.experiment(
            "sweep",
            {"variant": "a", "workload_size": 5, "flag": True, "none": None, "api_key": "s3cret", "owner": "me"},
            redact=["owner", "absent"],
        )
        self.assertEqual(block, {
            "group": "sweep",
            "metadata": {"variant": "a", "workload_size": 5, "flag": True, "none": None, "api_key": wr.REDACTED, "owner": wr.REDACTED},
        })
        with self.assertRaisesRegex(WorkloadResultError, "group must be a non-empty string or None"):
            wr.experiment("")
        with self.assertRaisesRegex(WorkloadResultError, "key 'Bad' must match"):
            wr.experiment("g", {"Bad": 1})
        with self.assertRaisesRegex(WorkloadResultError, "metadata.sizes must be a string, number, boolean, or null"):
            wr.experiment("g", {"sizes": [1]})
        with self.assertRaisesRegex(WorkloadResultError, "at most 256 characters"):
            wr.experiment("g", {"prompt": "x" * 257})

    def test_parse_key_value_and_metadata_arguments(self):
        self.assertEqual(wr.parse_key_value("line_count=256"), ("line_count", 256))
        self.assertEqual(wr.parse_key_value("mode=mcp"), ("mode", "mcp"))
        self.assertEqual(wr.parse_key_value("flag=true"), ("flag", True))
        self.assertEqual(wr.parse_key_value("missing=null"), ("missing", None))
        self.assertEqual(wr.parse_key_value("profile="), ("profile", ""))
        self.assertEqual(wr.parse_key_value("when=2026-09-18T10:00:00+00:00"), ("when", "2026-09-18T10:00:00+00:00"))
        with self.assertRaisesRegex(WorkloadResultError, "metadata 'mode' must be KEY=VALUE"):
            wr.parse_key_value("mode")
        with self.assertRaisesRegex(WorkloadResultError, "label filter '=x' must be KEY=VALUE"):
            wr.parse_key_value("=x", "label filter")
        with self.assertRaisesRegex(WorkloadResultError, "must be a string, number, boolean, or null"):
            wr.parse_key_value("sizes=[1, 2]")
        # A short SHA that looks like an exponent, and JSON's NaN, are not finite numbers.
        for text in ("repository_revision=12e5678", "variant=NaN", "x=-Infinity"):
            with self.assertRaisesRegex(WorkloadResultError, "must be a finite number"):
                wr.parse_key_value(text)
        self.assertEqual(wr.parse_metadata("variant=a"), ("variant", "a"))
        self.assertEqual(wr.parse_metadata("size=5"), ("size", 5))
        self.assertEqual(wr.parse_metadata("started_at=2026-09-18T10:00:00Z"), ("started_at", "2026-09-18T10:00:00Z"))
        # Credential-looking keys pass here because experiment() redacts them.
        self.assertEqual(wr.parse_metadata("api_key=s3cret"), ("api_key", "s3cret"))
        with self.assertRaisesRegex(WorkloadResultError, "metadata key 'Variant' must match"):
            wr.parse_metadata("Variant=a")
        with self.assertRaisesRegex(WorkloadResultError, "metadata.prompt must be at most 256 characters"):
            wr.parse_metadata("prompt=" + "x" * 257)
        with self.assertRaisesRegex(WorkloadResultError, "metadata.started_at must be an ISO 8601 timestamp"):
            wr.parse_metadata("started_at=2026-09-18 10:00")
        self.assertEqual(wr.experiment_group("sweep"), "sweep")
        with self.assertRaisesRegex(WorkloadResultError, "must not be empty"):
            wr.experiment_group("")
        self.assertEqual(wr.metadata_key("owner"), "owner")
        with self.assertRaisesRegex(WorkloadResultError, "metadata key 'Owner' must match"):
            wr.metadata_key("Owner")

    def test_value_and_experiment_matching(self):
        self.assertTrue(wr.value_matches(1, 1))
        self.assertTrue(wr.value_matches("a", "a"))
        self.assertTrue(wr.value_matches(None, None))
        self.assertTrue(wr.value_matches(True, True))
        self.assertFalse(wr.value_matches(True, False))
        self.assertFalse(wr.value_matches(True, 1))
        self.assertFalse(wr.value_matches(1, True))
        self.assertFalse(wr.value_matches(0, False))
        grouped = sample_result(experiment=wr.experiment("sweep", {"variant": "a", "size": 5, "flag": True}))
        ungrouped = sample_result()
        self.assertTrue(wr.experiment_matches(grouped))
        self.assertTrue(wr.experiment_matches(ungrouped))
        self.assertTrue(wr.experiment_matches(grouped, ["other", "sweep"]))
        self.assertFalse(wr.experiment_matches(grouped, ["other"]))
        self.assertFalse(wr.experiment_matches(ungrouped, ["sweep"]))
        self.assertTrue(wr.experiment_matches(grouped, where=[("variant", "a"), ("size", 5)]))
        self.assertTrue(wr.experiment_matches(grouped, ["sweep"], [("flag", True)]))
        self.assertFalse(wr.experiment_matches(grouped, where=[("variant", "b")]))
        self.assertFalse(wr.experiment_matches(grouped, where=[("size", True)]))
        # Absent metadata never matches, even a filter for null.
        self.assertFalse(wr.experiment_matches(grouped, where=[("missing", None)]))
        self.assertFalse(wr.experiment_matches(ungrouped, where=[("variant", "a")]))

    def test_experiment_timestamp_and_summaries(self):
        plain = sample_result()
        recorded = plain["environment"]["recorded_at"]
        self.assertEqual(wr.experiment_timestamp(plain), recorded)
        started = sample_result(experiment=wr.experiment("g", {"started_at": "2026-09-18T10:00:00+00:00", "owner": "me"}))
        self.assertEqual(wr.experiment_timestamp(started), "2026-09-18T10:00:00+00:00")
        # Offsets are normalised to UTC so documents order by instant, not by spelling.
        def at(started_at):
            return wr.experiment_timestamp(sample_result(experiment=wr.experiment("g", {"started_at": started_at})))
        self.assertEqual(at("2026-09-17T10:00:00-05:00"), "2026-09-17T15:00:00+00:00")
        self.assertEqual(at("2026-09-17T15:00:00Z"), "2026-09-17T15:00:00+00:00")
        self.assertLess(at("2026-09-17T10:00:00-05:00"), at("2026-09-17T16:00:00+00:00"))
        for unusable in ("", "2026-09-18 10:00", "2026-09-18T10:00:00", "yesterday", 5, None):
            with self.assertRaisesRegex(WorkloadResultError, "started_at must be an ISO 8601 timestamp"):
                wr.experiment("g", {"started_at": unusable})
        self.assertIsNone(wr.parse_timestamp("2026-09-18T10:00:00"))
        self.assertIsNone(wr.parse_timestamp("nope"))
        # recorded_at without an offset is read as UTC; a non-ISO one is used as written.
        naive = sample_result(environment={"python_version": "3", "platform": "p", "recorded_at": "2026-09-18T10:00:00"})
        self.assertEqual(wr.experiment_timestamp(naive), "2026-09-18T10:00:00+00:00")
        odd = sample_result(environment={"python_version": "3", "platform": "p", "recorded_at": "Thursday"})
        self.assertEqual(wr.experiment_timestamp(odd), "Thursday")
        self.assertEqual(wr.experiment_block(plain), {"group": None, "metadata": {}})
        self.assertIs(wr.experiment_block(started), started["experiment"])
        summary = wr.result_summary(started)
        self.assertEqual(summary["workload"], {"name": "sample", "kind": "benchmark", "producer": "test"})
        self.assertEqual(summary["group"], "g")
        self.assertEqual(summary["metadata"], {"started_at": "2026-09-18T10:00:00+00:00", "owner": "me"})
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["recorded_at"], recorded)
        self.assertEqual(summary["ordered_at"], "2026-09-18T10:00:00+00:00")
        self.assertEqual(summary["tool_version"], started["environment"]["tool"]["version"])
        self.assertEqual(summary["source_revision"], started["environment"]["source_revision"])
        self.assertEqual(
            [(item["id"], item["status"], item["errors"]) for item in summary["runs"]],
            [("lines-16", "success", []), ("lines-256", "failure", ["needle missed"])],
        )
        self.assertEqual(summary["runs"][0]["labels"]["line_count"], 16)
        self.assertNotIn("measurements", summary["runs"][0])
        json.dumps(summary, allow_nan=False)
        plain_summary = wr.result_summary(plain)
        self.assertIsNone(plain_summary["group"])
        self.assertEqual(plain_summary["metadata"], {})
        self.assertEqual(plain_summary["ordered_at"], recorded)
        bare = wr.result_summary(sample_result(environment={"python_version": "3", "platform": "p", "recorded_at": "t"}))
        self.assertIsNone(bare["tool_version"])
        self.assertIsNone(bare["source_revision"])
        redacted = wr.redact_summary(summary, ["owner", "absent"])
        self.assertEqual(redacted["metadata"], {"started_at": "2026-09-18T10:00:00+00:00", "owner": wr.REDACTED})
        self.assertEqual(summary["metadata"]["owner"], "me")
        self.assertEqual(redacted["runs"], summary["runs"])

    def test_format_helpers_show_groups_and_metadata(self):
        self.assertEqual(wr.format_metadata_value("a b"), "a b")
        self.assertEqual(wr.format_metadata_value(5), "5")
        self.assertEqual(wr.format_metadata_value(True), "true")
        self.assertEqual(wr.format_metadata_value(None), "null")
        self.assertEqual(wr.format_metadata({"variant": "a", "size": 5}), "variant=a size=5")
        self.assertEqual(wr.format_metadata({}), "")
        plain = wr.format_summary(sample_result())
        self.assertEqual(plain, "workload=sample kind=benchmark producer=test status=partial runs=2 errors=0")
        self.assertEqual(wr.format_summary(sample_result(experiment=wr.experiment("g", {"variant": "a"}))), plain + " group=g variant=a")
        self.assertEqual(wr.format_summary(sample_result(experiment=wr.experiment(None, {"variant": "a"}))), plain + " variant=a")
        self.assertEqual(wr.format_summary(sample_result(experiment=wr.experiment("g"))), plain + " group=g")
        self.assertEqual(wr.format_table([["a", "bb"], ["ccc", "d"]], ["h1", "h2"]), ["h1   h2", "a    bb", "ccc  d"])

    def test_iter_result_files_scans_directories_and_reports_invalid_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.json"
            write_result(sample_result(), good)
            nested = root / "sub" / "nested.json"
            write_result(sample_result(), nested, experiment=wr.experiment("g"))
            records = root / "records.json"
            records.write_text('{"schedule": []}', encoding="utf-8")
            (root / "list.json").write_text("[]", encoding="utf-8")
            broken = root / "broken.json"
            broken.write_text("{", encoding="utf-8")
            text = root / "notes.txt"
            text.write_text(json.dumps(sample_result()), encoding="utf-8")
            bad = root / "bad.json"
            bad.write_text(json.dumps({"format": wr.FORMAT, "format_version": wr.FORMAT_VERSION}), encoding="utf-8")
            found = [(path.name, result is not None, error) for path, result, error in wr.iter_result_files([root])]
            self.assertEqual(found, [
                ("bad.json", False, "result is missing environment, errors, measurements, runs, status, workload"),
                ("good.json", True, None),
                ("nested.json", True, None),
            ])
            explicit = list(wr.iter_result_files([str(good), records, broken, root / "missing.json", text]))
            self.assertEqual([(path.name, result is not None) for path, result, _ in explicit], [
                ("good.json", True), ("records.json", False), ("broken.json", False), ("missing.json", False), ("notes.txt", True),
            ])
            self.assertEqual(explicit[1][2], f"format must be {wr.FORMAT!r}")
            self.assertTrue(explicit[2][2].startswith("cannot read workload result"))
            self.assertTrue(explicit[3][2].startswith("cannot read workload result"))

    def test_result_arguments_build_experiment_blocks(self):
        parser = argparse.ArgumentParser()
        wr.add_result_argument(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.experiment)
        self.assertEqual((args.metadata, args.redact), ([], []))
        self.assertIsNone(wr.experiment_from_args(args))
        self.assertIsNone(wr.experiment_from_args(argparse.Namespace(result=None)))
        args = parser.parse_args([
            "--experiment", "sweep", "--metadata", "variant=a", "--metadata", "size=5",
            "--metadata", "owner=me", "--metadata", "api_key=s3cret", "--redact", "owner",
        ])
        self.assertEqual(args.metadata, [("variant", "a"), ("size", 5), ("owner", "me"), ("api_key", "s3cret")])
        self.assertEqual(wr.experiment_from_args(args), {
            "group": "sweep", "metadata": {"variant": "a", "size": 5, "owner": wr.REDACTED, "api_key": wr.REDACTED},
        })
        self.assertEqual(wr.experiment_from_args(parser.parse_args(["--metadata", "model=x"])), {"group": None, "metadata": {"model": "x"}})
        self.assertEqual(wr.experiment_from_args(parser.parse_args(["--experiment", "g"])), {"group": "g", "metadata": {}})
        for argv, message in (
            (["--metadata", "Bad=1"], "metadata key 'Bad' must match"),
            (["--metadata", "novalue"], "metadata 'novalue' must be KEY=VALUE"),
            (["--metadata", "prompt=" + "x" * 300], "at most 256 characters"),
            (["--metadata", "repository_revision=12e5678"], "must be a finite number"),
            (["--metadata", "started_at=2026"], "started_at must be an ISO 8601 timestamp"),
            (["--experiment", ""], "experiment group must not be empty"),
            (["--redact", "Bad"], "metadata key 'Bad' must match"),
        ):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as raised:
                parser.parse_args(argv)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, stderr.getvalue())

    def test_write_and_build_attach_experiment_blocks(self):
        block = wr.experiment("g", {"variant": "a"})
        built = build_result(workload="w", kind="benchmark", producer="p", experiment=block)
        self.assertEqual(built["experiment"], block)
        self.assertNotIn("experiment", build_result(workload="w", kind="benchmark", producer="p"))
        result = sample_result()
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            write_result(result, "-", experiment=block)
        self.assertEqual(json.loads(stdout.getvalue())["experiment"], block)
        # The caller's document is left untouched.
        self.assertNotIn("experiment", result)
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            write_result(result, "-")
        self.assertNotIn("experiment", json.loads(stdout.getvalue()))
        with self.assertRaisesRegex(WorkloadResultError, "looks like a credential"):
            write_result(result, "-", experiment={"group": "g", "metadata": {"api_key": "s3cret"}})
        # A producer's own block is merged with the command line, which wins key by key.
        produced = build_result(workload="w", kind="benchmark", producer="p", experiment=wr.experiment("nightly", {"tool_version": "0.4.0", "model": "old"}))
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            write_result(produced, "-", experiment=wr.experiment(None, {"model": "new"}))
        self.assertEqual(json.loads(stdout.getvalue())["experiment"], {"group": "nightly", "metadata": {"tool_version": "0.4.0", "model": "new"}})
        self.assertEqual(produced["experiment"]["metadata"]["model"], "old")
        self.assertEqual(wr.merge_experiment(None, None), None)
        self.assertEqual(wr.merge_experiment(None, block), block)
        self.assertEqual(wr.merge_experiment(block, None), block)
        self.assertEqual(wr.merge_experiment(block, wr.experiment("other")), {"group": "other", "metadata": {"variant": "a"}})


class TestSchemaFile(unittest.TestCase):
    def test_schema_constants_match_the_reference_validator(self):
        self.assertEqual(SCHEMA["properties"]["format"]["const"], wr.FORMAT)
        self.assertEqual(SCHEMA["properties"]["format_version"]["const"], wr.FORMAT_VERSION)
        self.assertEqual(SCHEMA["$defs"]["status"]["enum"], list(wr.STATUSES))
        self.assertEqual(SCHEMA["properties"]["workload"]["properties"]["kind"]["enum"], list(wr.KINDS))
        for definition in ("measurement", "measurement_fields"):
            properties = SCHEMA["$defs"][definition]["properties"]
            self.assertEqual(properties["unit"]["enum"], list(wr.UNITS))
            self.assertEqual([rule["required"][0] for rule in SCHEMA["$defs"][definition]["anyOf"]], list(wr.STATISTICS))
            self.assertTrue(set(wr.STATISTICS) <= set(properties))
        self.assertEqual(SCHEMA["$defs"]["name"]["pattern"], wr._NAME.pattern)
        self.assertTrue(set(wr.CANONICAL_MEASUREMENTS.values()) <= set(wr.UNITS))
        # Canonical names pin their unit in the schema exactly as the validator does.
        pinned = SCHEMA["$defs"]["measurements"]["properties"]
        self.assertEqual(set(pinned), set(wr.CANONICAL_MEASUREMENTS))
        for name, unit in wr.CANONICAL_MEASUREMENTS.items():
            self.assertEqual(pinned[name], {"$ref": "#/$defs/measurement", "properties": {"unit": {"const": unit}}})
        self.assertTrue(SCHEMA["properties"]["runs"]["uniqueItems"])
        experiment = SCHEMA["properties"]["experiment"]
        self.assertEqual(experiment["required"], ["group", "metadata"])
        metadata = experiment["properties"]["metadata"]
        self.assertEqual(metadata["propertyNames"]["pattern"], wr._NAME.pattern)
        self.assertEqual(metadata["additionalProperties"]["maxLength"], wr.MAX_METADATA_LENGTH)
        self.assertIn(wr.REDACTED, metadata["additionalProperties"]["description"])
        for key in wr.CANONICAL_METADATA:
            self.assertIn(key, metadata["additionalProperties"]["description"])

    def test_schema_accepts_sample_and_rejects_invalid_results(self):
        if importlib.util.find_spec("jsonschema") is None:
            self.skipTest("jsonschema is not installed")
        import jsonschema

        validator = jsonschema.Draft202012Validator(SCHEMA)
        validator.validate(sample_result())
        validator.validate(sample_result(experiment=wr.experiment("g", {"variant": "a", "size": 1, "flag": True, "none": None, "api_key": "x"})))
        validator.validate(sample_result(experiment=wr.experiment(None, {"model": "m"})))
        for invalid in (
            sample_result(format="other"),
            sample_result(measurements={"x": {"unit": "bytes"}}),
            sample_result(measurements={"wall_time_seconds": {"unit": "bytes", "value": 1}}),
            sample_result(runs=[run("r", measurements={"output_bytes": measurement("seconds", value=1)})]),
            sample_result(runs=[run("r"), run("r")]),
            sample_result(runs=[{**run("r"), "extra": 1}]),
            sample_result(runs=[run("r", phases=[{"name": "p", "unit": "bytes", "value": 1}])]),
            sample_result(experiment={"group": "g"}),
            sample_result(experiment={"group": "", "metadata": {}}),
            sample_result(experiment={"group": "g", "metadata": {"nested": {}}}),
            sample_result(experiment={"group": "g", "metadata": {"Bad": 1}}),
            sample_result(experiment={"group": "g", "metadata": {"prompt": "x" * 257}}),
        ):
            self.assertTrue(list(validator.iter_errors(invalid)), invalid)
        # A canonical measurement in a run is accepted when its unit matches.
        validator.validate(sample_result(runs=[run("r", measurements={"output_bytes": measurement("bytes", value=1)})]))
        # Run id uniqueness is beyond JSON Schema: two runs that share an id but differ
        # elsewhere pass the schema, so consumers need the reference validator for it.
        same_id = sample_result(runs=[run("r", labels={"variant": "a"}), run("r", labels={"variant": "b"})])
        validator.validate(same_id)
        with self.assertRaisesRegex(WorkloadResultError, "runs\\[1\\].id 'r' is duplicated"):
            wr.validate_result(same_id)


class TestCliHelpers(unittest.TestCase):
    def test_result_argument_and_streams(self):
        import argparse

        parser = argparse.ArgumentParser()
        wr.add_result_argument(parser)
        self.assertIsNone(parser.parse_args([]).result)
        self.assertEqual(parser.parse_args(["--result", "-"]).result, "-")
        self.assertIs(wr.report_stream("-"), wr.sys.stderr)
        self.assertIs(wr.report_stream(None), wr.sys.stdout)
        self.assertIs(wr.report_stream(Path("out.json")), wr.sys.stdout)
        self.assertFalse(wr.result_to_stdout(None))

    def test_write_result_targets_file_or_stdout_or_nothing(self):
        result = sample_result()
        write_result(result, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_result(result, path)
            self.assertEqual(wr.load_result(path), result)
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                write_result(result, "-")
            self.assertEqual(json.loads(stdout.getvalue()), result)
            self.assertTrue(stdout.getvalue().endswith("\n"))
        with self.assertRaisesRegex(WorkloadResultError, "format must be"):
            write_result(sample_result(format="x"), "-")

    def test_load_result_reports_unreadable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaisesRegex(WorkloadResultError, "cannot read"):
                wr.load_result(missing)
            bad = Path(directory) / "bad.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(WorkloadResultError, "cannot read"):
                wr.load_result(bad)

    def test_main_validates_files_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            write_result(sample_result(), good)
            bad = Path(directory) / "bad.json"
            bad.write_text("[]", encoding="utf-8")
            with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(wr.main([str(good), str(bad)]), 1)
            self.assertIn("workload=sample kind=benchmark producer=test status=partial runs=2 errors=0\n", stdout.getvalue())
            self.assertIn("INVALID: result must be an object", stderr.getvalue())
            grouped = Path(directory) / "grouped.json"
            write_result(sample_result(), grouped, experiment=wr.experiment("sweep", {"variant": "a"}))
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(wr.main([str(grouped)]), 0)
            self.assertIn("errors=0 group=sweep variant=a\n", stdout.getvalue())
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(wr.main([str(good)]), 0)

    def test_module_runs_as_a_script(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            write_result(sample_result(), good)
            with patch("sys.argv", ["workload_results.py", str(good)]), patch(
                "sys.stdout", new_callable=io.StringIO
            ) as stdout, self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(Path(__file__).with_name("workload_results.py")), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("workload=sample", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
