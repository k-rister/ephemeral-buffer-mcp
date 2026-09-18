"""Tests for the versioned coding-agent workload result format."""

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
        with patch.object(wr.Path, "open", side_effect=OSError("missing")):
            with patch.object(wr._metadata, "version", return_value="9.9.9"):
                self.assertEqual(wr._project_version(), "9.9.9")
            with patch.object(wr._metadata, "version", side_effect=wr._metadata.PackageNotFoundError("x")):
                self.assertIsNone(wr._project_version())

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

    def test_schema_accepts_sample_and_rejects_invalid_results(self):
        if importlib.util.find_spec("jsonschema") is None:
            self.skipTest("jsonschema is not installed")
        import jsonschema

        validator = jsonschema.Draft202012Validator(SCHEMA)
        validator.validate(sample_result())
        for invalid in (
            sample_result(format="other"),
            sample_result(measurements={"x": {"unit": "bytes"}}),
            sample_result(runs=[{**run("r"), "extra": 1}]),
            sample_result(runs=[run("r", phases=[{"name": "p", "unit": "bytes", "value": 1}])]),
        ):
            self.assertTrue(list(validator.iter_errors(invalid)), invalid)


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
            self.assertIn("workload=sample kind=benchmark producer=test status=partial runs=2 errors=0", stdout.getvalue())
            self.assertIn("INVALID: result must be an object", stderr.getvalue())
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
