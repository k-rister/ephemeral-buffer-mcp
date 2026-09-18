"""Tests for comparing coding-agent workload results."""

import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_workload_results as cw
import workload_results as wr
from compare_workload_results import ComparisonError


ENVIRONMENT = {
    "python_version": "3.12.14",
    "platform": "Linux-6.1-x86_64",
    "machine": "x86_64",
    "cpu_count": 8,
    "recorded_at": "2026-09-18T15:00:00+00:00",
    "tool": {"name": wr.PROJECT_NAME, "version": "0.4.0"},
    "source_revision": "a" * 40,
}


def latency_result(*, wall=0.010, p95=0.015, output=3840, tokens=None, phases=None, extra_runs=(), status=None, environment=None, samples=3, **workload):
    """A capture-latency style result whose numbers the tests vary."""
    measurements = {
        "wall_time_seconds": wr.measurement("seconds", median=wall, p95=p95, samples=samples),
        "output_bytes": wr.measurement("bytes", value=output),
        "estimated_tokens": wr.unavailable("tokens", "no model") if tokens is None else wr.measurement("tokens", value=tokens),
    }
    runs = [
        wr.run(
            "lines-256",
            labels={"cache_state": "warm", "line_count": 256},
            measurements=measurements,
            phases=phases if phases is not None else [wr.phase("command", median=0.004, p95=0.005), wr.phase("ingest", median=0.006, p95=0.008)],
        ),
        *extra_runs,
    ]
    return wr.build_result(
        workload=workload.pop("workload", "capture-latency"),
        kind="benchmark",
        producer="benchmark_latency.py",
        parameters={"line_counts": [256], "samples": samples},
        environment=environment or ENVIRONMENT,
        measurements={"cold_start_seconds": wr.measurement("seconds", value=1.5)},
        runs=runs,
        status=status,
        **workload,
    )


def agent_result(control_tokens=1000, mcp_tokens=600, control_status="success", mcp_status="success"):
    """An A/B summary style result with one run per configuration."""
    return wr.build_result(
        workload="agent-ab",
        kind="evaluation",
        producer="benchmark_agent_ab.py",
        parameters={"seed": 1, "repetitions": 2},
        environment=ENVIRONMENT,
        runs=[
            wr.run(
                "control",
                labels={"mode": "control"},
                status=control_status,
                measurements={
                    "input_tokens": wr.measurement("tokens", mean=control_tokens, stdev=10.0, samples=2),
                    "success_rate": wr.measurement("ratio", value=1.0, samples=2),
                    "wall_time_seconds": wr.measurement("seconds", mean=30.0, samples=2),
                },
                errors=[] if control_status == "success" else ["timeout: 1"],
            ),
            wr.run(
                "mcp",
                labels={"mode": "mcp"},
                status=mcp_status,
                measurements={
                    "input_tokens": wr.measurement("tokens", mean=mcp_tokens, stdev=10.0, samples=2),
                    "success_rate": wr.measurement("ratio", value=0.5 if mcp_status != "success" else 1.0, samples=2),
                    "wall_time_seconds": wr.measurement("seconds", mean=24.0, samples=2),
                },
            ),
        ],
    )


class ResultFiles(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def write(self, name, result):
        path = self.directory / name
        wr.write_result(result, path)
        return str(path)

    def compare(self, *results, **options):
        references = [self.write(f"r{index}.json", result) for index, result in enumerate(results)]
        return cw.compare(references, **options)


class TestParsing(unittest.TestCase):
    def test_parse_reference_prefers_existing_paths_over_run_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            hashed = Path(directory) / "result#1.json"
            hashed.write_text("{}", encoding="utf-8")
            self.assertEqual(cw.parse_reference(str(hashed)), (hashed, None))
            self.assertEqual(cw.parse_reference(f"{hashed}#control"), (hashed, "control"))
        self.assertEqual(cw.parse_reference("plain.json"), (Path("plain.json"), None))
        for bad in ("#run", "path#"):
            with self.assertRaisesRegex(ComparisonError, "must be PATH or PATH#RUN_ID"):
                cw.parse_reference(bad)

    def test_parse_label_filter_reads_json_values_or_strings(self):
        self.assertEqual(cw.parse_label_filter("line_count=256"), ("line_count", 256))
        self.assertEqual(cw.parse_label_filter("mode=mcp"), ("mode", "mcp"))
        self.assertEqual(cw.parse_label_filter("flag=true"), ("flag", True))
        self.assertEqual(cw.parse_label_filter("missing=null"), ("missing", None))
        self.assertEqual(cw.parse_label_filter("profile="), ("profile", ""))
        with self.assertRaisesRegex(ComparisonError, "must be KEY=VALUE"):
            cw.parse_label_filter("mode")
        with self.assertRaisesRegex(ComparisonError, "must be KEY=VALUE"):
            cw.parse_label_filter("=x")
        with self.assertRaisesRegex(ComparisonError, "string, number, boolean, or null"):
            cw.parse_label_filter("sizes=[1, 2]")

    def test_parse_direction(self):
        self.assertEqual(cw.parse_direction("needle_rank=lower"), ("needle_rank", "lower"))
        for bad in ("needle_rank", "=lower", "needle_rank=sideways"):
            with self.assertRaisesRegex(ComparisonError, "NAME=lower or NAME=higher"):
                cw.parse_direction(bad)


class TestDirectionsAndClassification(unittest.TestCase):
    def test_direction_for_uses_override_then_canonical_then_unit(self):
        self.assertEqual(cw.direction_for("wall_time_seconds", "seconds"), "lower")
        self.assertEqual(cw.direction_for("success_rate", "ratio"), "higher")
        self.assertIsNone(cw.direction_for("input_bytes", "bytes"))
        self.assertEqual(cw.direction_for("first_search_seconds", "seconds"), "lower")
        self.assertEqual(cw.direction_for("indexing_rate", "per_second"), "higher")
        self.assertIsNone(cw.direction_for("needle_rank", "count"))
        self.assertIsNone(cw.direction_for("overhead", "ratio"))
        self.assertIsNone(cw.direction_for("quality", "score"))
        self.assertEqual(cw.direction_for("needle_rank", "count", {"needle_rank": "lower"}), "lower")
        self.assertEqual(cw.direction_for("wall_time_seconds", "seconds", {"wall_time_seconds": "higher"}), "higher")
        self.assertEqual(set(cw.UNIT_DIRECTIONS), set(wr.UNITS))

    def test_label_matches_keeps_booleans_and_numbers_apart(self):
        self.assertTrue(cw.label_matches(1, 1))
        self.assertTrue(cw.label_matches(256, 256.0))
        self.assertTrue(cw.label_matches("mcp", "mcp"))
        self.assertTrue(cw.label_matches(None, None))
        self.assertTrue(cw.label_matches(True, True))
        self.assertFalse(cw.label_matches(True, 1))
        self.assertFalse(cw.label_matches(1, True))
        self.assertFalse(cw.label_matches(0, False))
        self.assertFalse(cw.label_matches(False, 0))

    def test_classify_covers_every_outcome(self):
        self.assertEqual(cw.classify(0, 0.0, "lower", 0.0), "unchanged")
        self.assertEqual(cw.classify(0.5, 5.0, "lower", 5.0), "unchanged")
        self.assertEqual(cw.classify(-0.5, -5.0, "lower", 5.0), "unchanged")
        self.assertEqual(cw.classify(0.6, 6.0, "lower", 5.0), "regressed")
        self.assertEqual(cw.classify(-0.6, -6.0, "lower", 5.0), "improved")
        self.assertEqual(cw.classify(0.6, 6.0, "higher", 5.0), "improved")
        self.assertEqual(cw.classify(-0.6, -6.0, "higher", 5.0), "regressed")
        self.assertEqual(cw.classify(0.6, 6.0, None, 5.0), "changed")
        # A zero baseline has no percentage; any change then counts.
        self.assertEqual(cw.classify(0.001, None, "lower", 50.0), "regressed")
        self.assertEqual(cw.classify(0.001, None, None, 50.0), "changed")


class TestCompareMeasurement(unittest.TestCase):
    def entries(self, baseline, candidate, **options):
        options.setdefault("statistics", cw.DEFAULT_STATISTICS)
        options.setdefault("tolerance_percent", 0.0)
        return cw.compare_measurement("metric", baseline, candidate, **options)

    def test_reports_absent_sides_and_unit_conflicts(self):
        seconds = wr.measurement("seconds", value=1.0)
        [entry] = self.entries(seconds, None)
        self.assertEqual((entry["outcome"], entry["reason"], entry["unit"]), ("missing", "absent in candidate", "seconds"))
        [entry] = self.entries(None, seconds)
        self.assertEqual((entry["outcome"], entry["reason"]), ("missing", "absent in baseline"))
        [entry] = self.entries(seconds, wr.measurement("bytes", value=1.0))
        self.assertEqual(entry["outcome"], "incompatible")
        self.assertEqual(entry["reason"], "unit seconds in baseline but bytes in candidate")
        self.assertIsNone(entry["statistic"])

    def test_reports_unavailable_statistics_per_side(self):
        entries = self.entries(
            wr.measurement("tokens", value=None, median=5.0, mean=2.0, samples=3),
            wr.measurement("tokens", value=4.0, median=None, p95=9.0),
        )
        by_statistic = {entry["statistic"]: entry for entry in entries}
        self.assertEqual(set(by_statistic), {"value", "median", "mean", "p95"})
        self.assertEqual(by_statistic["value"]["reason"], "unavailable in baseline")
        self.assertEqual(by_statistic["median"]["reason"], "unavailable in candidate")
        self.assertEqual(by_statistic["mean"]["reason"], "unavailable in candidate")
        self.assertEqual(by_statistic["p95"]["reason"], "unavailable in baseline")
        self.assertTrue(all(entry["outcome"] == "missing" for entry in entries))
        [entry] = self.entries(wr.unavailable("tokens"), wr.unavailable("tokens"))
        self.assertEqual((entry["statistic"], entry["reason"]), ("value", "unavailable in both"))
        # samples: 0 marks a numeric statistic unavailable too.
        [entry] = self.entries(wr.measurement("count", value=3, samples=0), wr.measurement("count", value=3, samples=1))
        self.assertEqual(entry["reason"], "unavailable in baseline")

    def test_computes_absolute_and_percentage_deltas(self):
        [entry] = self.entries(wr.measurement("seconds", median=2.0), wr.measurement("seconds", median=2.5))
        self.assertEqual(entry["delta"], 0.5)
        self.assertEqual(entry["delta_percent"], 25.0)
        self.assertEqual(entry["outcome"], "regressed")
        [entry] = self.entries(wr.measurement("seconds", median=2.0), wr.measurement("seconds", median=2.5), tolerance_percent=30)
        self.assertEqual(entry["outcome"], "unchanged")
        [entry] = self.entries(wr.measurement("count", value=0), wr.measurement("count", value=2))
        self.assertIsNone(entry["delta_percent"])
        self.assertEqual(entry["outcome"], "changed")
        [entry] = self.entries(wr.measurement("count", value=0), wr.measurement("count", value=2), directions={"metric": "higher"})
        self.assertEqual((entry["direction"], entry["outcome"]), ("higher", "improved"))
        # Only the requested statistics are compared.
        self.assertEqual(self.entries(wr.measurement("seconds", stdev=1.0), wr.measurement("seconds", stdev=2.0)), [])
        # Dispersion has no better direction, whatever the metric's level direction says.
        [entry] = self.entries(wr.measurement("seconds", stdev=1.0), wr.measurement("seconds", stdev=2.0), statistics=("stdev",))
        self.assertEqual((entry["direction"], entry["outcome"]), (None, "changed"))
        [entry] = self.entries(wr.measurement("ratio", stdev=0.1), wr.measurement("ratio", stdev=0.3), statistics=("stdev",), directions={"metric": "higher"})
        self.assertEqual((entry["direction"], entry["outcome"]), (None, "changed"))
        [entry] = self.entries(wr.measurement("ratio", stdev=0.1), wr.measurement("ratio", stdev=0.1), statistics=("stdev",))
        self.assertEqual(entry["outcome"], "unchanged")

    def test_non_finite_deltas_stay_json_serializable(self):
        # A subnormal baseline overflows the percentage; the absolute delta survives.
        [entry] = self.entries(wr.measurement("seconds", median=5e-324), wr.measurement("seconds", median=1.0))
        self.assertIsNone(entry["delta_percent"])
        self.assertAlmostEqual(entry["delta"], 1.0)
        self.assertEqual(entry["outcome"], "regressed")
        # Values at the float limits overflow the delta itself.
        [entry] = self.entries(wr.measurement("count", value=-1e308), wr.measurement("count", value=1e308))
        self.assertIsNone(entry["delta"])
        self.assertEqual((entry["outcome"], entry["reason"]), ("changed", "difference is too large to represent"))
        json.dumps(entry, allow_nan=False)


class TestPairRuns(unittest.TestCase):
    def document(self, run_ids, selected=None):
        return {"selected_run": selected, "runs": [wr.run(run_id) for run_id in run_ids]}

    def test_pairs_by_id_unless_both_sides_select_one_run(self):
        pairs = cw.pair_runs(self.document(["a", "b"]), self.document(["b", "c"]))
        self.assertEqual([(pair_id, before is not None, after is not None) for pair_id, before, after in pairs],
                         [("a", True, False), ("b", True, True), ("c", False, True)])
        [(pair_id, before, after)] = cw.pair_runs(self.document(["control"], "control"), self.document(["mcp"], "mcp"))
        self.assertEqual((pair_id, before["id"], after["id"]), ("control -> mcp", "control", "mcp"))
        [(pair_id, _, _)] = cw.pair_runs(self.document(["x"], "x"), self.document(["x"], "x"))
        self.assertEqual(pair_id, "x")
        # Selecting on one side only still pairs by id.
        pairs = cw.pair_runs(self.document(["control"], "control"), self.document(["control", "mcp"]))
        self.assertEqual([pair_id for pair_id, _, _ in pairs], ["control", "mcp"])


class TestCompare(ResultFiles):
    def test_equal_results_are_unchanged(self):
        comparison = self.compare(latency_result(), latency_result())
        self.assertEqual(comparison["format"], cw.FORMAT)
        self.assertEqual(comparison["format_version"], cw.FORMAT_VERSION)
        self.assertEqual(comparison["workload"], "capture-latency")
        [item] = comparison["comparisons"]
        summary = item["summary"]
        self.assertEqual(summary["regressed"], 0)
        self.assertEqual(summary["improved"], 0)
        self.assertEqual(summary["incompatible"], 0)
        self.assertEqual(summary["runs_compared"], 1)
        self.assertEqual(summary["runs_missing"], 0)
        self.assertTrue(summary["all_succeeded"])
        # cold_start, wall median+p95, output, two phases x two statistics
        self.assertEqual(summary["unchanged"], 1 + 2 + 1 + 4)
        # estimated_tokens is unavailable on both sides and is listed, not hidden.
        [missing] = [entry for entry in item["entries"] if entry["outcome"] == "missing"]
        self.assertEqual((missing["name"], missing["reason"]), ("estimated_tokens", "unavailable in both"))
        self.assertEqual(item["parameter_differences"], {})
        self.assertEqual(item["environment_differences"], {})
        self.assertTrue(cw.check_passes(comparison))
        self.assertEqual([document["role"] for document in comparison["documents"]], ["baseline", "candidate"])
        json.dumps(comparison, allow_nan=False)

    def test_latency_regression_is_identified_with_phase_attribution(self):
        slower = latency_result(wall=0.012, p95=0.030, phases=[wr.phase("command", median=0.004, p95=0.005), wr.phase("ingest", median=0.008, p95=0.023)])
        comparison = self.compare(latency_result(), slower, tolerance_percent=5.0)
        [item] = comparison["comparisons"]
        outcomes = {(entry["kind"], entry["name"], entry["statistic"]): entry["outcome"] for entry in item["entries"] if entry["run"]}
        self.assertEqual(outcomes[("measurement", "wall_time_seconds", "median")], "regressed")
        self.assertEqual(outcomes[("measurement", "wall_time_seconds", "p95")], "regressed")
        self.assertEqual(outcomes[("phase", "command", "median")], "unchanged")
        self.assertEqual(outcomes[("phase", "ingest", "median")], "regressed")
        self.assertEqual(outcomes[("phase", "ingest", "p95")], "regressed")
        self.assertEqual(comparison["summary"]["regressions"], 4)
        self.assertFalse(cw.check_passes(comparison))

    def test_latency_improvement_and_token_saving_are_identified(self):
        comparison = self.compare(latency_result(tokens=900, output=4000), latency_result(wall=0.008, tokens=600, output=3000))
        [item] = comparison["comparisons"]
        outcomes = {(entry["name"], entry["statistic"]): entry for entry in item["entries"] if entry["run"]}
        self.assertEqual(outcomes[("wall_time_seconds", "median")]["outcome"], "improved")
        self.assertEqual(outcomes[("estimated_tokens", "value")]["outcome"], "improved")
        self.assertAlmostEqual(outcomes[("estimated_tokens", "value")]["delta_percent"], -100 / 3)
        self.assertEqual(outcomes[("output_bytes", "value")]["delta"], -1000)
        self.assertEqual(item["summary"]["improved"], 3)
        self.assertEqual(item["summary"]["regressed"], 0)

    def test_token_regression_between_agent_configurations_in_one_document(self):
        path = self.write("ab.json", agent_result(control_tokens=1000, mcp_tokens=1300))
        comparison = cw.compare([f"{path}#control", f"{path}#mcp"])
        [item] = comparison["comparisons"]
        [record] = item["runs"]
        self.assertEqual((record["id"], record["baseline_run"], record["candidate_run"]), ("control -> mcp", "control", "mcp"))
        self.assertEqual(record["labels"], [{"mode": "control"}, {"mode": "mcp"}])
        by_name = {entry["name"]: entry for entry in item["entries"]}
        self.assertEqual(by_name["input_tokens"]["outcome"], "regressed")
        self.assertEqual(by_name["input_tokens"]["delta"], 300)
        self.assertEqual(by_name["input_tokens"]["delta_percent"], 30.0)
        self.assertEqual(by_name["wall_time_seconds"]["outcome"], "improved")
        self.assertEqual(by_name["success_rate"]["outcome"], "unchanged")
        self.assertEqual(comparison["documents"][0]["selected_run"], "control")
        self.assertEqual(comparison["documents"][1]["runs"], ["mcp"])

    def test_incomplete_comparisons_report_missing_runs_metrics_and_statuses(self):
        extra = wr.run("lines-2048", labels={"line_count": 2048}, status="timeout", errors=["exceeded 30 s"])
        baseline = latency_result(extra_runs=[wr.run("lines-16", labels={"line_count": 16}, measurements={"wall_time_seconds": wr.measurement("seconds", median=0.001)})])
        candidate = latency_result(extra_runs=[extra])
        # The candidate reports the same run with a different unit and a missing measurement.
        baseline["runs"][0]["measurements"]["payload"] = wr.measurement("bytes", value=3840)
        candidate["runs"][0]["measurements"]["payload"] = wr.measurement("count", value=3840)
        del candidate["runs"][0]["measurements"]["estimated_tokens"]
        candidate["runs"][0]["measurements"]["retained_summary_tokens"] = wr.measurement("tokens", value=12)
        candidate["status"] = "partial"
        comparison = self.compare(baseline, candidate)
        [item] = comparison["comparisons"]
        runs = {record["id"]: record for record in item["runs"]}
        self.assertEqual(runs["lines-16"]["outcome"], "missing")
        self.assertEqual(runs["lines-16"]["reason"], "absent in candidate")
        self.assertEqual(runs["lines-2048"]["reason"], "absent in baseline")
        self.assertEqual(runs["lines-2048"]["candidate_status"], "timeout")
        self.assertEqual(runs["lines-2048"]["candidate_errors"], ["exceeded 30 s"])
        self.assertIsNone(runs["lines-2048"]["baseline_status"])
        by_name = {entry["name"]: entry for entry in item["entries"] if entry["run"] == "lines-256"}
        self.assertEqual(by_name["payload"]["outcome"], "incompatible")
        self.assertEqual(by_name["estimated_tokens"]["reason"], "absent in candidate")
        self.assertEqual(by_name["retained_summary_tokens"]["reason"], "absent in baseline")
        self.assertEqual(item["summary"]["runs_missing"], 2)
        self.assertEqual(item["summary"]["incompatible"], 1)
        self.assertEqual(item["summary"]["missing"], 2)
        self.assertEqual(item["summary"]["statuses"], ["success", "partial"])
        self.assertFalse(item["summary"]["all_succeeded"])
        self.assertEqual(comparison["summary"]["regressions"], 0)
        # A non-success document fails the check even without regressions.
        self.assertFalse(cw.check_passes(comparison))

    def test_metadata_differences_and_metric_filters(self):
        other_environment = {**ENVIRONMENT, "cpu_count": 16, "recorded_at": "2026-09-19T00:00:00+00:00", "source_revision": "b" * 40, "embedding_threads": 4}
        candidate = latency_result(environment=other_environment, samples=5)
        comparison = self.compare(latency_result(), candidate, metrics=["wall_time_seconds", "ingest"])
        [item] = comparison["comparisons"]
        self.assertEqual(item["parameter_differences"], {"samples": [3, 5]})
        self.assertEqual(item["environment_differences"], {
            "cpu_count": [8, 16],
            "source_revision": ["a" * 40, "b" * 40],
            "embedding_threads": [None, 4],
        })
        self.assertEqual({entry["name"] for entry in item["entries"]}, {"wall_time_seconds", "ingest"})
        self.assertEqual(comparison["options"]["metrics"], ["ingest", "wall_time_seconds"])

    def test_metrics_iterator_applies_to_every_candidate(self):
        comparison = self.compare(latency_result(), latency_result(), latency_result(), metrics=iter(["wall_time_seconds", "wall_time_seconds"]))
        self.assertEqual(comparison["options"]["metrics"], ["wall_time_seconds"])
        for item in comparison["comparisons"]:
            self.assertEqual({entry["name"] for entry in item["entries"]}, {"wall_time_seconds"})

    def test_narrowed_documents_are_judged_by_their_selected_runs(self):
        path = self.write("ab.json", agent_result(control_status="partial", mcp_status="success"))
        whole = cw.compare([path, path])
        self.assertEqual(whole["comparisons"][0]["summary"]["statuses"], ["partial", "partial"])
        self.assertFalse(cw.check_passes(whole))
        narrowed = cw.compare([f"{path}#mcp", f"{path}#mcp"])
        self.assertEqual([document["status"] for document in narrowed["documents"]], ["success", "success"])
        self.assertEqual([document["document_status"] for document in narrowed["documents"]], ["partial", "partial"])
        self.assertEqual(narrowed["comparisons"][0]["summary"]["statuses"], ["success", "success"])
        self.assertTrue(cw.check_passes(narrowed))
        self.assertIn("status=success  document_status=partial  runs=1", cw.format_report(narrowed))
        filtered = cw.compare([path, path], label_filters=[("mode", "control")])
        self.assertEqual(filtered["comparisons"][0]["summary"]["statuses"], ["partial", "partial"])
        self.assertEqual(cw.selected_status([wr.run("a", status="timeout")], []), "timeout")
        self.assertEqual(cw.selected_status([wr.run("a", status="timeout"), wr.run("b", status="failure")], []), "failure")
        self.assertEqual(cw.selected_status([wr.run("a"), wr.run("b", status="failure")], []), "partial")
        self.assertEqual(cw.selected_status([wr.run("a")], ["boom"]), "error")
        self.assertEqual(cw.selected_status([], []), "success")
        # Top-level errors still make a narrowed document an error.
        errored = agent_result()
        errored["errors"] = ["harness crashed"]
        errored["status"] = "error"
        error_path = self.write("err.json", errored)
        self.assertEqual(cw.compare([f"{error_path}#mcp", f"{error_path}#mcp"])["documents"][0]["status"], "error")

    def test_same_file_is_loaded_once_per_comparison(self):
        path = self.write("ab.json", agent_result())
        with patch.object(cw.wr, "load_result", wraps=cw.wr.load_result) as load:
            cw.compare([f"{path}#control", f"{path}#mcp", path])
        self.assertEqual(load.call_count, 1)
        # Without a cache every call loads.
        self.assertEqual(cw.load_document(path)["runs"][0]["id"], "control")

    def test_label_filters_and_multiple_candidates(self):
        results = [
            agent_result(mcp_tokens=600),
            agent_result(mcp_tokens=500),
            agent_result(mcp_tokens=700),
        ]
        comparison = self.compare(*results, label_filters=[("mode", "mcp")], statistics=("mean",))
        self.assertEqual(len(comparison["comparisons"]), 2)
        self.assertEqual([document["runs"] for document in comparison["documents"]], [["mcp"], ["mcp"], ["mcp"]])
        self.assertEqual(comparison["options"]["label_filters"], [["mode", "mcp"]])
        improved, regressed = comparison["comparisons"]
        self.assertEqual({entry["outcome"] for entry in improved["entries"] if entry["name"] == "input_tokens"}, {"improved"})
        self.assertEqual({entry["outcome"] for entry in regressed["entries"] if entry["name"] == "input_tokens"}, {"regressed"})
        self.assertEqual(comparison["summary"]["regressions"], 1)
        # A filter that matches nothing leaves nothing to compare rather than failing.
        empty = self.compare(agent_result(), agent_result(), label_filters=[("mode", "other")])
        self.assertEqual(empty["comparisons"][0]["runs"], [])

    def test_rejects_unusable_inputs(self):
        path = self.write("a.json", latency_result())
        with self.assertRaisesRegex(ComparisonError, "at least two references"):
            cw.compare([path])
        with self.assertRaisesRegex(ComparisonError, "tolerance must not be negative"):
            cw.compare([path, path], tolerance_percent=-1)
        with self.assertRaisesRegex(ComparisonError, "unknown statistic average"):
            cw.compare([path, path], statistics=("median", "average"))
        with self.assertRaisesRegex(ComparisonError, "has no run 'lines-9' \\(available: lines-256\\)"):
            cw.compare([path, f"{path}#lines-9"])
        with self.assertRaisesRegex(wr.WorkloadResultError, "cannot read"):
            cw.compare([path, str(self.directory / "missing.json")])
        other = self.write("b.json", latency_result(workload="semantic-index"))
        with self.assertRaisesRegex(ComparisonError, "measures workload 'semantic-index' but the baseline .* measures 'capture-latency'"):
            cw.compare([path, other])
        forced = cw.compare([path, other], allow_workload_mismatch=True)
        self.assertEqual(forced["workload"], "capture-latency")
        self.assertEqual(forced["comparisons"][0]["summary"]["runs_compared"], 1)
        empty_reference = self.write("empty.json", wr.build_result(workload="capture-latency", kind="benchmark", producer="p", environment=ENVIRONMENT))
        with self.assertRaisesRegex(ComparisonError, "available: none"):
            cw.compare([path, f"{empty_reference}#lines-256"])


class TestReport(ResultFiles):
    def test_format_value_and_delta(self):
        self.assertEqual(cw.format_value(None, "seconds"), "n/a")
        self.assertEqual(cw.format_value(3840, "bytes"), "3,840 B")
        self.assertEqual(cw.format_value(0.012345, "seconds"), "0.01235 s")
        self.assertEqual(cw.format_value(1234.5, "tokens"), "1,234 tok")
        self.assertEqual(cw.format_value(0.95, "ratio"), "0.95")
        self.assertEqual(cw.format_value(12.5, "per_second"), "12.5/s")
        self.assertEqual(cw.format_value(7, None), "7")
        self.assertEqual(cw.format_delta({"delta": None, "delta_percent": None, "unit": None}), ("-", "-"))
        self.assertEqual(cw.format_delta({"delta": 0, "delta_percent": 0.0, "unit": "bytes"}), ("0 B", "0.0%"))
        self.assertEqual(cw.format_delta({"delta": 2, "delta_percent": None, "unit": "count"}), ("+2", "n/a"))
        self.assertEqual(cw.format_delta({"delta": -0.5, "delta_percent": -25.0, "unit": "seconds"}), ("-0.5 s", "-25.0%"))

    def test_report_lists_documents_differences_rows_and_summary(self):
        other_environment = {**ENVIRONMENT, "cpu_count": 16, "tool": {"name": wr.PROJECT_NAME, "version": None}, "source_revision": None}
        baseline = latency_result()
        baseline["runs"][0]["measurements"]["payload"] = wr.measurement("bytes", value=3840)
        candidate = latency_result(wall=0.020, environment=other_environment, samples=5)
        candidate["runs"][0]["measurements"]["payload"] = wr.measurement("count", value=3840)
        candidate["runs"].append(wr.run("lines-2048", labels={"cache_state": "warm"}, status="timeout", errors=["exceeded 30 s"]))
        candidate["status"] = "partial"
        comparison = self.compare(baseline, candidate, metrics=["wall_time_seconds", "payload", "ingest", "estimated_tokens"], label_filters=[("cache_state", "warm")], tolerance_percent=2.5)
        report = cw.format_report(comparison)
        self.assertIn("workload: capture-latency\n", report)
        self.assertIn("baseline: ", report)
        self.assertIn("tool=ephemeral-buffer-mcp 0.4.0  revision=aaaaaaaaaaaa", report)
        self.assertIn("candidate: ", report)
        self.assertIn("status=partial  runs=2", report)
        self.assertNotIn("revision=", report.split("candidate: ", 1)[1].split("\n", 1)[0])
        self.assertIn('statistics=value,median,mean,p95 tolerance=2.5% metrics=estimated_tokens,ingest,payload,wall_time_seconds filters=cache_state="warm"', report)
        self.assertIn("parameter differences: samples: 3 -> 5", report)
        self.assertIn("environment differences: cpu_count: 8 -> 16; source_revision:", report)
        self.assertIn('tool: {"name": "ephemeral-buffer-mcp", "version": "0.4.0"} -> {"name": "ephemeral-buffer-mcp", "version": null}', report)
        self.assertRegex(report, r"lines-2048\s+\(run\)\s+timeout\s+missing: absent in baseline \(exceeded 30 s\)\n")
        self.assertRegex(report, r"lines-256\s+wall_time_seconds\s+median\s+0\.01 s\s+0\.02 s\s+\+0\.01 s\s+\+100\.0%\s+regressed\n")
        self.assertRegex(report, r"lines-256\s+phase:ingest\s+median\s+0\.006 s\s+0\.006 s\s+0 s\s+0\.0%\s+unchanged\n")
        self.assertRegex(report, r"lines-256\s+payload\s+n/a\s+n/a\s+-\s+-\s+incompatible: unit bytes in baseline but count in candidate\n")
        self.assertRegex(report, r"lines-256\s+estimated_tokens\s+value\s+n/a\s+n/a\s+-\s+-\s+missing: unavailable in both\n")
        self.assertIn("summary: 0 improved, 1 regressed, 0 changed, 3 unchanged, 1 missing, 1 incompatible; runs compared=1 missing=1; status success -> partial", report)
        self.assertTrue(report.endswith("\n"))

    def test_report_shows_run_status_rows_and_empty_comparisons(self):
        path = self.write("ab.json", agent_result(control_status="partial", mcp_status="failure"))
        comparison = cw.compare([f"{path}#control", f"{path}#mcp"], statistics=("mean",))
        report = cw.format_report(comparison)
        self.assertRegex(report, r"control -> mcp\s+\(run status\)\s+partial\s+failure\s+timeout: 1\n")
        self.assertIn("timeout: 1", report)
        self.assertIn("input_tokens", report)
        without_errors = agent_result(mcp_status="partial")
        report = cw.format_report(self.compare(without_errors, without_errors, metrics=["nothing"]))
        self.assertRegex(report, r"\n\s+mcp\s+\(run status\)\s+partial\s+partial\s+non-success status\n")
        empty = self.compare(agent_result(), agent_result(), label_filters=[("mode", "other")])
        self.assertIn("  nothing to compare\n", cw.format_report(empty))


class TestMain(ResultFiles):
    def run_main(self, *argv):
        with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = cw.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_text_json_and_output_file(self):
        baseline = self.write("a.json", latency_result())
        candidate = self.write("b.json", latency_result(wall=0.011))
        code, stdout, stderr = self.run_main(baseline, candidate)
        self.assertEqual(code, cw.EXIT_OK)
        self.assertRegex(stdout, r"wall_time_seconds\s+median")
        self.assertIn("regressed", stdout)
        self.assertEqual(stderr, "")
        output = self.directory / "nested" / "comparison.json"
        code, stdout, _ = self.run_main(baseline, candidate, "--format", "json", "--output", str(output), "--tolerance", "20", "--statistic", "median", "--metric", "wall_time_seconds")
        self.assertEqual(code, cw.EXIT_OK)
        printed = json.loads(stdout)
        self.assertEqual(printed, json.loads(output.read_text(encoding="utf-8")))
        self.assertEqual(printed["options"], {
            "statistics": ["median"], "tolerance_percent": 20.0, "directions": {}, "metrics": ["wall_time_seconds"], "label_filters": [],
        })
        self.assertEqual(printed["comparisons"][0]["summary"]["unchanged"], 1)

    def test_statistic_all_directions_and_select(self):
        path = self.write("ab.json", agent_result(mcp_tokens=800))
        code, stdout, _ = self.run_main(f"{path}#control", f"{path}#mcp", "--statistic", "all", "--direction", "wall_time_seconds=higher", "--format", "json")
        self.assertEqual(code, cw.EXIT_OK)
        comparison = json.loads(stdout)
        self.assertEqual(comparison["options"]["statistics"], list(wr.STATISTICS))
        self.assertEqual(comparison["options"]["directions"], {"wall_time_seconds": "higher"})
        by_key = {(entry["name"], entry["statistic"]): entry["outcome"] for entry in comparison["comparisons"][0]["entries"]}
        self.assertEqual(by_key[("input_tokens", "stdev")], "unchanged")
        self.assertEqual(by_key[("wall_time_seconds", "mean")], "regressed")
        code, stdout, _ = self.run_main(path, path, "--select", "mode=mcp", "--select", "missing=null", "--format", "json")
        self.assertEqual(code, cw.EXIT_OK)
        self.assertEqual(json.loads(stdout)["comparisons"][0]["runs"], [])
        code, stdout, _ = self.run_main(path, path, "--select", "mode=mcp", "--format", "json")
        self.assertEqual(json.loads(stdout)["comparisons"][0]["runs"][0]["id"], "mcp")

    def test_check_mode_and_error_exit_codes(self):
        baseline = self.write("a.json", latency_result())
        code, stdout, stderr = self.run_main(baseline, baseline, "--check")
        self.assertEqual(code, cw.EXIT_OK)
        slower = self.write("b.json", latency_result(p95=0.030))
        code, stdout, stderr = self.run_main(baseline, slower, "--check")
        self.assertEqual(code, cw.EXIT_CHECK_FAILED)
        self.assertIn("regressed", stdout)
        self.assertEqual(stderr, "check failed: 1 regressed metric(s), all documents succeeded=true\n")
        code, _, stderr = self.run_main(baseline, slower, "--check", "--tolerance", "150")
        self.assertEqual(code, cw.EXIT_OK)
        failed = self.write("c.json", latency_result(status="failure"))
        code, _, stderr = self.run_main(baseline, failed, "--check")
        self.assertEqual(code, cw.EXIT_CHECK_FAILED)
        self.assertIn("all documents succeeded=false", stderr)
        for argv in (
            [baseline, f"{baseline}#nope"],
            [baseline, str(self.directory / "missing.json")],
            [baseline, baseline, "--select", "mode"],
            [baseline, baseline, "--direction", "x=up"],
            [baseline, baseline, "--statistic", "average"],
        ):
            code, stdout, stderr = self.run_main(*argv)
            self.assertEqual(code, cw.EXIT_INVALID, argv)
            self.assertEqual(stdout, "")
            self.assertTrue(stderr.startswith("error: "), stderr)

    def test_module_runs_as_a_script(self):
        path = self.write("a.json", latency_result())
        with patch("sys.argv", ["compare_workload_results.py", path, path]), patch("sys.stdout", new_callable=io.StringIO) as stdout, self.assertRaises(SystemExit) as raised:
            runpy.run_path(str(Path(__file__).with_name("compare_workload_results.py")), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("workload: capture-latency", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
