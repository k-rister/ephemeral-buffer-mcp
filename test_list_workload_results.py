"""Tests for listing coding-agent workload results by experiment group and metadata."""

import io
import json
import re
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import list_workload_results as lw
import workload_results as wr


ENVIRONMENT = {
    "python_version": "3.12.14",
    "platform": "Linux-6.1-x86_64",
    "machine": "x86_64",
    "cpu_count": 8,
    "recorded_at": "2026-09-18T15:00:00+00:00",
    "tool": {"name": wr.PROJECT_NAME, "version": "0.4.0"},
    "source_revision": "a" * 40,
}


def result(name="capture-latency", *, status=None, runs=None, errors=(), experiment=None, recorded_at="2026-09-18T15:00:00+00:00"):
    if runs is None:
        runs = [wr.run(
            "lines-256",
            labels={"cache_state": "warm", "line_count": 256},
            measurements={"wall_time_seconds": wr.measurement("seconds", median=0.01)},
        )]
    return wr.build_result(
        workload=name,
        kind="benchmark",
        producer="benchmark_latency.py",
        parameters={"samples": 3},
        environment={**ENVIRONMENT, "recorded_at": recorded_at},
        runs=runs,
        status=status,
        errors=errors,
        experiment=experiment,
    )


class ListingFixture(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.sweep_a = self.write("sweep/a.json", result(
            experiment=wr.experiment("sweep", {"variant": "a", "model": "m1", "started_at": "2026-09-18T12:00:00+00:00", "owner": "me"}),
        ))
        self.sweep_b = self.write("sweep/nested/b.json", result(
            experiment=wr.experiment("sweep", {"variant": "b", "model": "m1", "started_at": "2026-09-18T10:00:00+00:00"}),
            runs=[wr.run("lines-256", status="timeout", errors=["exceeded 30 s"]), wr.run("lines-16", labels={"line_count": 16})],
        ))
        self.other = self.write("other.json", result("semantic-index", experiment=wr.experiment("other", {"model": "m2", "size": 5, "flag": True})))
        self.plain = self.write("plain.json", result(runs=[], recorded_at="2026-09-18T09:00:00+00:00"))
        self.failed = self.write("failed.json", result(status="error", errors=["harness crashed"], experiment=wr.experiment(None, {"model": "m1"})))
        (self.root / "records.json").write_text('{"runs": []}', encoding="utf-8")
        (self.root / "notes.txt").write_text("not json", encoding="utf-8")

    def write(self, name, document):
        path = self.root / name
        wr.write_result(document, path)
        return str(path)

    def write_invalid(self, name="bad.json"):
        path = self.root / name
        path.write_text(json.dumps({"format": wr.FORMAT, "format_version": wr.FORMAT_VERSION}), encoding="utf-8")
        return str(path)

    @staticmethod
    def paths(listing):
        return [row["path"] for row in listing["results"]]


class TestBuildListing(ListingFixture):
    def test_lists_every_document_in_group_then_time_order(self):
        listing = lw.build_listing([str(self.root)])
        self.assertEqual(listing["format"], lw.FORMAT)
        self.assertEqual(listing["format_version"], lw.FORMAT_VERSION)
        self.assertEqual(self.paths(listing), [self.other, self.sweep_b, self.sweep_a, self.plain, self.failed])
        self.assertEqual(listing["summary"], {"listed": 5, "invalid": 0, "groups": ["other", "sweep"]})
        self.assertEqual(listing["filters"], {"groups": [], "where": [], "workloads": [], "statuses": [], "redact": []})
        row = listing["results"][2]
        self.assertEqual(row["path"], self.sweep_a)
        self.assertTrue(row["valid"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["group"], "sweep")
        # Documents are written with sorted keys, so metadata reads back alphabetically.
        self.assertEqual(list(row["metadata"]), ["model", "owner", "started_at", "variant"])
        self.assertEqual(row["metadata"], {"variant": "a", "model": "m1", "started_at": "2026-09-18T12:00:00+00:00", "owner": "me"})
        self.assertEqual(row["workload"], {"name": "capture-latency", "kind": "benchmark", "producer": "benchmark_latency.py"})
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["ordered_at"], "2026-09-18T12:00:00+00:00")
        self.assertEqual(row["recorded_at"], "2026-09-18T15:00:00+00:00")
        self.assertEqual(row["runs"], [{"id": "lines-256", "labels": {"cache_state": "warm", "line_count": 256}, "status": "success", "errors": []}])
        partial = listing["results"][1]
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([(item["id"], item["status"], item["errors"]) for item in partial["runs"]], [("lines-256", "timeout", ["exceeded 30 s"]), ("lines-16", "success", [])])
        self.assertEqual(listing["results"][4]["errors"], ["harness crashed"])
        self.assertIsNone(listing["results"][4]["group"])
        json.dumps(listing, allow_nan=False)

    def test_filters_by_group_metadata_workload_and_status(self):
        root = [str(self.root)]
        self.assertEqual(self.paths(lw.build_listing(root, groups=["sweep"])), [self.sweep_b, self.sweep_a])
        self.assertEqual(self.paths(lw.build_listing(root, groups=["sweep", "other"])), [self.other, self.sweep_b, self.sweep_a])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("model", "m1")])), [self.sweep_b, self.sweep_a, self.failed])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("model", "m1"), ("variant", "a")])), [self.sweep_a])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("size", 5)])), [self.other])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("size", True)])), [])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("flag", True)])), [self.other])
        self.assertEqual(self.paths(lw.build_listing(root, where=[("missing", None)])), [])
        self.assertEqual(self.paths(lw.build_listing(root, workloads=["semantic-index"])), [self.other])
        self.assertEqual(self.paths(lw.build_listing(root, statuses=["partial", "error"])), [self.sweep_b, self.failed])
        self.assertEqual(self.paths(lw.build_listing(root, groups=["sweep"], statuses=["success"], where=[("model", "m1")])), [self.sweep_a])
        filtered = lw.build_listing([self.plain, self.sweep_a], groups=["sweep"], where=iter([("model", "m1")]), workloads=["capture-latency"], statuses=["success"])
        self.assertEqual(self.paths(filtered), [self.sweep_a])
        self.assertEqual(filtered["filters"], {
            "groups": ["sweep"], "where": [["model", "m1"]], "workloads": ["capture-latency"], "statuses": ["success"], "redact": [],
        })

    def test_invalid_documents_are_listed_and_never_filtered_out(self):
        bad = self.write_invalid("sweep/bad.json")
        listing = lw.build_listing([str(self.root)], groups=["nope"])
        self.assertEqual(listing["results"], [{
            "path": bad, "valid": False, "status": "invalid",
            "error": "result is missing environment, errors, measurements, runs, status, workload",
        }])
        self.assertEqual(listing["summary"], {"listed": 0, "invalid": 1, "groups": []})
        listing = lw.build_listing([str(self.root / "missing.json"), str(self.root / "records.json")])
        self.assertEqual([row["status"] for row in listing["results"]], ["invalid", "invalid"])
        self.assertIn("cannot read", listing["results"][0]["error"])
        self.assertIn("format must be", listing["results"][1]["error"])
        # Invalid rows sort after every valid document.
        self.assertEqual(self.paths(lw.build_listing([str(self.root)]))[-1], bad)

    def test_redaction_hides_values_but_keeps_keys(self):
        listing = lw.build_listing([self.sweep_a], redact=["owner", "absent"])
        self.assertEqual(listing["results"][0]["metadata"]["owner"], wr.REDACTED)
        self.assertEqual(listing["results"][0]["metadata"]["variant"], "a")
        self.assertEqual(listing["filters"]["redact"], ["owner", "absent"])
        self.assertEqual(wr.load_result(Path(self.sweep_a))["experiment"]["metadata"]["owner"], "me")


class TestFormatListing(ListingFixture):
    def test_document_rows_show_group_status_time_and_metadata(self):
        text = lw.format_listing(lw.build_listing([str(self.root)]))
        self.assertEqual(text.splitlines()[0].split(), ["path", "group", "workload", "status", "runs", "time", "metadata", "errors"])
        self.assertRegex(text, re.escape(self.sweep_a) + r"\s+sweep\s+capture-latency\s+success\s+1\s+2026-09-18T12:00:00\+00:00\s+model=m1 owner=me started_at=2026-09-18T12:00:00\+00:00 variant=a\n")
        self.assertRegex(text, re.escape(self.sweep_b) + r"\s+sweep\s+capture-latency\s+partial\s+2\s+2026-09-18T10:00:00\+00:00\s+model=m1")
        self.assertRegex(text, re.escape(self.other) + r"\s+other\s+semantic-index\s+success\s+1\s+2026-09-18T15:00:00\+00:00\s+flag=true model=m2 size=5\n")
        self.assertRegex(text, re.escape(self.plain) + r"\s+-\s+capture-latency\s+success\s+0\s+2026-09-18T09:00:00\+00:00\n")
        self.assertRegex(text, re.escape(self.failed) + r"\s+-\s+capture-latency\s+error\s+1\s+2026-09-18T15:00:00\+00:00\s+model=m1\s+harness crashed\n")
        self.assertTrue(text.endswith("\n"))

    def test_field_columns_and_run_rows(self):
        listing = lw.build_listing([str(self.root)], groups=["sweep"])
        text = lw.format_listing(listing, fields=["variant", "size"])
        self.assertEqual(text.splitlines()[0].split(), ["path", "group", "workload", "status", "runs", "time", "variant", "size", "errors"])
        self.assertRegex(text, re.escape(self.sweep_a) + r"\s+sweep\s+capture-latency\s+success\s+1\s+\S+\s+a\s+-\n")
        text = lw.format_listing(listing, runs=True)
        self.assertEqual(text.splitlines()[0].split(), ["path", "group", "workload", "status", "run", "run_status", "labels", "metadata", "errors"])
        self.assertRegex(text, re.escape(self.sweep_b) + r"\s+sweep\s+capture-latency\s+partial\s+lines-256\s+timeout\s+model=m1 .*variant=b\s+exceeded 30 s\n")
        self.assertRegex(text, re.escape(self.sweep_b) + r"\s+sweep\s+capture-latency\s+partial\s+lines-16\s+success\s+line_count=16\s+model=m1")
        self.assertRegex(text, re.escape(self.sweep_a) + r"\s+sweep\s+capture-latency\s+success\s+lines-256\s+success\s+cache_state=warm line_count=256\s+model=m1")
        # A document without runs still gets one row, and document errors
        # appear beside every run of a document that has them.
        text = lw.format_listing(lw.build_listing([self.plain, self.failed]), runs=True, fields=["model"])
        self.assertRegex(text, re.escape(self.plain) + r"\s+-\s+capture-latency\s+success\s+\(none\)\s+-\s+-\n")
        self.assertRegex(text, re.escape(self.failed) + r"\s+-\s+capture-latency\s+error\s+lines-256\s+success\s+cache_state=warm line_count=256\s+m1\s+harness crashed\n")

    def test_empty_and_invalid_listings(self):
        self.assertEqual(lw.format_listing(lw.build_listing([str(self.root)], groups=["nope"])), "no workload results selected\n")
        bad = self.write_invalid()
        text = lw.format_listing(lw.build_listing([bad]), fields=["variant"])
        self.assertRegex(text, re.escape(bad) + r"\s+-\s+-\s+invalid\s+-\s+-\s+-\s+result is missing environment")
        text = lw.format_listing(lw.build_listing([bad]), runs=True)
        self.assertRegex(text, re.escape(bad) + r"\s+-\s+-\s+invalid\s+-\s+-\s+-\s+result is missing environment")
        text = lw.format_listing(lw.build_listing([bad, self.plain]))
        self.assertLess(text.index(self.plain), text.index(bad))


class TestMain(ListingFixture):
    def run_main(self, *argv):
        with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = lw.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_text_json_and_paths_output(self):
        code, stdout, stderr = self.run_main(str(self.root), "--group", "sweep", "--field", "variant")
        self.assertEqual((code, stderr), (lw.EXIT_OK, ""))
        self.assertIn("variant", stdout.splitlines()[0])
        self.assertIn(self.sweep_a, stdout)
        self.assertNotIn(self.other, stdout)
        code, stdout, _ = self.run_main(
            str(self.root), "--where", "model=m1", "--status", "success", "--workload", "capture-latency",
            "--redact", "owner", "--format", "json",
        )
        self.assertEqual(code, lw.EXIT_OK)
        listing = json.loads(stdout)
        self.assertEqual(self.paths(listing), [self.sweep_a])
        self.assertEqual(listing["filters"], {
            "groups": [], "where": [["model", "m1"]], "workloads": ["capture-latency"], "statuses": ["success"], "redact": ["owner"],
        })
        self.assertEqual(listing["results"][0]["metadata"]["owner"], wr.REDACTED)
        code, stdout, _ = self.run_main(str(self.root), "--group", "sweep", "--format", "paths")
        self.assertEqual((code, stdout), (lw.EXIT_OK, f"{self.sweep_b}\n{self.sweep_a}\n"))
        code, stdout, _ = self.run_main(str(self.root), "--runs", "--group", "sweep")
        self.assertIn("run_status", stdout.splitlines()[0])
        self.assertIn("lines-16", stdout)

    def test_invalid_documents_set_the_exit_code(self):
        bad = self.write_invalid()
        code, stdout, stderr = self.run_main(str(self.root), "--format", "paths")
        self.assertEqual(code, lw.EXIT_INVALID)
        self.assertNotIn(bad, stdout)
        self.assertIn(self.plain, stdout)
        self.assertEqual(stderr, f"{bad}: INVALID: result is missing environment, errors, measurements, runs, status, workload\n")
        code, stdout, stderr = self.run_main(str(self.root / "missing.json"))
        self.assertEqual(code, lw.EXIT_INVALID)
        self.assertIn("invalid", stdout)
        self.assertIn("cannot read", stderr)

    def test_argument_errors(self):
        for argv in (
            [str(self.root), "--where", "model"],
            [str(self.root), "--where", "sizes=[1]"],
            [str(self.root), "--field", "Bad"],
            [str(self.root), "--redact", "Bad"],
            [str(self.root), "--status", "meh"],
        ):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as raised:
                lw.main(argv)
            self.assertEqual(raised.exception.code, 2, argv)
            self.assertIn("error:", stderr.getvalue())

    def test_module_runs_as_a_script(self):
        with patch("sys.argv", ["list_workload_results.py", self.plain]), patch("sys.stdout", new_callable=io.StringIO) as stdout, self.assertRaises(SystemExit) as raised:
            runpy.run_path(str(Path(__file__).with_name("list_workload_results.py")), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(self.plain, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
