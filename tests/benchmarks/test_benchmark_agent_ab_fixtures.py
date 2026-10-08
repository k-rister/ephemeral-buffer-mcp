"""Tests for deterministic, privacy-safe agent A/B fixtures."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_agent_ab_fixtures import FIXTURE_VERSION, build_manifest, create_fixture


class TestAgentAbFixtures(unittest.TestCase):
    def test_manifest_has_balanced_tasks_and_success_criteria(self):
        manifest = build_manifest()
        self.assertEqual(manifest["fixture_version"], FIXTURE_VERSION)
        self.assertIn("synthetic", manifest["privacy"])
        self.assertEqual(len(manifest["tasks"]), 4)
        for task in manifest["tasks"].values():
            self.assertTrue(task["prompt"])
            self.assertTrue(task["signal_marker"])
            criteria = task["success_criteria"]
            self.assertTrue(criteria["description"])
            self.assertGreaterEqual(len(criteria["required_phrases"]), 2)
            self.assertIn(task["signal_marker"], criteria["required_phrases"])
            self.assertTrue(task["output_profile"])
        self.assertEqual(
            manifest["tasks"]["build-log-search"]["success_criteria"]["required_phrases"],
            ["BUILD_FAILURE_SIGNAL", "src/parser.c", "917"],
        )

    def test_create_fixture_is_deterministic_and_contains_no_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture"
            second_destination = Path(directory) / "fixture-copy"
            create_fixture(destination)
            create_fixture(second_destination)

            def snapshot(root):
                return {
                    path.relative_to(root): path.read_bytes()
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                }

            self.assertEqual(snapshot(destination), snapshot(second_destination))
            emitter = destination / "emit_log.py"
            self.assertTrue(emitter.is_file())
            self.assertTrue((destination / "FIXTURE.md").is_file())
            self.assertFalse((destination / "tests.log").exists())
            self.assertNotIn("TARGETED_SIGNAL", (destination / "FIXTURE.md").read_text(encoding="utf-8"))
            first = subprocess.check_output(
                ["python3", str(emitter), "build"], text=True
            )
            self.assertEqual(first.count("BUILD_FAILURE_SIGNAL"), 1)
            self.assertIn("src/parser.c:917", first)

    def test_create_fixture_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture"
            destination.mkdir()
            (destination / "existing").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_fixture(destination)


if __name__ == "__main__":
    unittest.main()
