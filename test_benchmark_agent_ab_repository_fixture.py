"""Tests for the repository-shaped agent A/B fixture."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_agent_ab_repository_fixture import PROFILE, build_manifest, create_fixture


class TestRepositoryFixture(unittest.TestCase):
    def test_manifest_describes_repository_workflows(self):
        manifest = build_manifest()
        self.assertEqual(manifest["profile"], PROFILE)
        self.assertIn("synthetic", manifest["privacy"])
        self.assertEqual(len(manifest["tasks"]), 4)
        self.assertTrue(all("repository_checks.py" in task["prompt"] for task in manifest["tasks"].values()))

    def test_fixture_has_layout_and_deterministic_workflow_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture"
            create_fixture(destination)
            self.assertTrue((destination / "pyproject.toml").is_file())
            self.assertTrue((destination / "src/packet_parser/parser.py").is_file())
            self.assertTrue((destination / "tests/test_parser.py").is_file())
            checks = destination / "tools/repository_checks.py"
            self.assertTrue(checks.stat().st_mode & 0o111)
            output = subprocess.check_output(["python3", str(checks), "targeted"], text=True)
            self.assertIn("TARGETED_SIGNAL in src/packet_parser/parser.py", output)
            self.assertIn(":12", output)

    def test_fixture_rejects_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture"
            destination.mkdir()
            (destination / "existing").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_fixture(destination)


if __name__ == "__main__":
    unittest.main()
