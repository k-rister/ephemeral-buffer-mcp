import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from release_checks import (
    ReleaseCheckError,
    changelog_version,
    extract_changelog_notes,
    git,
    is_ancestor,
    package_version,
    run,
    tag_version,
    validate_metadata,
    validate_source_state,
)


class TestReleaseChecks(unittest.TestCase):
    def test_tag_version_requires_semver(self):
        self.assertEqual(tag_version("v1.2.3"), "1.2.3")
        with self.assertRaises(ReleaseCheckError):
            tag_version("release-1.2.3")

    def test_metadata_requires_matching_changelog_section(self):
        validate_metadata("v1.2.3", "1.2.3", "## 1.2.3 - 2026-09-06\n\n- Notes\n")

        with self.assertRaisesRegex(ReleaseCheckError, "pyproject.toml"):
            validate_metadata("v1.2.4", "1.2.3", "## 1.2.3 - 2026-09-06")
        with self.assertRaisesRegex(ReleaseCheckError, "Unreleased"):
            changelog_version("## 1.2.3 - Unreleased", "1.2.3")
        with self.assertRaisesRegex(ReleaseCheckError, "no release section"):
            changelog_version("## 1.2.2 - 2026-09-06", "1.2.3")

    def test_bracketed_changelog_heading_is_supported(self):
        changelog_version("## [1.2.3] - 2026-09-06", "1.2.3")

    def test_release_notes_are_limited_to_the_requested_section(self):
        changelog = "## 1.2.3 - 2026-09-06\n\n- First note\n\n## 1.2.2 - 2026-08-01\n\n- Older note\n"

        self.assertEqual(
            extract_changelog_notes(changelog, "1.2.3"),
            "## 1.2.3 - 2026-09-06\n\n- First note",
        )

    def test_source_state_requires_clean_main_ancestry(self):
        validate_source_state("tag", "main", True, "")

        with self.assertRaisesRegex(ReleaseCheckError, "not clean"):
            validate_source_state("tag", "main", True, " M README.md")
        with self.assertRaisesRegex(ReleaseCheckError, "not contained"):
            validate_source_state("tag", "main", False, "")

    def test_changelog_can_be_read_from_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CHANGELOG.md"
            path.write_text("## 1.2.3 - 2026-09-06\n", encoding="utf-8")
            changelog_version(path.read_text(encoding="utf-8"), "1.2.3")

    def test_package_version_reports_missing_or_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.toml"
            with self.assertRaisesRegex(ReleaseCheckError, "unable to read"):
                package_version(missing)

            malformed = Path(directory) / "malformed.toml"
            malformed.write_text("not valid = [", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseCheckError, "unable to read"):
                package_version(malformed)

    def test_git_reports_command_failures(self):
        failure = subprocess.CalledProcessError(128, ["git", "status"], stderr="bad ref")
        with patch("release_checks.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(ReleaseCheckError, "git status failed: bad ref"):
                git("status")

    def test_is_ancestor_reports_both_git_results(self):
        with patch("release_checks.subprocess.run") as run_git:
            run_git.return_value.returncode = 0
            self.assertTrue(is_ancestor("tag", "main"))
            run_git.return_value.returncode = 1
            self.assertFalse(is_ancestor("tag", "main"))

    def test_cli_reports_missing_changelog(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            missing = Path(directory) / "missing.md"
            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                result = run([
                    "--tag", "v1.2.3",
                    "--pyproject", str(project),
                    "--changelog", str(missing),
                ])

        self.assertEqual(result, 1)
        self.assertIn("release guardrail failed", stderr.getvalue())

    def test_cli_reports_git_and_dirty_state_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            changelog = Path(directory) / "CHANGELOG.md"
            changelog.write_text("## 1.2.3 - 2026-09-06\n", encoding="utf-8")

            stderr = io.StringIO()
            with patch("release_checks.git", side_effect=ReleaseCheckError("git failed")), \
                    patch("sys.stderr", stderr):
                result = run([
                    "--tag", "v1.2.3",
                    "--pyproject", str(project),
                    "--changelog", str(changelog),
                ])
        self.assertEqual(result, 1)
        self.assertIn("git failed", stderr.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            changelog = Path(directory) / "CHANGELOG.md"
            changelog.write_text("## 1.2.3 - 2026-09-06\n", encoding="utf-8")
            stderr = io.StringIO()
            with patch("release_checks.git", side_effect=["tag", "main", " M file.py"]), \
                    patch("release_checks.is_ancestor", return_value=True), \
                    patch("sys.stderr", stderr):
                result = run([
                    "--tag", "v1.2.3",
                    "--pyproject", str(project),
                    "--changelog", str(changelog),
                ])
        self.assertEqual(result, 1)
        self.assertIn("not clean", stderr.getvalue())

    def test_cli_reports_non_ancestor_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            changelog = Path(directory) / "CHANGELOG.md"
            changelog.write_text("## 1.2.3 - 2026-09-06\n", encoding="utf-8")

            stderr = io.StringIO()
            with patch("release_checks.git", side_effect=["tag", "main", ""]), \
                    patch("release_checks.is_ancestor", return_value=False), \
                    patch("sys.stderr", stderr):
                result = run([
                    "--tag", "v1.2.3",
                    "--pyproject", str(project),
                    "--changelog", str(changelog),
                ])
        self.assertEqual(result, 1)
        self.assertIn("not contained", stderr.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            changelog = Path(directory) / "CHANGELOG.md"
            changelog.write_text("## 1.2.3 - 2026-09-06\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch("release_checks.git", side_effect=["tag", "main", ""]), \
                    patch("release_checks.is_ancestor", return_value=True), \
                    patch("sys.stdout", stdout):
                result = run([
                    "--tag", "v1.2.3",
                    "--pyproject", str(project),
                    "--changelog", str(changelog),
                ])
        self.assertEqual(result, 0)
        self.assertIn("Release guardrails passed", stdout.getvalue())

    def test_cli_requires_a_tag(self):
        with self.assertRaises(SystemExit):
            run([])


if __name__ == "__main__":
    unittest.main()
