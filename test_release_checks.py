import io
import runpy
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import release_checks
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
    def test_license_metadata_uses_compatible_file_table(self):
        project = Path(__file__).with_name("pyproject.toml")
        with project.open("rb") as stream:
            license_metadata = release_checks.tomllib.load(stream)["project"]["license"]

        self.assertEqual(license_metadata, {"file": "LICENSE"})

    def test_wheel_runtime_modules_include_metrics(self):
        project = Path(__file__).with_name("pyproject.toml")
        with project.open("rb") as stream:
            modules = release_checks.tomllib.load(stream)["tool"]["setuptools"]["py-modules"]

        self.assertIn("metrics", modules)

    def test_changelog_parser_skips_non_heading_lines(self):
        changelog_version("# Changelog\n\n## 1.2.3 - 2026-09-06", "1.2.3")

    def test_python_310_tomli_fallback_imports(self):
        source = Path(__file__).with_name("release_checks.py")
        original_import = __import__

        def block_tomllib(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("tomllib unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_tomllib):
            namespace = runpy.run_path(str(source), run_name="release_checks_fallback")

        self.assertEqual(namespace["tomllib"].__name__, "tomli")

    def test_tag_version_requires_semver(self):
        self.assertEqual(tag_version("v1.2.3"), "1.2.3")
        with self.assertRaises(ReleaseCheckError):
            tag_version("release-1.2.3")

    def test_metadata_requires_matching_changelog_section(self):
        validate_metadata("v1.2.3", "1.2.3", "## 1.2.3 - 2026-09-06\n\n- Notes\n")

        with self.assertRaisesRegex(ReleaseCheckError, "pyproject.toml"):
            validate_metadata("v1.2.4", "1.2.3", "## 1.2.3 - 2026-09-06")
        with self.assertRaisesRegex(ReleaseCheckError, "release date"):
            changelog_version("## 1.2.3 - Unreleased", "1.2.3")
        with self.assertRaisesRegex(ReleaseCheckError, "release date"):
            changelog_version("## 1.2.3", "1.2.3")
        with self.assertRaisesRegex(ReleaseCheckError, "release date"):
            changelog_version("## 1.2.3 - 2026-09-06 - TBD", "1.2.3")
        with self.assertRaisesRegex(ReleaseCheckError, "invalid release date"):
            changelog_version("## 1.2.3 - 2026-02-30", "1.2.3")
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

    def test_script_entrypoint_runs_successfully(self):
        def fake_git(command, *args, **kwargs):
            if command[:2] == ["git", "merge-base"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            outputs = {
                ("git", "rev-parse", "--verify", "v0.2.0^{commit}"): "tag",
                ("git", "rev-parse", "--verify", "main"): "main",
                ("git", "status", "--porcelain=v1", "--untracked-files=all"): "",
            }
            return SimpleNamespace(returncode=0, stdout=outputs[tuple(command)].strip() + "\n", stderr="")

        stdout = io.StringIO()
        with patch.object(sys, "argv", ["release_checks.py", "--tag", "v0.2.0", "--main-ref", "main"]), \
                patch.object(sys, "stdout", stdout), \
                patch.object(subprocess, "run", side_effect=fake_git):
            with self.assertRaisesRegex(SystemExit, "0"):
                runpy.run_path(str(Path(__file__).with_name("release_checks.py")), run_name="__main__")

        self.assertIn("Release guardrails passed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
