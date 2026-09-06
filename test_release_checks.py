import tempfile
import unittest
from pathlib import Path

from release_checks import (
    ReleaseCheckError,
    changelog_version,
    extract_changelog_notes,
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


if __name__ == "__main__":
    unittest.main()
