"""Tests for the temporary dependency audit exception and Pillow watch."""

import unittest

from scripts.dependency_audit import DEFAULT_EXCEPTIONS, load_exceptions, validate_report
from scripts.pillow_compatibility_watch import pillow_status


class TestDependencyAudit(unittest.TestCase):
    def setUp(self):
        self.exception = load_exceptions(DEFAULT_EXCEPTIONS)[0]

    def report(self, version=None, vulnerability_ids=None, other_vulns=None):
        version = version or self.exception["version"]
        vulnerability_ids = (
            vulnerability_ids
            if vulnerability_ids is not None
            else self.exception["vulnerability_ids"]
        )
        dependencies = [
            {
                "name": "Pillow",
                "version": version,
                "vulns": [{"id": vulnerability_id} for vulnerability_id in vulnerability_ids],
            }
        ]
        if other_vulns:
            dependencies.append(
                {
                    "name": "other-package",
                    "version": "1.0.0",
                    "vulns": [{"id": vulnerability_id} for vulnerability_id in other_vulns],
                }
            )
        return {"dependencies": dependencies}

    def test_accepts_only_the_exact_pillow_baseline(self):
        errors, matched = validate_report(self.report(), [self.exception])

        self.assertEqual(errors, [])
        self.assertEqual(len(matched), 1)
        self.assertIn("pillow==12.3.0", matched[0])
        self.assertIn("35 known advisories", matched[0])

    def test_rejects_an_unlisted_advisory(self):
        vulnerability_ids = self.exception["vulnerability_ids"] + ["GHSA-new-advisory"]

        errors, _matched = validate_report(
            self.report(vulnerability_ids=vulnerability_ids), [self.exception]
        )

        self.assertTrue(any("GHSA-new-advisory" in error for error in errors))

    def test_rejects_a_changed_pillow_version(self):
        errors, _matched = validate_report(self.report(version="13.0.0"), [self.exception])

        self.assertTrue(any("Unapproved Pillow version" in error for error in errors))

    def test_flags_a_stale_exception_when_findings_disappear(self):
        errors, _matched = validate_report(self.report(vulnerability_ids=[]), [self.exception])

        self.assertTrue(any("exception is stale" in error for error in errors))

    def test_rejects_vulnerabilities_in_other_packages(self):
        errors, _matched = validate_report(
            self.report(other_vulns=["CVE-2026-00001"]), [self.exception]
        )

        self.assertIn(
            "Unapproved vulnerability: other-package==1.0.0 CVE-2026-00001",
            errors,
        )

    def test_rejects_duplicate_pillow_records(self):
        report = self.report()
        report["dependencies"].append(report["dependencies"][0].copy())

        errors, _matched = validate_report(report, [self.exception])

        self.assertTrue(any("exactly one pillow dependency record" in error for error in errors))

    def test_pillow_watch_reports_clean_and_vulnerable_states(self):
        clean, detail = pillow_status(
            {"dependencies": [{"name": "Pillow", "version": "13.0.0", "vulns": []}]}
        )
        self.assertTrue(clean)
        self.assertIn("Pillow 13.0.0", detail)

        clean, detail = pillow_status(self.report())
        self.assertFalse(clean)
        self.assertIn("35 advisories", detail)

    def test_pillow_watch_requires_a_pillow_record(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            pillow_status({"dependencies": []})


if __name__ == "__main__":
    unittest.main()
