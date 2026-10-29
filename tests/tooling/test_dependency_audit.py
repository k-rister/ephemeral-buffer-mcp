"""Tests for the dependency audit policy and Pillow dependency watch."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.dependency_audit import (
    DEFAULT_EXCEPTIONS,
    exceptions_for_python,
    load_exceptions,
    validate_report,
)
from scripts.pillow_compatibility_watch import pillow_status


class TestDependencyAudit(unittest.TestCase):
    def test_filelock_exception_is_limited_to_python_310(self):
        exceptions = load_exceptions(DEFAULT_EXCEPTIONS)

        self.assertEqual(len(exceptions), 1)
        self.assertEqual(exceptions[0]["package"], "filelock")
        self.assertEqual(exceptions[0]["version"], "3.20.3")
        self.assertEqual(
            set(exceptions[0]["vulnerability_ids"]),
            {"CVE-2026-112909", "CVE-2026-112910"},
        )
        self.assertEqual(exceptions_for_python(exceptions, "3.10"), exceptions)
        self.assertEqual(exceptions_for_python(exceptions, "3.12"), [])

        report = {
            "dependencies": [
                {
                    "name": "filelock",
                    "version": "3.20.3",
                    "vulns": [
                        {"id": "CVE-2026-112909"},
                        {"id": "CVE-2026-112910"},
                    ],
                }
            ]
        }
        errors, matched = validate_report(
            report, exceptions_for_python(exceptions, "3.10")
        )
        self.assertEqual(errors, [])
        self.assertIn("filelock==3.20.3", matched[0])

    def test_unscoped_exceptions_apply_to_every_python_version(self):
        exception = {
            "package": "example-package",
            "version": "1.0.0",
            "vulnerability_ids": ["GHSA-example"],
            "issue": "https://example.com/issue/1",
        }

        self.assertEqual(exceptions_for_python([exception], "3.10"), [exception])

    def test_unapproved_advisory_fails_when_policy_is_empty(self):
        report = {
            "dependencies": [
                {"name": "filelock", "version": "3.32.5", "vulns": [{"id": "CVE-2026-112906"}]}
            ]
        }

        errors, matched = validate_report(report, [])

        self.assertEqual(matched, [])
        self.assertEqual(
            errors,
            ["Unapproved vulnerability: filelock==3.32.5 CVE-2026-112906"],
        )

    def test_clean_report_passes_without_exceptions(self):
        errors, matched = validate_report(
            {"dependencies": [{"name": "numpy", "version": "2.5.2", "vulns": []}]}, []
        )

        self.assertEqual(errors, [])
        self.assertEqual(matched, [])

    def test_exact_explicit_exception_still_matches(self):
        exception = {
            "package": "example-package",
            "version": "1.0.0",
            "vulnerability_ids": ["GHSA-example"],
            "issue": "https://example.com/issue/1",
            "reason": "A test fixture.",
        }
        report = {
            "dependencies": [
                {"name": "example-package", "version": "1.0.0", "vulns": [{"id": "GHSA-example"}]}
            ]
        }

        errors, matched = validate_report(report, [exception])

        self.assertEqual(errors, [])
        self.assertIn("example-package==1.0.0", matched[0])

    def test_report_without_pillow_is_clean(self):
        clean, detail = pillow_status({"dependencies": [{"name": "numpy", "version": "2.5.2", "vulns": []}]})

        self.assertTrue(clean)
        self.assertEqual(detail, "Pillow is not installed in this requirements resolution")

    def test_pillow_watch_reports_clean_and_vulnerable_states(self):
        clean, detail = pillow_status(
            {"dependencies": [{"name": "Pillow", "version": "13.0.0", "vulns": []}]}
        )
        self.assertTrue(clean)
        self.assertIn("Pillow 13.0.0", detail)

        clean, detail = pillow_status(
            {
                "dependencies": [
                    {
                        "name": "pillow",
                        "version": "12.3.0",
                        "vulns": [{"id": "CVE-2026-114501"}],
                    }
                ]
            }
        )
        self.assertFalse(clean)
        self.assertIn("CVE-2026-114501", detail)

    def test_pillow_watch_rejects_duplicate_pillow_records(self):
        report = {
            "dependencies": [
                {"name": "pillow", "version": "13.0.0", "vulns": []},
                {"name": "Pillow", "version": "13.0.0", "vulns": []},
            ]
        }

        with self.assertRaisesRegex(ValueError, "found 2"):
            pillow_status(report)

    def test_pillow_watch_rejects_malformed_reports(self):
        with self.assertRaisesRegex(ValueError, "missing its dependencies list"):
            pillow_status({})
        with self.assertRaisesRegex(ValueError, "malformed Pillow record"):
            pillow_status({"dependencies": [{"name": "Pillow", "version": "13.0.0"}]})

    def test_exception_policy_still_rejects_malformed_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            policy.write_text(json.dumps({"exceptions": None}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain an 'exceptions' list"):
                load_exceptions(policy)

    def test_exception_policy_rejects_malformed_python_version_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "exceptions": [
                            {
                                "package": "example-package",
                                "version": "1.0.0",
                                "vulnerability_ids": ["GHSA-example"],
                                "issue": "https://example.com/issue/1",
                                "python_version": "3",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "major.minor Python version"):
                load_exceptions(policy)


if __name__ == "__main__":
    unittest.main()
