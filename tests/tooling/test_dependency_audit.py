"""Tests for the dependency audit policy."""

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


class TestDependencyAudit(unittest.TestCase):
    def test_exception_policy_is_empty_for_supported_python_versions(self):
        exceptions = load_exceptions(DEFAULT_EXCEPTIONS)

        self.assertEqual(exceptions, [])
        self.assertEqual(exceptions_for_python(exceptions, "3.11"), [])
        self.assertEqual(exceptions_for_python(exceptions, "3.12"), [])

    def test_unscoped_exceptions_apply_to_every_python_version(self):
        exception = {
            "package": "example-package",
            "version": "1.0.0",
            "vulnerability_ids": ["GHSA-example"],
            "issue": "https://example.com/issue/1",
        }

        self.assertEqual(exceptions_for_python([exception], "3.11"), [exception])

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
