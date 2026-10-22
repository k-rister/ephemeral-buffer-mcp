"""Run pip-audit with the repository's narrowly scoped temporary exception."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEPTIONS = REPOSITORY_ROOT / ".github" / "dependency-audit-exceptions.json"


def load_exceptions(path: Path) -> list[dict[str, Any]]:
    """Load and validate explicit package/version/advisory exceptions."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read exception policy {path}: {exc}") from exc

    exceptions = document.get("exceptions") if isinstance(document, dict) else None
    if not isinstance(exceptions, list) or not exceptions:
        raise ValueError("Exception policy must contain a non-empty 'exceptions' list")

    seen_packages: set[str] = set()
    for exception in exceptions:
        if not isinstance(exception, dict):
            raise ValueError("Every exception must be a JSON object")
        package = exception.get("package")
        version = exception.get("version")
        vulnerability_ids = exception.get("vulnerability_ids")
        if not isinstance(package, str) or not package.strip():
            raise ValueError("Every exception must name a package")
        normalized_package = package.casefold()
        if normalized_package in seen_packages:
            raise ValueError(f"Only one exception per package is allowed: {package}")
        seen_packages.add(normalized_package)
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Exception for {package} must pin an exact version")
        if (
            not isinstance(vulnerability_ids, list)
            or not vulnerability_ids
            or any(not isinstance(item, str) or not item.strip() for item in vulnerability_ids)
        ):
            raise ValueError(f"Exception for {package} must list vulnerability IDs")
        if len(vulnerability_ids) != len(set(vulnerability_ids)):
            raise ValueError(f"Exception for {package} contains duplicate vulnerability IDs")
        if not isinstance(exception.get("issue"), str) or not exception["issue"].strip():
            raise ValueError(f"Exception for {package} must link a tracking issue")

    return exceptions


def validate_report(
    report: Any, exceptions: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Return policy errors and descriptions of the exact exceptions matched."""
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        return ["pip-audit JSON is missing its dependencies list"], []

    exception_by_package = {
        exception["package"].casefold(): exception for exception in exceptions
    }
    errors: list[str] = []
    observed_records: dict[str, int] = {name: 0 for name in exception_by_package}
    observed_ids: dict[str, set[str]] = {name: set() for name in exception_by_package}
    observed_versions: dict[str, set[str]] = {name: set() for name in exception_by_package}
    accepted_ids: dict[str, set[str]] = {name: set() for name in exception_by_package}

    for dependency in report["dependencies"]:
        if not isinstance(dependency, dict):
            errors.append("pip-audit returned a malformed dependency record")
            continue
        name = dependency.get("name")
        version = dependency.get("version")
        vulnerabilities = dependency.get("vulns")
        if not isinstance(name, str) or not isinstance(version, str):
            errors.append("pip-audit returned a dependency without a name and version")
            continue
        if not isinstance(vulnerabilities, list):
            errors.append(f"pip-audit returned malformed vulnerabilities for {name}=={version}")
            continue

        normalized_name = name.casefold()
        exception = exception_by_package.get(normalized_name)
        if exception is not None:
            observed_records[normalized_name] += 1
        for vulnerability in vulnerabilities:
            if (
                not isinstance(vulnerability, dict)
                or not isinstance(vulnerability.get("id"), str)
            ):
                errors.append(f"pip-audit returned a malformed vulnerability for {name}=={version}")
                continue
            vulnerability_id = vulnerability["id"]
            if exception is None:
                errors.append(
                    f"Unapproved vulnerability: {name}=={version} {vulnerability_id}"
                )
                continue

            observed_ids[normalized_name].add(vulnerability_id)
            observed_versions[normalized_name].add(version)
            if version != exception["version"]:
                errors.append(
                    f"Unapproved {name} version: expected {name}=={exception['version']}, "
                    f"observed {name}=={version} ({vulnerability_id})"
                )
            elif vulnerability_id not in exception["vulnerability_ids"]:
                errors.append(
                    f"Unapproved vulnerability for {name}=={version}: {vulnerability_id}"
                )
            else:
                accepted_ids[normalized_name].add(vulnerability_id)

    matched: list[str] = []
    for package, exception in exception_by_package.items():
        if observed_records[package] != 1:
            errors.append(
                f"Expected exactly one {package} dependency record, "
                f"found {observed_records[package]}"
            )
        expected_ids = set(exception["vulnerability_ids"])
        seen_ids = observed_ids[package]
        missing_ids = expected_ids - seen_ids
        if missing_ids:
            errors.append(
                f"The {package} exception is stale or the advisory baseline changed; "
                f"no longer observed: {', '.join(sorted(missing_ids))}"
            )
        if len(observed_versions[package]) > 1:
            errors.append(
                f"Multiple versions of {package} appeared in pip-audit output: "
                f"{', '.join(sorted(observed_versions[package]))}"
            )
        if not missing_ids and accepted_ids[package] == expected_ids:
            matched.append(
                f"{package}=={exception['version']} ({len(expected_ids)} known advisories; "
                f"tracking {exception['issue']})"
            )

    return errors, matched


def run_audit(requirements: Path, exceptions_path: Path = DEFAULT_EXCEPTIONS) -> int:
    """Run pip-audit, allowing only an exact and still-current exception baseline."""
    try:
        exceptions = load_exceptions(exceptions_path)
    except ValueError as exc:
        print(f"Dependency audit policy error: {exc}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--format=json",
        "--desc=off",
        "--aliases=off",
        "--progress-spinner=off",
        "-r",
        str(requirements),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        print(f"pip-audit failed with exit code {result.returncode}", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode or 2
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"pip-audit did not return valid JSON: {exc}", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return 2

    errors, matched = validate_report(report, exceptions)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if errors:
        print("Dependency audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "Review the findings and remove or update the exception in "
            ".github/dependency-audit-exceptions.json.",
            file=sys.stderr,
        )
        return 1

    for exception in matched:
        print(f"Temporary dependency audit exception active: {exception}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    args = parser.parse_args()
    return run_audit(args.requirements, args.exceptions)


if __name__ == "__main__":
    raise SystemExit(main())
