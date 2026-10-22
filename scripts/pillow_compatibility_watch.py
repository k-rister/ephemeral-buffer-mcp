"""Probe current unconstrained requirements for a Pillow audit fix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def pillow_status(report: Any) -> tuple[bool, str]:
    """Return whether the report has one Pillow dependency with no advisories."""
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise ValueError("pip-audit JSON is missing its dependencies list")
    pillow = [
        dependency
        for dependency in report["dependencies"]
        if isinstance(dependency, dict)
        and isinstance(dependency.get("name"), str)
        and dependency["name"].casefold() == "pillow"
    ]
    if len(pillow) != 1:
        raise ValueError(f"Expected one Pillow record in pip-audit output, found {len(pillow)}")
    dependency = pillow[0]
    version = dependency.get("version")
    vulnerabilities = dependency.get("vulns")
    if not isinstance(version, str) or not isinstance(vulnerabilities, list):
        raise ValueError("pip-audit returned a malformed Pillow record")
    if vulnerabilities:
        ids = sorted(
            vulnerability.get("id", "unknown")
            for vulnerability in vulnerabilities
            if isinstance(vulnerability, dict)
        )
        return (
            False,
            f"Pillow {version} still has {len(vulnerabilities)} advisories: {', '.join(ids)}",
        )
    return True, f"Pillow {version} has no reported advisories"


def inspect_interpreter(interpreter: Path, requirements: Path) -> tuple[bool, str]:
    """Resolve and audit current requirements with a selected Python interpreter."""
    command = [
        str(interpreter),
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
        details = result.stderr.strip() or f"pip-audit exited {result.returncode}"
        raise RuntimeError(f"Audit failed for {interpreter}: {details}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        details = result.stderr.strip() or "no diagnostic output"
        raise RuntimeError(f"Invalid pip-audit JSON for {interpreter}: {details}") from exc
    clean, detail = pillow_status(report)
    return clean, f"{interpreter}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        results = [inspect_interpreter(python, args.requirements) for python in args.python]
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Pillow compatibility watch failed: {exc}", file=sys.stderr)
        return 2

    for _clean, detail in results:
        print(detail)
    pillow_clean = all(clean for clean, _detail in results)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"pillow_clean={'true' if pillow_clean else 'false'}\n")
    if pillow_clean:
        print("Pillow is clean in every checked unpinned requirements resolution.")
    else:
        print("The Pillow compatibility blocker remains; the CI exception is still active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
