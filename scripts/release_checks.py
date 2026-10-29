#!/usr/bin/env python3
"""Validate the source state and metadata used to build a tagged release."""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the backport installed by build tooling.
    import tomli as tomllib


class ReleaseCheckError(ValueError):
    """Raised when a release guardrail fails."""


SEMVER_TAG = re.compile(
    r"^v(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$"
)
CHANGELOG_HEADING = re.compile(r"^##\s+(?:\[(?P<bracketed>[^\]]+)\]|(?P<plain>\S+))(?:\s+-\s+(?P<label>.*))?\s*$")
CHANGELOG_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def tag_version(tag: str) -> str:
    """Return the version encoded by a release tag."""
    match = SEMVER_TAG.fullmatch(tag)
    if not match:
        raise ReleaseCheckError(
            f"release tag {tag!r} must use the vMAJOR.MINOR.PATCH format"
        )
    return match.group("version")


def package_version(path: Path) -> str:
    """Read the PEP 621 project version from pyproject.toml."""
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseCheckError(f"unable to read project version from {path}: {exc}") from exc


def _project_metadata(path: Path) -> dict:
    """Read the project table from a distribution pyproject file."""
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)["project"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseCheckError(f"unable to read project metadata from {path}: {exc}") from exc


def validate_distribution_metadata(
    core_path: Path,
    full_path: Path,
    lexical_path: Path,
    version: str,
) -> None:
    """Validate names, versions, and dependency boundaries for all PyPI distributions."""
    core = _project_metadata(core_path)
    full = _project_metadata(full_path)
    lexical = _project_metadata(lexical_path)
    expected_names = {
        "core": "ephemeral-buffer-mcp-core",
        "full": "ephemeral-buffer-mcp",
        "lexical": "ephemeral-buffer-mcp-lexical",
    }
    projects = {"core": core, "full": full, "lexical": lexical}
    for kind, project in projects.items():
        if project.get("name") != expected_names[kind]:
            raise ReleaseCheckError(
                f"{kind} distribution must be named {expected_names[kind]!r}"
            )
        if project.get("version") != version:
            raise ReleaseCheckError(
                f"{expected_names[kind]} version {project.get('version')!r} "
                f"does not match release version {version!r}"
            )

    core_dependency = f"ephemeral-buffer-mcp-core=={version}"
    semantic_core_dependency = f"ephemeral-buffer-mcp-core[semantic]=={version}"
    if core.get("dependencies") != ["mcp>=2.3.0,<3", "urllib3>=2.8.0"]:
        raise ReleaseCheckError("core distribution must contain only the base runtime dependencies")
    semantic_extra = core.get("optional-dependencies", {}).get("semantic", [])
    expected_semantic_extra = {
        "filelock<3.20.4; python_version < '3.11'",
        "filelock>=4.2.1,<5; python_version >= '3.11'",
        "huggingface-hub>=0.20,<2",
        "numpy>=1.24.0",
        "onnxruntime>=1.17,<1.24; python_version < '3.11'",
        "onnxruntime>=1.17,<2; python_version >= '3.11'",
        "tokenizers>=0.15,<1",
    }
    if set(semantic_extra) != expected_semantic_extra:
        raise ReleaseCheckError("core semantic extra must declare the supported ONNX runtime stack")
    if full.get("dependencies") != [semantic_core_dependency]:
        raise ReleaseCheckError("full distribution must depend on the matching semantic core")
    if lexical.get("dependencies") != [core_dependency]:
        raise ReleaseCheckError("lexical distribution must depend on the matching base core")


def changelog_version(changelog: str, version: str) -> None:
    """Require a dated/non-unreleased changelog section for the version."""
    extract_changelog_notes(changelog, version)


def extract_changelog_notes(changelog: str, version: str) -> str:
    """Return the release heading and notes for a validated version."""
    lines = changelog.splitlines()
    for index, line in enumerate(lines):
        match = CHANGELOG_HEADING.fullmatch(line)
        if not match:
            continue
        heading_version = match.group("bracketed") or match.group("plain")
        if heading_version != version:
            continue
        label = (match.group("label") or "").strip()
        if not CHANGELOG_DATE.fullmatch(label):
            raise ReleaseCheckError(
                f"CHANGELOG.md section for {version} must use a release date in "
                "YYYY-MM-DD format"
            )
        try:
            datetime.date.fromisoformat(label)
        except ValueError as exc:
            raise ReleaseCheckError(
                f"CHANGELOG.md section for {version} has invalid release date {label!r}"
            ) from exc
        end = next(
            (candidate for candidate in range(index + 1, len(lines))
             if lines[candidate].startswith("## ")),
            len(lines),
        )
        return "\n".join(lines[index:end]).strip()
    raise ReleaseCheckError(
        f"CHANGELOG.md has no release section for {version}; add a heading such as "
        f"## {version} - YYYY-MM-DD"
    )


def validate_metadata(tag: str, project_version: str, changelog: str) -> None:
    """Validate tag, package, and changelog version agreement."""
    version = tag_version(tag)
    if version != project_version:
        raise ReleaseCheckError(
            f"release tag {tag} resolves to {version}, but pyproject.toml declares "
            f"{project_version}"
        )
    changelog_version(changelog, version)


def validate_source_state(tag_commit: str, main_commit: str, is_on_main: bool, status: str) -> None:
    """Validate that the tagged source is clean and belongs to the default branch."""
    if status:
        raise ReleaseCheckError(
            "release checkout is not clean; commit or remove these paths before tagging:\n"
            + status
        )
    if not is_on_main:
        raise ReleaseCheckError(
            f"tag commit {tag_commit} is not contained in the default branch commit "
            f"{main_commit}; create the tag from main"
        )


def git(*args: str) -> str:
    """Run Git and return stripped stdout, raising on command failure."""
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "unknown git error"
        raise ReleaseCheckError(f"git {' '.join(args)} failed: {detail}") from exc


def is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is reachable from ``descendant``."""
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
    ).returncode == 0


def run(argv: list[str] | None = None) -> int:
    """Validate release metadata and source-state guardrails from CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.1.1")
    parser.add_argument("--main-ref", default="origin/main", help="Fetched default-branch ref")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args(argv)

    try:
        version = package_version(args.pyproject)
        changelog = args.changelog.read_text(encoding="utf-8")
        validate_metadata(args.tag, version, changelog)
        repository_root = Path(__file__).resolve().parents[1]
        if args.pyproject.resolve() == repository_root / "pyproject.toml":
            validate_distribution_metadata(
                args.pyproject,
                repository_root / "packaging" / "full" / "pyproject.toml",
                repository_root / "packaging" / "lexical" / "pyproject.toml",
                version,
            )
        tag_commit = git("rev-parse", "--verify", f"{args.tag}^{{commit}}")
        main_commit = git("rev-parse", "--verify", args.main_ref)
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        validate_source_state(tag_commit, main_commit, is_ancestor(tag_commit, main_commit), status)
    except (OSError, ReleaseCheckError) as exc:
        print(f"release guardrail failed: {exc}", file=sys.stderr)
        return 1

    print(f"Release guardrails passed for {args.tag} ({version})")
    print(f"Tag commit {tag_commit} is contained in {args.main_ref} ({main_commit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
