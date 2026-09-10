#!/usr/bin/env python3
"""Create a deterministic, privacy-safe repository-shaped A/B fixture."""

import argparse
import json
from pathlib import Path
from typing import Any

FIXTURE_VERSION = 1
PROFILE = "repository-shaped-v1"

TASKS = {
    "targeted-inspection": {
        "category": "repository-inspection",
        "output_profile": "32 lines",
        "prompt": "From the repository root, run python3 tools/repository_checks.py targeted. Use the repository source and configuration to find the exact line containing TARGETED_SIGNAL and report that signal and its source line number. Do not paste the complete output.",
        "signal_marker": "TARGETED_SIGNAL",
        "success_criteria": "Final response contains TARGETED_SIGNAL and the correct source line number.",
    },
    "noisy-test-failure": {
        "category": "repository-test-output",
        "output_profile": "1,600 lines",
        "prompt": "From the repository root, run python3 tools/repository_checks.py tests. Locate the failed test, assertion, and TEST_FAILURE_SIGNAL, then report all three. Do not paste the complete output.",
        "signal_marker": "TEST_FAILURE_SIGNAL",
        "success_criteria": "Final response identifies test_case_1379, TEST_FAILURE_SIGNAL, and the assertion text.",
    },
    "build-log-search": {
        "category": "repository-build-output",
        "output_profile": "2,000 lines",
        "prompt": "From the repository root, run python3 tools/repository_checks.py build. Search the build output and report BUILD_FAILURE_SIGNAL with its source file and line number. Do not paste the complete output.",
        "signal_marker": "BUILD_FAILURE_SIGNAL",
        "success_criteria": "Final response contains BUILD_FAILURE_SIGNAL, src/parser.c, and line 917.",
    },
    "follow-up-context": {
        "category": "repository-search-and-retrieval",
        "output_profile": "1,600 lines plus follow-up question",
        "prompt": "From the repository root, run python3 tools/repository_checks.py tests, retain the result, then answer the follow-up: which test failed and what signal identifies it? Report test_case_1379 and TEST_FAILURE_SIGNAL without pasting the complete output.",
        "signal_marker": "TEST_FAILURE_SIGNAL",
        "success_criteria": "Final response preserves the earlier result and identifies test_case_1379 with TEST_FAILURE_SIGNAL.",
    },
}

FILES = {
    "pyproject.toml": "[project]\nname = \"fixture-parser\"\nversion = \"0.1.0\"\ndescription = \"Synthetic repository-shaped A/B fixture\"\n\n[tool.fixture]\nparser_mode = \"strict\"\ncache_key = \"parser-v2\"\n",
    "src/packet_parser/__init__.py": "\"\"\"Small parser package used by the repository-shaped fixture.\"\"\"\n\nfrom .parser import parse_packet\n\n__all__ = [\"parse_packet\"]\n",
    "src/packet_parser/parser.py": "\"\"\"Deterministic parser implementation for fixture inspection tasks.\"\"\"\n\n\ndef parse_packet(value: str) -> dict[str, str]:\n    \"\"\"Parse the fixture protocol's key=value packet format.\"\"\"\n    key, separator, payload = value.partition(\"=\")\n    if not separator or not key:\n        raise ValueError(\"packet must contain a key and value\")\n    return {\"key\": key, \"value\": payload}\n\n\nTARGETED_NOTE = \"TARGETED_SIGNAL cache key collision in parser\"\n",
    "tests/test_parser.py": "import unittest\n\nfrom packet_parser import parse_packet\n\n\nclass ParserTests(unittest.TestCase):\n    def test_parse_packet(self):\n        self.assertEqual(parse_packet(\"status=ready\")[\"value\"], \"ready\")\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
    "tools/repository_checks.py": "#!/usr/bin/env python3\n\"\"\"Run deterministic repository-shaped test, build, and inspection workflows.\"\"\"\n\nimport sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\n\ndef targeted() -> None:\n    source = ROOT / \"src/packet_parser/parser.py\"\n    for number in range(1, 32):\n        print(f\"inspection record {number:02d}: checked repository source\")\n    for number, line in enumerate(source.read_text(encoding=\"utf-8\").splitlines(), 1):\n        if \"TARGETED_SIGNAL\" in line:\n            print(f\"inspection result: TARGETED_SIGNAL in {source.relative_to(ROOT)}:{number}\")\n            return\n    raise SystemExit(\"targeted signal not found\")\n\ndef tests() -> None:\n    for number in range(1, 1601):\n        status = \"FAIL\" if number == 1379 else \"ok\"\n        print(f\"test_case_{number:04d} ... {status}\")\n    print(\"AssertionError: TEST_FAILURE_SIGNAL expected status=ready, got status=stalled\")\n    print(\"Ran 1600 tests in 12.4s\")\n\ndef build() -> None:\n    config = ROOT / \"pyproject.toml\"\n    for number in range(1, 2001):\n        print(f\"compiler progress module_{number:04d}.c (config={config.name})\")\n    print(\"src/parser.c:917: error: BUILD_FAILURE_SIGNAL missing terminating token\")\n    print(\"build failed: 1 error, 0 warnings\")\n\ndef main() -> None:\n    commands = {\"targeted\": targeted, \"tests\": tests, \"build\": build}\n    if len(sys.argv) != 2 or sys.argv[1] not in commands:\n        raise SystemExit(\"usage: repository_checks.py targeted|tests|build\")\n    commands[sys.argv[1]]()\n\nif __name__ == \"__main__\":\n    main()\n",
    "README.md": "# Fixture Parser\n\nThis small repository-shaped fixture contains source, tests, configuration, and workflow tooling. It is generated for privacy-safe agent evaluation only.\n",
}


def build_manifest() -> dict[str, Any]:
    return {
        "fixture_version": FIXTURE_VERSION,
        "profile": PROFILE,
        "privacy": "synthetic deterministic content only; no user prompts, logs, captures, or repository data",
        "tasks": TASKS,
    }


def create_fixture(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"fixture output must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative_path, content in FILES.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (destination / "tools/repository_checks.py").chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.fixture_output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(args.fixture_output), "manifest": str(args.manifest_output), "profile": PROFILE}))


if __name__ == "__main__":
    main()
