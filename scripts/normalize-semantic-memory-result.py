#!/usr/bin/env python3
"""Convert a legacy semantic-memory benchmark log to common result JSON."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_native_record(path: Path) -> dict[str, Any]:
    """Find the benchmark JSON object after any model-loader log lines."""
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {
            "configuration", "input", "rss_bytes", "timing_seconds"
        } <= value.keys():
            return value
    raise ValueError(f"no semantic-memory JSON record found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    benchmark = importlib.import_module("benchmark_semantic_memory")
    native = read_native_record(args.log)
    result = benchmark.workload_result(native)
    result["environment"]["source_revision"] = args.source_revision
    metadata = (args.revision_dir / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', metadata)
    if version:
        result["environment"]["tool"]["version"] = version.group(1)

    for path, payload in (
        (args.native_output, native),
        (args.result_output, result),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
