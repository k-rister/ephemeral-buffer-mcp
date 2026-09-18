#!/usr/bin/env python3
"""List and filter coding-agent workload results by experiment group and metadata.

Every producer in this repository can tag the ``coding-agent-workload-result``
document it writes (``--result``) with an experiment group and metadata
(``--experiment GROUP --metadata KEY=VALUE``; see ``workload_results.py``).
This command finds those documents under files and directories, filters them
by group, metadata, workload, and status, and prints one row per document (or
per run with ``--runs``) so a multi-run study can be organised without
tracking file names by hand.

Failed, timed-out, and partial documents are listed with their status and
errors; documents that do not satisfy the format are listed as ``invalid``
with the reason and make the command exit with status 1, so a broken file is
never mistaken for an absent one.  JSON files under a directory that are not
workload results (producer records, schedules, manifests) are ignored.

``--format paths`` prints only the selected paths, in listing order, for use
in shell substitution with ``compare_workload_results.py``.  ``--format json``
prints a ``coding-agent-workload-listing`` document (format version 1).
"""

import argparse
import json
import sys
from typing import Any, Iterable

import workload_results as wr


FORMAT = "coding-agent-workload-listing"
FORMAT_VERSION = 1

EXIT_OK = 0
EXIT_INVALID = 1


def _key_value(text: str) -> tuple[str, Any]:
    try:
        return wr.parse_key_value(text, "metadata filter")
    except wr.WorkloadResultError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _metadata_key(text: str) -> str:
    try:
        return wr.metadata_key(text)
    except wr.WorkloadResultError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_listing(
    paths: Iterable[str],
    *,
    groups: Iterable[str] = (),
    where: Iterable[tuple[str, Any]] = (),
    workloads: Iterable[str] = (),
    statuses: Iterable[str] = (),
    redact: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the listing document for every selected result under ``paths``.

    Filters combine with AND; ``groups``, ``workloads``, and ``statuses`` each
    accept any of their values.  Invalid documents are never filtered out.
    Rows are ordered by group (ungrouped last), then ``started_at`` metadata
    or ``recorded_at``, then path.
    """
    groups = list(groups)
    where = list(where)
    workloads = set(workloads)
    statuses = set(statuses)
    hidden = list(redact)
    rows: list[tuple[tuple[int, str, str, str], dict[str, Any]]] = []
    for path, result, error in wr.iter_result_files(paths):
        if result is None:
            rows.append(((2, "", "", str(path)), {"path": str(path), "valid": False, "error": error, "status": "invalid"}))
            continue
        if not wr.experiment_matches(result, groups, where):
            continue
        if workloads and result["workload"]["name"] not in workloads:
            continue
        if statuses and result["status"] not in statuses:
            continue
        summary = wr.redact_summary(wr.result_summary(result), hidden)
        group = summary["group"]
        order = (0 if group is not None else 1, group or "", summary["ordered_at"], str(path))
        rows.append((order, {"path": str(path), "valid": True, "error": None, **summary}))
    rows.sort(key=lambda item: item[0])
    results = [row for _, row in rows]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "filters": {
            "groups": groups,
            "where": [[key, value] for key, value in where],
            "workloads": sorted(workloads),
            "statuses": sorted(statuses),
            "redact": hidden,
        },
        "results": results,
        "summary": {
            "listed": sum(1 for row in results if row["valid"]),
            "invalid": sum(1 for row in results if not row["valid"]),
            "groups": sorted({row["group"] for row in results if row["valid"] and row["group"] is not None}),
        },
    }


def _metadata_cells(row: dict[str, Any], fields: list[str]) -> list[str]:
    if not fields:
        return [wr.format_metadata(row["metadata"]) if row["valid"] else ""]
    metadata = row["metadata"] if row["valid"] else {}
    return [wr.format_metadata_value(metadata[field]) if field in metadata else "-" for field in fields]


def format_listing(listing: dict[str, Any], *, fields: Iterable[str] = (), runs: bool = False) -> str:
    """Return the human-readable listing.

    Without ``fields`` the metadata column shows every key; with them, one
    column per named key (``-`` when absent).  ``runs`` prints one row per run
    with its labels and status beside the document's status, so failed and
    timed-out runs are visible inside a partial document.
    """
    fields = list(fields)
    metadata_headers = fields or ["metadata"]
    rows = []
    for row in listing["results"]:
        if not row["valid"]:
            filler = ["-"] * (3 if runs else 2)
            rows.append([row["path"], "-", "-", "invalid", *filler, *_metadata_cells(row, fields), row["error"]])
            continue
        group = row["group"]
        cells = [row["path"], group if group is not None else "-", row["workload"]["name"], row["status"]]
        if runs:
            for item in row["runs"]:
                rows.append([
                    *cells, item["id"], item["status"], wr.format_metadata(item["labels"]),
                    *_metadata_cells(row, fields), "; ".join(row["errors"] + item["errors"]),
                ])
            if not row["runs"]:
                rows.append([*cells, "(none)", "-", "", *_metadata_cells(row, fields), "; ".join(row["errors"])])
        else:
            rows.append([
                *cells, str(len(row["runs"])), row["ordered_at"],
                *_metadata_cells(row, fields), "; ".join(row["errors"]),
            ])
    headers = ["path", "group", "workload", "status"]
    headers += ["run", "run_status", "labels"] if runs else ["runs", "time"]
    headers += [*metadata_headers, "errors"]
    if not rows:
        return "no workload results selected\n"
    return "\n".join(wr.format_table(rows, headers)) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List coding-agent workload result documents by experiment group and metadata.",
        epilog=(
            "Directories are searched recursively for *.json workload results; other JSON files are ignored. "
            f"Conventional metadata keys: {', '.join(wr.CANONICAL_METADATA)}."
        ),
    )
    parser.add_argument("paths", nargs="+", help="Workload result files or directories to list")
    parser.add_argument("--group", action="append", metavar="NAME", default=[], help="Only list documents in this experiment group (repeatable; any matches)")
    parser.add_argument("--where", action="append", metavar="KEY=VALUE", type=_key_value, default=[], help="Only list documents whose metadata KEY equals VALUE (repeatable; all must match; values parse as JSON when possible)")
    parser.add_argument("--workload", action="append", metavar="NAME", default=[], help="Only list documents for this workload name (repeatable)")
    parser.add_argument("--status", action="append", metavar="STATUS", choices=wr.STATUSES, default=[], help="Only list documents with this status (repeatable)")
    parser.add_argument("--field", action="append", metavar="KEY", type=_metadata_key, default=[], help="Show this metadata key as its own column instead of the combined metadata column (repeatable)")
    parser.add_argument("--redact", action="append", metavar="KEY", type=_metadata_key, default=[], help=f"Show {wr.REDACTED} instead of this metadata key's value (repeatable)")
    parser.add_argument("--runs", action="store_true", help="Print one row per run with its labels and status")
    parser.add_argument("--format", choices=("text", "json", "paths"), default="text", help="What to print on stdout (default: text)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    listing = build_listing(
        args.paths,
        groups=args.group,
        where=args.where,
        workloads=args.workload,
        statuses=args.status,
        redact=args.redact,
    )
    if args.format == "json":
        sys.stdout.write(json.dumps(listing, indent=2, sort_keys=True, allow_nan=False) + "\n")
    elif args.format == "paths":
        sys.stdout.write("".join(f"{row['path']}\n" for row in listing["results"] if row["valid"]))
    else:
        sys.stdout.write(format_listing(listing, fields=args.field, runs=args.runs))
    sys.stdout.flush()
    invalid = [row for row in listing["results"] if not row["valid"]]
    for row in invalid:
        print(f"{row['path']}: INVALID: {row['error']}", file=sys.stderr)
    return EXIT_INVALID if invalid else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
