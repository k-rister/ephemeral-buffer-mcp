#!/usr/bin/env python3
"""Compare coding-agent workload results without rerunning the workloads.

Every benchmark, evaluation, and agent runner in this repository can emit a
``coding-agent-workload-result`` document (see ``workload_results.py``).  This
module compares two or more of those documents and reports, for every run and
measurement they share, the absolute and percentage delta and an outcome:

``improved``
    the change moved in the metric's better direction by more than the
    tolerance
``regressed``
    the change moved in the worse direction by more than the tolerance
``changed``
    the value changed but the metric has no known better direction
``unchanged``
    the change is within the tolerance
``missing``
    the metric, statistic, or run is absent or ``null`` on at least one side
``incompatible``
    both sides report the metric but with different units

Missing and incompatible data is always listed rather than silently skipped,
and the comparison carries each document's workload parameters and
environment so a reader can tell a regression from a different setup.

References name a whole document (``PATH``), one run inside it
(``PATH#RUN_ID``), or the documents of an experiment group recorded under a
directory (``DIR@GROUP``, narrowed by metadata with ``DIR@GROUP,KEY=VALUE``;
see ``parse_reference``).  Runs pair by ``id``; when the baseline and a
candidate each select exactly one run, those two runs pair regardless of their
ids, which compares two configurations recorded in the same document (for
example the ``control`` and ``mcp`` runs of an A/B summary).

The first reference is the baseline and every later reference is compared
against it.  The JSON output is a ``coding-agent-workload-comparison`` document
(format version 1).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import workload_results as wr


FORMAT = "coding-agent-workload-comparison"
FORMAT_VERSION = 1

OUTCOMES = ("improved", "regressed", "changed", "unchanged", "missing", "incompatible")
DIRECTIONS = ("lower", "higher")
DEFAULT_STATISTICS = ("value", "median", "mean", "p95")
# Dispersion statistics describe spread, not level, so a metric's better
# direction does not apply to them; their changes are reported as ``changed``.
DISPERSION_STATISTICS = ("stdev",)

# Fallback by unit for producer-specific names; canonical names take their
# direction from ``workload_results.CANONICAL_DIRECTIONS``.  Counts, ratios, and scores
# are ambiguous (a rank, an overhead ratio, and a quality score all differ),
# so they stay unjudged unless ``--direction`` says otherwise.
UNIT_DIRECTIONS: dict[str, str | None] = {
    "seconds": "lower",
    "bytes": "lower",
    "tokens": "lower",
    "per_second": "higher",
    "count": None,
    "ratio": None,
    "score": None,
}
UNIT_SUFFIXES = {"seconds": " s", "bytes": " B", "tokens": " tok", "per_second": "/s"}
# Environment fields that always differ between recordings and therefore do
# not explain a delta.
UNINFORMATIVE_ENVIRONMENT = frozenset({"recorded_at"})

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_CHECK_FAILED = 2


class ComparisonError(ValueError):
    """Raised when references cannot be compared."""


# --------------------------------------------------------------------------
# References and documents


class Reference(NamedTuple):
    """A parsed reference: a file or a group of documents under a directory."""

    path: Path
    run_id: str | None
    group: str | None
    where: tuple[tuple[str, Any], ...]


REFERENCE_FORMS = "PATH, PATH#RUN_ID, DIR@GROUP, or DIR@GROUP,KEY=VALUE[,KEY=VALUE...]"


def parse_reference(text: str) -> Reference:
    """Parse ``PATH[#RUN_ID]`` or ``DIR@GROUP[,KEY=VALUE...][#RUN_ID]``.

    A path that exists as written wins over the ``#`` and ``@`` splits, so
    file names containing those characters still work when nothing is
    selected.  ``DIR@GROUP`` names every result document under ``DIR``
    (searched recursively) whose ``experiment.group`` is ``GROUP``, ordered by
    ``started_at`` metadata or ``recorded_at``; each ``KEY=VALUE`` keeps only
    documents whose metadata has that value, and ``#RUN_ID`` applies to every
    selected document.
    """
    if Path(text).exists() or ("#" not in text and "@" not in text):
        return Reference(Path(text), None, None, ())
    rest, run_id = text, None
    if "#" in text:
        rest, _, run_id = text.rpartition("#")
        if not rest or not run_id:
            raise ComparisonError(f"reference {text!r} must be {REFERENCE_FORMS}")
    if "@" not in rest or Path(rest).exists():
        return Reference(Path(rest), run_id, None, ())
    path, _, selector = rest.partition("@")
    group, *selectors = selector.split(",")
    if not path or not group:
        raise ComparisonError(f"reference {text!r} must be {REFERENCE_FORMS}")
    try:
        where = tuple(wr.parse_key_value(item, "metadata selector") for item in selectors)
    except wr.WorkloadResultError as exc:
        raise ComparisonError(f"reference {text!r}: {exc}") from exc
    return Reference(Path(path), run_id, group, where)


def parse_label_filter(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` where value is JSON when possible, else a string."""
    try:
        return wr.parse_key_value(text, "label filter")
    except wr.WorkloadResultError as exc:
        raise ComparisonError(str(exc)) from exc


def label_matches(label: Any, wanted: Any) -> bool:
    """Return whether a label equals a filter value (booleans only match booleans)."""
    return wr.value_matches(label, wanted)


def parse_direction(text: str) -> tuple[str, str]:
    """Parse ``NAME=lower`` or ``NAME=higher``."""
    name, separator, direction = text.partition("=")
    if not separator or not name or direction not in DIRECTIONS:
        raise ComparisonError(f"direction {text!r} must be NAME=lower or NAME=higher")
    return name, direction


def selected_status(runs: list[dict[str, Any]], errors: list[str]) -> str:
    """Return the status of a narrowed selection of runs.

    Top-level errors still mean ``error``; runs that all share one status keep
    it (one ``partial`` run stays ``partial``); mixed selections follow
    ``workload_results.derive_status``.
    """
    statuses = {item["status"] for item in runs}
    if not errors and len(statuses) == 1:
        return statuses.pop()
    return wr.derive_status(runs, errors)


Scanned = list[tuple[Path, dict[str, Any] | None, str | None]]


def _claimed_group(path: Path) -> Any:
    """Return the ``experiment.group`` an invalid document claims, or ``None`` when it cannot be read."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["experiment"]["group"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def scan_directory(directory: Path, scanned: dict[Path, Scanned] | None = None) -> Scanned:
    """Return every result document under ``directory``, reading it once per comparison."""
    key = directory.resolve()
    if scanned is not None and key in scanned:
        return scanned[key]
    entries: Scanned = list(wr.iter_result_files([directory]))
    if scanned is not None:
        scanned[key] = entries
    return entries


def resolve_group(
    reference: str,
    parsed: Reference,
    loaded: dict[Path, dict[str, Any]] | None = None,
    scanned: dict[Path, Scanned] | None = None,
) -> list[Path]:
    """Return the documents a ``DIR@GROUP`` reference selects, in group order.

    Valid documents are cached in ``loaded`` and the directory scan in
    ``scanned``.  An invalid document is an error only when it claims the
    requested group; invalid documents in other groups, or with no group,
    cannot be selected and are ignored here (``list_workload_results.py``
    reports them).
    """
    if not parsed.path.is_dir():
        raise ComparisonError(f"reference {reference!r} names group {parsed.group!r} but {parsed.path} is not a directory")
    selected: list[tuple[str, Path]] = []
    groups: set[str] = set()
    for path, result, error in scan_directory(parsed.path, scanned):
        if result is None:
            if _claimed_group(path) == parsed.group:
                raise wr.WorkloadResultError(f"{path}: {error}")
            continue
        if loaded is not None:
            loaded[path] = result
        group = wr.experiment_block(result)["group"]
        if group is not None:
            groups.add(group)
        if wr.experiment_matches(result, [parsed.group], parsed.where):
            selected.append((wr.experiment_timestamp(result), path))
    if not selected:
        available = ", ".join(sorted(groups)) or "none"
        raise ComparisonError(f"{reference!r} selects no result document under {parsed.path} (groups there: {available})")
    return [path for _, path in sorted(selected)]


def _load_selected(
    reference: str,
    path: Path,
    run_id: str | None,
    label_filters: list[tuple[str, Any]],
    loaded: dict[Path, dict[str, Any]] | None,
    selected_by: str | None,
) -> dict[str, Any]:
    if loaded is None or path not in loaded:
        result = wr.load_result(path)
        if loaded is not None:
            loaded[path] = result
    else:
        result = loaded[path]
    runs = list(result["runs"])
    if run_id is not None:
        runs = [item for item in runs if item["id"] == run_id]
        if not runs:
            available = ", ".join(item["id"] for item in result["runs"]) or "none"
            raise ComparisonError(f"{path} has no run {run_id!r} (available: {available})")
    filters = list(label_filters)
    for key, value in filters:
        runs = [item for item in runs if key in item["labels"] and label_matches(item["labels"][key], value)]
    narrowed = run_id is not None or bool(filters)
    return {
        "reference": reference,
        "selected_by": selected_by,
        "path": str(path),
        "selected_run": run_id,
        "workload": result["workload"],
        "environment": result["environment"],
        "experiment": wr.experiment_block(result),
        "status": selected_status(runs, result["errors"]) if narrowed else result["status"],
        "document_status": result["status"],
        "errors": list(result["errors"]),
        "measurements": result["measurements"],
        "runs": runs,
    }


def load_documents(
    reference: str,
    label_filters: Iterable[tuple[str, Any]] = (),
    loaded: dict[Path, dict[str, Any]] | None = None,
    scanned: dict[Path, Scanned] | None = None,
) -> list[dict[str, Any]]:
    """Load one reference into comparison documents with their selected runs.

    A file reference yields one document; a ``DIR@GROUP`` reference yields one
    per selected result, each with the concrete ``PATH[#RUN_ID]`` as its
    ``reference`` and the group reference as ``selected_by``.  ``loaded``
    caches validated results by path so several references into the same file
    (``PATH#control PATH#mcp``) read and validate it once, and ``scanned``
    caches directory scans so several group references into one directory
    walk it once.  When the reference or the filters narrow the document to
    some of its runs, ``status`` is derived from those runs alone and the
    whole document's status is kept as ``document_status``.
    """
    parsed = parse_reference(reference)
    filters = list(label_filters)
    if parsed.group is None:
        return [_load_selected(reference, parsed.path, parsed.run_id, filters, loaded, None)]
    documents = []
    for path in resolve_group(reference, parsed, loaded, scanned):
        concrete = str(path) + (f"#{parsed.run_id}" if parsed.run_id else "")
        documents.append(_load_selected(concrete, path, parsed.run_id, filters, loaded, reference))
    return documents


# --------------------------------------------------------------------------
# Deltas


def direction_for(name: str, unit: str, overrides: dict[str, str] | None = None) -> str | None:
    """Return the better direction for a measurement, if one is known."""
    if overrides and name in overrides:
        return overrides[name]
    if name in wr.CANONICAL_DIRECTIONS:
        return wr.CANONICAL_DIRECTIONS[name]
    return UNIT_DIRECTIONS[unit]


def validate_tolerance(tolerance_percent: float) -> float:
    """Return ``tolerance_percent`` after checking it is a finite, non-negative percentage."""
    if not math.isfinite(tolerance_percent):
        raise ComparisonError(f"tolerance must be a finite percentage, not {tolerance_percent}")
    if tolerance_percent < 0:
        raise ComparisonError("tolerance must not be negative")
    return tolerance_percent


def classify(delta: float, delta_percent: float | None, direction: str | None, tolerance_percent: float) -> str:
    """Return the outcome for one numeric delta."""
    if delta == 0:
        return "unchanged"
    if delta_percent is not None and abs(delta_percent) <= tolerance_percent:
        return "unchanged"
    if direction is None:
        return "changed"
    better = delta < 0 if direction == "lower" else delta > 0
    return "improved" if better else "regressed"


def _entry(**fields: Any) -> dict[str, Any]:
    entry = {
        "run": None,
        "kind": "measurement",
        "name": None,
        "statistic": None,
        "unit": None,
        "baseline": None,
        "candidate": None,
        "delta": None,
        "delta_percent": None,
        "direction": None,
        "outcome": None,
        "reason": None,
    }
    entry.update(fields)
    return entry


def compare_measurement(
    name: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    statistics: Iterable[str],
    tolerance_percent: float,
    directions: dict[str, str] | None = None,
    run: str | None = None,
    kind: str = "measurement",
) -> list[dict[str, Any]]:
    """Return comparison entries for one measurement name on both sides."""
    common = dict(run=run, kind=kind, name=name)
    if baseline is None or candidate is None:
        present = baseline if candidate is None else candidate
        side = "candidate" if candidate is None else "baseline"
        return [_entry(**common, unit=present["unit"], outcome="missing", reason=f"absent in {side}")]
    if baseline["unit"] != candidate["unit"]:
        return [_entry(
            **common,
            outcome="incompatible",
            reason=f"unit {baseline['unit']} in baseline but {candidate['unit']} in candidate",
        )]
    unit = baseline["unit"]
    level_direction = direction_for(name, unit, directions)
    entries = []
    for statistic in statistics:
        if statistic not in baseline and statistic not in candidate:
            continue
        before = wr.statistic(baseline, statistic)
        after = wr.statistic(candidate, statistic)
        direction = None if statistic in DISPERSION_STATISTICS else level_direction
        common_stat = dict(common, statistic=statistic, unit=unit, baseline=before, candidate=after, direction=direction)
        if before is None or after is None:
            if before is None and after is None:
                reason = "unavailable in both"
            else:
                reason = "unavailable in baseline" if before is None else "unavailable in candidate"
            entries.append(_entry(**common_stat, outcome="missing", reason=reason))
            continue
        delta = after - before
        if not math.isfinite(delta):
            entries.append(_entry(**common_stat, outcome="changed", reason="difference is too large to represent"))
            continue
        delta_percent = (delta / abs(before)) * 100.0 if before != 0 else None
        if delta_percent is not None and not math.isfinite(delta_percent):
            delta_percent = None
        entries.append(_entry(
            **common_stat,
            delta=delta,
            delta_percent=delta_percent,
            outcome=classify(delta, delta_percent, direction, tolerance_percent),
        ))
    return entries


def compare_measurement_blocks(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    run: str | None,
    kind: str,
    statistics: Iterable[str],
    tolerance_percent: float,
    directions: dict[str, str] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    names = list(baseline) + [name for name in candidate if name not in baseline]
    for name in names:
        entries.extend(compare_measurement(
            name,
            baseline.get(name),
            candidate.get(name),
            statistics=statistics,
            tolerance_percent=tolerance_percent,
            directions=directions,
            run=run,
            kind=kind,
        ))
    return entries


def _phases(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: {key: value for key, value in entry.items() if key != "name"} for entry in run["phases"]}


def pair_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]]:
    """Return ``(pair id, baseline run, candidate run)`` for every run on either side."""
    if baseline["selected_run"] and candidate["selected_run"] and len(baseline["runs"]) == 1 and len(candidate["runs"]) == 1:
        before, after = baseline["runs"][0], candidate["runs"][0]
        pair_id = before["id"] if before["id"] == after["id"] else f"{before['id']} -> {after['id']}"
        return [(pair_id, before, after)]
    after_by_id = {item["id"]: item for item in candidate["runs"]}
    pairs = [(item["id"], item, after_by_id.pop(item["id"], None)) for item in baseline["runs"]]
    pairs.extend((item["id"], None, item) for item in after_by_id.values())
    return pairs


def _differences(before: dict[str, Any], after: dict[str, Any], ignore: frozenset[str] = frozenset()) -> dict[str, list[Any]]:
    keys = [key for key in before if key not in ignore] + [key for key in after if key not in before and key not in ignore]
    return {key: [before.get(key), after.get(key)] for key in keys if before.get(key) != after.get(key)}


def _flatten_experiment(block: dict[str, Any]) -> dict[str, Any]:
    """Return ``group`` and the metadata keys side by side for difference reports."""
    flat = {"group": block["group"]}
    for key, value in block["metadata"].items():
        flat["metadata.group" if key == "group" else key] = value
    return flat


def compare_documents(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    statistics: Iterable[str] = DEFAULT_STATISTICS,
    tolerance_percent: float = 0.0,
    directions: dict[str, str] | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare one candidate document against the baseline."""
    tolerance_percent = validate_tolerance(tolerance_percent)
    statistics = tuple(statistics)
    wanted = set(metrics) if metrics is not None else None
    metrics = wanted
    options = dict(statistics=statistics, tolerance_percent=tolerance_percent, directions=directions)
    entries = compare_measurement_blocks(
        baseline["measurements"], candidate["measurements"], run=None, kind="measurement", **options
    )
    runs = []
    for pair_id, before, after in pair_runs(baseline, candidate):
        record = {
            "id": pair_id,
            "baseline_run": before["id"] if before else None,
            "candidate_run": after["id"] if after else None,
            "baseline_status": before["status"] if before else None,
            "candidate_status": after["status"] if after else None,
            "baseline_errors": list(before["errors"]) if before else [],
            "candidate_errors": list(after["errors"]) if after else [],
            "labels": [before["labels"] if before else None, after["labels"] if after else None],
        }
        if before is None or after is None:
            record["outcome"] = "missing"
            record["reason"] = "absent in candidate" if after is None else "absent in baseline"
        else:
            record["outcome"] = "compared"
            record["reason"] = None
            entries.extend(compare_measurement_blocks(
                before["measurements"], after["measurements"], run=pair_id, kind="measurement", **options
            ))
            entries.extend(compare_measurement_blocks(_phases(before), _phases(after), run=pair_id, kind="phase", **options))
        runs.append(record)
    if wanted is not None:
        entries = [entry for entry in entries if entry["name"] in wanted]
    summary = {outcome: sum(1 for entry in entries if entry["outcome"] == outcome) for outcome in OUTCOMES}
    summary["runs_compared"] = sum(1 for record in runs if record["outcome"] == "compared")
    summary["runs_missing"] = sum(1 for record in runs if record["outcome"] == "missing")
    summary["statuses"] = [baseline["status"], candidate["status"]]
    summary["all_succeeded"] = baseline["status"] == "success" and candidate["status"] == "success"
    return {
        "baseline": baseline["reference"],
        "candidate": candidate["reference"],
        "parameter_differences": _differences(baseline["workload"]["parameters"], candidate["workload"]["parameters"]),
        "environment_differences": _differences(baseline["environment"], candidate["environment"], UNINFORMATIVE_ENVIRONMENT),
        "experiment_differences": _differences(_flatten_experiment(baseline["experiment"]), _flatten_experiment(candidate["experiment"])),
        "runs": runs,
        "entries": entries,
        "summary": summary,
    }


def compare(
    references: list[str],
    *,
    label_filters: Iterable[tuple[str, Any]] = (),
    statistics: Iterable[str] = DEFAULT_STATISTICS,
    tolerance_percent: float = 0.0,
    directions: dict[str, str] | None = None,
    metrics: Iterable[str] | None = None,
    allow_workload_mismatch: bool = False,
) -> dict[str, Any]:
    """Compare every later reference against the first and return the comparison document.

    A single ``DIR@GROUP`` reference is enough when the group holds at least
    two documents: the earliest is the baseline.
    """
    if not references:
        raise ComparisonError("at least two documents are required")
    tolerance_percent = validate_tolerance(tolerance_percent)
    statistics = tuple(statistics)
    unknown = [name for name in statistics if name not in wr.STATISTICS]
    if unknown:
        raise ComparisonError(f"unknown statistic {', '.join(unknown)}; expected one of {', '.join(wr.STATISTICS)}")
    filters = list(label_filters)
    wanted = sorted(set(metrics)) if metrics is not None else None
    loaded: dict[Path, dict[str, Any]] = {}
    scanned: dict[Path, Scanned] = {}
    documents = [document for reference in references for document in load_documents(reference, filters, loaded, scanned)]
    if len(documents) < 2:
        raise ComparisonError(f"at least two documents are required; {', '.join(map(repr, references))} selects {len(documents)}")
    baseline = documents[0]
    for document in documents[1:]:
        if document["workload"]["name"] != baseline["workload"]["name"] and not allow_workload_mismatch:
            raise ComparisonError(
                f"{document['reference']} measures workload {document['workload']['name']!r} but the baseline "
                f"{baseline['reference']} measures {baseline['workload']['name']!r}; pass --allow-workload-mismatch to compare anyway"
            )
    comparisons = [
        compare_documents(
            baseline,
            document,
            statistics=statistics,
            tolerance_percent=tolerance_percent,
            directions=directions,
            metrics=wanted,
        )
        for document in documents[1:]
    ]
    regressions = sum(item["summary"]["regressed"] for item in comparisons)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "workload": baseline["workload"]["name"],
        "options": {
            "statistics": list(statistics),
            "tolerance_percent": tolerance_percent,
            "directions": dict(directions or {}),
            "metrics": wanted,
            "label_filters": [[key, value] for key, value in filters],
        },
        "documents": [
            {
                "role": "baseline" if index == 0 else "candidate",
                "reference": document["reference"],
                "selected_by": document["selected_by"],
                "path": document["path"],
                "selected_run": document["selected_run"],
                "workload": document["workload"],
                "environment": document["environment"],
                "experiment": document["experiment"],
                "status": document["status"],
                "document_status": document["document_status"],
                "errors": document["errors"],
                "runs": [item["id"] for item in document["runs"]],
            }
            for index, document in enumerate(documents)
        ],
        "comparisons": comparisons,
        "summary": {
            "regressions": regressions,
            "all_succeeded": all(document["status"] == "success" for document in documents),
        },
    }


def check_passes(comparison: dict[str, Any]) -> bool:
    """Return whether a regression check should pass: no regressions and every compared document succeeded.

    A document narrowed with ``PATH#RUN_ID`` or a label filter is judged by
    its selected runs, so failures elsewhere in the file do not fail the check.
    """
    return comparison["summary"]["regressions"] == 0 and comparison["summary"]["all_succeeded"]


# --------------------------------------------------------------------------
# Human-readable report


def format_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "n/a"
    text = f"{int(value):,}" if float(value).is_integer() else f"{value:,.4g}"
    return text + UNIT_SUFFIXES.get(unit or "", "")


def format_delta(entry: dict[str, Any]) -> tuple[str, str]:
    if entry["delta"] is None:
        return "-", "-"
    sign = "+" if entry["delta"] > 0 else ""
    absolute = sign + format_value(entry["delta"], entry["unit"])
    if entry["delta"] == 0:
        percent = "0.0%"
    elif entry["delta_percent"] is None:
        percent = "n/a"
    else:
        percent = f"{entry['delta_percent']:+.1f}%"
    return absolute, percent


_table = wr.format_table


def _describe_document(document: dict[str, Any]) -> str:
    env = document["environment"]
    tool = env.get("tool") or {}
    revision = env.get("source_revision")
    group = document["experiment"]["group"]
    parts = [
        f"{document['role']}: {document['reference']}",
        *([f"group={group}"] if group is not None else []),
        f"producer={document['workload']['producer']}",
        f"kind={document['workload']['kind']}",
        f"status={document['status']}",
        *([f"document_status={document['document_status']}"] if document["document_status"] != document["status"] else []),
        f"runs={len(document['runs'])}",
        f"recorded={env['recorded_at']}",
        f"python={env['python_version']}",
    ]
    if tool.get("version"):
        parts.append(f"tool={tool.get('name')} {tool['version']}")
    if revision:
        parts.append(f"revision={revision[:12]}")
    return "  ".join(parts)


def format_report(comparison: dict[str, Any]) -> str:
    """Return the human-readable comparison."""
    lines = [f"workload: {comparison['workload']}"]
    lines.extend("  " + _describe_document(document) for document in comparison["documents"])
    options = comparison["options"]
    lines.append(
        f"  statistics={','.join(options['statistics'])} tolerance={options['tolerance_percent']:g}%"
        + (f" metrics={','.join(options['metrics'])}" if options["metrics"] else "")
        + (f" filters={' '.join(f'{key}={json.dumps(value)}' for key, value in options['label_filters'])}" if options["label_filters"] else "")
    )
    for item in comparison["comparisons"]:
        lines.append("")
        lines.append(f"{item['baseline']} -> {item['candidate']}")
        for title, differences in (
            ("parameter", item["parameter_differences"]),
            ("environment", item["environment_differences"]),
            ("experiment", item["experiment_differences"]),
        ):
            if differences:
                described = "; ".join(f"{key}: {json.dumps(before)} -> {json.dumps(after)}" for key, (before, after) in differences.items())
                lines.append(f"  {title} differences: {described}")
        rows = []
        for record in item["runs"]:
            errors = "; ".join(record["baseline_errors"] + record["candidate_errors"])
            if record["outcome"] == "missing":
                rows.append([
                    record["id"], "(run)", "", record["baseline_status"] or "", record["candidate_status"] or "", "", "",
                    f"missing: {record['reason']}" + (f" ({errors})" if errors else ""),
                ])
            elif record["baseline_status"] != "success" or record["candidate_status"] != "success":
                rows.append([
                    record["id"], "(run status)", "", record["baseline_status"], record["candidate_status"], "", "",
                    errors or "non-success status",
                ])
        for entry in item["entries"]:
            absolute, percent = format_delta(entry)
            name = entry["name"] if entry["kind"] == "measurement" else f"phase:{entry['name']}"
            outcome = entry["outcome"] + (f": {entry['reason']}" if entry["reason"] else "")
            rows.append([
                entry["run"] or "(workload)",
                name,
                entry["statistic"] or "",
                format_value(entry["baseline"], entry["unit"]),
                format_value(entry["candidate"], entry["unit"]),
                absolute,
                percent,
                outcome,
            ])
        if rows:
            lines.extend("  " + line for line in _table(rows, ["run", "metric", "stat", "baseline", "candidate", "delta", "change", "outcome"]))
        else:
            lines.append("  nothing to compare")
        summary = item["summary"]
        lines.append(
            "  summary: " + ", ".join(f"{summary[outcome]} {outcome}" for outcome in OUTCOMES)
            + f"; runs compared={summary['runs_compared']} missing={summary['runs_missing']}"
            + f"; status {summary['statuses'][0]} -> {summary['statuses'][1]}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare coding-agent workload result documents.",
        epilog=(
            f"References are {REFERENCE_FORMS}; the first is the baseline. DIR@GROUP expands to every "
            "result document under DIR in that experiment group, ordered by started_at metadata or recorded_at."
        ),
    )
    parser.add_argument("references", nargs="+", help="Workload result files or DIR@GROUP experiment groups to compare (baseline first)")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="What to print on stdout (default: text)")
    parser.add_argument("--output", metavar="PATH", help="Also write the JSON comparison document to PATH")
    parser.add_argument(
        "--statistic", action="append", metavar="NAME", dest="statistics",
        help=f"Statistic to compare (repeatable; default: {', '.join(DEFAULT_STATISTICS)}; 'all' compares every statistic)",
    )
    parser.add_argument("--metric", action="append", metavar="NAME", dest="metrics", help="Only report this measurement or phase name (repeatable)")
    parser.add_argument("--select", action="append", metavar="KEY=VALUE", default=[], help="Only compare runs whose label KEY equals VALUE (repeatable)")
    parser.add_argument("--direction", action="append", metavar="NAME=lower|higher", default=[], help="Declare which direction is better for a measurement (repeatable)")
    parser.add_argument("--tolerance", type=float, default=0.0, metavar="PERCENT", help="Changes within this percentage are unchanged (default: 0)")
    parser.add_argument("--allow-workload-mismatch", action="store_true", help="Compare documents whose workload names differ")
    parser.add_argument(
        "--check", action="store_true",
        help=(
            f"Exit with status {EXIT_CHECK_FAILED} when any metric regressed or any document (its selected runs, "
            "when narrowed with PATH#RUN_ID or --select) has a non-success status"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.statistics and "all" in args.statistics:
            statistics: tuple[str, ...] = wr.STATISTICS
        else:
            statistics = tuple(args.statistics) if args.statistics else DEFAULT_STATISTICS
        directions = dict(parse_direction(text) for text in args.direction)
        comparison = compare(
            args.references,
            label_filters=[parse_label_filter(text) for text in args.select],
            statistics=statistics,
            tolerance_percent=args.tolerance,
            directions=directions,
            metrics=args.metrics,
            allow_workload_mismatch=args.allow_workload_mismatch,
        )
    except (ComparisonError, wr.WorkloadResultError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    document = json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    if args.format == "json":
        sys.stdout.write(document)
    else:
        sys.stdout.write(format_report(comparison))
    sys.stdout.flush()
    if args.check and not check_passes(comparison):
        summary = comparison["summary"]
        print(
            f"check failed: {summary['regressions']} regressed metric(s), "
            f"all documents succeeded={str(summary['all_succeeded']).lower()}",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
