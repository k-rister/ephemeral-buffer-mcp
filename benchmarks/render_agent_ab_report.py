#!/usr/bin/env python3
"""Append privacy-safe task and criterion tables to a release benchmark report."""

import argparse
import json
from pathlib import Path
from typing import Any


SOURCES = ("search_capture", "get_capture_slice")
RESPONSE_STATES = (
    "criterion_hit",
    "response_without_criterion_hit",
    "no_successful_response",
)
LEGACY_RESPONSE_STATES = {
    "criterion_hit": "phrase_hit",
    "response_without_criterion_hit": "response_without_phrase_hit",
}


def _read_summary(revision: str, path: Path) -> tuple[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {revision} summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{revision} summary root must be an object")
    return revision, value


def _fraction(item: Any, numerator: str) -> str:
    if not isinstance(item, dict) or not item.get("count"):
        return "n/a"
    return f"{item.get(numerator, 0)}/{item['count']}"


def _paired_delta(item: Any) -> str:
    if not isinstance(item, dict) or not item.get("available"):
        return "n/a"
    return f"{item['mean'] * 100:+.1f} pp ± {item['ci95_half_width'] * 100:.1f}"


def _display_number(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "n/a"


def _display_ci(interval: Any) -> str:
    if not isinstance(interval, dict):
        return "n/a"
    return f"[{interval['lower'] * 100:+.1f}, {interval['upper'] * 100:+.1f}] pp"


def _display_status(value: Any) -> str:
    labels = {
        "evidence_ready": "evidence-ready",
        "powered_but_inconclusive": "powered-but-inconclusive",
        "direction_established": "direction established",
        "within_practical_range": "within practical range",
        "collect_more": "collect more",
    }
    return labels.get(value, value.replace("_", " ") if isinstance(value, str) else "unknown")


def _task_ids(summary: dict[str, Any]) -> set[str]:
    task_ids: set[str] = set()
    for mode_summary in summary.get("task_success_by_mode", {}).values():
        task_ids.update(mode_summary.get("tasks", {}))
    for mode_summary in summary.get("criterion_summaries_by_mode", {}).values():
        task_ids.update(mode_summary.get("tasks", {}))
    for source_summary in (
        summary.get("criterion_exposure_summaries_by_mode", {})
        .get("mcp", {})
        .get("sources", {})
        .values()
    ):
        task_ids.update(source_summary.get("tasks", {}))
    return task_ids


def _response_state_outcome(outcomes: dict[str, Any], state: str) -> Any:
    """Read current state names and the phrase-only names from schema v3."""
    outcome = outcomes.get(state)
    if isinstance(outcome, dict):
        return outcome
    legacy_state = LEGACY_RESPONSE_STATES.get(state)
    return outcomes.get(legacy_state) if legacy_state is not None else None


def render_agent_ab_sections(summaries: list[tuple[str, dict[str, Any]]]) -> str:
    """Render task success, criterion outcomes, and response/outcome cross-tabs."""
    lines = ["", "## Agent A/B statistical design", ""]
    lines.append(
        "Task-success planning uses matched binary outcomes, exact two-sided "
        "McNemar tests, and Holm correction across tasks. The schedule reports "
        "the 5/10/20-point and 80%/90% scenario matrix, paired score-interval "
        "precision, and projected invocation costs. Runtime and provider token "
        "totals are shown only when current-protocol measurements are interpretable."
    )
    lines.extend((
        "",
        "| Revision | Run evidence label | Design repetitions | Stage | Pilot evidence |",
        "|---|---|---:|---|---|",
    ))
    for revision, summary in summaries:
        design = summary.get("statistical_design")
        adequacy = summary.get("statistical_adequacy")
        if not isinstance(design, dict):
            lines.append(f"| {revision} | unknown (legacy summary) | n/a | n/a | n/a |")
            continue
        status = adequacy.get("status", "unknown") if isinstance(adequacy, dict) else "unknown"
        stage = adequacy.get("extension_stage") if isinstance(adequacy, dict) else None
        stage_label = (
            f"{stage.get('percent')}% ({stage.get('repetitions')} reps)"
            if isinstance(stage, dict) else "n/a"
        )
        evidence = design.get("pilot_evidence_source", "unknown")
        if design.get("pilot_required"):
            evidence = f"{evidence}; 20/task pilot required"
        lines.append(
            f"| {revision} | {_display_status(status)} | "
            f"{design.get('selected_repetitions', 'n/a')} | {stage_label} | {evidence} |"
        )

    lines.extend((
        "",
        "| Revision | Difference target | Target power | Validated minimum power (adjusted lower bound) | Repetitions/task/mode | Paired score CI half-width (pp) | Invocations/variant | Runtime/variant (s) | Input tokens/variant | Output tokens/variant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for revision, summary in summaries:
        design = summary.get("statistical_design")
        scenarios = design.get("scenario_matrix", []) if isinstance(design, dict) else []
        if not scenarios:
            lines.append(f"| {revision} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        for scenario in scenarios:
            powers = scenario.get("simulated_power_by_task", {})
            lower_bounds = scenario.get("power_lower_bound_by_task", {})
            design_version = design.get("schema_version") if isinstance(design, dict) else None
            power_is_validated = (
                isinstance(design_version, int)
                and not isinstance(design_version, bool)
                and design_version >= 4
            )
            simulated_power = (
                f"{min(powers.values()) * 100:.1f}% (LB {min(lower_bounds.values()) * 100:.1f}%)"
                if power_is_validated and powers and lower_bounds else "n/a"
            )
            lines.append(
                f"| {revision} | {scenario.get('effect_pp', 'n/a')} pp | "
                f"{_display_number(scenario.get('target_power') * 100 if isinstance(scenario.get('target_power'), (int, float)) else None)}% | "
                f"{simulated_power} | "
                f"{scenario.get('repetitions_per_task_and_mode', 'n/a')} | "
                f"{_display_number(scenario.get('expected_paired_score_ci95_half_width_pp'))} | "
                f"{scenario.get('invocations_per_variant', 'n/a')} | "
                f"{_display_number(scenario.get('estimated_runtime_seconds_per_variant'))} | "
                f"{_display_number(scenario.get('estimated_input_tokens_per_variant'), 0)} | "
                f"{_display_number(scenario.get('estimated_output_tokens_per_variant'), 0)} |"
            )
    missing_runtime = any(
        scenario.get("estimated_runtime_seconds_per_variant") is None
        for _, summary in summaries
        for scenario in (summary.get("statistical_design") or {}).get("scenario_matrix", [])
    )
    missing_tokens = any(
        scenario.get("estimated_input_tokens_per_variant") is None
        or scenario.get("estimated_output_tokens_per_variant") is None
        for _, summary in summaries
        for scenario in (summary.get("statistical_design") or {}).get("scenario_matrix", [])
    )
    notes = []
    if missing_runtime:
        notes.append("`n/a` runtime projections mean current-protocol timing data is unavailable.")
    if missing_tokens:
        notes.append("Provider token estimates remain unavailable until usage semantics are interpretable.")
    if notes:
        lines.extend(("", " ".join(notes)))

    lines.extend(("", "## Agent A/B task success", ""))
    lines.append(
        "Counts are objective task successes over completed runs. Paired deltas are "
        "within-task MCP minus control percentage points with a 95% confidence "
        "interval half-width. `n/a` means the summary predates that metric."
    )
    lines.extend((
        "",
        "| Revision | Task | Control success | MCP success | MCP − control (pp ± 95% CI half-width) | Evidence status | Adjusted paired interval | Holm p |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ))

    for revision, summary in summaries:
        by_mode = summary.get("task_success_by_mode", {})
        paired = summary.get("paired_task_success_deltas_mcp_minus_control", {}).get("tasks", {})
        adequacy = summary.get("statistical_adequacy", {})
        adequacy_tasks = adequacy.get("tasks", {}) if isinstance(adequacy, dict) else {}
        task_ids = sorted(_task_ids(summary))
        if not task_ids:
            lines.append(f"| {revision} | task-level data unavailable | n/a | n/a | n/a | unknown (legacy summary) | n/a | n/a |")
            continue
        for task_id in task_ids:
            control = by_mode.get("control", {}).get("tasks", {}).get(task_id)
            mcp = by_mode.get("mcp", {}).get("tasks", {}).get(task_id)
            delta = paired.get(task_id)
            evidence = adequacy_tasks.get(task_id, {})
            interval = evidence.get("paired_score_interval") if isinstance(evidence, dict) else None
            adjusted_p = evidence.get("holm_adjusted_p") if isinstance(evidence, dict) else None
            task_status = evidence.get("status", "unknown") if isinstance(evidence, dict) else "unknown"
            lines.append(
                f"| {revision} | {task_id} | {_fraction(control, 'successes')} "
                f"| {_fraction(mcp, 'successes')} | {_paired_delta(delta)} | "
                f"{_display_status(task_status)} | {_display_ci(interval)} | {_display_number(adjusted_p, 3)} |"
            )

    lines.extend(("", "## Agent A/B criterion results", ""))
    lines.append(
        "Answer columns show passing runs over scheduled runs. Search and slice hit "
        "columns show MCP runs where a criterion matched in at least one "
        "successful response over scheduled MCP runs. Matching case-folds text, "
        "normalizes whitespace, checks phrase boundaries, and accepts explicit "
        "line-number references. Response totals/runs "
        "show successful responses and runs with at least one response. Response text "
        "is not stored."
    )
    lines.extend((
        "",
        "| Revision | Task | Criterion | Control answer | MCP answer | MCP − control (pp ± 95% CI half-width) | Search hit runs | Search responses (total/runs) | Slice hit runs | Slice responses (total/runs) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))

    for revision, summary in summaries:
        criteria = summary.get("criterion_summaries_by_mode", {})
        deltas = summary.get("paired_criterion_deltas_mcp_minus_control", {}).get("tasks", {})
        exposure = (
            summary.get("criterion_exposure_summaries_by_mode", {})
            .get("mcp", {})
            .get("sources", {})
        )
        task_ids = sorted(
            set(criteria.get("control", {}).get("tasks", {}))
            | set(criteria.get("mcp", {}).get("tasks", {}))
        )
        for task_id in task_ids:
            control_items = criteria.get("control", {}).get("tasks", {}).get(task_id, [])
            mcp_items = criteria.get("mcp", {}).get("tasks", {}).get(task_id, [])
            count = max(len(control_items), len(mcp_items))
            search_items = exposure.get("search_capture", {}).get("tasks", {}).get(task_id, [])
            slice_items = exposure.get("get_capture_slice", {}).get("tasks", {}).get(task_id, [])
            search_counts = exposure.get("search_capture", {}).get("response_counts_by_task", {}).get(task_id, {})
            slice_counts = exposure.get("get_capture_slice", {}).get("response_counts_by_task", {}).get(task_id, {})
            task_deltas = deltas.get(task_id, [])
            for index in range(count):
                control = control_items[index] if index < len(control_items) else None
                mcp = mcp_items[index] if index < len(mcp_items) else None
                search = search_items[index] if index < len(search_items) else None
                sliced = slice_items[index] if index < len(slice_items) else None
                delta = task_deltas[index] if index < len(task_deltas) else None
                search_response_text = (
                    f"{search_counts.get('total', 0)}/{search_counts.get('runs_with_response', 0)}"
                    if search_counts else "n/a"
                )
                slice_response_text = (
                    f"{slice_counts.get('total', 0)}/{slice_counts.get('runs_with_response', 0)}"
                    if slice_counts else "n/a"
                )
                lines.append(
                    f"| {revision} | {task_id} | {index + 1} | {_fraction(control, 'passed')} "
                    f"| {_fraction(mcp, 'passed')} | {_paired_delta(delta)} "
                    f"| {_fraction(search, 'exposed')} | {search_response_text} "
                    f"| {_fraction(sliced, 'exposed')} | {slice_response_text} |"
                )

    lines.extend(("", "## MCP response exposure and answer outcomes", ""))
    lines.append(
        "Each cell is `answer pass/answer fail` for runs in that response state. "
        "`criterion_hit` means a criterion matched in a successful response. "
        "`response_without_criterion_hit` means one or more successful responses were "
        "returned without a criterion match. `no_successful_response` means the "
        "run returned zero successful responses from that tool; it does not establish "
        "whether the tool was called. Criterion absence does not establish semantic "
        "irrelevance, and these cross-tabs show association rather than causation."
    )
    lines.extend((
        "",
        "| Revision | Task | Criterion | Source | Criterion hit (pass/fail) | Response without criterion hit (pass/fail) | No successful response (pass/fail) |",
        "|---|---|---:|---|---:|---:|---:|",
    ))
    for revision, summary in summaries:
        criteria_by_mode = summary.get("criterion_summaries_by_mode", {})
        sources = (
            summary.get("criterion_exposure_summaries_by_mode", {})
            .get("mcp", {})
            .get("sources", {})
        )
        task_ids = sorted(
            set(criteria_by_mode.get("mcp", {}).get("tasks", {}))
            | set().union(*(set(sources.get(source, {}).get("tasks", {})) for source in SOURCES))
        )
        if not task_ids:
            lines.append(f"| {revision} | task-level data unavailable | n/a | unavailable | n/a | n/a | n/a |")
            continue
        for task_id in task_ids:
            mcp_criteria = criteria_by_mode.get("mcp", {}).get("tasks", {}).get(task_id, [])
            for source in SOURCES:
                source_criteria = sources.get(source, {}).get("tasks", {}).get(task_id, [])
                count = max(len(mcp_criteria), len(source_criteria))
                for index in range(count):
                    criterion = source_criteria[index] if index < len(source_criteria) else None
                    cross_tab = (
                        criterion.get("answer_outcomes_by_response_state", {})
                        if isinstance(criterion, dict) else {}
                    )
                    values = []
                    for state in RESPONSE_STATES:
                        outcome = _response_state_outcome(cross_tab, state)
                        values.append(
                            f"{outcome.get('passed', 0)}/{outcome.get('failed', 0)}"
                            if isinstance(outcome, dict) else "n/a"
                        )
                    lines.append(
                        f"| {revision} | {task_id} | {index + 1} | {source} "
                        f"| {values[0]} | {values[1]} | {values[2]} |"
                    )
    return "\n".join(lines) + "\n"


def append_agent_ab_report(baseline_path: Path, candidate_path: Path, report_path: Path) -> None:
    """Append report sections for whichever revision summaries are available."""
    summaries = [
        result
        for revision, path in (("baseline", baseline_path), ("candidate", candidate_path))
        if (result := _read_summary(revision, path)) is not None
    ]
    if summaries:
        with report_path.open("a", encoding="utf-8") as report:
            report.write(render_agent_ab_sections(summaries))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    append_agent_ab_report(args.baseline_summary, args.candidate_summary, args.report)


if __name__ == "__main__":
    main()
