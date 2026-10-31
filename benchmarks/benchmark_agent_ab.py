#!/usr/bin/env python3
"""Plan and summarize privacy-safe agent-level MCP A/B evaluations."""

import argparse
import json
import math
import random
import statistics
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import benchmarks.workload_results as wr
from benchmarks.agent_ab_statistics import (
    bootstrap_discordance_upper,
    exact_mcnemar_pvalue,
    holm_adjust,
    newcombe_paired_score_interval,
    normal_approximate_repetitions,
    select_simulated_repetitions,
    simulate_paired_mcnemar_power,
)


SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 5
RECORDS_SCHEMA_VERSION = 10
TASK_FIXTURE_VERSION = 4
TASK_RUBRIC_VERSION = 4
MODES = ("control", "mcp")
DEFAULT_AGENT_AB_SEED = 20260909
DEFAULT_DESIGN_SEED = 20261031
DEFAULT_POWER_SIMULATIONS = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_EFFECT = 0.10
DEFAULT_TARGET_POWER = 0.80
DEFAULT_PRACTICAL_EFFECT = 0.10
STAGE_PERCENTAGES = (50, 75, 100)
STAGE_ALPHA_SPENDING = (0.005, 0.015, 0.030)
FINAL_LOOK_ALPHA = STAGE_ALPHA_SPENDING[-1]
SCENARIO_EFFECTS = (0.05, 0.10, 0.20)
SCENARIO_POWERS = (0.80, 0.90)
EXPOSURE_STATES = (
    "criterion_hit",
    "response_without_criterion_hit",
    "no_successful_response",
)
TASKS = (
    {"id": "targeted-inspection", "category": "small-targeted-output"},
    {"id": "noisy-test-failure", "category": "noisy-test-output"},
    {"id": "build-log-search", "category": "noisy-build-output"},
    {"id": "follow-up-context", "category": "search-and-retrieval"},
)
RUN_KEYS = {
    "task_id",
    "repetition",
    "mode",
    "completed",
    "signal_retrieved",
    "duration_seconds",
    "tool_calls",
    "repeated_commands",
    "context_bytes",
    "peak_rss_bytes",
}
RUN_KEYS_V2 = RUN_KEYS - {"context_bytes"} | {
    "context_bytes_proxy",
    "exit_code",
    "failure_reason",
    "mcp_tool_calls",
    "input_tokens",
    "output_tokens",
}
RUN_KEYS_V3 = RUN_KEYS_V2 | {"input_token_samples", "output_token_samples"}
RUN_KEYS_V4 = RUN_KEYS_V3 | {"prompt_bytes_proxy", "output_bytes_proxy"}
DATA_PATH_BYTE_FIELDS = (
    "capture_input_bytes",
    "capture_retained_bytes",
    "capture_original_bytes",
    "tool_response_bytes",
    "search_response_bytes",
    "retrieval_response_bytes",
    "socket_request_bytes",
    "socket_response_bytes",
)
RUN_KEYS_V5 = RUN_KEYS_V4 | set(DATA_PATH_BYTE_FIELDS)
RUN_KEYS_V6 = RUN_KEYS_V5 | {"task_success"}
RUN_KEYS_V7 = RUN_KEYS_V6 | {"criterion_passes"}
RUN_KEYS_V8 = RUN_KEYS_V7 | {
    "criterion_search_response_hits",
    "criterion_slice_response_hits",
    "search_capture_responses",
    "get_capture_slice_responses",
}
RUN_KEYS_V9 = RUN_KEYS_V8
METRICS = (
    "completed",
    "task_success",
    "signal_retrieved",
    "duration_seconds",
    "tool_calls",
    "mcp_tool_calls",
    "repeated_commands",
    "context_bytes_proxy",
    "peak_rss_bytes",
)
PROTOCOL_FIELDS = (
    "model_config",
    "repository_fixture",
    "environment",
    "reset_policy",
    "agent_adapter",
    "embedding_mode",
    "embedding_model",
    "embedding_cache",
)
LEGACY_PROTOCOL_FIELDS = PROTOCOL_FIELDS[:5]
CONTEXT_PROXY_NOTE = "prompt plus event envelope bytes; the agent CLI does not expose context size"
# Record and summary field -> (workload measurement name, unit).
RECORD_MEASUREMENTS = {
    "duration_seconds": ("wall_time_seconds", "seconds"),
    "tool_calls": ("tool_calls", "count"),
    "mcp_tool_calls": ("mcp_tool_calls", "count"),
    "repeated_commands": ("repeated_commands", "count"),
    "context_bytes_proxy": ("context_bytes", "bytes"),
    "prompt_bytes_proxy": ("prompt_bytes", "bytes"),
    "output_bytes_proxy": ("output_bytes", "bytes"),
    "peak_rss_bytes": ("peak_rss_bytes", "bytes"),
    "input_tokens": ("input_tokens", "tokens"),
    "output_tokens": ("output_tokens", "tokens"),
    **{field: (field, "bytes") for field in DATA_PATH_BYTE_FIELDS},
}
PAIRED_MEASUREMENTS = {
    "completed": ("invocation_completion_rate", "ratio"),
    "task_success": ("success_rate", "ratio"),
    "signal_retrieved": ("signal_retrieval_rate", "ratio"),
    **RECORD_MEASUREMENTS,
}


def stage_repetitions(repetitions: int) -> list[dict[str, int | float]]:
    """Return the preplanned 50%, 75%, and 100% cumulative looks."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    values = []
    prior = 0
    cumulative_alpha = 0.0
    for percentage, alpha in zip(STAGE_PERCENTAGES, STAGE_ALPHA_SPENDING):
        count = max(prior, math.ceil(repetitions * percentage / 100))
        cumulative_alpha += alpha
        if count == prior and values:
            values[-1]["percent"] = percentage
            values[-1]["alpha_spending"] = float(values[-1]["alpha_spending"]) + alpha
            values[-1]["cumulative_alpha"] = cumulative_alpha
            continue
        values.append({
            "percent": percentage,
            "repetitions": count,
            "alpha_spending": alpha,
            "cumulative_alpha": cumulative_alpha,
        })
        prior = count
    return values


def _bootstrap_model_inputs(
    payload: dict[str, Any] | None,
    schedule: dict[str, Any] | None,
    *,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], str, bool]:
    """Build task-specific paired-model inputs from current-rubric records."""
    defaults = {
        task["id"]: {
            "baseline_success_rate": 0.5,
            "discordance_rate": None,
            "matched_repetitions": 0,
            "source": "maximum-discordance-stress-model",
            "planning_assumption": "baseline success 0.5; use the maximum feasible discordance for each target effect",
        }
        for task in TASKS
    }
    if payload is None and schedule is None:
        return defaults, "conservative-maximum-discordance", True
    if payload is None or schedule is None:
        raise ValueError("power pilot payload and schedule must be supplied together")
    if (
        payload.get("task_fixture_version") != TASK_FIXTURE_VERSION
        or payload.get("task_rubric_version") != TASK_RUBRIC_VERSION
        or schedule.get("task_fixture_version") != TASK_FIXTURE_VERSION
        or schedule.get("task_rubric_version", TASK_RUBRIC_VERSION) != TASK_RUBRIC_VERSION
        or payload.get("records_schema_version", 1) < 6
    ):
        raise ValueError("power pilot records must contain current fixture-v4/rubric-v4 paired task-success outcomes")
    _resolve_pilot_identity(
        payload,
        schedule,
        model_config=None,
        repository_fixture=None,
    )
    records = validate_records(payload, schedule)
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["repetition"], record["task_id"]), {})[record["mode"]] = record
    results = {}
    insufficient = False
    for task_index, task in enumerate(TASKS):
        task_pairs = [
            (pair["control"]["task_success"], pair["mcp"]["task_success"])
            for (_, task_id), pair in grouped.items()
            if task_id == task["id"] and set(pair) == set(MODES)
        ]
        if not task_pairs:
            insufficient = True
            results[task["id"]] = {
                **defaults[task["id"]],
                "source": "missing-pilot-conservative-maximum-discordance",
            }
            continue
        if len(task_pairs) < 20:
            insufficient = True
            results[task["id"]] = {
                **defaults[task["id"]],
                "matched_repetitions": len(task_pairs),
                "source": "under-20-pair-conservative-maximum-discordance",
                "planning_assumption": (
                    "fewer than 20 matched pilot pairs; use baseline success 0.5 "
                    "and maximum feasible discordance for each target effect"
                ),
            }
            continue
        bootstrap = bootstrap_discordance_upper(
            task_pairs,
            simulations=5000,
            seed=seed + task_index,
            quantile=0.95,
        )
        results[task["id"]] = {
            **bootstrap,
            "discordance_rate": bootstrap["discordance_upper_bootstrap"],
            "source": "paired-records-bootstrap-95th-percentile",
        }
    evidence_source = (
        "paired-records-bootstrap-with-conservative-fallback"
        if insufficient
        else "paired-records-bootstrap"
    )
    return results, evidence_source, insufficient


def _scenario_discordance(model: dict[str, Any], effect: float) -> float:
    """Use the largest feasible discordance supported by pilot/bootstrap data."""
    p0 = float(model.get("baseline_success_rate", 0.5))
    observed_upper = model.get("discordance_rate")
    if observed_upper is None:
        return 1.0 - abs(effect)
    upper = float(observed_upper)
    feasible_caps = []
    for signed_effect in (abs(effect), -abs(effect)):
        cap = min(
            2.0 * (1.0 - p0) - signed_effect,
            2.0 * p0 + signed_effect,
        )
        if cap >= abs(effect):
            feasible_caps.append(cap)
    if not feasible_caps:
        return 1.0 - abs(effect)
    return max(abs(effect), min(upper, max(feasible_caps), 1.0))


def _pilot_cost_profile(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Estimate scale from current records without guessing token semantics."""
    if payload is None:
        return {
            "mean_duration_seconds_per_invocation": None,
            "mean_input_tokens_per_invocation": None,
            "mean_output_tokens_per_invocation": None,
            "runtime_note": "unavailable without current-protocol timing records",
            "token_estimates_note": "unavailable until provider usage semantics are calibrated",
        }
    runs = payload.get("runs", [])
    durations = [
        float(record["duration_seconds"])
        for record in runs
        if isinstance(record, dict)
        and _is_nonnegative_finite_number(record.get("duration_seconds"))
    ]
    usage = payload.get("usage_interpretation", {})
    means: dict[str, float | None] = {}
    for field in ("input_tokens", "output_tokens"):
        declared = isinstance(usage, dict) and usage.get(field) == "per-invocation-total"
        values = [
            float(record[field])
            for record in runs
            if isinstance(record, dict) and _is_nonnegative_finite_number(record.get(field))
        ]
        means[field] = statistics.mean(values) if declared and values and len(values) == len(runs) else None
    return {
        "mean_duration_seconds_per_invocation": statistics.mean(durations) if durations else None,
        "mean_input_tokens_per_invocation": means["input_tokens"],
        "mean_output_tokens_per_invocation": means["output_tokens"],
        "runtime_note": "mean duration from supplied current-protocol records" if durations else "unavailable without current-protocol timing records",
        "token_estimates_note": (
            "provider usage declared per-invocation-total"
            if all(value is not None for value in means.values())
            else "unavailable until provider usage semantics are calibrated"
        ),
    }


def build_power_design(
    pilot_payload: dict[str, Any] | None = None,
    pilot_schedule: dict[str, Any] | None = None,
    *,
    seed: int = DEFAULT_DESIGN_SEED,
    simulations: int = DEFAULT_POWER_SIMULATIONS,
) -> dict[str, Any]:
    """Create a reproducible paired-McNemar design and scenario cost table."""
    model_inputs, evidence_source, pilot_required = _bootstrap_model_inputs(
        pilot_payload, pilot_schedule, seed=seed
    )
    cost_profile = _pilot_cost_profile(pilot_payload)
    task_ids = [task["id"] for task in TASKS]
    default_discordance = [
        _scenario_discordance(model_inputs[task_id], DEFAULT_EFFECT)
        for task_id in task_ids
    ]
    selected = select_simulated_repetitions(
        effect=DEFAULT_EFFECT,
        target_power=DEFAULT_TARGET_POWER,
        discordance_rates=default_discordance,
        simulations=simulations,
        family_alpha=FINAL_LOOK_ALPHA,
        seed=seed,
    )
    scenarios = []
    for effect_index, effect in enumerate(SCENARIO_EFFECTS):
        discordance_rates = [
            _scenario_discordance(model_inputs[task_id], effect)
            for task_id in task_ids
        ]
        for power_index, target_power in enumerate(SCENARIO_POWERS):
            scenario_seed = seed + 100 + effect_index * len(SCENARIO_POWERS) + power_index
            if effect == DEFAULT_EFFECT and target_power == DEFAULT_TARGET_POWER:
                repetitions = selected["repetitions"]
                simulated = selected["power_simulation"]
            else:
                repetitions = normal_approximate_repetitions(
                    effect=effect,
                    target_power=target_power,
                    discordance_rates=discordance_rates,
                    family_alpha=FINAL_LOOK_ALPHA,
                )
                # The exact paired test and Holm correction verify each scenario.
                # Increase in 2% steps until the lower Monte Carlo bound clears target.
                while True:
                    simulated = simulate_paired_mcnemar_power(
                        repetitions=repetitions,
                        effect=effect,
                        discordance_rates=discordance_rates,
                        simulations=simulations,
                        alpha=FINAL_LOOK_ALPHA,
                        seed=scenario_seed + repetitions,
                    )
                    if min(simulated["power_lower_95_by_task"]) >= target_power:
                        break
                    repetitions += max(1, math.ceil(repetitions * 0.02))
            variance = max(
                discordance - effect * effect for discordance in discordance_rates
            )
            expected_ci = 1.96 * math.sqrt(variance / repetitions)
            adjusted_z = NormalDist().inv_cdf(1.0 - FINAL_LOOK_ALPHA / (2.0 * len(task_ids)))
            adjusted_ci = adjusted_z * math.sqrt(variance / repetitions)
            invocations = repetitions * len(task_ids) * len(MODES)
            scenarios.append({
                "effect_pp": round(effect * 100),
                "target_power": target_power,
                "family_alpha": DEFAULT_ALPHA,
                "final_look_alpha": FINAL_LOOK_ALPHA,
                "task_count": len(task_ids),
                "discordance_rates": dict(zip(task_ids, discordance_rates)),
                "repetitions_per_task_and_mode": repetitions,
                "simulated_power_by_task": dict(zip(task_ids, simulated["power_by_task"])),
                "power_lower_95_by_task": dict(zip(task_ids, simulated["power_lower_95_by_task"])),
                "expected_paired_score_ci95_half_width_pp": expected_ci * 100,
                "holm_first_step_adjusted_ci_half_width_pp": adjusted_ci * 100,
                "invocations_per_variant": invocations,
                "estimated_runtime_seconds_per_variant": (
                    cost_profile["mean_duration_seconds_per_invocation"] * invocations
                    if cost_profile["mean_duration_seconds_per_invocation"] is not None else None
                ),
                "estimated_input_tokens_per_variant": (
                    cost_profile["mean_input_tokens_per_invocation"] * invocations
                    if cost_profile["mean_input_tokens_per_invocation"] is not None else None
                ),
                "estimated_output_tokens_per_variant": (
                    cost_profile["mean_output_tokens_per_invocation"] * invocations
                    if cost_profile["mean_output_tokens_per_invocation"] is not None else None
                ),
                "token_estimates_note": cost_profile["token_estimates_note"],
            })
    return {
        "schema_version": 1,
        "method": "seeded paired binary simulation; exact two-sided McNemar tests; Holm familywise correction",
        "effect_target": DEFAULT_EFFECT,
        "power_target": DEFAULT_TARGET_POWER,
        "family_alpha": DEFAULT_ALPHA,
        "task_count": len(task_ids),
        "pilot_evidence_source": evidence_source,
        "pilot_required": pilot_required,
        "pilot_minimum_repetitions_per_task": 20,
        "bootstrap_method": "resample matched current-rubric pairs 5000 times; use the 95th percentile discordance rate",
        "paired_model_by_task": model_inputs,
        "simulation_seed": seed,
        "power_simulations": simulations,
        "power_confidence_rule": "per-task 95% Wilson lower bound must meet target at the final-look Holm alpha allocation",
        "final_look_alpha": FINAL_LOOK_ALPHA,
        "selected_repetitions": selected["repetitions"],
        "selected_power_by_task": dict(zip(task_ids, selected["power_simulation"]["power_by_task"])),
        "selected_power_lower_95_by_task": dict(zip(task_ids, selected["power_simulation"]["power_lower_95_by_task"])),
        "selected_power_rule": selected["selection_rule"],
        "stage_percentages": list(STAGE_PERCENTAGES),
        "stage_alpha_spending": list(STAGE_ALPHA_SPENDING),
        "scenario_matrix": scenarios,
        "cost_basis": "one invocation for each task, mode, and repetition; costs are per variant",
        "runtime_note": cost_profile["runtime_note"],
        "token_estimates_note": cost_profile["token_estimates_note"],
    }


@lru_cache(maxsize=1)
def _default_power_design() -> dict[str, Any]:
    """Cache the default design so smoke schedules do not resimulate it."""
    return build_power_design()


def build_schedule(
    repetitions: int | None = None,
    seed: int = DEFAULT_AGENT_AB_SEED,
    *,
    design: dict[str, Any] | None = None,
    model_config: str | None = None,
    repository_fixture: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic counterbalanced task/mode schedule."""
    is_explicit = repetitions is not None
    power_design = design if design is not None else _default_power_design()
    selected_repetitions = repetitions if repetitions is not None else power_design["selected_repetitions"]
    if selected_repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    schedule = []
    sequence = 1
    for repetition in range(1, selected_repetitions + 1):
        task_order = [task["id"] for task in TASKS]
        rng.shuffle(task_order)
        for task_id in task_order:
            mode_order = list(MODES)
            rng.shuffle(mode_order)
            for mode in mode_order:
                schedule.append({
                    "sequence": sequence,
                    "repetition": repetition,
                    "task_id": task_id,
                    "mode": mode,
                })
                sequence += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "agent-ab",
        "task_fixture_version": TASK_FIXTURE_VERSION,
        "task_rubric_version": TASK_RUBRIC_VERSION,
        "seed": seed,
        "repetitions": selected_repetitions,
        "selected_design_repetitions": power_design["selected_repetitions"],
        "repetitions_explicit": is_explicit,
        "requested_repetitions": repetitions,
        "run_label": (
            "exploratory"
            if is_explicit and selected_repetitions < power_design["selected_repetitions"]
            else "evidence-ready-schedule"
        ),
        "stage_schedule": (
            stage_repetitions(selected_repetitions)
            if not is_explicit
            else [{
                "percent": 100,
                "repetitions": selected_repetitions,
                "alpha_spending": DEFAULT_ALPHA,
                "cumulative_alpha": DEFAULT_ALPHA,
            }]
        ),
        "model_config": model_config,
        "repository_fixture": repository_fixture,
        "statistical_design": power_design,
        "modes": list(MODES),
        "tasks": list(TASKS),
        "schedule": schedule,
        "privacy": "schedule contains task IDs and categories only; no prompts, captures, commands, or user content",
    }


def _resolve_pilot_identity(
    pilot_payload: dict[str, Any],
    pilot_schedule: dict[str, Any] | None,
    *,
    model_config: str | None,
    repository_fixture: str | None,
) -> tuple[str, str]:
    """Resolve intended run identities from the pilot and reject mismatches."""
    protocol = pilot_payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("pilot records protocol metadata is required")
    if pilot_schedule is not None and not isinstance(pilot_schedule, dict):
        raise ValueError("pilot schedule metadata must be an object")

    resolved = {}
    for field, requested in (
        ("model_config", model_config),
        ("repository_fixture", repository_fixture),
    ):
        value = protocol.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pilot protocol field {field} must be a non-empty identifier")
        if requested is not None and requested != value:
            raise ValueError(f"pilot records {field} does not match --{field.replace('_', '-')}")
        scheduled = pilot_schedule.get(field) if pilot_schedule is not None else None
        if scheduled is not None and scheduled != value:
            raise ValueError(f"pilot schedule {field} does not match pilot records protocol")
        resolved[field] = value
    return resolved["model_config"], resolved["repository_fixture"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root in {path} must be an object")
    return value


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("records protocol metadata is required")
    for field in LEGACY_PROTOCOL_FIELDS:
        value = protocol.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"records protocol field {field} must be a non-empty identifier")
    # Embedding metadata was added after records schema versions 1-4. Preserve
    # validation of those historical records while validating new metadata
    # whenever it is present.
    for field in PROTOCOL_FIELDS[len(LEGACY_PROTOCOL_FIELDS):]:
        if field in protocol:
            value = protocol[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"records protocol field {field} must be a non-empty identifier")


def _run_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (int(record["repetition"]), record["task_id"], record["mode"])


def _is_nonnegative_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def validate_records(
    payload: dict[str, Any],
    schedule: dict[str, Any],
    *,
    allow_staged_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Validate balanced metadata-only agent records against a schedule."""
    if payload.get("schema_version") not in (1, SCHEMA_VERSION) or payload.get("benchmark") != "agent-ab":
        raise ValueError("records schema or benchmark name is invalid")
    if payload.get("task_fixture_version") != schedule.get("task_fixture_version"):
        raise ValueError("records task_fixture_version does not match schedule")
    payload_rubric = payload.get("task_rubric_version")
    schedule_rubric = schedule.get("task_rubric_version")
    if payload_rubric is not None and schedule_rubric is not None and payload_rubric != schedule_rubric:
        raise ValueError("records task_rubric_version does not match schedule")
    _validate_protocol(payload.get("protocol"))
    records_schema_version = payload.get("records_schema_version", 1)
    if records_schema_version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, RECORDS_SCHEMA_VERSION):
        raise ValueError("records schema version is unsupported")
    if records_schema_version >= 10 and (
        payload_rubric != TASK_RUBRIC_VERSION
        or schedule_rubric != TASK_RUBRIC_VERSION
    ):
        raise ValueError("current records require matching fixture-v4/rubric-v4 metadata")
    run_keys = (
        RUN_KEYS
        if records_schema_version == 1
        else RUN_KEYS_V2
        if records_schema_version == 2
        else RUN_KEYS_V3
        if records_schema_version == 3
        else RUN_KEYS_V4
        if records_schema_version == 4
        else RUN_KEYS_V5
        if records_schema_version == 5
        else RUN_KEYS_V6
        if records_schema_version == 6
        else RUN_KEYS_V7
        if records_schema_version == 7
        else RUN_KEYS_V8
        if records_schema_version == 8
        else RUN_KEYS_V9
    )
    expected = {_run_key(item): item for item in schedule.get("schedule", [])}
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("records runs must be a list")
    actual = {}
    criterion_counts_by_task: dict[str, int] = {}
    for record in runs:
        if not isinstance(record, dict) or set(record) != run_keys:
            raise ValueError("each run must contain exactly the documented metadata fields")
        key = _run_key(record)
        if key not in expected:
            raise ValueError(f"run is not present in schedule: {key}")
        if key in actual:
            raise ValueError(f"duplicate run: {key}")
        if not isinstance(record["task_id"], str) or record["mode"] not in MODES:
            raise ValueError(f"invalid task or mode in run: {key}")
        if not isinstance(record["repetition"], int) or record["repetition"] < 1:
            raise ValueError(f"invalid repetition in run: {key}")
        for field in ("completed", "signal_retrieved"):
            if not isinstance(record[field], bool):
                raise ValueError(f"{field} must be boolean in run: {key}")
        if records_schema_version >= 6 and not isinstance(record["task_success"], bool):
            raise ValueError(f"task_success must be boolean in run: {key}")
        if records_schema_version >= 7:
            criterion_passes = record["criterion_passes"]
            if (
                not isinstance(criterion_passes, list)
                or len(criterion_passes) < 2
                or any(not isinstance(passed, bool) for passed in criterion_passes)
            ):
                raise ValueError(f"criterion_passes must contain at least two booleans in run: {key}")
            prior_count = criterion_counts_by_task.setdefault(record["task_id"], len(criterion_passes))
            if len(criterion_passes) != prior_count:
                raise ValueError(f"criterion_passes length must be consistent for task: {record['task_id']}")
            if record["task_success"] and not all(criterion_passes):
                raise ValueError(f"task_success requires every criterion to pass in run: {key}")
        if records_schema_version >= 8:
            exposure_fields = (
                "criterion_search_response_hits",
                "criterion_slice_response_hits",
            )
            if record["mode"] == "control":
                if any(record[field] is not None for field in exposure_fields):
                    raise ValueError(f"MCP response exposure must be null for control runs: {key}")
                if record["search_capture_responses"] != 0 or record["get_capture_slice_responses"] != 0:
                    raise ValueError(f"MCP response counts must be zero for control runs: {key}")
            else:
                for field in exposure_fields:
                    values = record[field]
                    if (
                        not isinstance(values, list)
                        or len(values) != len(record["criterion_passes"])
                        or any(not isinstance(value, bool) for value in values)
                    ):
                        raise ValueError(f"{field} must align with criterion_passes in run: {key}")
            for field in ("search_capture_responses", "get_capture_slice_responses"):
                value = record[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{field} must be a non-negative integer in run: {key}")
        numeric_fields = ("duration_seconds", "tool_calls", "repeated_commands", "peak_rss_bytes")
        if records_schema_version == 1:
            numeric_fields += ("context_bytes",)
        elif records_schema_version < 4:
            numeric_fields += ("context_bytes_proxy", "mcp_tool_calls")
        else:
            numeric_fields += (
                "context_bytes_proxy",
                "prompt_bytes_proxy",
                "output_bytes_proxy",
                "mcp_tool_calls",
            )
        for field in numeric_fields:
            value = record[field]
            if not _is_nonnegative_finite_number(value):
                raise ValueError(f"{field} must be a non-negative number in run: {key}")
        if records_schema_version >= 2:
            if record["exit_code"] is not None and (isinstance(record["exit_code"], bool) or not isinstance(record["exit_code"], int)):
                raise ValueError(f"exit_code must be an integer or null in run: {key}")
            if record["failure_reason"] is not None and not isinstance(record["failure_reason"], str):
                raise ValueError(f"failure_reason must be a string or null in run: {key}")
            for field in ("input_tokens", "output_tokens"):
                value = record[field]
                if value is not None and not _is_nonnegative_finite_number(value):
                    raise ValueError(f"{field} must be a non-negative number or null in run: {key}")
        if records_schema_version >= 3:
            for field in ("input_token_samples", "output_token_samples"):
                samples = record[field]
                if not isinstance(samples, list):
                    raise ValueError(f"{field} must be a list in run: {key}")
                if any(not _is_nonnegative_finite_number(value) for value in samples):
                    raise ValueError(f"{field} must contain non-negative numbers in run: {key}")
        if records_schema_version >= 4:
            if record["prompt_bytes_proxy"] + record["output_bytes_proxy"] != record["context_bytes_proxy"]:
                raise ValueError(f"context byte proxy components must sum to the total in run: {key}")
        if records_schema_version >= 5:
            for field in DATA_PATH_BYTE_FIELDS:
                value = record[field]
                if not _is_nonnegative_finite_number(value):
                    raise ValueError(f"{field} must be a non-negative number in run: {key}")
        actual[key] = record
    missing = sorted(set(expected) - set(actual))
    if missing:
        if not allow_staged_prefix or not actual:
            raise ValueError(f"records are missing scheduled runs: {missing[0]}")
        completed_repetitions = sorted({record["repetition"] for record in actual.values()})
        last_repetition = completed_repetitions[-1]
        if completed_repetitions != list(range(1, last_repetition + 1)):
            raise ValueError("staged records must contain a contiguous repetition prefix")
        expected_prefix = {
            key for key, item in expected.items() if item["repetition"] <= last_repetition
        }
        if set(actual) != expected_prefix:
            raise ValueError("staged records must contain every matched run through the latest repetition")
    return list(actual.values())


def _metric_value(record: dict[str, Any], metric: str) -> float:
    """Read a normalized metric from either records schema version."""
    if metric == "context_bytes_proxy":
        return float(record.get("context_bytes_proxy", record.get("context_bytes", 0)))
    return float(record.get(metric, 0))


def _invocation_completed(record: dict[str, Any], records_schema_version: int) -> bool | None:
    """Return invocation completion when the records schema can distinguish it."""
    if records_schema_version >= 6:
        return record["completed"]
    if records_schema_version >= 2:
        exit_code = record.get("exit_code")
        return exit_code == 0 if exit_code is not None else None
    return None


def _optional_stats(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record[field] for record in records if record.get(field) is not None]
    if not values:
        return {"available": False, "count": 0}
    result = _stats([float(value) for value in values])
    result["available"] = True
    return result


def _optional_positive_stats(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Summarize measurements where zero is the documented unavailable sentinel."""
    values = [record[field] for record in records if record.get(field, 0) > 0]
    if not values:
        return {"available": False, "count": 0}
    result = _stats([float(value) for value in values])
    result["available"] = True
    return result


def _stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.mean(values) if values else 0.0
    deviation = statistics.stdev(values) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "stdev": deviation,
        "ci95_half_width": 1.96 * deviation / (count ** 0.5) if count > 1 else 0.0,
    }


def _usage_accounting(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Describe observed usage sequences without asserting their semantics."""
    sequences = [record.get(field, []) for record in records if record.get(field)]
    sample_counts = [len(sequence) for sequence in sequences]
    multi_sample_sequences = [sequence for sequence in sequences if len(sequence) > 1]
    monotonic_count = sum(
        all(current >= previous for previous, current in zip(sequence, sequence[1:]))
        for sequence in multi_sample_sequences
    )
    non_monotonic_count = len(multi_sample_sequences) - monotonic_count
    if not sequences:
        observation = "unavailable"
    elif not multi_sample_sequences:
        observation = "insufficient_samples"
    elif non_monotonic_count:
        observation = "non_monotonic_samples"
    else:
        observation = "monotonic_samples_inconclusive"
    empty_stats = {"count": 0, "mean": 0.0, "stdev": 0.0, "ci95_half_width": 0.0}
    return {
        "runs_with_samples": len(sequences),
        "runs_with_multiple_samples": len(multi_sample_sequences),
        "sample_count": _stats([float(value) for value in sample_counts]),
        "monotonic_runs": monotonic_count,
        "non_monotonic_runs": non_monotonic_count,
        "first_to_last_delta": (
            _stats([float(sequence[-1] - sequence[0]) for sequence in multi_sample_sequences])
            if multi_sample_sequences else empty_stats
        ),
        "observation": observation,
        "interpretation": (
            "Monotonic samples may be cumulative or per-turn; use a controlled calibration matrix."
            if observation == "monotonic_samples_inconclusive"
            else "Inspect sample boundaries and event types before interpreting token totals."
        ),
    }


def validate_schedule_extension(
    previous_schedule: dict[str, Any],
    extended_schedule: dict[str, Any],
) -> None:
    """Require an extension to preserve the seed, rubric, and full old prefix."""
    stable_fields = (
        "benchmark",
        "seed",
        "task_fixture_version",
        "task_rubric_version",
        "tasks",
        "modes",
        "model_config",
        "repository_fixture",
        "selected_design_repetitions",
        "repetitions_explicit",
        "statistical_design",
    )
    for field in stable_fields:
        if previous_schedule.get(field) != extended_schedule.get(field):
            raise ValueError(f"schedule extension changed {field}")
    if (
        not previous_schedule.get("repetitions_explicit", True)
        and previous_schedule.get("stage_schedule") != extended_schedule.get("stage_schedule")
    ):
        raise ValueError("schedule extension changed stage_schedule")
    previous_items = previous_schedule.get("schedule")
    extended_items = extended_schedule.get("schedule")
    if not isinstance(previous_items, list) or not isinstance(extended_items, list):
        raise ValueError("schedule extension requires schedule item lists")
    if len(extended_items) < len(previous_items) or extended_items[:len(previous_items)] != previous_items:
        raise ValueError("schedule extension must preserve the previous schedule as an exact prefix")
    if extended_schedule.get("repetitions", 0) < previous_schedule.get("repetitions", 0):
        raise ValueError("schedule extension cannot reduce planned repetitions")


def _statistical_adequacy(
    runs: list[dict[str, Any]],
    schedule: dict[str, Any],
    records_schema_version: int,
    records_rubric_version: int | None,
) -> dict[str, Any]:
    """Summarize paired binary evidence and staged stopping decisions."""
    design = schedule.get("statistical_design")
    if records_schema_version < 6 or not isinstance(design, dict):
        return {
            "status": "unknown",
            "adequacy_known": False,
            "continue": False,
            "reason": (
                "legacy records lack paired objective outcomes"
                if records_schema_version < 6
                else "summary predates the statistical design metadata"
            ),
            "tasks": {},
        }
    if (
        schedule.get("task_fixture_version") != TASK_FIXTURE_VERSION
        or schedule.get("task_rubric_version") != TASK_RUBRIC_VERSION
        or records_rubric_version != TASK_RUBRIC_VERSION
    ):
        return {
            "status": "unknown",
            "adequacy_known": False,
            "continue": False,
            "reason": "statistical evidence cannot be pooled across fixture or rubric versions",
            "tasks": {},
        }

    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault((record["repetition"], record["task_id"]), {})[record["mode"]] = record
    pairs_by_task: dict[str, list[dict[str, Any]]] = {task["id"]: [] for task in TASKS}
    for (_, task_id), pair in grouped.items():
        if task_id in pairs_by_task and set(pair) == set(MODES):
            pairs_by_task[task_id].append(pair)
    counts = {task_id: len(pairs) for task_id, pairs in pairs_by_task.items()}
    completed = min(counts.values(), default=0)
    planned = int(schedule.get("repetitions", 0))
    target = int(schedule.get("selected_design_repetitions", design.get("selected_repetitions", planned)))
    explicit = bool(schedule.get("repetitions_explicit", False))
    exploratory_override = explicit and planned < target
    looks = schedule.get("stage_schedule", [])
    look = next(
        (item for item in reversed(looks) if completed >= item.get("repetitions", 0)),
        None,
    )
    if look is None and looks:
        look = looks[0]
    alpha_increment = float(look.get("alpha_spending", DEFAULT_ALPHA)) if look else DEFAULT_ALPHA
    cumulative_alpha = float(look.get("cumulative_alpha", alpha_increment)) if look else DEFAULT_ALPHA
    alpha = alpha_increment
    simultaneous_confidence = 1.0 - alpha / max(1, len(pairs_by_task))
    p_values: list[float | None] = []
    interim: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_id = task["id"]
        pairs = sorted(pairs_by_task[task_id], key=lambda pair: pair["mcp"]["repetition"])
        if not pairs:
            p_values.append(None)
            interim[task_id] = {
                "matched_repetitions": 0,
                "status": "unknown",
                "paired_score_interval": None,
                "exact_mcnemar_p": None,
            }
            continue
        both_successes = sum(
            pair["control"]["task_success"] and pair["mcp"]["task_success"]
            for pair in pairs
        )
        mcp_only = sum(
            pair["mcp"]["task_success"] and not pair["control"]["task_success"]
            for pair in pairs
        )
        control_only = sum(
            pair["control"]["task_success"] and not pair["mcp"]["task_success"]
            for pair in pairs
        )
        interval = newcombe_paired_score_interval(
            both_successes=both_successes,
            mcp_only_successes=mcp_only,
            control_only_successes=control_only,
            count=len(pairs),
            confidence=simultaneous_confidence,
        )
        p_value = exact_mcnemar_pvalue(mcp_only, control_only)
        p_values.append(p_value)
        interim[task_id] = {
            "matched_repetitions": len(pairs),
            "control_successes": both_successes + control_only,
            "mcp_successes": both_successes + mcp_only,
            "both_successes": both_successes,
            "mcp_only_successes": mcp_only,
            "control_only_successes": control_only,
            "difference": (mcp_only - control_only) / len(pairs),
            "paired_score_interval": interval,
            "exact_mcnemar_p": p_value,
            "interval_confidence_simultaneous": simultaneous_confidence,
        }
    adjusted = holm_adjust(p_values)
    task_results = {}
    at_cap = completed >= planned and planned >= target
    for task, adjusted_p in zip(TASKS, adjusted):
        task_id = task["id"]
        result = interim[task_id]
        interval = result.get("paired_score_interval")
        interval_excludes_zero = (
            isinstance(interval, dict)
            and (interval["lower"] > 0.0 or interval["upper"] < 0.0)
        )
        direction_established = (
            adjusted_p is not None
            and adjusted_p <= alpha
            and result.get("difference") is not None
            and result["difference"] != 0
            and interval_excludes_zero
        )
        equivalent = (
            isinstance(interval, dict)
            and interval["lower"] >= -DEFAULT_PRACTICAL_EFFECT
            and interval["upper"] <= DEFAULT_PRACTICAL_EFFECT
        )
        if exploratory_override:
            task_status = "exploratory"
        elif direction_established:
            task_status = "direction_established"
        elif equivalent:
            task_status = "within_practical_range"
        elif at_cap:
            task_status = "powered_but_inconclusive"
        else:
            task_status = "collect_more"
        task_results[task_id] = {
            **result,
            "holm_adjusted_p": adjusted_p,
            "direction_established": direction_established,
            "within_practical_range": equivalent,
            "status": task_status,
            "target_repetitions": target,
            "adequate_repetitions": result["matched_repetitions"] >= target,
        }

    if exploratory_override:
        status = "exploratory"
        should_continue = False
    elif all(item["status"] in {"direction_established", "within_practical_range"} for item in task_results.values()):
        status = "evidence_ready"
        should_continue = False
    elif at_cap:
        status = "powered_but_inconclusive"
        should_continue = False
    else:
        status = "collecting"
        should_continue = True
    return {
        "status": status,
        "adequacy_known": True,
        "continue": should_continue,
        "planned_repetitions": planned,
        "selected_design_repetitions": target,
        "completed_matched_repetitions_by_task": counts,
        "extension_stage": look,
        "look_alpha_spending": alpha_increment,
        "look_cumulative_alpha": cumulative_alpha,
        "family_alpha": DEFAULT_ALPHA,
        "multiplicity_method": "Holm exact McNemar p-values across tasks",
        "interval_method": "Newcombe 1998 method 10 paired score interval with Bonferroni simultaneous confidence",
        "practical_effect_range": [-DEFAULT_PRACTICAL_EFFECT, DEFAULT_PRACTICAL_EFFECT],
        "tasks": task_results,
    }


def summarize_records(payload: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate and paired outcome summaries without raw agent data."""
    runs = validate_records(
        payload,
        schedule,
        allow_staged_prefix=not schedule.get("repetitions_explicit", True),
    )
    records_schema_version = payload.get("records_schema_version", 1)
    by_mode = {}
    for mode in MODES:
        selected = [record for record in runs if record["mode"] == mode]
        invocation_statuses = [
            _invocation_completed(record, records_schema_version) for record in selected
        ]
        by_mode[mode] = {
            "runs": len(selected),
            "completion_rate": sum(record["completed"] for record in selected) / len(selected),
            "invocation_completion_rate": (
                sum(invocation_statuses) / len(invocation_statuses)
                if all(status is not None for status in invocation_statuses)
                else None
            ),
            "task_success_rate": (
                sum(record["task_success"] for record in selected) / len(selected)
                if all("task_success" in record for record in selected)
                else None
            ),
            "signal_retrieval_rate": (
                sum(record["signal_retrieved"] for record in selected) / len(selected)
                if all("task_success" in record for record in selected)
                else None
            ),
            "duration_seconds": _stats([_metric_value(record, "duration_seconds") for record in selected]),
            "tool_calls": _stats([_metric_value(record, "tool_calls") for record in selected]),
            "mcp_tool_calls": _stats([_metric_value(record, "mcp_tool_calls") for record in selected]),
            "repeated_commands": _stats([_metric_value(record, "repeated_commands") for record in selected]),
            "context_bytes_proxy": _stats([_metric_value(record, "context_bytes_proxy") for record in selected]),
            "prompt_bytes_proxy": _optional_stats(selected, "prompt_bytes_proxy"),
            "output_bytes_proxy": _optional_stats(selected, "output_bytes_proxy"),
            "peak_rss_bytes": _optional_positive_stats(selected, "peak_rss_bytes"),
            "data_path_bytes": {
                field: _optional_stats(selected, field)
                for field in DATA_PATH_BYTE_FIELDS
            },
            "input_tokens": _optional_stats(selected, "input_tokens"),
            "output_tokens": _optional_stats(selected, "output_tokens"),
            "usage_accounting": {
                "input_tokens": _usage_accounting(selected, "input_token_samples"),
                "output_tokens": _usage_accounting(selected, "output_token_samples"),
            },
            "failure_reasons": {
                reason: sum(record.get("failure_reason") == reason for record in selected)
                for reason in sorted({record.get("failure_reason") for record in selected if record.get("failure_reason")})
            },
        }

    paired = {}
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault((record["repetition"], record["task_id"]), {})[record["mode"]] = record

    task_success_by_mode = {
        mode: {"available": False, "tasks": {}}
        for mode in MODES
    }
    paired_task_success_deltas: dict[str, Any] = {"available": False, "tasks": {}}
    if records_schema_version >= 6:
        task_success_by_mode = {}
        for mode in MODES:
            mode_records = [record for record in runs if record["mode"] == mode]
            task_summaries = {}
            for task_id in sorted({record["task_id"] for record in mode_records}):
                task_records = [record for record in mode_records if record["task_id"] == task_id]
                successes = sum(record["task_success"] for record in task_records)
                task_summaries[task_id] = {
                    "successes": successes,
                    "count": len(task_records),
                    "success_rate": successes / len(task_records),
                }
            task_success_by_mode[mode] = {"available": True, "tasks": task_summaries}

        paired_task_success = {}
        for task_id in sorted({task_id for _, task_id in grouped}):
            task_pairs = [
                pair for (repetition, paired_task_id), pair in grouped.items()
                if paired_task_id == task_id
            ]
            deltas = [
                int(pair["mcp"]["task_success"]) - int(pair["control"]["task_success"])
                for pair in task_pairs
            ]
            stats = _stats(deltas)
            stats["available"] = bool(deltas)
            paired_task_success[task_id] = stats
        paired_task_success_deltas = {"available": True, "tasks": paired_task_success}

    criterion_summaries_by_mode = {
        mode: {"available": False, "tasks": {}}
        for mode in MODES
    }
    paired_criterion_deltas: dict[str, Any] = {"available": False, "tasks": {}}
    if records_schema_version >= 7:
        criterion_summaries_by_mode = {}
        for mode in MODES:
            mode_records = [record for record in runs if record["mode"] == mode]
            task_summaries = {}
            for task_id in sorted({record["task_id"] for record in mode_records}):
                task_records = [record for record in mode_records if record["task_id"] == task_id]
                criterion_count = len(task_records[0]["criterion_passes"])
                task_summaries[task_id] = [
                    {
                        "criterion_index": index + 1,
                        "passed": sum(record["criterion_passes"][index] for record in task_records),
                        "count": len(task_records),
                        "pass_rate": sum(record["criterion_passes"][index] for record in task_records) / len(task_records),
                    }
                    for index in range(criterion_count)
                ]
            criterion_summaries_by_mode[mode] = {"available": True, "tasks": task_summaries}

        paired_tasks = {}
        for task_id in sorted({task_id for _, task_id in grouped}):
            task_pairs = [pair for (repetition, paired_task_id), pair in grouped.items() if paired_task_id == task_id]
            criterion_count = len(task_pairs[0]["control"]["criterion_passes"])
            paired_tasks[task_id] = []
            for index in range(criterion_count):
                deltas = [
                    int(pair["mcp"]["criterion_passes"][index])
                    - int(pair["control"]["criterion_passes"][index])
                    for pair in task_pairs
                ]
                stats = _stats(deltas)
                stats.update({
                    "available": bool(deltas),
                    "criterion_index": index + 1,
                })
                paired_tasks[task_id].append(stats)
        paired_criterion_deltas = {"available": True, "tasks": paired_tasks}

    criterion_exposure_summaries_by_mode = {
        mode: {"available": False, "sources": {}}
        for mode in MODES
    }
    if records_schema_version >= 8:
        source_fields = (
            ("search_capture", "criterion_search_response_hits", "search_capture_responses"),
            ("get_capture_slice", "criterion_slice_response_hits", "get_capture_slice_responses"),
        )
        criterion_exposure_summaries_by_mode["mcp"] = {"available": True, "sources": {}}
        for source, hit_field, response_count_field in source_fields:
            source_tasks = {}
            response_counts_by_task = {}
            mcp_task_ids = {
                record["task_id"] for record in runs if record["mode"] == "mcp"
            }
            for task_id in sorted(mcp_task_ids):
                task_records = [
                    record for record in runs
                    if record["mode"] == "mcp" and record["task_id"] == task_id
                ]
                response_counts = [record[response_count_field] for record in task_records]
                criterion_count = len(task_records[0][hit_field])
                response_counts_by_task[task_id] = {
                    "total": sum(response_counts),
                    "runs": len(response_counts),
                    "runs_with_response": sum(count > 0 for count in response_counts),
                }
                criteria = []
                for index in range(criterion_count):
                    exposure_count = sum(record[hit_field][index] for record in task_records)
                    outcomes_by_state = {
                        state: {"passed": 0, "failed": 0, "count": 0, "pass_rate": None}
                        for state in EXPOSURE_STATES
                    }
                    for record in task_records:
                        if record[hit_field][index]:
                            state = "criterion_hit"
                        elif record[response_count_field] > 0:
                            state = "response_without_criterion_hit"
                        else:
                            state = "no_successful_response"
                        outcome = outcomes_by_state[state]
                        passed = record["criterion_passes"][index]
                        outcome["passed" if passed else "failed"] += 1
                        outcome["count"] += 1
                    for outcome in outcomes_by_state.values():
                        if outcome["count"]:
                            outcome["pass_rate"] = outcome["passed"] / outcome["count"]
                    criteria.append({
                        "criterion_index": index + 1,
                        "exposed": exposure_count,
                        "count": len(task_records),
                        "exposure_rate": exposure_count / len(task_records),
                        "answer_outcomes_by_response_state": outcomes_by_state,
                    })
                source_tasks[task_id] = criteria
            criterion_exposure_summaries_by_mode["mcp"]["sources"][source] = {
                "response_counts_by_task": response_counts_by_task,
                "tasks": source_tasks,
            }

    for metric in METRICS:
        deltas = []
        for pair in grouped.values():
            if metric in {"task_success", "signal_retrieved"} and not all(
                metric in pair[mode] and "task_success" in pair[mode]
                for mode in MODES
            ):
                continue
            if metric == "peak_rss_bytes" and (
                pair["control"].get(metric, 0) <= 0 or pair["mcp"].get(metric, 0) <= 0
            ):
                continue
            if metric == "completed":
                control = _invocation_completed(pair["control"], records_schema_version)
                mcp = _invocation_completed(pair["mcp"], records_schema_version)
                if control is None or mcp is None:
                    continue
            else:
                control = _metric_value(pair["control"], metric)
                mcp = _metric_value(pair["mcp"], metric)
            deltas.append(mcp - control)
        paired[metric] = _stats(deltas)
        if metric in {"completed", "task_success", "signal_retrieved"}:
            paired[metric]["available"] = bool(deltas)
        if metric == "peak_rss_bytes":
            paired[metric]["available"] = bool(deltas)
    for metric in ("prompt_bytes_proxy", "output_bytes_proxy"):
        if all(metric in record for record in runs):
            deltas = []
            for pair in grouped.values():
                deltas.append(_metric_value(pair["mcp"], metric) - _metric_value(pair["control"], metric))
            paired[metric] = _stats(deltas)
    for field in DATA_PATH_BYTE_FIELDS:
        if all(field in record for record in runs):
            paired[field] = _stats([
                _metric_value(pair["mcp"], field) - _metric_value(pair["control"], field)
                for pair in grouped.values()
            ])

    recommendations = [
        "Treat objectively scored task success and affirmative signal retrieval as primary outcomes; interpret invocation completion and cost metrics as secondary.",
        "Repeat with representative privacy-reviewed agent tasks before changing defaults.",
    ]
    if not paired["task_success"].get("available"):
        recommendations.insert(1, "Legacy records lack objective task-success criteria and cannot support task-success or signal-retrieval comparisons.")
    elif paired["task_success"]["mean"] > 0 or paired["signal_retrieved"]["mean"] > 0:
        recommendations.insert(1, "MCP improved at least one paired task outcome; inspect task-level variance before generalizing.")
    elif paired["task_success"]["mean"] < 0 or paired["signal_retrieved"]["mean"] < 0:
        recommendations.insert(1, "MCP reduced at least one paired task outcome; inspect failures and routing choices before enabling broader use.")

    adequacy = _statistical_adequacy(
        runs,
        schedule,
        records_schema_version,
        payload.get("task_rubric_version"),
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "benchmark": "agent-ab",
        "records_schema_version": max(payload.get("records_schema_version", 1), 1),
        "task_fixture_version": schedule["task_fixture_version"],
        "task_rubric_version": schedule.get("task_rubric_version"),
        "seed": schedule["seed"],
        "repetitions": schedule["repetitions"],
        "planned_repetitions": schedule["repetitions"],
        "completed_repetitions_by_task": adequacy.get("completed_matched_repetitions_by_task", {}),
        "extension_stage": adequacy.get("extension_stage"),
        "statistical_design": schedule.get("statistical_design"),
        "statistical_adequacy": adequacy,
        "protocol": payload["protocol"],
        "mode_summaries": by_mode,
        "paired_deltas_mcp_minus_control": paired,
        "task_success_by_mode": task_success_by_mode,
        "paired_task_success_deltas_mcp_minus_control": paired_task_success_deltas,
        "criterion_summaries_by_mode": criterion_summaries_by_mode,
        "paired_criterion_deltas_mcp_minus_control": paired_criterion_deltas,
        "criterion_exposure_summaries_by_mode": criterion_exposure_summaries_by_mode,
        "recommendations": recommendations,
        "privacy": "summary contains aggregate metadata and task-scoped criterion indices only; raw prompts, criterion text, tool responses, commands, captures, transcripts, and user content are excluded",
    }


def _stats_measurement(stats: dict[str, Any], unit: str, note: str | None = None) -> dict[str, Any]:
    """Map an aggregate ``_stats`` block onto a workload measurement."""
    if stats.get("available") is False or not stats.get("count"):
        return wr.unavailable(unit, note)
    return wr.measurement(unit, mean=stats["mean"], stdev=stats["stdev"], samples=stats["count"], note=note)


def _rate_status(rate: float) -> str:
    return "success" if rate >= 1.0 else "failure" if rate <= 0.0 else "partial"


def summary_workload_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-agnostic workload result for an aggregate A/B summary."""
    runs = []
    for mode, block in summary["mode_summaries"].items():
        count = block["runs"]
        invocation_completion_rate = block["invocation_completion_rate"]
        measurements = {
            "invocation_completion_rate": (
                wr.measurement("ratio", value=invocation_completion_rate, samples=count)
                if invocation_completion_rate is not None
                else wr.unavailable("ratio", "legacy records lack exit codes to determine invocation completion")
            ),
            "success_rate": (
                wr.measurement("ratio", value=block["task_success_rate"], samples=count)
                if block["task_success_rate"] is not None
                else wr.unavailable("ratio", "legacy records lack objective task-success criteria")
            ),
            "signal_retrieval_rate": (
                wr.measurement("ratio", value=block["signal_retrieval_rate"], samples=count)
                if block["signal_retrieval_rate"] is not None
                else wr.unavailable("ratio", "legacy records use the prior marker-only retrieval scoring")
            ),
        }
        for field, (name, unit) in RECORD_MEASUREMENTS.items():
            stats = block["data_path_bytes"].get(field) if field in DATA_PATH_BYTE_FIELDS else block.get(field)
            if stats is None:
                continue
            note = CONTEXT_PROXY_NOTE if field == "context_bytes_proxy" else None
            measurements[name] = _stats_measurement(stats, unit, note)
        runs.append(wr.run(
            mode,
            labels={"mode": mode},
            status=(
                _rate_status(block["task_success_rate"])
                if block["task_success_rate"] is not None
                else "partial"
            ),
            measurements=measurements,
            errors=[f"{reason}: {total}" for reason, total in block["failure_reasons"].items()],
        ))
    paired = {}
    for metric, stats in summary["paired_deltas_mcp_minus_control"].items():
        name, unit = PAIRED_MEASUREMENTS[metric]
        paired[name] = _stats_measurement(stats, unit, "mcp minus control, paired by task and repetition")
    runs.append(wr.run("paired-delta", labels={"comparison": "mcp_minus_control"}, measurements=paired))
    return wr.build_result(
        workload="agent-ab",
        kind="evaluation",
        producer="benchmark_agent_ab.py",
        producer_schema_version=summary["records_schema_version"],
        fixture_version=summary["task_fixture_version"],
        parameters={
            "seed": summary["seed"],
            "repetitions": summary["repetitions"],
            "task_rubric_version": summary.get("task_rubric_version"),
            "protocol": summary["protocol"],
        },
        runs=runs,
        details=summary,
        privacy=summary["privacy"],
    )


def records_workload_result(
    payload: dict[str, Any], schedule: dict[str, Any] | None = None, producer: str = "benchmark_agent_ab.py"
) -> dict[str, Any]:
    """Return one workload run per agent execution in a records envelope."""
    schedule = schedule if schedule is not None else payload["schedule"]
    records_schema_version = payload.get("records_schema_version", 1)
    runs = []
    for record in validate_records(payload, schedule):
        failure_reason = record.get("failure_reason")
        task_success = record.get("task_success")
        invocation_completed = _invocation_completed(record, records_schema_version)
        if task_success is True:
            status = "success"
        elif failure_reason == "timeout":
            status = "timeout"
        elif task_success is None:
            status = "failure" if invocation_completed is False else "partial"
        else:
            status = "failure"
        measurements = {
            "invocation_completion_rate": (
                wr.measurement("ratio", value=1.0 if invocation_completed else 0.0, samples=1)
                if invocation_completed is not None
                else wr.unavailable("ratio", "legacy record lacks an exit code to determine invocation completion")
            ),
            "success_rate": (
                wr.measurement("ratio", value=1.0 if task_success else 0.0, samples=1)
                if task_success is not None
                else wr.unavailable("ratio", "legacy record has no objective task-success score")
            ),
            "signal_retrieval_rate": (
                wr.measurement("ratio", value=1.0 if record["signal_retrieved"] else 0.0, samples=1)
                if task_success is not None
                else wr.unavailable("ratio", "legacy record uses marker-only retrieval scoring")
            ),
            "context_bytes": wr.measurement(
                "bytes", value=_metric_value(record, "context_bytes_proxy"), samples=1, note=CONTEXT_PROXY_NOTE
            ),
        }
        for field, (name, unit) in RECORD_MEASUREMENTS.items():
            if field == "context_bytes_proxy" or field not in record:
                continue
            value = record[field]
            if field == "peak_rss_bytes" and (value is None or value <= 0):
                measurements[name] = wr.unavailable(unit, "zero peak RSS means the adapter could not measure it")
            elif value is None:
                measurements[name] = wr.unavailable(unit)
            else:
                measurements[name] = wr.measurement(unit, value=value, samples=1)
        runs.append(wr.run(
            f"{record['task_id']}-r{record['repetition']}-{record['mode']}",
            labels={"task_id": record["task_id"], "repetition": record["repetition"], "mode": record["mode"]},
            status=status,
            measurements=measurements,
            errors=[failure_reason] if failure_reason else [],
        ))
    return wr.build_result(
        workload="agent-ab",
        kind="agent-run",
        producer=producer,
        producer_schema_version=payload.get("records_schema_version", 1),
        fixture_version=schedule["task_fixture_version"],
        parameters={
            "seed": schedule["seed"],
            "repetitions": schedule["repetitions"],
            "task_rubric_version": schedule.get("task_rubric_version"),
            "protocol": payload["protocol"],
        },
        runs=runs,
        privacy=payload.get("privacy", "records contain metadata only"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-output", type=Path, help="Write a deterministic run schedule")
    parser.add_argument("--repetitions", type=int, help="Explicit smoke/evaluation override; omitted selects the powered A/B design")
    parser.add_argument("--seed", type=int, default=DEFAULT_AGENT_AB_SEED)
    parser.add_argument("--model-config", help="Model identifier used to validate resumable stages")
    parser.add_argument("--repository-fixture", help="Fixture profile used to validate resumable stages")
    parser.add_argument("--pilot-records", type=Path, help="Current fixture-v4/rubric-v4 records used to bootstrap paired outcomes")
    parser.add_argument("--pilot-schedule", type=Path, help="Schedule for --pilot-records when not embedded")
    parser.add_argument("--records", type=Path, help="Analyze an external metadata-only records envelope")
    parser.add_argument("--schedule", type=Path, help="Schedule JSON when it is not embedded in records")
    parser.add_argument("--output", type=Path, help="Write an aggregate summary JSON")
    wr.add_result_argument(parser)
    args = parser.parse_args()
    if bool(args.schedule_output) == bool(args.records):
        parser.error("provide exactly one of --schedule-output or --records")
    if args.schedule_output and (args.result or args.experiment or args.metadata or args.redact):
        parser.error("--result, --experiment, --metadata, and --redact require --records")
    if args.schedule_output:
        pilot_payload = _read_json(args.pilot_records) if args.pilot_records else None
        pilot_schedule = (
            _read_json(args.pilot_schedule)
            if args.pilot_schedule
            else pilot_payload.get("schedule") if pilot_payload is not None else None
        )
        if args.pilot_schedule and not args.pilot_records:
            parser.error("--pilot-schedule requires --pilot-records")
        resolved_model_config = args.model_config
        resolved_repository_fixture = args.repository_fixture
        if pilot_payload is not None:
            try:
                resolved_model_config, resolved_repository_fixture = _resolve_pilot_identity(
                    pilot_payload,
                    pilot_schedule,
                    model_config=args.model_config,
                    repository_fixture=args.repository_fixture,
                )
            except ValueError as exc:
                parser.error(str(exc))
        design = (
            build_power_design(pilot_payload, pilot_schedule)
            if pilot_payload is not None
            else _default_power_design()
        )
        schedule = build_schedule(
            args.repetitions,
            args.seed,
            design=design,
            model_config=resolved_model_config,
            repository_fixture=resolved_repository_fixture,
        )
        args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
        args.schedule_output.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "schedule": str(args.schedule_output),
            "repetitions": schedule["repetitions"],
            "selected_design_repetitions": schedule["selected_design_repetitions"],
            "run_label": schedule["run_label"],
            "pilot_required": design["pilot_required"],
        }, sort_keys=True))
        return

    payload = _read_json(args.records)
    schedule = _read_json(args.schedule) if args.schedule else payload.get("schedule")
    if not isinstance(schedule, dict):
        parser.error("records must embed a schedule or provide --schedule")
    summary = summarize_records(payload, schedule)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), file=wr.report_stream(args.result))
    if args.result:
        wr.write_result(summary_workload_result(summary), args.result, experiment=wr.experiment_from_args(args))


if __name__ == "__main__":
    main()
