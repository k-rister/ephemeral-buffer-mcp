"""Tests for the privacy-safe agent-level A/B evaluation harness."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmarks.benchmark_agent_ab as benchmark_agent_ab
import benchmarks.workload_results as wr
from benchmarks.render_agent_ab_report import append_agent_ab_report, render_agent_ab_sections
from benchmarks.benchmark_agent_ab import (
    DATA_PATH_BYTE_FIELDS,
    RECORDS_SCHEMA_VERSION,
    TASK_RUBRIC_VERSION,
    build_power_design,
    build_schedule,
    records_workload_result,
    stage_repetitions,
    summarize_records,
    summary_workload_result,
    validate_records,
    validate_schedule_extension,
)


def _records_payload(repetitions=2, design=None, powered=False):
    schedule = build_schedule(repetitions=repetitions, seed=17, design=design, powered=powered)
    runs = []
    for item in schedule["schedule"]:
        mcp = item["mode"] == "mcp"
        runs.append({
            "task_id": item["task_id"],
            "repetition": item["repetition"],
            "mode": item["mode"],
            "completed": mcp,
            "signal_retrieved": mcp,
            "duration_seconds": 2 if mcp else 3,
            "tool_calls": 2 if mcp else 4,
            "repeated_commands": 0 if mcp else 1,
            "context_bytes": 100 if mcp else 400,
            "peak_rss_bytes": 1000,
        })
    return schedule, {
        "schema_version": 1,
        "benchmark": "agent-ab",
        "task_fixture_version": schedule["task_fixture_version"],
        "task_rubric_version": schedule["task_rubric_version"],
        "protocol": {
            "model_config": "test-model",
            "repository_fixture": "synthetic-repo-v1",
            "environment": "test-environment",
            "reset_policy": "fresh-worktree-per-run",
            "agent_adapter": "test-adapter",
            "embedding_mode": "test",
            "embedding_model": "deterministic-test",
            "embedding_cache": "not-applicable",
            "timeout_seconds": 10,
            "effective_sandbox": "read-only",
            "mcp_launch_fingerprint": "sha256:" + "0" * 64,
        },
        "schedule": schedule,
        "runs": runs,
    }


def _scored_records_payload(repetitions=2, design=None, powered=False):
    schedule, payload = _records_payload(repetitions=repetitions, design=design, powered=powered)
    payload["records_schema_version"] = RECORDS_SCHEMA_VERSION
    for record in payload["runs"]:
        context_bytes = record.pop("context_bytes")
        record.update({
            "context_bytes_proxy": context_bytes,
            "exit_code": 0,
            "failure_reason": None,
            "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
            "input_tokens": None,
            "output_tokens": None,
            "input_token_samples": [],
            "output_token_samples": [],
            "prompt_bytes_proxy": 40,
            "output_bytes_proxy": context_bytes - 40,
            "task_success": record["mode"] == "mcp" and record["repetition"] == 2,
        })
        record["criterion_passes"] = [record["task_success"], record["task_success"]]
        record["line_reference_diagnostics"] = (
            [{
                "criterion_index": 2,
                "category": (
                    "recognized_reference"
                    if record["criterion_passes"][1]
                    else "expected_number_absent"
                ),
            }]
            if record["task_id"] == "targeted-inspection"
            else []
        )
        record["criterion_search_response_hits"] = [False, False] if record["mode"] == "mcp" else None
        record["criterion_slice_response_hits"] = [False, False] if record["mode"] == "mcp" else None
        record["search_capture_responses"] = 0
        record["get_capture_slice_responses"] = 0
        record.update({field: 0 for field in DATA_PATH_BYTE_FIELDS})
        record["completed"] = True
        record["signal_retrieved"] = record["task_success"] or (
            record["mode"] == "mcp" and record["repetition"] == 1 and record["task_id"] == "targeted-inspection"
        )
    return schedule, payload


def _legacy_v6_records_payload():
    schedule, payload = _scored_records_payload(design={"selected_repetitions": 20})
    payload["records_schema_version"] = 6
    for record in payload["runs"]:
        record.pop("criterion_passes")
        record.pop("line_reference_diagnostics")
        record.pop("criterion_search_response_hits")
        record.pop("criterion_slice_response_hits")
        record.pop("search_capture_responses")
        record.pop("get_capture_slice_responses")
    return schedule, payload


def _legacy_v10_records_payload():
    schedule, payload = _scored_records_payload()
    payload["records_schema_version"] = 10
    payload["task_rubric_version"] = 4
    schedule["task_rubric_version"] = 4
    for record in payload["runs"]:
        record.pop("line_reference_diagnostics")
    return schedule, payload


def _legacy_v5_records_payload():
    schedule, payload = _records_payload()
    payload["records_schema_version"] = 5
    for record in payload["runs"]:
        context_bytes = record.pop("context_bytes")
        record.update({
            "context_bytes_proxy": context_bytes,
            "exit_code": 0,
            "failure_reason": None if record["completed"] else "mcp_not_used",
            "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
            "input_tokens": None,
            "output_tokens": None,
            "input_token_samples": [],
            "output_token_samples": [],
            "prompt_bytes_proxy": 40,
            "output_bytes_proxy": context_bytes - 40,
        })
        record.update({field: 0 for field in DATA_PATH_BYTE_FIELDS})
    return schedule, payload


class TestAgentAbBenchmark(unittest.TestCase):
    def test_historical_rubric_v5_records_remain_readable_under_schema_v12(self):
        schedule, payload = _scored_records_payload(design={"selected_repetitions": 2})
        schedule["task_rubric_version"] = 5
        payload["task_rubric_version"] = 5

        summary = summarize_records(payload, schedule)

        self.assertEqual(summary["task_rubric_version"], 5)
        self.assertEqual(summary["records_schema_version"], 12)

    def test_historical_v3_fixture_records_remain_readable(self):
        schedule, payload = _legacy_v10_records_payload()
        schedule["task_fixture_version"] = 3
        payload["task_fixture_version"] = 3
        summary = summarize_records(payload, schedule)
        self.assertEqual(summary["task_fixture_version"], 3)
        self.assertEqual(summary["records_schema_version"], 10)
        self.assertEqual(summary["statistical_adequacy"]["status"], "unknown")

    def test_default_schedule_is_five_exploratory_repetitions_in_one_look(self):
        schedule = build_schedule(seed=20261031, design={"selected_repetitions": 12})
        self.assertEqual(schedule["repetitions"], 5)
        self.assertFalse(schedule["powered"])
        self.assertEqual(schedule["run_label"], "exploratory")
        self.assertEqual(len(schedule["schedule"]), 5 * len(benchmark_agent_ab.TASKS) * len(benchmark_agent_ab.MODES))
        self.assertEqual([item["repetitions"] for item in schedule["stage_schedule"]], [5])

    def test_powered_schedule_uses_powered_design_and_three_looks(self):
        schedule = build_schedule(seed=20261031, powered=True)
        design = schedule["statistical_design"]
        self.assertEqual(design["schema_version"], 5)
        self.assertEqual(
            design["pilot_evidence_source"],
            "least-favorable-unknown-baseline-full-discordance",
        )
        self.assertIn("Clopper-Pearson", design["baseline_interval_method"])
        self.assertIn("95% joint coverage", design["pilot_parameter_confidence_rule"])
        selected = next(
            scenario for scenario in design["scenario_matrix"]
            if scenario["effect_pp"] == 10 and scenario["target_power"] == 0.8
        )
        stress = next(
            scenario for scenario in design["scenario_matrix"]
            if scenario["effect_pp"] == 20 and scenario["target_power"] == 0.9
        )
        self.assertEqual(schedule["repetitions"], design["selected_repetitions"])
        self.assertGreater(schedule["repetitions"], 5)
        self.assertTrue(all(
            case["discordance_rates"] == [1.0] * len(benchmark_agent_ab.TASKS)
            for case in stress["paired_model_cases"]
        ))
        self.assertEqual(
            stress["paired_model_cases"][0]["baseline_success_rates"],
            [0.4] * len(benchmark_agent_ab.TASKS),
        )
        self.assertEqual(
            stress["paired_model_cases"][1]["baseline_success_rates"],
            [0.6] * len(benchmark_agent_ab.TASKS),
        )
        self.assertTrue(all(value >= 0.8 for value in selected["power_lower_bound_by_task"].values()))
        self.assertTrue(all(
            all(
                scenario["power_lower_bound_by_task"][task_id] >= scenario["target_power"]
                for task_id in scenario["power_lower_bound_by_task"]
            )
            and all(
                0.95 < confidence < 1.0
                for confidence in scenario["power_lower_confidence_by_task"].values()
            )
            for scenario in design["scenario_matrix"]
        ))
        self.assertEqual(
            [item["percent"] for item in schedule["stage_schedule"]],
            [50, 75, 100],
        )
        self.assertEqual(len(design["scenario_matrix"]), 6)
        self.assertTrue(all(item["invocations_per_variant"] > 0 for item in design["scenario_matrix"]))

    def test_explicit_small_override_is_exploratory_and_stage_counts_are_cumulative(self):
        design = {"selected_repetitions": 12}
        schedule = build_schedule(repetitions=3, seed=4, design=design)
        self.assertTrue(schedule["repetitions_explicit"])
        self.assertEqual(schedule["run_label"], "exploratory")
        self.assertEqual(schedule["stage_schedule"], [{
            "percent": 100,
            "repetitions": 3,
            "alpha_spending": 0.05,
            "cumulative_alpha": 0.05,
        }])
        self.assertEqual(
            [
                (item["percent"], item["repetitions"], item["alpha_spending"], item["cumulative_alpha"])
                for item in stage_repetitions(12)
            ],
            [(50, 6, 0.005, 0.005), (75, 9, 0.015, 0.02), (100, 12, 0.03, 0.05)],
        )

    def test_explicit_count_at_or_above_powered_design_stays_exploratory(self):
        design = {"selected_repetitions": 5}
        schedule = build_schedule(repetitions=7, seed=4, design=design)
        self.assertEqual(schedule["run_label"], "exploratory")
        self.assertFalse(schedule["powered"])
        self.assertEqual([item["repetitions"] for item in schedule["stage_schedule"]], [7])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            build_schedule(repetitions=7, powered=True, design=design)

    def test_schedule_helpers_keep_legacy_power_and_staging_semantics(self):
        legacy_powered = build_schedule(powered=True, design={"selected_repetitions": 8})
        legacy_powered.pop("powered")
        self.assertTrue(benchmark_agent_ab.schedule_is_powered(legacy_powered))
        self.assertTrue(benchmark_agent_ab.schedule_uses_staged_looks(legacy_powered))

        legacy_exploratory = build_schedule(
            repetitions=3,
            design={"selected_repetitions": 8},
        )
        legacy_exploratory.pop("powered")
        self.assertFalse(benchmark_agent_ab.schedule_is_powered(legacy_exploratory))
        self.assertFalse(benchmark_agent_ab.schedule_uses_staged_looks(legacy_exploratory))
        self.assertTrue(benchmark_agent_ab.schedule_is_powered({
            "repetitions": 5,
            "repetitions_explicit": True,
        }))

    def test_exploratory_adequacy_never_claims_power_from_large_count(self):
        schedule, payload = _scored_records_payload(
            repetitions=5,
            design={"selected_repetitions": 5},
        )
        adequacy = summarize_records(payload, schedule)["statistical_adequacy"]
        self.assertEqual(adequacy["status"], "exploratory")
        self.assertFalse(adequacy["continue"])
        self.assertTrue(all(not task["adequate_repetitions"] for task in adequacy["tasks"].values()))

    def test_schedule_extension_requires_exact_prefix_and_matching_versions(self):
        design = {"selected_repetitions": 20}
        original = build_schedule(seed=17, powered=True, design=design)
        extended = build_schedule(seed=17, powered=True, design=design)
        validate_schedule_extension(original, extended)
        explicit = build_schedule(repetitions=4, seed=17, design=design)
        explicit_extension = build_schedule(repetitions=8, seed=17, design=design)
        with self.assertRaisesRegex(ValueError, "exploratory schedules cannot be extended"):
            validate_schedule_extension(explicit, explicit_extension)
        validate_schedule_extension(explicit, explicit)
        changed_seed = build_schedule(seed=18, powered=True, design=design)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_schedule_extension(original, changed_seed)
        changed_model = build_schedule(
            seed=17, powered=True, design=design, model_config="different-model"
        )
        with self.assertRaisesRegex(ValueError, "model_config"):
            validate_schedule_extension(original, changed_model)
        changed_fixture = build_schedule(
            seed=17, powered=True, design=design, repository_fixture="different-fixture"
        )
        with self.assertRaisesRegex(ValueError, "repository_fixture"):
            validate_schedule_extension(original, changed_fixture)
        changed_rubric = dict(extended, task_rubric_version=TASK_RUBRIC_VERSION - 1)
        with self.assertRaisesRegex(ValueError, "task_rubric_version"):
            validate_schedule_extension(original, changed_rubric)

    def test_schedule_binds_profile_identity_and_rejects_changes(self):
        profile = {
            "name": "self-hosted",
            "model_id": "custom-model",
            "provider_id": "gateway",
            "config_sha256": "sha256:" + "a" * 64,
        }
        schedule = build_schedule(
            repetitions=1,
            seed=19,
            design={"selected_repetitions": 1},
            model_config=profile["model_id"],
            codex_profile=profile,
        )
        self.assertEqual(schedule["codex_profile"], profile)
        self.assertEqual(
            benchmark_agent_ab._parse_codex_profile_json(json.dumps(profile)),
            profile,
        )

        changed_profile = {**profile, "config_sha256": "sha256:" + "b" * 64}
        changed_schedule = build_schedule(
            repetitions=1,
            seed=19,
            design={"selected_repetitions": 1},
            model_config=profile["model_id"],
            codex_profile=changed_profile,
        )
        with self.assertRaisesRegex(ValueError, "codex_profile"):
            validate_schedule_extension(schedule, changed_schedule)
        with self.assertRaisesRegex(ValueError, "profile identity"):
            build_schedule(
                repetitions=1,
                seed=19,
                design={"selected_repetitions": 1},
                codex_profile={**profile, "config_sha256": "unhashed"},
            )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            benchmark_agent_ab._parse_codex_profile_json("null")

    def test_power_pilot_rejects_mixed_fixture_or_rubric_versions(self):
        schedule, payload = _scored_records_payload()
        payload["task_rubric_version"] = TASK_RUBRIC_VERSION
        with self.assertRaisesRegex(ValueError, "fixture-v4/rubric-v6"):
            build_power_design(dict(payload, task_fixture_version=3), schedule, simulations=100)
        with self.assertRaisesRegex(ValueError, "fixture-v4/rubric-v6"):
            build_power_design(dict(payload, task_rubric_version=3), schedule, simulations=100)

    def test_power_design_bootstraps_paired_pilot_and_marks_small_samples(self):
        small_schedule, small_payload = _scored_records_payload(repetitions=2)
        for record in small_payload["runs"]:
            record["task_success"] = False
            record["criterion_passes"] = [False, False]
        small_design = build_power_design(small_payload, small_schedule, simulations=100, seed=73)
        conservative_design = build_power_design(seed=73, simulations=100)
        self.assertEqual(
            small_design["pilot_evidence_source"],
            "paired-records-exact-bounds-with-conservative-fallback",
        )
        self.assertTrue(small_design["pilot_required"])
        self.assertEqual(small_design["selected_repetitions"], conservative_design["selected_repetitions"])
        self.assertEqual(
            small_design["paired_model_by_task"]["targeted-inspection"]["matched_repetitions"],
            2,
        )
        small_task_model = small_design["paired_model_by_task"]["targeted-inspection"]
        self.assertIsNone(small_task_model["baseline_success_rate"])
        self.assertIsNone(small_task_model["discordance_rate"])
        self.assertEqual(small_task_model["source"], "under-20-pair-least-favorable-full-discordance")
        self.assertAlmostEqual(benchmark_agent_ab._scenario_discordance(small_task_model, 0.1), 1.0)

        pilot_schedule, pilot_payload = _scored_records_payload(repetitions=20)
        pilot = build_power_design(pilot_payload, pilot_schedule, simulations=100, seed=73)
        self.assertEqual(pilot["pilot_evidence_source"], "paired-records-exact-bounds")
        self.assertFalse(pilot["pilot_required"])
        self.assertEqual(pilot["pilot_parameter_family_alpha"], benchmark_agent_ab.DEFAULT_ALPHA)
        self.assertEqual(pilot["pilot_parameter_bound_count"], 2 * len(benchmark_agent_ab.TASKS))
        self.assertEqual(
            pilot["paired_model_by_task"]["targeted-inspection"]["matched_repetitions"],
            20,
        )
        self.assertEqual(
            pilot["paired_model_by_task"]["targeted-inspection"]["discordance_upper_method"],
            "one-sided-clopper-pearson",
        )
        self.assertEqual(
            pilot["paired_model_by_task"]["targeted-inspection"]["baseline_success_rate_interval"]["confidence"],
            1.0 - 0.05 / (2 * len(benchmark_agent_ab.TASKS)),
        )

    def test_zero_discordance_pilot_uses_exact_upper_bound_for_power_design(self):
        schedule, payload = _scored_records_payload(repetitions=20)
        for record in payload["runs"]:
            shared_success = record["repetition"] <= 10
            record["task_success"] = shared_success
            record["criterion_passes"] = [shared_success, shared_success]

        design = build_power_design(payload, schedule, simulations=100, seed=83)
        pilot_bound_alpha = 0.05 / (2 * len(benchmark_agent_ab.TASKS))
        expected = 1.0 - pilot_bound_alpha ** (1.0 / 20)
        for model in design["paired_model_by_task"].values():
            self.assertEqual(model["observed_discordance_rate"], 0.0)
            self.assertEqual(model["discordance_upper_bootstrap"], 0.0)
            self.assertAlmostEqual(model["discordance_rate"], expected)
            self.assertAlmostEqual(benchmark_agent_ab._scenario_discordance(model, 0.1), expected)
            self.assertEqual(model["discordance_upper_method"], "one-sided-clopper-pearson")
            self.assertEqual(model["source"], "paired-records-familywise-exact-clopper-pearson-upper-bound")
            self.assertEqual(
                model["baseline_success_rate_interval"]["confidence"],
                1.0 - pilot_bound_alpha,
            )

    def test_near_boundary_baseline_does_not_clip_pilot_discordance_upper(self):
        schedule, payload = _scored_records_payload(repetitions=20)
        for record in payload["runs"]:
            mcp_only_success = record["mode"] == "mcp" and record["repetition"] <= 2
            record["task_success"] = mcp_only_success
            record["criterion_passes"] = [mcp_only_success, mcp_only_success]

        design = build_power_design(payload, schedule, simulations=100, seed=89)
        pilot_bound_alpha = 0.05 / (2 * len(benchmark_agent_ab.TASKS))
        default_scenario = next(
            scenario for scenario in design["scenario_matrix"]
            if scenario["effect_pp"] == 10 and scenario["target_power"] == 0.8
        )
        for model in design["paired_model_by_task"].values():
            self.assertEqual(model["baseline_success_rate"], 0.0)
            self.assertEqual(model["observed_discordance_rate"], 0.1)
            self.assertGreater(model["discordance_rate"], 0.1)
            self.assertGreater(model["baseline_success_rate_interval"]["upper"], 0.05)
            self.assertEqual(
                model["baseline_success_rate_interval"]["confidence"],
                1.0 - pilot_bound_alpha,
            )
            self.assertAlmostEqual(
                benchmark_agent_ab._scenario_discordance(model, 0.1),
                model["discordance_rate"],
            )
        self.assertTrue(all(
            rate > 0.1 for rate in default_scenario["discordance_rates"].values()
        ))
        self.assertEqual(
            [case["effect"] for case in default_scenario["paired_model_cases"]],
            [0.1, -0.1],
        )
        positive, negative = default_scenario["paired_model_cases"]
        task_index = 0
        self.assertGreater(positive["baseline_success_rates"][task_index], 0.05)
        self.assertGreater(positive["discordance_rates"][task_index], 0.1)
        self.assertGreater(negative["discordance_rates"][task_index], 0.1)

    def test_infeasible_pilot_baseline_uses_conservative_directional_fallback(self):
        cases = benchmark_agent_ab._scenario_model_cases(
            {
                "task": {
                    "baseline_success_rate": 0.95,
                    "baseline_success_rate_interval": {"lower": 0.90, "upper": 0.99},
                    "discordance_rate": 0.99,
                },
            },
            ["task"],
            0.2,
        )
        self.assertEqual(cases[0]["baseline_source_by_task"]["task"], "least-favorable-no-feasible-pilot-baseline")
        self.assertEqual(cases[0]["baseline_success_rates"], [0.4])
        self.assertAlmostEqual(cases[0]["discordance_rates"][0], 1.0)
        self.assertEqual(cases[1]["baseline_source_by_task"]["task"], "exact-baseline-interval")

    def test_unknown_baseline_uses_full_discordance_at_directional_extremes(self):
        cases = benchmark_agent_ab._scenario_model_cases(
            {"task": {"baseline_success_rate": None, "discordance_rate": None}},
            ["task"],
            0.2,
        )
        self.assertEqual(cases[0]["baseline_success_rates"], [0.4])
        self.assertEqual(cases[0]["discordance_rates"], [1.0])
        self.assertEqual(cases[1]["baseline_success_rates"], [0.6])
        self.assertEqual(cases[1]["discordance_rates"], [1.0])
        self.assertEqual(
            cases[0]["baseline_source_by_task"]["task"],
            "least-favorable-unknown-baseline-full-discordance",
        )
        self.assertAlmostEqual(cases[0]["baseline_success_rates"][0] + 0.2, 0.6)

    def test_adequate_pilot_tasks_keep_bootstrap_when_another_task_is_small(self):
        schedule, payload = _scored_records_payload(repetitions=20)
        task_id = "targeted-inspection"
        schedule["schedule"] = [
            item for item in schedule["schedule"]
            if item["task_id"] != task_id or item["repetition"] <= 2
        ]
        payload["runs"] = [
            record for record in payload["runs"]
            if record["task_id"] != task_id or record["repetition"] <= 2
        ]

        design = build_power_design(payload, schedule, simulations=100, seed=79)
        models = design["paired_model_by_task"]
        self.assertTrue(design["pilot_required"])
        self.assertEqual(models[task_id]["matched_repetitions"], 2)
        self.assertIsNone(models[task_id]["discordance_rate"])
        self.assertEqual(models["noisy-test-failure"]["matched_repetitions"], 20)
        self.assertIsNotNone(models["noisy-test-failure"]["discordance_rate"])

    def test_pilot_identity_is_derived_and_explicit_mismatches_are_rejected(self):
        schedule, payload = _scored_records_payload(repetitions=20)
        resolved = benchmark_agent_ab._resolve_pilot_identity(
            payload,
            schedule,
            model_config=None,
            repository_fixture=None,
        )
        self.assertEqual(resolved, ("test-model", "synthetic-repo-v1"))
        self.assertEqual(
            benchmark_agent_ab._resolve_pilot_identity(
                payload,
                schedule,
                model_config="test-model",
                repository_fixture="synthetic-repo-v1",
            ),
            resolved,
        )
        with self.assertRaisesRegex(ValueError, "model_config"):
            benchmark_agent_ab._resolve_pilot_identity(
                payload,
                schedule,
                model_config="different-model",
                repository_fixture=None,
            )
        with self.assertRaisesRegex(ValueError, "repository_fixture"):
            benchmark_agent_ab._resolve_pilot_identity(
                payload,
                schedule,
                model_config=None,
                repository_fixture="different-fixture",
            )
        with self.assertRaisesRegex(ValueError, "repository_fixture.*does not match pilot records protocol"):
            benchmark_agent_ab._resolve_pilot_identity(
                payload,
                dict(schedule, repository_fixture="different-fixture"),
                model_config=None,
                repository_fixture=None,
            )
        missing_identity = dict(payload, protocol={**payload["protocol"], "model_config": ""})
        with self.assertRaisesRegex(ValueError, "pilot protocol field model_config"):
            benchmark_agent_ab._resolve_pilot_identity(
                missing_identity,
                schedule,
                model_config="test-model",
                repository_fixture=None,
            )

    def test_direction_requires_holm_significance_and_simultaneous_interval(self):
        design = {"selected_repetitions": 12}
        schedule, payload = _scored_records_payload(repetitions=None, design=design, powered=True)
        for record in payload["runs"]:
            record["task_success"] = record["mode"] == "mcp"
            record["criterion_passes"] = [record["task_success"], record["task_success"]]

        adequacy = summarize_records(payload, schedule)["statistical_adequacy"]
        self.assertEqual(adequacy["status"], "evidence_ready")
        self.assertEqual(adequacy["look_cumulative_alpha"], 0.05)
        self.assertTrue(all(
            task["status"] == "direction_established"
            and task["holm_adjusted_p"] <= 0.05
            and task["paired_score_interval"]["lower"] > 0.0
            for task in adequacy["tasks"].values()
        ))

    def test_staged_summary_continues_then_reports_powered_inconclusive_at_cap(self):
        design = {"selected_repetitions": 12}
        schedule, payload = _scored_records_payload(repetitions=None, design=design, powered=True)
        payload["runs"] = [record for record in payload["runs"] if record["repetition"] <= 6]
        first_look = summarize_records(payload, schedule)
        adequacy = first_look["statistical_adequacy"]
        self.assertEqual(adequacy["status"], "collecting")
        self.assertTrue(adequacy["continue"])
        self.assertEqual(adequacy["extension_stage"]["percent"], 50)
        self.assertEqual(adequacy["look_cumulative_alpha"], 0.005)

        payload["runs"] = [record for record in _scored_records_payload(repetitions=None, design=design, powered=True)[1]["runs"]]
        final_look = summarize_records(payload, schedule)
        self.assertEqual(final_look["statistical_adequacy"]["status"], "powered_but_inconclusive")
        self.assertFalse(final_look["statistical_adequacy"]["continue"])
        self.assertTrue(all(
            task["status"] == "powered_but_inconclusive"
            for task in final_look["statistical_adequacy"]["tasks"].values()
        ))

    def test_staged_summary_rejects_unplanned_or_undefined_looks(self):
        design = {"selected_repetitions": 12}
        schedule, payload = _scored_records_payload(repetitions=None, design=design, powered=True)
        payload["runs"] = [record for record in payload["runs"] if record["repetition"] <= 7]
        with self.assertRaisesRegex(ValueError, "planned stage look"):
            summarize_records(payload, schedule)
        with self.assertRaisesRegex(ValueError, "planned stage look"):
            records_workload_result(payload, schedule)

        payload["runs"] = [record for record in payload["runs"] if record["repetition"] <= 6]
        schedule_without_looks = dict(schedule)
        schedule_without_looks.pop("stage_schedule")
        with self.assertRaisesRegex(ValueError, "planned stage look"):
            validate_records(payload, schedule_without_looks, allow_staged_prefix=True)
        with self.assertRaisesRegex(ValueError, "planned stage look"):
            records_workload_result(payload, schedule_without_looks)

        explicit_schedule, explicit_payload = _scored_records_payload(repetitions=2)
        explicit_payload["runs"] = [
            record for record in explicit_payload["runs"] if record["repetition"] == 1
        ]
        with self.assertRaisesRegex(ValueError, "records are missing scheduled runs"):
            records_workload_result(explicit_payload, explicit_schedule)

    def test_schedule_is_reproducible_and_balanced(self):
        first = build_schedule(repetitions=3, seed=17)
        self.assertEqual(first, build_schedule(repetitions=3, seed=17))
        self.assertEqual(len(first["schedule"]), 24)
        self.assertEqual({item["mode"] for item in first["schedule"]}, {"control", "mcp"})
        serialized = json_text(first)
        self.assertNotIn('"prompt":', serialized)
        self.assertNotIn('"commands":', serialized)
        self.assertNotIn('"content":', serialized)

    def test_summary_reports_mode_and_paired_metrics(self):
        schedule, payload = _records_payload()
        summary = summarize_records(payload, schedule)
        self.assertEqual(summary["mode_summaries"]["mcp"]["runs"], 8)
        self.assertEqual(summary["mode_summaries"]["mcp"]["completion_rate"], 1.0)
        self.assertIsNone(summary["mode_summaries"]["mcp"]["invocation_completion_rate"])
        self.assertFalse(summary["paired_deltas_mcp_minus_control"]["completed"]["available"])
        self.assertIsNone(summary["mode_summaries"]["mcp"]["task_success_rate"])
        self.assertFalse(summary["paired_deltas_mcp_minus_control"]["task_success"]["available"])
        self.assertTrue(summary["recommendations"])

    def test_legacy_v5_invocation_completion_uses_exit_code(self):
        schedule, payload = _legacy_v5_records_payload()
        summary = summarize_records(payload, schedule)

        control = summary["mode_summaries"]["control"]
        self.assertEqual(control["completion_rate"], 0.0)
        self.assertEqual(control["invocation_completion_rate"], 1.0)
        self.assertEqual(summary["mode_summaries"]["mcp"]["invocation_completion_rate"], 1.0)
        paired_completion = summary["paired_deltas_mcp_minus_control"]["completed"]
        self.assertEqual(paired_completion["mean"], 0.0)
        self.assertTrue(paired_completion["available"])

        summary_result = summary_workload_result(summary)
        control_result = next(item for item in summary_result["runs"] if item["id"] == "control")
        self.assertEqual(control_result["measurements"]["invocation_completion_rate"]["value"], 1.0)

        run_result = records_workload_result(payload, schedule)
        control_run = next(item for item in run_result["runs"] if item["labels"]["mode"] == "control")
        self.assertEqual(control_run["status"], "partial")
        self.assertEqual(control_run["measurements"]["invocation_completion_rate"]["value"], 1.0)

    def test_summary_separates_invocation_completion_from_objective_task_success(self):
        schedule, payload = _scored_records_payload()
        summary = summarize_records(payload, schedule)
        mcp = summary["mode_summaries"]["mcp"]
        self.assertEqual(mcp["invocation_completion_rate"], 1.0)
        self.assertEqual(mcp["task_success_rate"], 0.5)
        self.assertEqual(mcp["signal_retrieval_rate"], 0.625)
        paired = summary["paired_deltas_mcp_minus_control"]
        self.assertEqual(paired["completed"]["mean"], 0.0)
        self.assertEqual(paired["task_success"]["mean"], 0.5)
        self.assertEqual(summary["schema_version"], 6)
        task_success = summary["task_success_by_mode"]
        self.assertEqual(
            task_success["mcp"]["tasks"]["targeted-inspection"],
            {"successes": 1, "count": 2, "success_rate": 0.5},
        )
        task_delta = summary["paired_task_success_deltas_mcp_minus_control"]["tasks"]["targeted-inspection"]
        self.assertEqual(task_delta["mean"], 0.5)
        self.assertEqual(task_delta["count"], 2)
        self.assertTrue(task_delta["available"])
        self.assertTrue(summary["criterion_summaries_by_mode"]["mcp"]["available"])
        self.assertEqual(
            summary["criterion_summaries_by_mode"]["mcp"]["tasks"]["targeted-inspection"][0]["pass_rate"],
            0.5,
        )
        self.assertTrue(summary["paired_criterion_deltas_mcp_minus_control"]["available"])
        result = summary_workload_result(summary)
        mcp_result = next(item for item in result["runs"] if item["id"] == "mcp")
        self.assertEqual(mcp_result["status"], "partial")
        self.assertEqual(mcp_result["measurements"]["success_rate"]["value"], 0.5)
        self.assertEqual(mcp_result["measurements"]["invocation_completion_rate"]["value"], 1.0)

    def test_criterion_summary_and_paired_deltas_show_task_scoped_indices(self):
        schedule, payload = _scored_records_payload()
        targeted_results = {
            ("control", 1): [True, False],
            ("control", 2): [False, False],
            ("mcp", 1): [True, False],
            ("mcp", 2): [True, True],
        }
        for record in payload["runs"]:
            if record["task_id"] == "targeted-inspection":
                record["criterion_passes"] = targeted_results[(record["mode"], record["repetition"])]

        summary = summarize_records(payload, schedule)
        control = summary["criterion_summaries_by_mode"]["control"]["tasks"]["targeted-inspection"]
        mcp = summary["criterion_summaries_by_mode"]["mcp"]["tasks"]["targeted-inspection"]
        self.assertEqual(control[0], {"criterion_index": 1, "passed": 1, "count": 2, "pass_rate": 0.5})
        self.assertEqual(control[1], {"criterion_index": 2, "passed": 0, "count": 2, "pass_rate": 0.0})
        self.assertEqual(mcp[0], {"criterion_index": 1, "passed": 2, "count": 2, "pass_rate": 1.0})
        self.assertEqual(mcp[1], {"criterion_index": 2, "passed": 1, "count": 2, "pass_rate": 0.5})

        paired = summary["paired_criterion_deltas_mcp_minus_control"]["tasks"]["targeted-inspection"]
        self.assertEqual(paired[0]["mean"], 0.5)
        self.assertEqual(paired[0]["count"], 2)
        self.assertEqual(paired[1]["mean"], 0.5)
        serialized = json.dumps(summary)
        self.assertNotIn("TARGETED_SIGNAL", serialized)
        self.assertNotIn("line 32", serialized)

    def test_response_exposure_cross_tabs_criterion_outcomes_without_content(self):
        schedule, payload = _scored_records_payload()
        targeted_mcp = sorted(
            (
                record for record in payload["runs"]
                if record["task_id"] == "targeted-inspection" and record["mode"] == "mcp"
            ),
            key=lambda record: record["repetition"],
        )
        first, second = targeted_mcp
        first["task_success"] = False
        first["criterion_passes"] = [False, True]
        first["criterion_search_response_hits"] = [True, False]
        first["search_capture_responses"] = 1
        first["criterion_slice_response_hits"] = [False, False]
        first["get_capture_slice_responses"] = 0
        second["task_success"] = True
        second["criterion_passes"] = [True, True]
        second["criterion_search_response_hits"] = [False, True]
        second["search_capture_responses"] = 1
        second["criterion_slice_response_hits"] = [False, True]
        second["get_capture_slice_responses"] = 1

        summary = summarize_records(payload, schedule)
        self.assertEqual(summary["records_schema_version"], RECORDS_SCHEMA_VERSION)
        search_criteria = summary["criterion_exposure_summaries_by_mode"]["mcp"]["sources"]["search_capture"]["tasks"]["targeted-inspection"]
        criterion_one_states = search_criteria[0]["answer_outcomes_by_response_state"]
        self.assertEqual(criterion_one_states["criterion_hit"]["failed"], 1)
        self.assertEqual(criterion_one_states["criterion_hit"]["passed"], 0)
        self.assertEqual(criterion_one_states["response_without_criterion_hit"]["passed"], 1)
        self.assertEqual(criterion_one_states["no_successful_response"]["count"], 0)
        self.assertNotIn("phrase_hit", criterion_one_states)

        slice_criteria = summary["criterion_exposure_summaries_by_mode"]["mcp"]["sources"]["get_capture_slice"]["tasks"]["targeted-inspection"]
        criterion_one_slice = slice_criteria[0]["answer_outcomes_by_response_state"]
        self.assertEqual(criterion_one_slice["no_successful_response"]["failed"], 1)
        self.assertEqual(criterion_one_slice["response_without_criterion_hit"]["passed"], 1)
        self.assertNotIn("TARGETED_SIGNAL", json.dumps(summary))
        self.assertNotIn("line 32", json.dumps(summary))

    def test_agent_ab_report_renders_new_and_legacy_summaries(self):
        report_design = {
            "schema_version": 3,
            "selected_repetitions": 20,
            "scenario_matrix": [{
                "effect_pp": 10,
                "target_power": 0.8,
                "repetitions_per_task_and_mode": 20,
                "expected_paired_score_ci95_half_width_pp": None,
                "invocations_per_variant": 160,
                "estimated_runtime_seconds_per_variant": None,
                "estimated_input_tokens_per_variant": None,
                "estimated_output_tokens_per_variant": None,
            }],
        }
        schedule, payload = _scored_records_payload(design=report_design)
        targeted_mcp = sorted(
            (
                record for record in payload["runs"]
                if record["task_id"] == "targeted-inspection" and record["mode"] == "mcp"
            ),
            key=lambda record: record["repetition"],
        )
        targeted_mcp[0]["task_success"] = False
        targeted_mcp[0]["criterion_passes"] = [False, True]
        targeted_mcp[0]["criterion_search_response_hits"] = [True, False]
        targeted_mcp[0]["search_capture_responses"] = 1
        targeted_mcp[0]["get_capture_slice_responses"] = 0
        targeted_mcp[1]["criterion_passes"] = [True, True]
        targeted_mcp[1]["criterion_search_response_hits"] = [False, True]
        targeted_mcp[1]["search_capture_responses"] = 1
        targeted_mcp[1]["criterion_slice_response_hits"] = [False, True]
        targeted_mcp[1]["get_capture_slice_responses"] = 1
        candidate = summarize_records(payload, schedule)

        legacy_schedule, legacy_payload = _legacy_v6_records_payload()
        legacy = summarize_records(legacy_payload, legacy_schedule)
        legacy["schema_version"] = 2
        legacy.pop("task_success_by_mode")
        legacy.pop("paired_task_success_deltas_mcp_minus_control")
        legacy.pop("criterion_exposure_summaries_by_mode")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            report_path = root / "REPORT.md"
            baseline_path.write_text(json.dumps(legacy), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            report_path.write_text("# Report\n", encoding="utf-8")
            append_agent_ab_report(baseline_path, candidate_path, report_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("## Agent A/B task success", report)
        self.assertIn("## MCP response exposure and answer outcomes", report)
        self.assertIn("## Unbound line-reference diagnostics", report)
        self.assertIn(
            "| baseline | n/a | diagnostics unavailable | n/a | n/a | n/a | n/a |",
            report,
        )
        self.assertIn(
            "| candidate | mcp | targeted-inspection | 2 | 1/2 (50.0%) | 0/2 (0.0%) | 1/2 (50.0%) |",
            report,
        )
        self.assertIn("`criterion_hit` means", report)
        self.assertIn("response_without_criterion_hit", report)
        self.assertIn("no_successful_response", report)
        self.assertIn("unknown (legacy summary)", report)
        self.assertIn("| baseline | task-level data unavailable | n/a | n/a | n/a | unknown (legacy summary) | n/a | n/a |", report)
        self.assertIn("| baseline | task-level data unavailable | n/a | unavailable | n/a | n/a | n/a |", report)
        self.assertIn("| candidate | targeted-inspection | 0/2 | 1/2 | +50.0 pp", report)
        self.assertIn("exploratory", report)
        self.assertIn("Provider token estimates remain unavailable", report)
        self.assertNotIn("TARGETED_SIGNAL", report)
        self.assertNotIn("line 32", report)

    def test_agent_ab_report_reads_legacy_v3_exposure_state_names(self):
        summary = {
            "schema_version": 3,
            "criterion_summaries_by_mode": {
                "mcp": {"tasks": {"targeted-inspection": [{"passed": 1, "count": 1}]}},
            },
            "criterion_exposure_summaries_by_mode": {
                "mcp": {"sources": {"search_capture": {"tasks": {
                    "targeted-inspection": [{
                        "answer_outcomes_by_response_state": {
                            "phrase_hit": {"passed": 1, "failed": 0, "count": 1},
                            "response_without_phrase_hit": {"passed": 0, "failed": 1, "count": 1},
                            "no_successful_response": {"passed": 0, "failed": 0, "count": 0},
                        },
                    }],
                }}}},
            },
        }

        report = render_agent_ab_sections([("legacy", summary)])

        self.assertIn("| Criterion hit (pass/fail) | Response without criterion hit (pass/fail) |", report)
        self.assertIn("| legacy | targeted-inspection | 1 | search_capture | 1/0 | 0/1 | 0/0 |", report)
        self.assertNotIn("phrase_hit", report)

    def test_agent_ab_report_distinguishes_no_unbound_lines_from_legacy_data(self):
        summary = {
            "line_reference_diagnostics_by_mode": {
                mode: {"available": True, "tasks": {"file-bound-task": []}}
                for mode in ("control", "mcp")
            },
        }

        report = render_agent_ab_sections([("current", summary)])

        self.assertIn(
            "| current | n/a | no unbound line criteria | n/a | n/a | n/a | n/a |",
            report,
        )

    def test_agent_ab_report_labels_evidence_ready_inconclusive_and_missing_costs(self):
        def summary(status):
            return {
                "statistical_design": {
                    "schema_version": 5,
                    "selected_repetitions": 900,
                    "pilot_evidence_source": "least-favorable-unknown-baseline-full-discordance",
                    "pilot_required": True,
                    "scenario_matrix": [{
                        "effect_pp": 10,
                        "target_power": 0.8,
                        "simulated_power_by_task": {"example-task": 0.9},
                        "power_lower_bound_by_task": {"example-task": 0.82},
                        "repetitions_per_task_and_mode": 900,
                        "expected_paired_score_ci95_half_width_pp": 6.2,
                        "invocations_per_variant": 7200,
                        "estimated_runtime_seconds_per_variant": None,
                        "estimated_input_tokens_per_variant": None,
                        "estimated_output_tokens_per_variant": None,
                    }],
                },
                "statistical_adequacy": {
                    "status": status,
                    "extension_stage": {"percent": 100, "repetitions": 900},
                    "tasks": {
                        "example-task": {
                            "status": status,
                            "paired_score_interval": {"lower": -0.12, "upper": 0.17},
                            "holm_adjusted_p": 0.31,
                        },
                    },
                },
                "task_success_by_mode": {
                    "control": {"tasks": {"example-task": {"successes": 10, "count": 20}}},
                    "mcp": {"tasks": {"example-task": {"successes": 12, "count": 20}}},
                },
                "paired_task_success_deltas_mcp_minus_control": {
                    "tasks": {"example-task": {"available": True, "mean": 0.1, "ci95_half_width": 0.2}},
                },
            }

        report = render_agent_ab_sections([
            ("resolved", summary("evidence_ready")),
            ("cap", summary("powered_but_inconclusive")),
        ])
        self.assertIn("evidence-ready", report)
        self.assertIn("powered-but-inconclusive", report)
        self.assertIn("Validated minimum power (adjusted lower bound)", report)
        self.assertIn("| resolved | 10 pp | 80.0% | 90.0% (LB 82.0%) | 900 | 6.2 | 7200 | n/a | n/a | n/a |", report)
        self.assertIn("Provider token estimates remain unavailable", report)

    def test_legacy_design_power_is_not_rendered_as_validated(self):
        summary = {
            "statistical_design": {
                "schema_version": 4,
                "scenario_matrix": [{
                    "effect_pp": 10,
                    "target_power": 0.8,
                    "simulated_power_by_task": {"task": 0.95},
                    "power_lower_bound_by_task": {"task": 0.91},
                    "repetitions_per_task_and_mode": 900,
                    "expected_paired_score_ci95_half_width_pp": 6.2,
                    "invocations_per_variant": 7200,
                }],
            },
        }
        report = render_agent_ab_sections([("legacy-design", summary)])
        self.assertIn("| legacy-design | 10 pp | 80.0% | n/a | 900 | 6.2 | 7200 |", report)
        self.assertNotIn("95.0% (LB 91.0%)", report)

    def test_legacy_v6_records_keep_criterion_results_unavailable(self):
        schedule, payload = _legacy_v6_records_payload()
        summary = summarize_records(payload, schedule)
        self.assertEqual(summary["records_schema_version"], 6)
        self.assertEqual(
            summary["criterion_summaries_by_mode"]["mcp"],
            {"available": False, "tasks": {}},
        )
        self.assertEqual(
            summary["paired_criterion_deltas_mcp_minus_control"],
            {"available": False, "tasks": {}},
        )

    def test_validation_rejects_invalid_or_inconsistent_criterion_scores(self):
        schedule, payload = _scored_records_payload()
        payload["runs"][0]["criterion_passes"] = ["private phrase", False]
        with self.assertRaisesRegex(ValueError, "criterion_passes must contain at least two booleans"):
            validate_records(payload, schedule)

        schedule, payload = _scored_records_payload()
        payload["runs"][0]["criterion_passes"].append(False)
        with self.assertRaisesRegex(ValueError, "criterion_passes length must be consistent"):
            validate_records(payload, schedule)

        schedule, payload = _scored_records_payload()
        successful = next(record for record in payload["runs"] if record["task_success"])
        successful["criterion_passes"][0] = False
        with self.assertRaisesRegex(ValueError, "task_success requires every criterion to pass"):
            validate_records(payload, schedule)

    def test_line_reference_diagnostics_are_validated_and_legacy_v11_is_readable(self):
        schedule, payload = _scored_records_payload()
        record = next(
            item for item in payload["runs"]
            if item["task_id"] == "targeted-inspection" and item["mode"] == "mcp"
        )
        original = record["line_reference_diagnostics"]
        invalid_values = (
            [{"criterion_index": 2, "category": "answer text"}],
            [{"criterion_index": 2, "category": {"answer": "text"}}],
            [{"criterion_index": 2, "category": "recognized_reference", "text": "line 32"}],
            [{"criterion_index": True, "category": "recognized_reference"}],
            [{"criterion_index": 3, "category": "recognized_reference"}],
            [
                {"criterion_index": 2, "category": "recognized_reference"},
                {"criterion_index": 2, "category": "expected_number_absent"},
            ],
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                record["line_reference_diagnostics"] = invalid
                with self.assertRaises(ValueError):
                    validate_records(payload, schedule)
        record["line_reference_diagnostics"] = original

        wrong_fixture_schedule = copy.deepcopy(schedule)
        wrong_fixture_payload = copy.deepcopy(payload)
        wrong_fixture_schedule["task_fixture_version"] = 3
        wrong_fixture_payload["task_fixture_version"] = 3
        with self.assertRaisesRegex(ValueError, "fixture-v4/rubric-v5 or rubric-v6"):
            validate_records(wrong_fixture_payload, wrong_fixture_schedule)

        legacy_schedule, legacy_payload = _legacy_v10_records_payload()
        legacy_payload["records_schema_version"] = 11
        summary = summarize_records(legacy_payload, legacy_schedule)
        self.assertEqual(summary["records_schema_version"], 11)
        self.assertEqual(
            summary["line_reference_diagnostics_by_mode"]["mcp"],
            {"available": False, "tasks": {}},
        )

    def test_line_reference_summary_reports_counts_and_rates_by_mode(self):
        schedule, payload = _scored_records_payload()
        for record in payload["runs"]:
            if record["task_id"] != "targeted-inspection":
                continue
            if record["mode"] == "control":
                category = (
                    "recognized_reference"
                    if record["repetition"] == 1
                    else "expected_number_absent"
                )
            else:
                category = (
                    "expected_number_present_unrecognized_format"
                    if record["repetition"] == 1
                    else "expected_number_absent"
                )
            record["line_reference_diagnostics"] = [
                {"criterion_index": 2, "category": category}
            ]

        summary = summarize_records(payload, schedule)
        control = summary["line_reference_diagnostics_by_mode"]["control"]["tasks"]["targeted-inspection"][0]
        mcp = summary["line_reference_diagnostics_by_mode"]["mcp"]["tasks"]["targeted-inspection"][0]
        self.assertEqual(control["count"], 2)
        self.assertEqual(control["categories"]["recognized_reference"], {"count": 1, "rate": 0.5})
        self.assertEqual(control["categories"]["expected_number_absent"], {"count": 1, "rate": 0.5})
        self.assertEqual(mcp["categories"]["expected_number_present_unrecognized_format"], {"count": 1, "rate": 0.5})
        self.assertEqual(mcp["categories"]["expected_number_absent"], {"count": 1, "rate": 0.5})
        self.assertEqual(
            summary["paired_task_success_deltas_mcp_minus_control"]["tasks"]["targeted-inspection"]["count"],
            2,
        )
        self.assertIn("ci95_half_width", summary["paired_task_success_deltas_mcp_minus_control"]["tasks"]["targeted-inspection"])

    def test_summary_excludes_unavailable_rss_measurements(self):
        schedule, payload = _records_payload()
        for record in payload["runs"]:
            if record["mode"] == "control":
                record["peak_rss_bytes"] = 2000
            elif record["repetition"] == 1:
                record["peak_rss_bytes"] = 0
            else:
                record["peak_rss_bytes"] = 3000

        summary = summarize_records(payload, schedule)
        mcp_rss = summary["mode_summaries"]["mcp"]["peak_rss_bytes"]
        paired_rss = summary["paired_deltas_mcp_minus_control"]["peak_rss_bytes"]
        self.assertTrue(mcp_rss["available"])
        self.assertEqual(mcp_rss["count"], 4)
        self.assertEqual(mcp_rss["mean"], 3000.0)
        self.assertTrue(paired_rss["available"])
        self.assertEqual(paired_rss["count"], 4)
        self.assertEqual(paired_rss["mean"], 1000.0)

        for record in payload["runs"]:
            record["peak_rss_bytes"] = 0
        unavailable = summarize_records(payload, schedule)
        self.assertEqual(
            unavailable["mode_summaries"]["mcp"]["peak_rss_bytes"],
            {"available": False, "count": 0},
        )
        self.assertFalse(unavailable["paired_deltas_mcp_minus_control"]["peak_rss_bytes"]["available"])

    def test_validation_rejects_missing_runs(self):
        schedule, payload = _records_payload()
        payload["runs"] = payload["runs"][:-1]
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)

    def test_validation_rejects_raw_fields(self):
        schedule, payload = _records_payload()
        payload["runs"][0]["raw_output"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_records(payload, schedule)

    def test_validation_rejects_json_nonfinite_measurements(self):
        schedule, payload = _records_payload()
        payload["runs"][0]["duration_seconds"] = json.loads("1e999")

        with self.assertRaisesRegex(ValueError, "duration_seconds must be a non-negative number"):
            validate_records(payload, schedule)

    def test_validation_accepts_legacy_protocol_without_embedding_metadata(self):
        schedule, payload = _records_payload()
        for field in ("embedding_mode", "embedding_model", "embedding_cache"):
            del payload["protocol"][field]
        self.assertEqual(len(validate_records(payload, schedule)), len(payload["runs"]))

    def test_current_records_require_resume_protocol_identity(self):
        schedule, payload = _scored_records_payload(design={"selected_repetitions": 2})
        del payload["protocol"]["timeout_seconds"]
        with self.assertRaisesRegex(ValueError, "requires timeout_seconds"):
            validate_records(payload, schedule)

        payload["records_schema_version"] = 10
        payload["task_rubric_version"] = 4
        schedule["task_rubric_version"] = 4
        for record in payload["runs"]:
            record.pop("line_reference_diagnostics")
        self.assertEqual(len(validate_records(payload, schedule)), len(payload["runs"]))

    def test_validation_accepts_version_four_records_and_breaks_out_proxy(self):
        schedule, payload = _records_payload()
        payload["records_schema_version"] = 4
        payload["runs"] = [
            {
                "task_id": record["task_id"],
                "repetition": record["repetition"],
                "mode": record["mode"],
                "completed": record["completed"],
                "signal_retrieved": record["signal_retrieved"],
                "duration_seconds": record["duration_seconds"],
                "tool_calls": record["tool_calls"],
                "repeated_commands": record["repeated_commands"],
                "context_bytes_proxy": record["context_bytes"],
                "peak_rss_bytes": record["peak_rss_bytes"],
                "exit_code": 0,
                "failure_reason": None,
                "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_samples": [100] if record["mode"] == "control" else [100, 120],
                "output_token_samples": [20] if record["mode"] == "control" else [20, 24],
                "prompt_bytes_proxy": 40,
                "output_bytes_proxy": 60 if record["mode"] == "mcp" else 360,
            }
            for record in payload["runs"]
        ]
        summary = summarize_records(payload, schedule)
        self.assertIn("context_bytes_proxy", summary["mode_summaries"]["mcp"])
        self.assertEqual(summary["mode_summaries"]["mcp"]["prompt_bytes_proxy"]["mean"], 40.0)
        self.assertEqual(summary["mode_summaries"]["mcp"]["output_bytes_proxy"]["mean"], 60.0)
        self.assertIn("prompt_bytes_proxy", summary["paired_deltas_mcp_minus_control"])
        usage = summary["mode_summaries"]["mcp"]["usage_accounting"]["input_tokens"]
        self.assertEqual(usage["runs_with_multiple_samples"], 8)
        self.assertEqual(usage["observation"], "monotonic_samples_inconclusive")
        self.assertEqual(summary["records_schema_version"], 4)

    def test_summary_reports_session_data_path_bytes_for_version_five(self):
        schedule, payload = _records_payload()
        payload["records_schema_version"] = 5
        for record in payload["runs"]:
            record.pop("context_bytes", None)
            record.update({
                "context_bytes_proxy": 100 if record["mode"] == "mcp" else 400,
                "exit_code": 0,
                "failure_reason": None,
                "mcp_tool_calls": 1 if record["mode"] == "mcp" else 0,
                "input_tokens": None,
                "output_tokens": None,
                "input_token_samples": [],
                "output_token_samples": [],
                "prompt_bytes_proxy": 40,
                "output_bytes_proxy": 60 if record["mode"] == "mcp" else 360,
            })
            for field in DATA_PATH_BYTE_FIELDS:
                record[field] = 0
            if record["mode"] == "mcp":
                record["capture_input_bytes"] = 1000
                record["capture_retained_bytes"] = 600
                record["tool_response_bytes"] = 120
                record["search_response_bytes"] = 80
                record["retrieval_response_bytes"] = 40

        summary = summarize_records(payload, schedule)
        self.assertEqual(
            summary["mode_summaries"]["mcp"]["data_path_bytes"]["capture_input_bytes"]["mean"],
            1000.0,
        )
        self.assertEqual(
            summary["paired_deltas_mcp_minus_control"]["tool_response_bytes"]["mean"],
            120.0,
        )

    def test_build_schedule_rejects_invalid_repetitions(self):
        with self.assertRaises(ValueError):
            build_schedule(repetitions=0)


def json_text(value):
    """Serialize test metadata for a simple privacy assertion."""
    return json.dumps(value, sort_keys=True)


class TestAgentAbWorkloadResults(unittest.TestCase):
    def test_summary_workload_result_reports_modes_and_paired_deltas(self):
        schedule, payload = _records_payload()
        summary = summarize_records(payload, schedule)
        result = summary_workload_result(summary)
        self.assertEqual(result["workload"]["name"], "agent-ab")
        self.assertEqual(result["workload"]["kind"], "evaluation")
        self.assertEqual(result["workload"]["parameters"]["protocol"]["agent_adapter"], "test-adapter")
        self.assertEqual([item["id"] for item in result["runs"]], ["control", "mcp", "paired-delta"])
        control, mcp, paired = result["runs"]
        self.assertEqual(control["status"], "partial")
        self.assertEqual(mcp["status"], "partial")
        self.assertEqual(mcp["measurements"]["success_rate"], {"unit": "ratio", "value": None, "samples": 0, "note": "legacy records lack objective task-success criteria"})
        self.assertEqual(mcp["measurements"]["wall_time_seconds"]["mean"], 2.0)
        self.assertEqual(mcp["measurements"]["context_bytes"]["note"], benchmark_agent_ab.CONTEXT_PROXY_NOTE)
        # Version-one records carry no token or data-path fields: they are absent, not zero.
        self.assertEqual(mcp["measurements"]["input_tokens"]["value"], None)
        self.assertEqual(mcp["measurements"]["capture_input_bytes"]["value"], None)
        self.assertIsNone(mcp["measurements"]["invocation_completion_rate"]["value"])
        self.assertEqual(mcp["measurements"]["invocation_completion_rate"]["samples"], 0)
        self.assertEqual(paired["labels"], {"comparison": "mcp_minus_control"})
        self.assertIsNone(paired["measurements"]["success_rate"]["value"])
        self.assertIsNone(paired["measurements"]["invocation_completion_rate"]["value"])
        self.assertEqual(paired["measurements"]["wall_time_seconds"]["mean"], -1.0)
        self.assertEqual(result["details"], summary)

        records_result = records_workload_result(payload, schedule)
        control_record = next(item for item in records_result["runs"] if item["labels"]["mode"] == "control")
        self.assertEqual(control_record["status"], "partial")
        self.assertIsNone(control_record["measurements"]["invocation_completion_rate"]["value"])

    def test_records_workload_result_keeps_timeouts_and_unavailable_values(self):
        schedule = build_schedule(repetitions=1, seed=3)
        runs = []
        for item in schedule["schedule"]:
            mcp = item["mode"] == "mcp"
            runs.append({
                "task_id": item["task_id"],
                "repetition": item["repetition"],
                "mode": item["mode"],
                "completed": mcp,
                "signal_retrieved": mcp,
                "duration_seconds": 1.5,
                "tool_calls": 3,
                "repeated_commands": 0,
                "context_bytes_proxy": 250,
                "peak_rss_bytes": 4096 if mcp else 0,
                "exit_code": 0 if mcp else 124,
                "failure_reason": None if mcp else "timeout",
                "mcp_tool_calls": 1 if mcp else 0,
                "input_tokens": 100 if mcp else None,
                "output_tokens": 20 if mcp else None,
            })
        payload = {
            "schema_version": 1,
            "benchmark": "agent-ab",
            "records_schema_version": 2,
            "task_fixture_version": schedule["task_fixture_version"],
            "protocol": {
                "model_config": "m", "repository_fixture": "f", "environment": "e",
                "reset_policy": "r", "agent_adapter": "a",
            },
            "runs": runs,
        }
        result = records_workload_result(payload, schedule, producer="custom-runner")
        self.assertEqual(result["workload"]["kind"], "agent-run")
        self.assertEqual(result["workload"]["producer"], "custom-runner")
        self.assertEqual(result["workload"]["producer_schema_version"], 2)
        self.assertEqual(result["status"], "failure")
        self.assertEqual(len(result["runs"]), 8)
        control = next(item for item in result["runs"] if item["labels"]["mode"] == "control")
        self.assertEqual(control["status"], "timeout")
        self.assertEqual(control["errors"], ["timeout"])
        self.assertIsNone(control["measurements"]["peak_rss_bytes"]["value"])
        self.assertIsNone(control["measurements"]["input_tokens"]["value"])
        mcp = next(item for item in result["runs"] if item["labels"]["mode"] == "mcp")
        self.assertEqual(mcp["id"], f"{mcp['labels']['task_id']}-r1-mcp")
        self.assertEqual(mcp["status"], "partial")
        self.assertIsNone(mcp["measurements"]["success_rate"]["value"])
        self.assertEqual(mcp["measurements"]["invocation_completion_rate"]["value"], 1.0)
        self.assertEqual(mcp["measurements"]["peak_rss_bytes"]["value"], 4096)
        self.assertEqual(mcp["measurements"]["input_tokens"]["value"], 100)
        self.assertEqual(mcp["measurements"]["wall_time_seconds"]["value"], 1.5)
        self.assertNotIn("details", result)
        self.assertNotIn("prompt_bytes", mcp["measurements"])

    def test_schedule_generation_derives_identity_from_pilot_protocol(self):
        _, payload = _scored_records_payload(repetitions=20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot_path = root / "pilot.json"
            output_path = root / "schedule.json"
            pilot_path.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "benchmark_agent_ab.py",
                "--schedule-output", str(output_path),
                "--pilot-records", str(pilot_path),
            ]
            with patch("benchmarks.benchmark_agent_ab.build_power_design", return_value={
                "selected_repetitions": 12,
                "pilot_required": False,
            }), patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO):
                benchmark_agent_ab.main()
            generated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(generated["model_config"], "test-model")
        self.assertEqual(generated["repository_fixture"], "synthetic-repo-v1")

    def test_schedule_cli_defaults_to_exploratory_and_requires_explicit_power_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default_path = root / "default.json"
            powered_path = root / "powered.json"
            design = {"selected_repetitions": 9, "pilot_required": False}
            with patch("benchmarks.benchmark_agent_ab._default_power_design", return_value=design):
                with patch("sys.argv", ["benchmark_agent_ab.py", "--schedule-output", str(default_path)]), patch(
                    "sys.stdout", new_callable=io.StringIO
                ):
                    benchmark_agent_ab.main()
                with patch("sys.argv", ["benchmark_agent_ab.py", "--schedule-output", str(powered_path), "--powered"]), patch(
                    "sys.stdout", new_callable=io.StringIO
                ):
                    benchmark_agent_ab.main()
            default = json.loads(default_path.read_text(encoding="utf-8"))
            powered = json.loads(powered_path.read_text(encoding="utf-8"))
            self.assertEqual(default["repetitions"], 5)
            self.assertEqual(default["run_label"], "exploratory")
            self.assertEqual(default["stage_schedule"][0]["repetitions"], 5)
            self.assertEqual(powered["repetitions"], 9)
            self.assertTrue(powered["powered"])
            self.assertEqual([item["percent"] for item in powered["stage_schedule"]], [50, 75, 100])

            for argv, expected_error in (
                (["benchmark_agent_ab.py", "--schedule-output", str(root / "bad.json"), "--powered", "--repetitions", "2"], "cannot be combined"),
                (["benchmark_agent_ab.py", "--records", str(root / "records.json"), "--powered"], "only valid when generating a schedule"),
            ):
                with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as raised:
                    benchmark_agent_ab.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_error, stderr.getvalue())

    def test_main_result_flag_requires_records_mode(self):
        schedule, payload = _records_payload()
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.json"
            records.write_text(json.dumps(payload), encoding="utf-8")
            argv = ["benchmark_agent_ab.py", "--records", str(records), "--result", "-"]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                benchmark_agent_ab.main()
            result = wr.validate_result(json.loads(stdout.getvalue()))
            self.assertEqual(result["workload"]["name"], "agent-ab")
            self.assertEqual(json.loads(stderr.getvalue())["benchmark"], "agent-ab")

            for extra in (["--result", "-"], ["--experiment", "policy-ab"], ["--metadata", "model=x"], ["--redact", "owner"]):
                argv = ["benchmark_agent_ab.py", "--schedule-output", str(Path(directory) / "s.json"), *extra]
                with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as raised:
                    benchmark_agent_ab.main()
                self.assertEqual(raised.exception.code, 2, extra)
                self.assertIn("require --records", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
