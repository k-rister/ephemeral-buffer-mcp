"""Tests for paired binary A/B evidence calculations."""

import unittest
from unittest.mock import patch

from benchmarks.agent_ab_statistics import (
    bootstrap_discordance_upper,
    estimate_pair_probabilities,
    exact_mcnemar_pvalue,
    holm_adjust,
    newcombe_paired_score_interval,
    select_simulated_repetitions,
    simulate_paired_interval_coverage,
    simulate_paired_mcnemar_power,
)


class TestAgentAbStatistics(unittest.TestCase):
    def test_exact_mcnemar_is_symmetric_and_uses_only_discordant_pairs(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_pvalue(3, 1), exact_mcnemar_pvalue(1, 3))
        self.assertAlmostEqual(exact_mcnemar_pvalue(10, 0), 2.0 / 1024)
        self.assertAlmostEqual(exact_mcnemar_pvalue(2113, 1902), 0.0009168552, places=9)

    def test_holm_adjustment_preserves_order_and_controls_family(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.01, None, 0.03]), [0.02, None, 0.03])

    def test_paired_model_and_newcombe_interval(self):
        model = estimate_pair_probabilities(0.5, 0.1, 0.9)
        self.assertAlmostEqual(sum(model.values()), 1.0)
        self.assertAlmostEqual(model["mcp_only_success"] - model["control_only_success"], 0.1)
        interval = newcombe_paired_score_interval(
            both_successes=40,
            mcp_only_successes=20,
            control_only_successes=10,
            count=100,
        )
        self.assertGreater(interval["estimate"], 0.0)
        self.assertLess(interval["lower"], interval["estimate"])
        self.assertGreater(interval["upper"], interval["estimate"])
        self.assertEqual(interval["method"], "Newcombe-1998-method-10-paired-score")

    def test_holm_familywise_error_is_bounded_under_paired_null(self):
        result = simulate_paired_mcnemar_power(
            repetitions=24,
            effect=0.0,
            discordance_rates=[0.4] * 4,
            simulations=2500,
            seed=17,
        )
        self.assertLessEqual(result["familywise_rejection_rate"], 0.065)
        self.assertGreaterEqual(result["familywise_rejection_rate"], 0.02)

    def test_power_simulation_is_seeded_and_selects_an_adequate_size(self):
        first = simulate_paired_mcnemar_power(
            repetitions=80,
            effect=0.1,
            discordance_rates=[0.9] * 4,
            simulations=300,
            seed=29,
        )
        second = simulate_paired_mcnemar_power(
            repetitions=80,
            effect=0.1,
            discordance_rates=[0.9] * 4,
            simulations=300,
            seed=29,
        )
        self.assertEqual(first, second)
        design = select_simulated_repetitions(
            effect=0.1,
            target_power=0.8,
            discordance_rates=[0.9] * 4,
            simulations=150,
            seed=31,
        )
        self.assertGreater(design["repetitions"], 0)
        self.assertGreaterEqual(design["selected_power_lower_95"], 0.8)

    def test_repetition_search_returns_first_qualifying_candidate(self):
        calls = 0

        def nonmonotonic_rejections(_p_values, _alpha):
            nonlocal calls
            candidate = calls // 100 + 1
            calls += 1
            # Candidate 2 qualifies, candidate 3 would fail, and candidate 4
            # would qualify again. The search must stop at the first pass.
            return [candidate == 2 or candidate == 4] * 2

        with patch(
            "benchmarks.agent_ab_statistics._holm_rejections",
            side_effect=nonmonotonic_rejections,
        ):
            design = select_simulated_repetitions(
                effect=0.1,
                target_power=0.8,
                discordance_rates=[0.5, 0.5],
                simulations=100,
                seed=7,
            )
        self.assertEqual(design["repetitions"], 2)
        self.assertEqual(calls, 200)
        self.assertIn("ascending scan", design["selection_rule"])

    def test_paired_score_interval_has_simulated_coverage(self):
        coverage = simulate_paired_interval_coverage(
            repetitions=80,
            effect=0.15,
            discordance=0.55,
            simulations=2000,
            seed=8,
        )
        self.assertGreaterEqual(coverage["coverage"], 0.90)
        self.assertLessEqual(coverage["coverage"], 0.98)

    def test_bootstrap_uses_matched_pairs_and_is_reproducible(self):
        pairs = [(True, True), (True, False), (False, True), (False, False)] * 10
        first = bootstrap_discordance_upper(pairs, simulations=1000, seed=41)
        second = bootstrap_discordance_upper(pairs, simulations=1000, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(first["matched_repetitions"], 40)
        self.assertAlmostEqual(first["baseline_success_rate"], 0.5)
        self.assertGreaterEqual(first["discordance_upper_bootstrap"], 0.5)
        self.assertEqual(first["discordance_upper_method"], "paired-bootstrap-percentile")
        self.assertEqual(first["discordance_upper_estimate"], first["discordance_upper_bootstrap"])

    def test_zero_discordance_uses_one_sided_exact_upper_bound(self):
        pairs = [(True, True)] * 10 + [(False, False)] * 10
        result = bootstrap_discordance_upper(pairs, simulations=1000, seed=19)
        expected = 1.0 - 0.05 ** (1.0 / len(pairs))
        self.assertEqual(result["observed_discordance_rate"], 0.0)
        self.assertEqual(result["discordance_upper_bootstrap"], 0.0)
        self.assertAlmostEqual(result["discordance_upper_estimate"], expected)
        self.assertEqual(result["discordance_upper_method"], "one-sided-clopper-pearson")


if __name__ == "__main__":
    unittest.main()
