"""Tests for paired binary A/B evidence calculations."""

import unittest
from math import comb
from unittest.mock import patch

from benchmarks.agent_ab_statistics import (
    bootstrap_discordance_upper,
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
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
        self.assertGreaterEqual(design["selected_power_lower_bound"], 0.8)
        self.assertEqual(design["power_simulation"]["power_bound_method"], "one-sided-clopper-pearson")

    def test_selected_power_has_independent_validation_and_retry_alpha_spending(self):
        validation_calls = []

        def validation(**kwargs):
            validation_calls.append(kwargs)
            rejected = [0] if len(validation_calls) == 1 else [kwargs["simulations"]]
            return {
                "repetitions": kwargs["repetitions"],
                "effect": kwargs["effect"],
                "discordance_rates": list(kwargs["discordance_rates"]),
                "simulations": kwargs["simulations"],
                "seed": kwargs["seed"],
                "alpha": kwargs["alpha"],
                "power_by_task": [count / kwargs["simulations"] for count in rejected],
                "rejection_counts_by_task": rejected,
                "familywise_rejection_rate": max(rejected) / kwargs["simulations"],
            }

        with (
            patch("benchmarks.agent_ab_statistics.normal_approximate_repetitions", return_value=6),
            patch("benchmarks.agent_ab_statistics.simulate_paired_mcnemar_power", side_effect=validation),
        ):
            design = select_simulated_repetitions(
                effect=1.0,
                target_power=0.8,
                discordance_rates=[1.0],
                simulations=100,
                seed=7,
            )
        self.assertEqual(design["discovery_candidate_repetitions"], 6)
        self.assertEqual(design["repetitions"], 7)
        self.assertEqual([call["repetitions"] for call in validation_calls], [6, 7])
        self.assertNotEqual(validation_calls[0]["seed"], 7)
        self.assertNotEqual(validation_calls[0]["seed"], validation_calls[1]["seed"])
        self.assertAlmostEqual(design["validation_alpha_spending"], 0.05 / 6)
        self.assertEqual(design["selected_power_confidence_by_task"], [1.0 - 0.05 / 6])

    def test_exact_discordance_upper_bound_has_finite_sample_coverage(self):
        count = 20
        true_probability = 0.2
        coverage = sum(
            comb(count, successes)
            * true_probability**successes
            * (1.0 - true_probability) ** (count - successes)
            for successes in range(count + 1)
            if clopper_pearson_upper_bound(successes, count, 0.975) >= true_probability
        )
        self.assertGreaterEqual(coverage, 0.975)
        # The single-discordance case identified in review exceeds the true
        # 0.20 rate, unlike the under-covering bootstrap percentile.
        self.assertGreater(clopper_pearson_upper_bound(1, count, 0.975), true_probability)

    def test_exact_power_lower_bound_inverts_binomial_tail(self):
        lower = clopper_pearson_lower_bound(90, 100, 0.95)
        self.assertAlmostEqual(lower, 0.8362823767, places=9)

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
        self.assertEqual(first["discordance_upper_method"], "one-sided-clopper-pearson")
        self.assertEqual(first["discordance_upper_estimate"], clopper_pearson_upper_bound(20, 40, 0.95))
        self.assertGreaterEqual(first["discordance_upper_estimate"], first["discordance_upper_bootstrap"])

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
