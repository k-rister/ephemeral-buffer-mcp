import json
import tempfile
import unittest
from pathlib import Path

from benchmark_effectiveness import (
    run_ab_evaluation,
    run_baseline,
    run_benchmark,
    run_consolidation_benchmark,
    run_mcp,
    scenarios,
    write_results,
)
from engine import EphemeralEngine


class TestEffectivenessBenchmark(unittest.TestCase):
    def test_fixtures_are_reproducible_and_content_free(self):
        first = scenarios()
        second = scenarios()
        self.assertEqual(first, second)
        self.assertTrue(all("synthetic" in "\n".join(item["lines"]) for item in first))

    def test_baseline_resolves_every_fixture(self):
        results = [run_baseline(scenario) for scenario in scenarios()]
        self.assertTrue(all(result["success"] for result in results))
        self.assertTrue(all(result["bytes_examined"] == result["bytes_retrieved"] for result in results))

    def test_mcp_workflow_resolves_every_fixture_with_useful_searches(self):
        engine = EphemeralEngine(max_captures=len(scenarios()))
        results = [run_mcp(scenario, engine) for scenario in scenarios()]
        self.assertTrue(all(result["success"] for result in results))
        self.assertTrue(all(result["search_useful"] for result in results))
        self.assertTrue(all(result["retrievals"] == 1 for result in results))

    def test_both_mode_has_machine_readable_comparison(self):
        record = run_benchmark()
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["aggregate"]["scenario_count"], len(scenarios()))
        self.assertEqual(record["aggregate"]["mcp_successes"], len(scenarios()))
        self.assertGreater(record["aggregate"]["mean_bytes_reduction"], 0)

    def test_results_can_be_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "effectiveness.json"
            write_results(path, run_benchmark("baseline"))
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["mode"], "baseline")
        self.assertIsNone(record["aggregate"])

    def test_ab_evaluation_is_reproducible_and_paired(self):
        first = run_ab_evaluation(repetitions=3, seed=17)
        second = run_ab_evaluation(repetitions=3, seed=17)
        self.assertEqual(first["schedules"], second["schedules"])
        self.assertEqual(len(first["records"]), 3 * len(scenarios()) * 2)
        self.assertTrue(all(comparison["baseline"]["runs"] == 3 for comparison in first["comparisons"]))
        self.assertTrue(all(comparison["mcp"]["runs"] == 3 for comparison in first["comparisons"]))
        self.assertEqual(first["controls"]["telemetry"], "none; all measurements are local")

    def test_ab_evaluation_reports_variation_and_recommendations(self):
        record = run_ab_evaluation(repetitions=2)
        comparison = record["comparisons"][0]
        self.assertIn("stdev_time_seconds", comparison["baseline"])
        self.assertIn("mean_bytes_reduction", comparison)
        self.assertTrue(record["recommendations"])

    def test_ab_evaluation_rejects_non_positive_repetitions(self):
        with self.assertRaises(ValueError):
            run_ab_evaluation(repetitions=0)

    def test_consolidation_benchmark_is_paired_and_successful(self):
        first = run_consolidation_benchmark(repetitions=2, seed=17)
        second = run_consolidation_benchmark(repetitions=2, seed=17)

        self.assertEqual(first["schedules"], second["schedules"])
        self.assertEqual(len(first["records"]), 4)
        self.assertEqual(first["summaries"]["sequential"]["success_rate"], 1.0)
        self.assertEqual(first["summaries"]["consolidated"]["success_rate"], 1.0)
        self.assertGreater(first["comparison"]["overview_bytes_reduction"], 0)
        self.assertTrue(all(record["retrievals"] == len(scenarios()) for record in first["records"]))

    def test_consolidation_benchmark_rejects_non_positive_repetitions(self):
        with self.assertRaises(ValueError):
            run_consolidation_benchmark(repetitions=0)


if __name__ == "__main__":
    unittest.main()
