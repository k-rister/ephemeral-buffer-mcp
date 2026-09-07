import json
import tempfile
import unittest
from pathlib import Path

from benchmark_effectiveness import run_baseline, run_benchmark, run_mcp, scenarios, write_results
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


if __name__ == "__main__":
    unittest.main()
