import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_effectiveness
import workload_results as wr
from benchmark_effectiveness import (
    run_ab_evaluation,
    run_baseline,
    run_benchmark,
    run_consolidation_benchmark,
    run_mcp,
    run_summary_benchmark,
    scenarios,
    workload_result,
    write_results,
)
from engine import EphemeralEngine
import server


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

    def test_summary_benchmark_measures_representative_capture_outcomes(self):
        record = run_summary_benchmark()
        self.assertEqual(record["benchmark"], "capture-summary")
        self.assertEqual(record["aggregate"]["task_count"], 5)
        self.assertGreater(record["aggregate"]["mean_byte_reduction"], 0)
        self.assertGreater(record["aggregate"]["mean_token_proxy_reduction"], 0)
        self.assertGreater(record["aggregate"]["mean_prompt_byte_reduction"], 0)
        self.assertGreater(record["aggregate"]["mean_prompt_token_proxy_reduction"], 0)
        self.assertTrue(all(item["full_output_available_for_retrieval"] for item in record["records"]))
        self.assertTrue(all(item["retrieval_verified"] for item in record["records"]))
        self.assertTrue(all(item["payload_shapes_aligned"] for item in record["records"]))
        self.assertEqual(
            {item["task_id"] for item in record["records"]},
            {"successful-test", "failed-test", "noisy-build", "truncated-command", "timed-out-command"},
        )

    def test_summary_benchmark_preserves_caller_engine_on_success_and_failure(self):
        original_engine = EphemeralEngine(
            max_captures=2,
            embedding_warmup=False,
            semantic_prefetch=False,
        )
        previous_engine = server.engine
        server.engine = original_engine
        try:
            capture = original_engine.ingest("caller capture", label="caller")
            run_summary_benchmark()
            self.assertIs(server.engine, original_engine)
            self.assertIsNotNone(original_engine.get_capture(capture.capture_id))
            with patch(
                "benchmark_effectiveness.server.execute_and_capture",
                side_effect=RuntimeError("benchmark failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "benchmark failure"):
                    run_summary_benchmark()
            self.assertIs(server.engine, original_engine)
            self.assertIsNotNone(original_engine.get_capture(capture.capture_id))
        finally:
            server.engine = previous_engine
            original_engine.shutdown()


    def test_smoke_workload_result_reports_scenarios_and_aggregate(self):
        result = workload_result(run_benchmark())
        self.assertEqual(result["workload"]["name"], "mcp-effectiveness-smoke")
        self.assertEqual(result["workload"]["kind"], "evaluation")
        self.assertEqual(len(result["runs"]), 2 * len(scenarios()))
        self.assertEqual(result["runs"][0]["id"], "large-build-baseline")
        self.assertEqual(result["runs"][0]["measurements"]["estimated_tokens"]["value"], None)
        self.assertEqual(result["measurements"]["mcp_success_rate"]["value"], 1.0)
        self.assertGreater(result["measurements"]["bytes_examined_reduction"]["value"], 0)
        baseline_only = workload_result(run_benchmark("baseline"))
        self.assertEqual(baseline_only["measurements"], {})
        self.assertEqual(len(baseline_only["runs"]), len(scenarios()))

    def test_paired_ab_workload_result_reports_both_modes_per_scenario(self):
        result = workload_result(run_ab_evaluation(repetitions=2, seed=5))
        self.assertEqual(result["workload"]["name"], "mcp-effectiveness-paired-ab")
        self.assertEqual(result["workload"]["parameters"]["seed"], 5)
        self.assertEqual(len(result["runs"]), 2 * len(scenarios()))
        mcp = next(item for item in result["runs"] if item["labels"]["mode"] == "mcp")
        self.assertEqual(mcp["measurements"]["wall_time_seconds"]["samples"], 2)
        self.assertIn("bytes_examined_reduction", mcp["measurements"])
        self.assertIn("local_overhead_ratio", mcp["measurements"])
        baseline = next(item for item in result["runs"] if item["labels"]["mode"] == "baseline")
        self.assertNotIn("bytes_examined_reduction", baseline["measurements"])
        self.assertEqual(benchmark_effectiveness._rate_status(0.5), "partial")
        self.assertEqual(benchmark_effectiveness._rate_status(0.0), "failure")

    def test_consolidation_workload_result_reports_reductions(self):
        result = workload_result(run_consolidation_benchmark(repetitions=2, seed=5))
        self.assertEqual(result["workload"]["name"], "mcp-effectiveness-consolidation")
        self.assertEqual([item["id"] for item in result["runs"]], ["sequential", "consolidated"])
        self.assertEqual(set(result["measurements"]), {"overview_bytes_reduction", "retrieval_bytes_reduction", "time_ratio"})
        self.assertEqual(result["runs"][0]["measurements"]["success_rate"]["value"], 1.0)

    def test_summary_workload_result_reports_token_proxies_and_verification(self):
        record = run_summary_benchmark()
        result = workload_result(record)
        self.assertEqual(result["workload"]["name"], "capture-summary")
        self.assertEqual(result["status"], "success")
        run = result["runs"][0]
        self.assertEqual(run["labels"]["task_id"], record["records"][0]["task_id"])
        self.assertEqual(run["measurements"]["retained_summary_tokens"]["value"], record["records"][0]["compact_token_proxy"])
        self.assertEqual(run["measurements"]["estimated_tokens"]["note"], benchmark_effectiveness.TOKEN_PROXY_NOTE)
        self.assertEqual(result["measurements"]["task_count"]["value"], len(record["records"]))
        record["records"][0]["retrieval_verified"] = False
        failed = workload_result(record)
        self.assertEqual(failed["status"], "partial")
        self.assertEqual(failed["runs"][0]["status"], "failure")
        self.assertTrue(failed["runs"][0]["errors"])

    def test_result_flag_emits_json_on_stdout(self):
        argv = ["benchmark_effectiveness.py", "--mode", "baseline", "--result", "-"]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            benchmark_effectiveness.main()
        result = wr.validate_result(json.loads(stdout.getvalue()))
        self.assertEqual(result["workload"]["parameters"]["mode"], "baseline")
        self.assertEqual(json.loads(stderr.getvalue())["mode"], "baseline")


if __name__ == "__main__":
    unittest.main()
