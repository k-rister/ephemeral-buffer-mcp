"""Tests for the GitHub Actions CI workflow policy."""

import re
import unittest
from pathlib import Path
from tests.paths import REPOSITORY_ROOT


CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "collect-release-benchmarks.sh"
AGENT_AB_REPORTER = REPOSITORY_ROOT / "benchmarks" / "render_agent_ab_report.py"


class TestCiWorkflow(unittest.TestCase):
    def test_pull_request_concurrency_cancels_only_superseded_pr_runs(self):
        source = CI_WORKFLOW.read_text()

        self.assertRegex(
            source,
            re.compile(
                r"(?ms)^concurrency:\n"
                r"  group: \$\{\{ github\.workflow \}\}-"
                r"\$\{\{ github\.event_name \}\}-"
                r"\$\{\{ github\.event\.pull_request\.number \|\| github\.ref \}\}\n"
                r"  cancel-in-progress: \$\{\{ github\.event_name == 'pull_request' \}\}\n"
            ),
        )

    def test_ci_event_triggers_remain_explicit(self):
        source = CI_WORKFLOW.read_text()

        self.assertIn("  pull_request:\n", source)
        self.assertIn("  workflow_dispatch:\n", source)
        self.assertIn("  schedule:\n", source)

    def test_ci_compiles_and_executes_resumable_execution_paths(self):
        source = CI_WORKFLOW.read_text()

        self.assertIn("compileall -q src benchmarks scripts", source)
        self.assertIn("tests.unit.test_execution", source)
        self.assertIn("tests.unit.test_execution_server", source)

    def test_fastmcp_compatibility_matrix_checks_supported_1x_endpoints(self):
        source = CI_WORKFLOW.read_text()

        self.assertIn("fastmcp-compatibility:", source)
        self.assertIn('mcp-spec: ["mcp==1.29.1", "mcp<2"]', source)
        self.assertIn("tests.unit.test_fastmcp_adapter", source)

    def test_ci_compiles_and_executes_workload_result_tooling(self):
        source = CI_WORKFLOW.read_text()

        for module in (
            "tests.benchmarks.test_workload_results",
            "tests.benchmarks.test_compare_workload_results",
            "tests.benchmarks.test_list_workload_results",
        ):
            self.assertIn(module, source)

    def test_release_benchmark_runner_uses_agent_reporter_in_benchmarks_directory(self):
        source = RELEASE_BENCHMARK_SCRIPT.read_text()

        self.assertTrue(AGENT_AB_REPORTER.is_file())
        self.assertIn('"$repo_dir/benchmarks/render_agent_ab_report.py"', source)

    def test_release_notes_import_release_checks_from_scripts_package(self):
        source = RELEASE_WORKFLOW.read_text()

        self.assertIn(
            "from scripts.release_checks import extract_changelog_notes, tag_version",
            source,
        )
        self.assertNotIn("from release_checks import", source)

    def test_runner_temp_paths_are_scoped_to_steps(self):
        source = CI_WORKFLOW.read_text()
        steps = source[source.index("    steps:"):]

        self.assertNotIn("runner.temp", source[:source.index("    steps:")])
        for variable in (
            "EPHEMERAL_SOCKET_PATH",
            "EPHEMERAL_EXECUTION_STATE_DIR",
            "EPHEMERAL_METRICS_FILE",
        ):
            self.assertIn(f"{variable}: ${{{{ runner.temp }}}}", steps)


if __name__ == "__main__":
    unittest.main()
