"""Tests for the GitHub Actions CI workflow policy."""

import re
import unittest
from pathlib import Path


class TestCiWorkflow(unittest.TestCase):
    def test_pull_request_concurrency_cancels_only_superseded_pr_runs(self):
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()

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
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()

        self.assertIn("  pull_request:\n", source)
        self.assertIn("  workflow_dispatch:\n", source)
        self.assertIn("  schedule:\n", source)

    def test_ci_compiles_and_executes_resumable_execution_paths(self):
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()

        self.assertIn("execution.py", source)
        self.assertIn("test_execution.py", source)
        self.assertIn("test_execution_server.py", source)

    def test_fastmcp_compatibility_matrix_checks_supported_1x_endpoints(self):
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()

        self.assertIn("fastmcp-compatibility:", source)
        self.assertIn('mcp-spec: ["mcp==1.29.1", "mcp<2"]', source)
        self.assertIn("test_fastmcp_adapter.py", source)

    def test_ci_compiles_and_executes_workload_result_tooling(self):
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()

        for module in ("workload_results.py", "compare_workload_results.py", "list_workload_results.py"):
            self.assertIn(f" {module}", source)
            self.assertIn(f" test_{module}", source)

    def test_runner_temp_paths_are_scoped_to_steps(self):
        workflow = Path(__file__).with_name(".github") / "workflows" / "ci.yml"
        source = workflow.read_text()
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
