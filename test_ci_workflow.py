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


if __name__ == "__main__":
    unittest.main()
