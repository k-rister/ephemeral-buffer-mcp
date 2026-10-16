"""Tests for the GitHub Actions CI workflow policy."""

import inspect
import re
import unittest
from pathlib import Path

import ephemeral_buffer_mcp.server as server

from tests.paths import REPOSITORY_ROOT


CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "collect-release-benchmarks.sh"
AGENT_AB_REPORTER = REPOSITORY_ROOT / "benchmarks" / "render_agent_ab_report.py"
TOOL_REFERENCE = REPOSITORY_ROOT / "docs" / "tool-reference.md"


def _registered_signature(function, name):
    """Render the registered argument names and defaults in reference format."""
    parameters = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is inspect.Parameter.empty:
            parameters.append(parameter.name)
        else:
            parameters.append(f"{parameter.name}={parameter.default!r}")
    return f"{name}({', '.join(parameters)})"


class TestCiWorkflow(unittest.TestCase):
    def test_tool_reference_signatures_match_registered_tools(self):
        source = TOOL_REFERENCE.read_text()
        documented = {}
        row_pattern = re.compile(
            r"^\| `(?P<signature>[A-Za-z_]\w*\([^`]*\))` \|",
            re.MULTILINE,
        )
        for match in row_pattern.finditer(source):
            signature = match.group("signature")
            name = signature[:signature.index("(")]
            self.assertNotIn(name, documented, f"duplicate tool-reference row: {name}")
            documented[name] = signature

        registered = {
            name: _registered_signature(function, name)
            for function, name, _category, _result_factory in server._MCP_TOOL_REGISTRATIONS
        }
        self.assertEqual(set(documented), set(registered))
        for name in sorted(registered):
            with self.subTest(tool=name):
                self.assertEqual(documented[name], registered[name])

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

    def test_ci_coverage_suite_includes_transport_decomposition(self):
        source = CI_WORKFLOW.read_text()

        self.assertIn("tests.unit.test_transport_decomposition", source)

    def test_mcp_sdk_compatibility_matrix_checks_supported_2x_endpoints(self):
        source = CI_WORKFLOW.read_text()

        self.assertIn("mcp-sdk-compatibility:", source)
        self.assertIn('mcp-spec: ["mcp==2.3.0", "mcp>=2.3.0,<3"]', source)
        self.assertIn("tests.unit.test_mcpserver_adapter", source)

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
