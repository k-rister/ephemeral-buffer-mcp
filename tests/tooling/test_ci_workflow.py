"""Tests for the GitHub Actions CI workflow policy."""

import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

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
        self.assertIn("tests.unit.test_semantic_index", source)

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

    def test_release_benchmark_runner_skips_missing_baselines_and_runs_present_assets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = temporary_directory
            repository = os.path.join(root, "repository")
            scripts = os.path.join(repository, "scripts")
            benchmarks = os.path.join(repository, "benchmarks")
            data = os.path.join(benchmarks, "data")
            os.makedirs(scripts)
            os.makedirs(data)
            shutil.copyfile(RELEASE_BENCHMARK_SCRIPT, os.path.join(scripts, "collect-release-benchmarks.sh"))
            for name in ("benchmark_concurrency.py", "benchmark_relevance.py"):
                with open(os.path.join(benchmarks, name), "w", encoding="utf-8"):
                    pass

            def git(*arguments):
                return subprocess.run(
                    ["git", *arguments], cwd=repository, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci-test@example.invalid")
            git("add", ".")
            git("commit", "--quiet", "-m", "baseline without benchmark assets")
            baseline_revision = git("rev-parse", "HEAD")
            for name in ("benchmark_baseline.json", "benchmark_relevance_baseline.json"):
                with open(os.path.join(data, name), "w", encoding="utf-8") as stream:
                    stream.write("{}\n")
            git("add", ".")
            git("commit", "--quiet", "-m", "add benchmark assets")

            fake_python = os.path.join(root, "python")
            fake_python_log = os.path.join(root, "python.log")
            with open(fake_python, "w", encoding="utf-8") as stream:
                stream.write(
                    "#!/usr/bin/env python3\n"
                    "import json, os, sys\n"
                    "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
                    "    print(os.path.realpath(sys.argv[3]))\n"
                    "    raise SystemExit(0)\n"
                    "with open(os.environ['FAKE_PYTHON_LOG'], 'a', encoding='utf-8') as log:\n"
                    "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                )
            os.chmod(fake_python, 0o700)
            codex_home = os.path.join(root, "codex-home")
            binary_directory = os.path.join(root, "bin")
            os.makedirs(codex_home)
            os.makedirs(binary_directory)
            fake_codex = os.path.join(binary_directory, "codex")
            with open(fake_codex, "w", encoding="utf-8") as stream:
                stream.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake_codex, 0o700)
            output_directory = os.path.join(root, "bundle")
            environment = os.environ.copy()
            environment.update({
                "PYTHON_BIN": fake_python,
                "FAKE_PYTHON_LOG": fake_python_log,
                "CODEX_HOME": codex_home,
                "PATH": binary_directory + os.pathsep + environment.get("PATH", ""),
            })
            result = subprocess.run(
                [
                    "bash", os.path.join(scripts, "collect-release-benchmarks.sh"),
                    "--baseline-ref", baseline_revision,
                    "--model", "gpt-5.6-luna",
                    "--output-dir", output_directory,
                    "--samples", "1",
                    "--semantic-samples", "1",
                    "--repetitions", "1",
                    "--agent-ab-repetitions", "1",
                ],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(output_directory, "STATUS.tsv"), encoding="utf-8") as stream:
                statuses = stream.read()
            self.assertIn(
                "skipped\tbaseline-concurrency\t-\tbaseline asset absent at this revision",
                statuses,
            )
            self.assertIn(
                "skipped\tbaseline-relevance\t-\tbaseline asset absent at this revision",
                statuses,
            )
            self.assertIn(
                "skipped\tbaseline-admission\t-\tbenchmark_admission.py is absent at this revision",
                statuses,
            )
            self.assertIn("success\tcandidate-concurrency\t0\t", statuses)
            self.assertIn("success\tcandidate-relevance\t0\t", statuses)
            with open(fake_python_log, encoding="utf-8") as stream:
                invocations = stream.read()
            self.assertIn("benchmarks.benchmark_concurrency", invocations)
            self.assertIn("benchmarks.benchmark_relevance", invocations)
            self.assertIn("benchmark_baseline.json", invocations)
            self.assertIn("benchmark_relevance_baseline.json", invocations)

    def test_agent_ab_runner_reports_missing_codex_authentication(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            codex_home = os.path.join(temporary_directory, "codex-home")
            binary_directory = os.path.join(temporary_directory, "bin")
            os.makedirs(codex_home)
            os.makedirs(binary_directory)
            fake_codex = os.path.join(binary_directory, "codex")
            with open(fake_codex, "w", encoding="utf-8") as stream:
                stream.write("#!/bin/sh\nexit 1\n")
            os.chmod(fake_codex, 0o700)
            environment = os.environ.copy()
            environment.update({
                "CODEX_HOME": codex_home,
                "PATH": binary_directory + os.pathsep + environment.get("PATH", ""),
                "PYTHON_BIN": sys.executable,
            })
            result = subprocess.run(
                ["bash", str(REPOSITORY_ROOT / "benchmarks" / "run_agent_ab_experiment.sh")],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                f"codex CLI is not logged in (CODEX_HOME={codex_home})",
                result.stderr,
            )

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
