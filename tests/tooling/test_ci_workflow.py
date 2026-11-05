"""Tests for the GitHub Actions CI workflow policy."""

import inspect
import json
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


def _write_codex_test_profile(codex_home, key_file):
    os.makedirs(codex_home, exist_ok=True)
    with open(os.path.join(codex_home, "local-eval.config.toml"), "w", encoding="utf-8") as stream:
        stream.write('model = "test-model"\nmodel_provider = "self_hosted"\n')
    with open(os.path.join(codex_home, "config.toml"), "w", encoding="utf-8") as stream:
        stream.write(
            '[model_providers.self_hosted]\n'
            'name = "Self-hosted"\n'
            'base_url = "https://models.example.invalid/v1"\n'
            'wire_api = "responses"\n'
            'env_key = "TEST_MODEL_API_KEY"\n'
        )
    with open(key_file, "w", encoding="utf-8") as stream:
        stream.write("test-model-key\n")
    return {
        "AGENT_AB_CODEX_PROFILE": "local-eval",
        "AGENT_AB_CODEX_API_KEY_FILE": key_file,
        "AGENT_AB_CODEX_API_KEY_ENV": "TEST_MODEL_API_KEY",
    }


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

    def test_real_model_compatibility_lanes_are_schedule_and_manual_only(self):
        source = CI_WORKFLOW.read_text()
        job = source.split("  real-model-compatibility:\n", 1)[1].split("\n  benchmark:", 1)[0]

        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'",
            job,
        )
        self.assertNotIn("pull_request", job.split("    env:\n", 1)[0])
        for lane, python_version, dependency_set in (
            ("py312-hashed-runtime", '"3.12"', "hashed-runtime"),
            ("py311-constrained", '"3.11"', "constrained"),
            ("py312-floating", '"3.12"', "floating"),
        ):
            with self.subTest(lane=lane):
                self.assertRegex(
                    job,
                    re.compile(
                        rf"(?m)^          - lane: {re.escape(lane)}\n"
                        rf"            python-version: {re.escape(python_version)}\n"
                        rf"            dependency-set: {re.escape(dependency_set)}$"
                    ),
                )

        self.assertIn("python -m pip install --require-hashes -r requirements-lock-py312.txt", job)
        self.assertIn(
            "python -m pip install -r requirements.txt -c constraints.txt coverage build pip-audit twine",
            job,
        )
        self.assertRegex(
            job,
            re.compile(r"(?s)floating\)\s+python -m pip install -r requirements\.txt\s+;;"),
        )
        self.assertIn("--ensure-failure-report", job)
        self.assertIn("if: always()", job)
        self.assertIn("actions/upload-artifact", job)
        self.assertIn("benchmark-embedding-${{ runner.os }}-python-", job)

    def test_required_ci_matrix_uses_python_311_and_312(self):
        source = CI_WORKFLOW.read_text()
        test_job = source.split("  test:\n", 1)[1].split("\n  mcp-sdk-compatibility:", 1)[0]

        self.assertIn('python-version: ["3.11", "3.12"]', test_job)
        self.assertNotIn('"3.10"', source)

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
                    "    if len(sys.argv) > 3 and 'json.loads' in sys.argv[2]:\n"
                    "        metadata = json.loads(sys.argv[3])\n"
                    "        field = next(name for name in ('model_id', 'provider_id', 'config_sha256', 'name') if name in sys.argv[2])\n"
                    "        print(metadata[field])\n"
                    "        raise SystemExit(0)\n"
                    "    print(os.path.realpath(sys.argv[3]))\n"
                    "    raise SystemExit(0)\n"
                    "if len(sys.argv) > 2 and sys.argv[1:3] == ['-m', 'benchmarks.codex_profile']:\n"
                    "    print(json.dumps({'name': 'local-eval', 'model_id': 'test-model', 'provider_id': 'self_hosted', 'config_sha256': 'sha256:' + 'a' * 64}))\n"
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
                stream.write("#!/bin/sh\n[ \"$1\" = login ] && exit 1\nexit 0\n")
            os.chmod(fake_codex, 0o700)
            profile_environment = _write_codex_test_profile(
                codex_home, os.path.join(root, "api-key")
            )
            output_directory = os.path.join(root, "bundle")
            environment = os.environ.copy()
            environment.update({
                "PYTHON_BIN": fake_python,
                "FAKE_PYTHON_LOG": fake_python_log,
                "CODEX_HOME": codex_home,
                "PATH": binary_directory + os.pathsep + environment.get("PATH", ""),
                **profile_environment,
            })
            result = subprocess.run(
                [
                    "bash", os.path.join(scripts, "collect-release-benchmarks.sh"),
                    "--baseline-ref", baseline_revision,
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

    def test_agent_ab_runner_requires_profile_before_generating_fixture(self):
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
                "AGENT_AB_RUN_DIR": os.path.join(temporary_directory, "run-output"),
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
            self.assertIn("AGENT_AB_CODEX_PROFILE", result.stderr)
            self.assertFalse(os.path.exists(os.path.join(temporary_directory, "run-output")))

    def test_agent_ab_runner_rejects_bad_key_and_provider_before_generating_fixture(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            codex_home = os.path.join(temporary_directory, "codex-home")
            binary_directory = os.path.join(temporary_directory, "bin")
            key_file = os.path.join(temporary_directory, "api-key")
            os.makedirs(binary_directory)
            profile_environment = _write_codex_test_profile(codex_home, key_file)
            fake_codex = os.path.join(binary_directory, "codex")
            with open(fake_codex, "w", encoding="utf-8") as stream:
                stream.write("#!/bin/sh\n[ \"$1\" = login ] && exit 1\nexit 0\n")
            os.chmod(fake_codex, 0o700)
            run_directory = os.path.join(temporary_directory, "run-output")
            environment = os.environ.copy()
            environment.update({
                "CODEX_HOME": codex_home,
                "PATH": binary_directory + os.pathsep + environment.get("PATH", ""),
                "PYTHON_BIN": sys.executable,
                "AGENT_AB_RUN_DIR": run_directory,
                **profile_environment,
            })

            with open(key_file, "w", encoding="utf-8") as stream:
                stream.write("  \n")
            empty_key_result = subprocess.run(
                ["bash", str(REPOSITORY_ROOT / "benchmarks" / "run_agent_ab_experiment.sh")],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(empty_key_result.returncode, 2)
            self.assertIn("API key file", empty_key_result.stderr)
            self.assertFalse(os.path.exists(run_directory))

            with open(key_file, "w", encoding="utf-8") as stream:
                stream.write("test-model-key\n")
            environment["AGENT_AB_CODEX_API_KEY_ENV"] = "WRONG_API_KEY"
            provider_result = subprocess.run(
                ["bash", str(REPOSITORY_ROOT / "benchmarks" / "run_agent_ab_experiment.sh")],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(provider_result.returncode, 2)
            self.assertIn("env_key", provider_result.stderr)
            self.assertFalse(os.path.exists(run_directory))

    def test_agent_ab_wrapper_defaults_to_exploratory_and_forwards_opt_ins(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = os.path.abspath(temporary_directory)
            codex_home = os.path.join(root, "codex-home")
            binary_directory = os.path.join(root, "bin")
            os.makedirs(codex_home)
            os.makedirs(binary_directory)
            fake_codex = os.path.join(binary_directory, "codex")
            with open(fake_codex, "w", encoding="utf-8") as stream:
                stream.write("#!/bin/sh\n[ \"$1\" = login ] && exit 1\nexit 0\n")
            os.chmod(fake_codex, 0o700)

            python_shim = os.path.join(root, "python-shim")
            with open(python_shim, "w", encoding="utf-8") as stream:
                stream.write(f"#!{sys.executable}\n")
                stream.write(
                    "import json, os, pathlib, subprocess, sys\n"
                    "args = sys.argv[1:]\n"
                    "if args and args[0] == '-c':\n"
                    "    raise SystemExit(subprocess.call([os.environ['REAL_PYTHON'], *args]))\n"
                    "if args and args[0] == '-B':\n"
                    "    args = args[1:]\n"
                    "if len(args) < 2 or args[0] != '-m':\n"
                    "    raise SystemExit(2)\n"
                    "module, call_args = args[1], args[2:]\n"
                    "with open(os.environ['AGENT_AB_TEST_CALLS'], 'a', encoding='utf-8') as log:\n"
                    "    log.write(json.dumps({'module': module, 'args': call_args}) + '\\n')\n"
                    "if module == 'benchmarks.codex_profile':\n"
                    "    raise SystemExit(subprocess.call([os.environ['REAL_PYTHON'], '-m', module, *call_args]))\n"
                    "def value(flag):\n"
                    "    return call_args[call_args.index(flag) + 1]\n"
                    "if module in {'benchmarks.benchmark_agent_ab_fixtures', 'benchmarks.benchmark_agent_ab_repository_fixture'}:\n"
                    "    pathlib.Path(value('--fixture-output')).mkdir(parents=True, exist_ok=True)\n"
                    "    pathlib.Path(value('--manifest-output')).write_text('{}', encoding='utf-8')\n"
                    "elif module == 'benchmarks.benchmark_agent_ab' and '--schedule-output' in call_args:\n"
                    "    count = 8 if '--powered' in call_args else int(value('--repetitions')) if '--repetitions' in call_args else 5\n"
                    "    looks = [4, 8] if '--powered' in call_args else [count]\n"
                    "    pathlib.Path(value('--schedule-output')).write_text(json.dumps({'stage_schedule': [{'repetitions': n} for n in looks]}), encoding='utf-8')\n"
                    "elif module == 'benchmarks.run_codex_agent_ab':\n"
                    "    output = pathlib.Path(value('--output'))\n"
                    "    output.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    output.write_text('{}', encoding='utf-8')\n"
                    "    pathlib.Path(value('--result')).write_text('{}', encoding='utf-8')\n"
                    "    with open(os.environ['AGENT_AB_TEST_CALLS'], 'a', encoding='utf-8') as log:\n"
                    "        log.write(json.dumps({'runner_profile': os.environ.get('AGENT_AB_CODEX_PROFILE'), 'key_env': os.environ.get('AGENT_AB_CODEX_API_KEY_ENV'), 'key_file_configured': bool(os.environ.get('AGENT_AB_CODEX_API_KEY_FILE'))}) + '\\n')\n"
                    "elif module == 'benchmarks.benchmark_agent_ab' and '--records' in call_args:\n"
                    "    pathlib.Path(value('--output')).write_text(json.dumps({'statistical_adequacy': {'continue': False}}), encoding='utf-8')\n"
                    "    pathlib.Path(value('--result')).write_text('{}', encoding='utf-8')\n"
                    "else:\n"
                    "    raise SystemExit(f'unexpected fake module: {module}')\n"
                )
            os.chmod(python_shim, 0o700)
            api_key_file = os.path.join(root, "api-key")
            profile_environment = _write_codex_test_profile(codex_home, api_key_file)

            def run_case(name, overrides):
                run_dir = os.path.join(root, name)
                call_log = os.path.join(root, f"{name}.jsonl")
                environment = os.environ.copy()
                for variable in ("AGENT_AB_POWERED", "AGENT_AB_REPETITIONS"):
                    environment.pop(variable, None)
                environment.update({
                    "CODEX_HOME": codex_home,
                    "PATH": binary_directory + os.pathsep + environment.get("PATH", ""),
                    "PYTHON_BIN": python_shim,
                    "REAL_PYTHON": sys.executable,
                    "AGENT_AB_TEST_CALLS": call_log,
                    "AGENT_AB_RUN_DIR": run_dir,
                    **profile_environment,
                    **overrides,
                })
                result = subprocess.run(
                    ["bash", str(REPOSITORY_ROOT / "benchmarks" / "run_agent_ab_experiment.sh")],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return result, call_log

            for name, overrides, expected_size, expected_flag, expected_count in (
                ("default", {}, "5", None, None),
                ("custom", {"AGENT_AB_REPETITIONS": "3"}, "3", "--repetitions", "3"),
                ("powered", {"AGENT_AB_POWERED": "1"}, "powered", "--powered", None),
            ):
                with self.subTest(name=name):
                    result, call_log = run_case(name, overrides)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    with open(call_log, encoding="utf-8") as call_stream:
                        calls = [json.loads(line) for line in call_stream]
                    schedule = next(
                        call for call in calls
                        if call["module"] == "benchmarks.benchmark_agent_ab"
                        and "--schedule-output" in call["args"]
                    )
                    run_call = next(call for call in calls if call["module"] == "benchmarks.run_codex_agent_ab")
                    self.assertIn(f"workload_size={expected_size}", run_call["args"])
                    if expected_flag is None:
                        self.assertNotIn("--powered", schedule["args"])
                        self.assertNotIn("--repetitions", schedule["args"])
                    else:
                        index = schedule["args"].index(expected_flag)
                        if expected_count is not None:
                            self.assertEqual(schedule["args"][index + 1], expected_count)
                    runner_settings = [call for call in calls if "runner_profile" in call]
                    self.assertTrue(runner_settings)
                    self.assertTrue(all(call["runner_profile"] == "local-eval" for call in runner_settings))
                    self.assertTrue(all(call["key_env"] == "TEST_MODEL_API_KEY" for call in runner_settings))
                    self.assertTrue(all(call["key_file_configured"] for call in runner_settings))

            conflict_result, conflict_log = run_case(
                "conflict",
                {"AGENT_AB_POWERED": "1", "AGENT_AB_REPETITIONS": "3"},
            )
            self.assertEqual(conflict_result.returncode, 2)
            self.assertIn("cannot be combined", conflict_result.stderr)
            self.assertFalse(os.path.exists(conflict_log))

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
