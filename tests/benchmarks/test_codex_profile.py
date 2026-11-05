"""Tests for profile-backed Codex authentication preflight."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.codex_profile import (
    read_api_key,
    validate_codex_profile,
    validate_codex_profile_and_mcp_servers,
)


class TestCodexProfile(unittest.TestCase):
    def _write_configuration(self, home: Path, *, env_key: str = "RUNNER_API_KEY", openai_auth: bool = False) -> None:
        (home / "config.toml").write_text(
            '[model_providers.self_hosted]\n'
            'name = "Self-hosted"\n'
            'base_url = "https://models.example.invalid/v1"\n'
            'wire_api = "responses"\n'
            f'env_key = "{env_key}"\n'
            f'requires_openai_auth = {str(openai_auth).lower()}\n',
            encoding="utf-8",
        )
        (home / "local-eval.config.toml").write_text(
            'model = "custom-model-v1"\n'
            'model_provider = "self_hosted"\n',
            encoding="utf-8",
        )

    def _key_file(self, root: Path, value: str = "test-secret-key\n") -> Path:
        path = root / "api-key"
        path.write_text(value, encoding="utf-8")
        return path

    def _validate(self, home: Path, key_file: Path, *, env: str = "RUNNER_API_KEY") -> dict[str, str]:
        return validate_codex_profile(
            codex_home=str(home),
            profile_name="local-eval",
            api_key_file=str(key_file),
            api_key_env=env,
        )

    def test_returns_only_profile_identity_and_hashes_base_and_profile_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "codex-home"
            home.mkdir()
            self._write_configuration(home)
            key_file = self._key_file(root)

            identity = self._validate(home, key_file)

            self.assertEqual(identity["name"], "local-eval")
            self.assertEqual(identity["model_id"], "custom-model-v1")
            self.assertEqual(identity["provider_id"], "self_hosted")
            self.assertRegex(identity["config_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertNotIn(str(key_file), json.dumps(identity))
            self.assertNotIn("test-secret-key", json.dumps(identity))

            first_hash = identity["config_sha256"]
            (home / "config.toml").write_text(
                (home / "config.toml").read_text(encoding="utf-8")
                .replace("example.invalid", "changed.invalid"),
                encoding="utf-8",
            )
            self.assertNotEqual(first_hash, self._validate(home, key_file)["config_sha256"])

    def test_rejects_missing_empty_and_unreadable_key_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "codex-home"
            home.mkdir()
            self._write_configuration(home)
            with self.assertRaisesRegex(ValueError, "API_KEY_FILE is required"):
                validate_codex_profile(
                    codex_home=str(home),
                    profile_name="local-eval",
                    api_key_file=None,
                    api_key_env="RUNNER_API_KEY",
                )
            with self.assertRaisesRegex(ValueError, "API key file"):
                self._validate(home, root / "missing-key")

            key_file = self._key_file(root, "  \n")
            with self.assertRaisesRegex(ValueError, "API key file"):
                self._validate(home, key_file)

            key_file.write_text("secret", encoding="utf-8")
            key_file.chmod(0)
            with self.assertRaisesRegex(ValueError, "API key file"):
                self._validate(home, key_file)

    def test_rejects_missing_profile_mismatched_env_key_and_openai_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "codex-home"
            home.mkdir()
            key_file = self._key_file(root)

            with self.assertRaisesRegex(ValueError, "profile configuration"):
                self._validate(home, key_file)

            self._write_configuration(home)
            with self.assertRaisesRegex(ValueError, "env_key"):
                self._validate(home, key_file, env="OTHER_API_KEY")

            self._write_configuration(home, openai_auth=True)
            with self.assertRaisesRegex(ValueError, "account authentication"):
                self._validate(home, key_file)

    def test_key_reader_returns_value_without_including_it_in_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._key_file(root, "  secret-value\n")
            self.assertEqual(read_api_key(str(path)), "secret-value")
            with self.assertRaisesRegex(ValueError, "API key file") as caught:
                read_api_key(str(root / "missing"))
            self.assertNotIn(str(root), str(caught.exception))

    def test_profile_validation_returns_merged_mcp_server_names_without_persisting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "codex-home"
            home.mkdir()
            self._write_configuration(home)
            with (home / "config.toml").open("a", encoding="utf-8") as config:
                config.write('\n[mcp_servers.base_only]\ncommand = "base"\n')
                config.write('\n[mcp_servers."vendor.mcp"]\ncommand = "vendor"\n')
            with (home / "local-eval.config.toml").open("a", encoding="utf-8") as profile:
                profile.write('\n[mcp_servers."ephemeral-buffer"]\ncommand = "profile"\n')
                profile.write('\n[mcp_servers.profile_only]\ncommand = "profile"\n')
            key_file = self._key_file(root)

            identity, server_names = validate_codex_profile_and_mcp_servers(
                codex_home=str(home),
                profile_name="local-eval",
                api_key_file=str(key_file),
                api_key_env="RUNNER_API_KEY",
            )

            self.assertEqual(
                server_names,
                ("base_only", "ephemeral-buffer", "profile_only", "vendor.mcp"),
            )
            self.assertEqual(set(identity), {"name", "model_id", "provider_id", "config_sha256"})
            self.assertNotIn("vendor.mcp", json.dumps(identity))

    def test_profile_validation_rejects_malformed_mcp_server_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "codex-home"
            home.mkdir()
            self._write_configuration(home)
            with (home / "local-eval.config.toml").open("a", encoding="utf-8") as profile:
                profile.write('\nmcp_servers = "invalid"\n')
            key_file = self._key_file(root)

            with self.assertRaisesRegex(ValueError, "mcp_servers configuration"):
                validate_codex_profile_and_mcp_servers(
                    codex_home=str(home),
                    profile_name="local-eval",
                    api_key_file=str(key_file),
                    api_key_env="RUNNER_API_KEY",
                )


if __name__ == "__main__":
    unittest.main()
