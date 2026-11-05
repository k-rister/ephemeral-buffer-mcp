"""Validate a Codex profile for agent A/B runs without exposing credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from pathlib import Path
from typing import Any


_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MODEL_ID_RE = re.compile(r"^\S+$")


def _read_config(path: Path, *, required: bool) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        if required:
            raise ValueError("Codex profile configuration was not found") from None
        return {}, b""
    except OSError:
        raise ValueError("Codex configuration is not readable") from None
    try:
        payload = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise ValueError("Codex configuration is not valid TOML") from None
    if not isinstance(payload, dict):
        raise ValueError("Codex configuration must be a TOML table")
    return payload, content


def _merge_tables(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_tables(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_key_file(path_value: str | None) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("AGENT_AB_CODEX_API_KEY_FILE is required")
    path = Path(path_value).expanduser()
    try:
        file_stat = path.stat()
    except OSError:
        raise ValueError("API key file must be readable and nonempty") from None
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or not (file_stat.st_mode & 0o444)
        or not os.access(path, os.R_OK)
    ):
        raise ValueError("API key file must be readable and nonempty")
    try:
        # Validate content without returning or persisting it. The Codex runner
        # reads the key again immediately before creating the Codex child.
        with path.open("rb") as source:
            content = source.read()
            if not content.strip() or b"\0" in content:
                raise ValueError("API key file must be readable and nonempty")
    except OSError:
        raise ValueError("API key file must be readable and nonempty") from None


def validate_codex_profile_and_mcp_servers(
    *,
    codex_home: str | None = None,
    profile_name: str | None = None,
    api_key_file: str | None = None,
    api_key_env: str | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Validate the profile and return its safe identity and MCP server names."""
    codex_home = codex_home or os.environ.get("CODEX_HOME")
    profile_name = profile_name or os.environ.get("AGENT_AB_CODEX_PROFILE")
    api_key_file = api_key_file or os.environ.get("AGENT_AB_CODEX_API_KEY_FILE")
    api_key_env = api_key_env or os.environ.get("AGENT_AB_CODEX_API_KEY_ENV")

    if not isinstance(codex_home, str) or not codex_home.strip():
        raise ValueError("CODEX_HOME must point to a writable Codex home")
    home = Path(codex_home).expanduser()
    if not home.is_dir() or not os.access(home, os.W_OK):
        raise ValueError("CODEX_HOME must point to a writable Codex home")
    if not isinstance(profile_name, str) or not _PROFILE_NAME_RE.fullmatch(profile_name):
        raise ValueError("AGENT_AB_CODEX_PROFILE must be a valid profile name")
    if not isinstance(api_key_env, str) or not _ENVIRONMENT_NAME_RE.fullmatch(api_key_env):
        raise ValueError("AGENT_AB_CODEX_API_KEY_ENV must be a valid environment-variable name")
    _validate_key_file(api_key_file)

    base, base_bytes = _read_config(home / "config.toml", required=False)
    profile, profile_bytes = _read_config(
        home / f"{profile_name}.config.toml",
        required=True,
    )
    effective = _merge_tables(base, profile)
    configured_mcp_servers = effective.get("mcp_servers", {})
    if not isinstance(configured_mcp_servers, dict) or any(
        not isinstance(name, str) or not name
        for name in configured_mcp_servers
    ):
        raise ValueError("Codex mcp_servers configuration must be a table with named servers")

    model_id = effective.get("model")
    provider_id = effective.get("model_provider")
    if not isinstance(model_id, str) or not model_id or not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("Codex profile must select a model")
    if not isinstance(provider_id, str) or not _PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError("Codex profile must select a model provider")
    providers = effective.get("model_providers")
    provider = providers.get(provider_id) if isinstance(providers, dict) else None
    if not isinstance(provider, dict):
        raise ValueError("Codex profile model provider is not configured")
    if provider.get("env_key") != api_key_env:
        raise ValueError("Codex profile provider env_key does not match AGENT_AB_CODEX_API_KEY_ENV")
    if provider.get("requires_openai_auth") is True:
        raise ValueError(
            "Codex profile provider requires OpenAI account authentication instead of env_key"
        )

    digest = hashlib.sha256()
    digest.update(len(base_bytes).to_bytes(8, "big"))
    digest.update(base_bytes)
    digest.update(len(profile_bytes).to_bytes(8, "big"))
    digest.update(profile_bytes)
    return (
        {
            "name": profile_name,
            "model_id": model_id,
            "provider_id": provider_id,
            "config_sha256": f"sha256:{digest.hexdigest()}",
        },
        tuple(sorted(configured_mcp_servers)),
    )


def validate_codex_profile(
    *,
    codex_home: str | None = None,
    profile_name: str | None = None,
    api_key_file: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, str]:
    """Check the profile/provider/key setup and return privacy-safe identity."""
    identity, _ = validate_codex_profile_and_mcp_servers(
        codex_home=codex_home,
        profile_name=profile_name,
        api_key_file=api_key_file,
        api_key_env=api_key_env,
    )
    return identity


def read_api_key(path_value: str | None) -> str:
    """Read a key immediately before launching Codex; never include it in errors."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("AGENT_AB_CODEX_API_KEY_FILE is required")
    path = Path(path_value).expanduser()
    _validate_key_file(path_value)
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        raise ValueError("API key file must be readable and nonempty") from None
    if not key or any(character in key for character in "\0\r\n"):
        raise ValueError("API key file must be readable and nonempty")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    try:
        identity = validate_codex_profile()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    if args.format == "json":
        print(json.dumps(identity, sort_keys=True))
    else:
        print("\t".join(identity[field] for field in (
            "name", "model_id", "provider_id", "config_sha256",
        )))


if __name__ == "__main__":
    main()
