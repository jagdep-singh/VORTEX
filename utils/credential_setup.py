from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from config.config import Config

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_API_KEY_ERROR_PREFIX = "No API key found"


def split_config_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    credential_errors: list[str] = []
    other_errors: list[str] = []

    for error in errors:
        if error.startswith(_API_KEY_ERROR_PREFIX):
            credential_errors.append(error)
        else:
            other_errors.append(error)

    return credential_errors, other_errors


def resolve_api_key_env_name(config: Config) -> str:
    profile = config.active_profile
    if profile and profile.api_key_env:
        return profile.api_key_env
    return "API_KEY"


def should_prompt_for_base_url(config: Config) -> bool:
    profile = config.active_profile
    return profile is None or not bool(profile.base_url)


def suggested_base_url(config: Config) -> str:
    candidate = (config.base_url or DEFAULT_BASE_URL).strip()
    return normalize_base_url(candidate) or DEFAULT_BASE_URL


def normalize_base_url(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    return candidate.rstrip("/")


def validate_base_url(value: str) -> str | None:
    normalized = normalize_base_url(value)
    if not normalized:
        return "Provider URL cannot be empty."

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Provider URL must start with http:// or https:// and include a host."

    return None


def upsert_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = {key: value for key, value in values.items()}
    target_keys = list(values)
    rewritten: list[str] = []

    for line in lines:
        matched_key = _match_env_key(line, target_keys)
        if matched_key is None:
            rewritten.append(line)
            continue

        if matched_key in pending:
            rewritten.append(_format_env_assignment(matched_key, pending.pop(matched_key)))

    if rewritten and rewritten[-1].strip():
        rewritten.append("")

    for key, value in pending.items():
        rewritten.append(_format_env_assignment(key, value))

    content = "\n".join(rewritten).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")


def _match_env_key(line: str, keys: Iterable[str]) -> str | None:
    for key in keys:
        pattern = rf"^\s*(?:export\s+)?{re.escape(str(key))}\s*="
        if re.match(pattern, line):
            return str(key)
    return None


def _format_env_assignment(key: str, value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'{key}="{escaped}"'
