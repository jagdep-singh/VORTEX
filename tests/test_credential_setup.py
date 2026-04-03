from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config.config import Config
from utils.credential_setup import (
    resolve_api_key_env_name,
    should_prompt_for_base_url,
    split_config_errors,
    suggested_base_url,
    upsert_env_file,
    validate_base_url,
)


class CredentialSetupTests(unittest.TestCase):
    def test_split_config_errors_separates_missing_key_messages(self) -> None:
        credential_errors, other_errors = split_config_errors(
            [
                "No API key found. Set API_KEY or configure [models.<name>] with api_key/api_key_env.",
                "Working directory does not exist: /tmp/missing",
            ]
        )

        self.assertEqual(len(credential_errors), 1)
        self.assertEqual(other_errors, ["Working directory does not exist: /tmp/missing"])

    def test_resolve_api_key_env_name_prefers_active_profile_env(self) -> None:
        config = Config(
            cwd=Path("."),
            active_model_profile="openai",
            models={
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
        )

        self.assertEqual(resolve_api_key_env_name(config), "OPENAI_API_KEY")

    def test_should_prompt_for_base_url_skips_profile_with_explicit_url(self) -> None:
        config = Config(
            cwd=Path("."),
            active_model_profile="gateway",
            models={
                "gateway": {
                    "base_url": "https://example.com/v1",
                }
            },
        )

        self.assertFalse(should_prompt_for_base_url(config))
        self.assertEqual(suggested_base_url(config), "https://example.com/v1")

    def test_validate_base_url_requires_http_scheme_and_host(self) -> None:
        self.assertIsNone(validate_base_url("https://api.openai.com/v1"))
        self.assertIsNotNone(validate_base_url("api.openai.com/v1"))

    def test_upsert_env_file_updates_values_and_deduplicates_targets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                'EXTRA="keep"\nAPI_KEY="old-value"\nAPI_KEY="older-value"\n',
                encoding="utf-8",
            )

            upsert_env_file(
                env_path,
                {
                    "API_KEY": 'new"value',
                    "BASE_URL": "https://api.openai.com/v1",
                },
            )

            content = env_path.read_text(encoding="utf-8")

        self.assertIn('EXTRA="keep"', content)
        self.assertIn('API_KEY="new\\"value"', content)
        self.assertIn('BASE_URL="https://api.openai.com/v1"', content)
        self.assertEqual(content.count("API_KEY="), 1)


if __name__ == "__main__":
    unittest.main()
