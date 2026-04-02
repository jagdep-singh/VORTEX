from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config.config import Config
from utils.model_discovery import (
    ModelCatalogResult,
    ModelCatalogSource,
    ModelCatalogStore,
    build_model_catalog_sources,
    flatten_model_catalog,
)


class ModelDiscoveryConfigTests(unittest.TestCase):
    def test_profile_prefers_its_own_env_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"API_KEY": "global-key", "OPENAI_API_KEY": "profile-key"},
            clear=True,
        ):
            config = Config(
                cwd=Path("."),
                models={
                    "openai": {
                        "api_key_env": "OPENAI_API_KEY",
                    }
                },
            )

            self.assertEqual(
                config.resolve_profile_api_key("openai"),
                "profile-key",
            )
            self.assertEqual(
                config.resolve_profile_key_source_label("openai"),
                "env:OPENAI_API_KEY",
            )

    def test_profile_without_explicit_key_uses_global_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"API_KEY": "global-key", "BASE_URL": "https://example.com/v1"},
            clear=True,
        ):
            config = Config(
                cwd=Path("."),
                models={
                    "shared": {
                        "model": {"name": "shared-model"},
                    }
                },
            )

            self.assertEqual(config.resolve_profile_api_key("shared"), "global-key")
            self.assertEqual(
                config.resolve_profile_base_url("shared"),
                "https://example.com/v1",
            )
            self.assertEqual(
                config.resolve_profile_key_source_label("shared"),
                "env:API_KEY",
            )

    def test_build_sources_falls_back_to_default_profile(self) -> None:
        with patch.dict(
            "os.environ",
            {"API_KEY": "default-key"},
            clear=True,
        ):
            config = Config(cwd=Path("."))
            sources = build_model_catalog_sources(config)

            self.assertEqual(len(sources), 1)
            self.assertIsNone(sources[0].profile_name)
            self.assertEqual(sources[0].display_name, "default")
            self.assertEqual(sources[0].api_key, "default-key")

    def test_flatten_model_catalog_assigns_stable_indexes(self) -> None:
        results = [
            ModelCatalogResult(
                source=ModelCatalogSource(
                    profile_name="openai",
                    display_name="openai",
                    base_url="https://api.openai.com/v1",
                    api_key="key-1",
                    key_source_label="env:OPENAI_API_KEY",
                    is_active=True,
                ),
                models=["gpt-5-mini", "gpt-5"],
                checked_at="2026-04-01T00:00:00+00:00",
            ),
            ModelCatalogResult(
                source=ModelCatalogSource(
                    profile_name="other",
                    display_name="other",
                    base_url="https://example.com/v1",
                    api_key="key-2",
                    key_source_label="env:OTHER_API_KEY",
                    is_active=False,
                ),
                models=["other-model"],
                checked_at="2026-04-01T00:00:00+00:00",
            ),
        ]

        entries = flatten_model_catalog(results)

        self.assertEqual([entry.index for entry in entries], [1, 2, 3])
        self.assertEqual(
            [(entry.display_name, entry.model_name) for entry in entries],
            [
                ("openai", "gpt-5-mini"),
                ("openai", "gpt-5"),
                ("other", "other-model"),
            ],
        )
        self.assertEqual([entry.status for entry in entries], ["working", "working", "working"])

    def test_store_returns_cached_result_for_same_source(self) -> None:
        source = ModelCatalogSource(
            profile_name="openai",
            display_name="openai",
            base_url="https://api.openai.com/v1",
            api_key="key-1",
            key_source_label="env:OPENAI_API_KEY",
            is_active=True,
        )

        with TemporaryDirectory() as tmp_dir, patch(
            "utils.model_discovery.get_data_dir",
            return_value=Path(tmp_dir),
        ):
            store = ModelCatalogStore()
            store.save(
                ModelCatalogResult(
                    source=source,
                    models=["gpt-5-mini"],
                    checked_at="2026-04-01T00:00:00+00:00",
                )
            )

            cached = store.get(source)

        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertTrue(cached.cached)
        self.assertEqual(cached.models, ["gpt-5-mini"])
        self.assertEqual(cached.checked_at, "2026-04-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
