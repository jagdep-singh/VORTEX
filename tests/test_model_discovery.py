from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config.config import Config
from utils.model_discovery import (
    MODEL_BUCKET_ORDER,
    ModelCatalogResult,
    ModelCatalogSource,
    ModelCatalogStore,
    build_model_catalog_sources,
    flatten_model_catalog,
    group_model_catalog_entries_by_bucket,
    model_status_bucket,
    order_model_catalog_entries,
    select_best_working_model,
)
from utils.model_health import ModelHealthRecord


class ModelDiscoveryConfigTests(unittest.TestCase):
    def test_model_status_bucket_collapses_detailed_errors(self) -> None:
        self.assertEqual(model_status_bucket("working"), "working")
        self.assertEqual(model_status_bucket("limited"), "limited")
        self.assertEqual(model_status_bucket("quota"), "quota")
        self.assertEqual(model_status_bucket("rate-limited"), "quota")
        self.assertEqual(model_status_bucket("auth-error"), "not-working")
        self.assertEqual(model_status_bucket("unknown"), "not-working")
        self.assertEqual(MODEL_BUCKET_ORDER[:2], ("working", "limited"))

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
        self.assertEqual([entry.status for entry in entries], ["unknown", "unknown", "unknown"])

    def test_flatten_model_catalog_uses_probe_health_when_available(self) -> None:
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
            )
        ]

        health_records = {
            ("openai", "gpt-5-mini"): ModelHealthRecord(
                provider="openai",
                model="gpt-5-mini",
                status="working",
                checked_at="2026-04-01T01:00:00+00:00",
                detail=None,
            ),
            ("openai", "gpt-5"): ModelHealthRecord(
                provider="openai",
                model="gpt-5",
                status="quota",
                checked_at="2026-04-01T01:05:00+00:00",
                detail="Quota exceeded",
            ),
        }

        entries = flatten_model_catalog(results, health_records=health_records)

        self.assertEqual([entry.status for entry in entries], ["working", "quota"])
        self.assertEqual(
            [entry.checked_at for entry in entries],
            ["2026-04-01T01:00:00+00:00", "2026-04-01T01:05:00+00:00"],
        )
        self.assertEqual([entry.note for entry in entries], [None, "Quota exceeded"])

    def test_select_best_working_model_prefers_exact_current_match(self) -> None:
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
                models=["gpt-5", "gpt-5-mini"],
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
        health_records = {
            ("openai", "gpt-5"): ModelHealthRecord(
                provider="openai",
                model="gpt-5",
                status="working",
                checked_at="2026-04-01T01:00:00+00:00",
            ),
            ("openai", "gpt-5-mini"): ModelHealthRecord(
                provider="openai",
                model="gpt-5-mini",
                status="working",
                checked_at="2026-04-01T01:05:00+00:00",
            ),
            ("other", "other-model"): ModelHealthRecord(
                provider="other",
                model="other-model",
                status="working",
                checked_at="2026-04-01T01:10:00+00:00",
            ),
        }

        selected = select_best_working_model(
            results,
            health_records=health_records,
            preferred_profile_name="openai",
            preferred_model_name="gpt-5-mini",
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.profile_name, "openai")
        self.assertEqual(selected.model_name, "gpt-5-mini")

    def test_select_best_working_model_falls_back_to_any_working_model(self) -> None:
        results = [
            ModelCatalogResult(
                source=ModelCatalogSource(
                    profile_name="broken",
                    display_name="broken",
                    base_url="https://broken.example/v1",
                    api_key="key-1",
                    key_source_label="env:BROKEN_API_KEY",
                    is_active=True,
                ),
                models=["broken-model"],
                checked_at="2026-04-01T00:00:00+00:00",
            ),
            ModelCatalogResult(
                source=ModelCatalogSource(
                    profile_name="healthy",
                    display_name="healthy",
                    base_url="https://healthy.example/v1",
                    api_key="key-2",
                    key_source_label="env:HEALTHY_API_KEY",
                    is_active=False,
                ),
                models=["healthy-model"],
                checked_at="2026-04-01T00:00:00+00:00",
            ),
        ]
        health_records = {
            ("broken", "broken-model"): ModelHealthRecord(
                provider="broken",
                model="broken-model",
                status="quota",
                checked_at="2026-04-01T01:00:00+00:00",
            ),
            ("healthy", "healthy-model"): ModelHealthRecord(
                provider="healthy",
                model="healthy-model",
                status="working",
                checked_at="2026-04-01T01:05:00+00:00",
            ),
        }

        selected = select_best_working_model(
            results,
            health_records=health_records,
            preferred_profile_name="broken",
            preferred_model_name="broken-model",
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.profile_name, "healthy")
        self.assertEqual(selected.model_name, "healthy-model")

    def test_group_model_catalog_entries_by_bucket_preserves_bucket_order(self) -> None:
        entries = [
            *flatten_model_catalog(
                [
                    ModelCatalogResult(
                        source=ModelCatalogSource(
                            profile_name="demo",
                            display_name="demo",
                            base_url="https://example.com/v1",
                            api_key="key-1",
                            key_source_label="env:API_KEY",
                            is_active=True,
                        ),
                        models=["quota-model", "broken-model", "working-model"],
                        checked_at="2026-04-01T00:00:00+00:00",
                    )
                ],
                health_records={
                    ("demo", "quota-model"): ModelHealthRecord(
                        provider="demo",
                        model="quota-model",
                        status="quota",
                        checked_at="2026-04-01T01:00:00+00:00",
                    ),
                    ("demo", "broken-model"): ModelHealthRecord(
                        provider="demo",
                        model="broken-model",
                        status="auth-error",
                        checked_at="2026-04-01T01:05:00+00:00",
                    ),
                    ("demo", "working-model"): ModelHealthRecord(
                        provider="demo",
                        model="working-model",
                        status="working",
                        checked_at="2026-04-01T01:10:00+00:00",
                    ),
                },
            )
        ]
        ordered_entries = order_model_catalog_entries(
            entries,
            active_profile_name="demo",
            active_model_name="working-model",
        )

        grouped = group_model_catalog_entries_by_bucket(ordered_entries)

        self.assertEqual(
            [bucket for bucket, _entries in grouped],
            [bucket for bucket in MODEL_BUCKET_ORDER if any(model_status_bucket(entry.status) == bucket for entry in ordered_entries)],
        )
        self.assertEqual(
            [[entry.model_name for entry in bucket_entries] for _bucket, bucket_entries in grouped],
            [["working-model"], ["quota-model"], ["broken-model"]],
        )

    def test_order_model_catalog_entries_sorts_by_status_priority(self) -> None:
        results = [
            ModelCatalogResult(
                source=ModelCatalogSource(
                    profile_name="mixed",
                    display_name="mixed",
                    base_url="https://example.com/v1",
                    api_key="key-1",
                    key_source_label="env:MIXED_API_KEY",
                    is_active=True,
                ),
                models=["quota-model", "working-model", "unavailable-model"],
                checked_at="2026-04-01T00:00:00+00:00",
            )
        ]
        health_records = {
            ("mixed", "quota-model"): ModelHealthRecord(
                provider="mixed",
                model="quota-model",
                status="quota",
                checked_at="2026-04-01T01:00:00+00:00",
            ),
            ("mixed", "working-model"): ModelHealthRecord(
                provider="mixed",
                model="working-model",
                status="working",
                checked_at="2026-04-01T01:05:00+00:00",
            ),
            ("mixed", "unavailable-model"): ModelHealthRecord(
                provider="mixed",
                model="unavailable-model",
                status="unavailable",
                checked_at="2026-04-01T01:10:00+00:00",
            ),
        }

        ordered = order_model_catalog_entries(
            flatten_model_catalog(results, health_records=health_records),
            active_profile_name="mixed",
            active_model_name="working-model",
        )

        self.assertEqual(
            [(entry.index, entry.model_name, entry.status) for entry in ordered],
            [
                (1, "working-model", "working"),
                (2, "quota-model", "quota"),
                (3, "unavailable-model", "unavailable"),
            ],
        )

    def test_order_model_catalog_entries_prefers_active_model_within_status_group(self) -> None:
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
                models=["gpt-5", "gpt-5-mini"],
                checked_at="2026-04-01T00:00:00+00:00",
            )
        ]
        health_records = {
            ("openai", "gpt-5"): ModelHealthRecord(
                provider="openai",
                model="gpt-5",
                status="working",
                checked_at="2026-04-01T01:00:00+00:00",
            ),
            ("openai", "gpt-5-mini"): ModelHealthRecord(
                provider="openai",
                model="gpt-5-mini",
                status="working",
                checked_at="2026-04-01T01:05:00+00:00",
            ),
        }

        ordered = order_model_catalog_entries(
            flatten_model_catalog(results, health_records=health_records),
            active_profile_name="openai",
            active_model_name="gpt-5-mini",
        )

        self.assertEqual([entry.model_name for entry in ordered], ["gpt-5-mini", "gpt-5"])
        self.assertEqual([entry.index for entry in ordered], [1, 2])

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
