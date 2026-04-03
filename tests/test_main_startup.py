from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import main
from utils.model_health import ModelHealthRecord


class MainStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_environ = dict(os.environ)
        self.original_active_workspace_env = dict(main.ACTIVE_WORKSPACE_ENV)

    def tearDown(self) -> None:
        for key in list(os.environ):
            if key not in self.original_environ:
                os.environ.pop(key, None)
        for key, value in self.original_environ.items():
            os.environ[key] = value
        main.ACTIVE_WORKSPACE_ENV = dict(self.original_active_workspace_env)

    def test_activate_workspace_env_replaces_previous_workspace_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / ".env").write_text(
                "API_KEY=first-key\nBASE_URL=https://first.example/v1\n",
                encoding="utf-8",
            )

            os.environ.pop("API_KEY", None)
            os.environ.pop("BASE_URL", None)

            main._activate_workspace_env(first)
            self.assertEqual(os.environ.get("API_KEY"), "first-key")
            self.assertEqual(os.environ.get("BASE_URL"), "https://first.example/v1")

            main._activate_workspace_env(second)
            self.assertIsNone(os.environ.get("API_KEY"))
            self.assertIsNone(os.environ.get("BASE_URL"))

    def test_temporary_workspace_env_hides_previous_workspace_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / ".env").write_text(
                "API_KEY=first-key\nBASE_URL=https://first.example/v1\n",
                encoding="utf-8",
            )

            os.environ.pop("API_KEY", None)
            os.environ.pop("BASE_URL", None)

            main._activate_workspace_env(first)
            with main._temporary_workspace_env(second):
                self.assertIsNone(os.environ.get("API_KEY"))
                self.assertIsNone(os.environ.get("BASE_URL"))

            self.assertEqual(os.environ.get("API_KEY"), "first-key")
            self.assertEqual(os.environ.get("BASE_URL"), "https://first.example/v1")

    def test_should_repair_api_credentials_for_missing_or_rejected_keys(self) -> None:
        self.assertTrue(
            main._should_repair_api_credentials(
                ModelHealthRecord(
                    provider="default",
                    model="openrouter/free",
                    status="missing-key",
                    checked_at="2026-04-03T00:00:00+00:00",
                )
            )
        )
        self.assertTrue(
            main._should_repair_api_credentials(
                ModelHealthRecord(
                    provider="default",
                    model="openrouter/free",
                    status="auth-error",
                    checked_at="2026-04-03T00:00:00+00:00",
                )
            )
        )
        self.assertFalse(
            main._should_repair_api_credentials(
                ModelHealthRecord(
                    provider="default",
                    model="openrouter/free",
                    status="quota",
                    checked_at="2026-04-03T00:00:00+00:00",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
