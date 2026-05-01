from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

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

    def test_provider_prompt_helpers_support_local_shortcuts(self) -> None:
        self.assertEqual(
            main._provider_prompt_default_value("http://localhost:11434/v1"),
            "1",
        )
        self.assertEqual(
            main._provider_prompt_default_value("https://openrouter.ai/api/v1"),
            "2",
        )
        self.assertEqual(
            main._resolve_provider_prompt_input("1"),
            main._LOCAL_PROVIDER_CHOICE,
        )
        self.assertEqual(
            main._resolve_provider_prompt_input("openrouter"),
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            main._resolve_provider_prompt_input("https://example.com/v1"),
            "https://example.com/v1",
        )

    def test_should_repair_local_ollama_setup_only_for_default_local_workspaces(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BASE_URL": "http://localhost:11434/v1",
                "MODEL_NAME": "qwen2.5-coder:1.5b",
            },
            clear=True,
        ):
            config = main.Config(cwd=Path("."))
            self.assertTrue(
                main._should_repair_local_ollama_setup(
                    config=config,
                    record=None,
                )
            )
            self.assertTrue(
                main._should_repair_local_ollama_setup(
                    config=config,
                    record=ModelHealthRecord(
                        provider="default",
                        model="qwen2.5-coder:1.5b",
                        status="offline",
                        checked_at="2026-05-01T00:00:00+00:00",
                    ),
                )
            )
            self.assertFalse(
                main._should_repair_local_ollama_setup(
                    config=config,
                    record=ModelHealthRecord(
                        provider="default",
                        model="qwen2.5-coder:1.5b",
                        status="limited",
                        checked_at="2026-05-01T00:00:00+00:00",
                    ),
                )
            )

        profiled_config = main.Config(
            cwd=Path("."),
            active_model_profile="local",
            models={
                "local": {
                    "base_url": "http://localhost:11434/v1",
                }
            },
        )
        self.assertFalse(
            main._should_repair_local_ollama_setup(
                config=profiled_config,
                record=None,
            )
        )


class _FakeSetupTUI:
    def __init__(
        self,
        *,
        local_choices: list[str] | None = None,
        setup_choices: list[str] | None = None,
    ) -> None:
        self.local_choices = list(local_choices or [])
        self.setup_choices = list(setup_choices or [])
        self.status_messages: list[str] = []

    async def prompt_local_model_choice(self, **_: object) -> str:
        return self.local_choices.pop(0) if self.local_choices else "1"

    async def prompt_setup_decision(self, **kwargs: object) -> str:
        if self.setup_choices:
            return self.setup_choices.pop(0)
        return str(kwargs.get("default_choice", "1"))

    def show_status(self, message: str) -> None:
        self.status_messages.append(message)

    def clear_status(self) -> None:
        self.status_messages.append("CLEAR")


class MainLocalSetupAsyncTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_complete_local_ollama_setup_persists_workspace_config(self) -> None:
        option = main.get_local_model_option("1")
        assert option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI()

            with patch("main.is_ollama_installed", return_value=True), patch(
                "main.list_installed_models",
                new=AsyncMock(return_value=[option.model_name]),
            ):
                status, config = await main._complete_local_ollama_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    option=option,
                )

            self.assertEqual(status, "configured")
            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.base_url, main.OLLAMA_OPENAI_BASE_URL)
            self.assertEqual(config.model_name, option.model_name)
            env_text = (workspace / ".env").read_text(encoding="utf-8")
            self.assertIn('BASE_URL="http://localhost:11434/v1"', env_text)
            self.assertIn(f'MODEL_NAME="{option.model_name}"', env_text)
            self.assertIn('API_KEY=""', env_text)

    async def test_complete_local_ollama_setup_pulls_missing_model_when_space_allows(self) -> None:
        option = main.get_local_model_option("1")
        assert option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI(setup_choices=["1"])

            with patch("main.is_ollama_installed", return_value=True), patch(
                "main.list_installed_models",
                new=AsyncMock(side_effect=[[], [option.model_name]]),
            ), patch(
                "main.get_free_space_bytes",
                return_value=option.size_bytes + (1024 * 1024 * 1024),
            ), patch(
                "main.pull_model",
                new=AsyncMock(return_value=None),
            ) as pull_model_mock:
                status, config = await main._complete_local_ollama_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    option=option,
                )

            self.assertEqual(status, "configured")
            self.assertIsNotNone(config)
            pull_model_mock.assert_awaited_once()
            self.assertTrue(
                any("Local setup: pulling" in message for message in tui.status_messages)
            )

    async def test_complete_local_ollama_setup_can_install_ollama_when_missing(self) -> None:
        option = main.get_local_model_option("1")
        assert option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI(setup_choices=["1"])

            with patch("main.supports_automatic_ollama_install", return_value=True), patch(
                "main.is_ollama_installed",
                side_effect=[False, True],
            ), patch(
                "main._install_ollama_with_permission",
                new=AsyncMock(return_value=(True, None)),
            ) as install_mock, patch(
                "main.list_installed_models",
                new=AsyncMock(return_value=[option.model_name]),
            ):
                status, config = await main._complete_local_ollama_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    option=option,
                )

            self.assertEqual(status, "configured")
            self.assertIsNotNone(config)
            install_mock.assert_awaited_once()

    async def test_complete_local_ollama_setup_can_start_ollama_when_not_running(self) -> None:
        option = main.get_local_model_option("1")
        assert option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI(setup_choices=["1"])

            with patch("main.is_ollama_installed", return_value=True), patch(
                "main.list_installed_models",
                new=AsyncMock(side_effect=[RuntimeError("offline"), [option.model_name]]),
            ), patch(
                "main._start_ollama_with_permission",
                new=AsyncMock(return_value=(True, None)),
            ) as start_mock:
                status, config = await main._complete_local_ollama_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    option=option,
                )

            self.assertEqual(status, "configured")
            self.assertIsNotNone(config)
            start_mock.assert_awaited_once()

    async def test_complete_local_ollama_setup_can_fall_back_to_external(self) -> None:
        option = main.get_local_model_option("1")
        assert option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI(setup_choices=["3"])

            with patch("main.is_ollama_installed", return_value=False), patch(
                "main.supports_automatic_ollama_install",
                return_value=False,
            ):
                status, config = await main._complete_local_ollama_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    option=option,
                )

            self.assertEqual(status, "external")
            self.assertIsNone(config)

    async def test_requested_local_model_can_fall_back_to_smaller_fitting_option(self) -> None:
        preferred_option = main.get_local_model_option("2")
        smaller_option = main.get_local_model_option("1")
        assert preferred_option is not None
        assert smaller_option is not None

        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            tui = _FakeSetupTUI(setup_choices=["1", "1"])

            with patch("main.is_ollama_installed", return_value=True), patch(
                "main.list_installed_models",
                new=AsyncMock(
                    side_effect=[
                        [],
                        [],
                        [smaller_option.model_name],
                    ]
                ),
            ), patch(
                "main.get_free_space_bytes",
                return_value=smaller_option.size_bytes
                + (512 * 1024 * 1024)
                + (32 * 1024 * 1024),
            ), patch(
                "main.get_total_memory_bytes",
                return_value=4 * 1024 * 1024 * 1024,
            ), patch(
                "main.pull_model",
                new=AsyncMock(return_value=None),
            ) as pull_model_mock:
                status, config = await main._complete_requested_local_ollama_model_setup(
                    requested_cwd=workspace,
                    tui=tui,
                    model_name=preferred_option.model_name,
                    allow_external=False,
                )

            self.assertEqual(status, "configured")
            self.assertIsNotNone(config)
            pull_model_mock.assert_awaited_once_with(
                smaller_option.model_name,
                progress_callback=unittest.mock.ANY,
            )
            env_text = (workspace / ".env").read_text(encoding="utf-8")
            self.assertIn(f'MODEL_NAME="{smaller_option.model_name}"', env_text)


if __name__ == "__main__":
    unittest.main()
