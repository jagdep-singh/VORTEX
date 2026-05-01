from __future__ import annotations

import unittest

from unittest.mock import patch

from utils.ollama_setup import (
    OLLAMA_OPENAI_BASE_URL,
    OLLAMA_SPACE_RESERVE_BYTES,
    build_ollama_install_shell_command,
    format_bytes,
    get_local_model_option,
    has_enough_space,
    is_ollama_base_url,
    ollama_install_docs_url,
    supports_automatic_ollama_install,
)


class OllamaSetupTests(unittest.TestCase):
    def test_get_local_model_option_returns_expected_small_model(self) -> None:
        option = get_local_model_option("1")

        self.assertIsNotNone(option)
        assert option is not None
        self.assertEqual(option.model_name, "qwen2.5-coder:1.5b")
        self.assertEqual(option.title, "Fast + light")

    def test_has_enough_space_includes_reserve_buffer(self) -> None:
        required = 900 * 1024 * 1024
        enough = required + OLLAMA_SPACE_RESERVE_BYTES + 1
        not_enough = required + OLLAMA_SPACE_RESERVE_BYTES - 1

        self.assertTrue(has_enough_space(required_bytes=required, free_bytes=enough))
        self.assertFalse(
            has_enough_space(required_bytes=required, free_bytes=not_enough)
        )

    def test_is_ollama_base_url_matches_local_openai_endpoint(self) -> None:
        self.assertTrue(is_ollama_base_url(OLLAMA_OPENAI_BASE_URL))
        self.assertTrue(is_ollama_base_url("http://localhost:11434/v1/"))
        self.assertTrue(is_ollama_base_url("http://127.0.0.1:11434/v1"))
        self.assertFalse(is_ollama_base_url("https://api.openai.com/v1"))

    def test_format_bytes_uses_readable_units(self) -> None:
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(1024 * 1024), "1.0 MB")

    def test_macos_install_helpers(self) -> None:
        with patch("utils.ollama_setup.os.name", "posix"), patch(
            "utils.ollama_setup.sys.platform",
            "darwin",
        ):
            self.assertIn("docs.ollama.com/macos", ollama_install_docs_url())
            self.assertTrue(supports_automatic_ollama_install())
            self.assertIsNotNone(build_ollama_install_shell_command())

    def test_linux_install_helpers(self) -> None:
        with patch("utils.ollama_setup.os.name", "posix"), patch(
            "utils.ollama_setup.sys.platform",
            "linux",
        ):
            self.assertIn("docs.ollama.com/linux", ollama_install_docs_url())
            self.assertTrue(supports_automatic_ollama_install())
            self.assertIsNotNone(build_ollama_install_shell_command())

    def test_windows_install_helpers_use_official_powershell_path(self) -> None:
        with patch("utils.ollama_setup.os.name", "nt"), patch(
            "utils.ollama_setup.sys.platform",
            "win32",
        ):
            self.assertTrue(supports_automatic_ollama_install())
            self.assertIn("docs.ollama.com/windows", ollama_install_docs_url())
            command = build_ollama_install_shell_command()
            self.assertIsNotNone(command)
            assert command is not None
            self.assertIn("install.ps1", command)
            self.assertIn("powershell", command.lower())


if __name__ == "__main__":
    unittest.main()
