from __future__ import annotations

import io
import unittest
from pathlib import Path

from rich.console import Console
from rich.style import Style

from config.config import Config
from ui.tui import AGENT_THEME, TUI
from utils.versioning import ReleaseInfo


class AssistantRenderingTests(unittest.TestCase):
    def _make_tui(self) -> TUI:
        console = Console(
            file=io.StringIO(),
            theme=AGENT_THEME,
            highlight=False,
            force_terminal=True,
            color_system="truecolor",
            width=80,
            record=True,
        )
        return TUI(Config(cwd=Path(".")), console=console)

    def test_render_assistant_text_preserves_content_without_markdown_fences(self) -> None:
        tui = self._make_tui()

        rendered = tui._render_assistant_text(
            "Hello `name`\n```python\nprint(1)\n```\nDone"
        )

        self.assertEqual(rendered.plain, "Hello name\n\nprint(1)\n\nDone")

    def test_render_assistant_text_applies_gradient_background_styles(self) -> None:
        tui = self._make_tui()

        rendered = tui._render_assistant_text("Gradient test")

        self.assertTrue(len(rendered.spans) > 0)

    def test_print_welcome_includes_version_and_update_notice(self) -> None:
        tui = self._make_tui()

        tui.print_welcome(
            release_info=ReleaseInfo(
                current_version="0.1.0.2",
                latest_version="0.1.0.3",
                checked_at="2026-04-03T00:00:00+00:00",
                source="live",
                install_mode="pipx",
            )
        )

        output = tui.console.export_text()
        self.assertIn("Version", output)
        self.assertIn("0.1.0.2", output)
        self.assertIn("0.1.0.3 available", output)


if __name__ == "__main__":
    unittest.main()
