from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.style import Style

from config.config import Config
from ui.tui import AGENT_THEME, HIGH_CONTRAST_THEME, TUI
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

    def test_status_line_shows_error_and_metrics(self) -> None:
        tui = self._make_tui()
        tui._supports_live_status = False  # force inline output
        tui.set_run_metrics(1.23, None)
        tui.record_error("boom")

        tui.show_status("Ready")
        output = tui.console.export_text()
        self.assertIn("Ready", output)
        self.assertIn("1.2s", output)
        self.assertIn("tok n/a", output)
        self.assertIn("ERR", output)

    def test_high_contrast_toggle_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(cwd=Path(tmpdir))
            console = Console(
                file=io.StringIO(),
                theme=AGENT_THEME,
                highlight=False,
                force_terminal=True,
                color_system="truecolor",
                width=80,
                record=True,
            )
            tui = TUI(cfg, console=console)
            tui.set_high_contrast(True, persist=True)
            self.assertTrue(tui._high_contrast)
            self.assertTrue((Path(tmpdir) / ".env").exists())

    def test_env_change_panel_lists_keys_not_values(self) -> None:
        tui = self._make_tui()
        tui._supports_live_status = False
        previous = {"FOO": "one"}
        new = {"FOO": "one", "BAR": "secret123"}

        tui.show_env_change(previous, new)
        output = tui.console.export_text()
        self.assertIn("BAR", output)
        self.assertNotIn("secret123", output)


if __name__ == "__main__":
    unittest.main()
