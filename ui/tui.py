from __future__ import annotations

import os
import re
import sys
import time
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from collections import Counter

from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from client.response import TokenUsage

from config.config import Config
from tools.base import ToolConfirmation
from utils.credential_setup import upsert_env_file
from utils.model_discovery import (
    MODEL_BUCKET_ORDER,
    ModelCatalogResult,
    flatten_model_catalog,
    group_model_catalog_entries_by_bucket,
    model_status_bucket,
    order_model_catalog_entries,
)
from utils.model_health import ModelHealthRecord
from utils.paths import display_path_rel_to_cwd
from utils.text import truncate_text
from utils.versioning import ReleaseInfo

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import PathCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
except Exception:
    PromptSession = None
    PathCompleter = None
    FormattedText = None
    InMemoryHistory = None

def _truthy_env(value: str | None) -> bool:
    if not value:
        return False
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _build_theme(*, high_contrast: bool = False) -> Theme:
    base = {
        "info": "#00e5e5",
        "warning": "#ffd166",
        "error": "bold #ff6b6b",
        "success": "#00e5e5",
        "muted": "#555555",
        "dim": "#333333",
        "surface": "#0d0d0d",
        "surface.alt": "#121212",
        "panel.border": "#1a1a1a",
        "panel.title": "bold #00e5e5",
        "panel.subtitle": "#555555",
        "brand": "bold #00e5e5",
        "brand.badge": "bold #00e5e5",
        "chip": "#00e5e5",
        "chip.border": "#1a4a4a",
        "prompt": "#00e5e5",
        "prompt.hint": "#555555",
        "user": "#c9c9c9",
        "user.badge": "#555555",
        "assistant": "#c9c9c9",
        "assistant.badge": "#00e5e5",
        "assistant.inline_code": "bold #00e5e5",
        "assistant.code_block": "#d6f5ff",
        "assistant.panel.border": "#2c6476",
        "assistant.panel.title": "bold #e7fbff",
        "assistant.panel.subtitle": "#7fc9d8",
        "thinking": "italic #777777",
        "thinking.badge": "#00e5e5",
        "meta.label": "#00e5e5",
        "meta.value": "#c9c9c9",
        "tool": "bold #00e5e5",
        "tool.badge": "#00e5e5",
        "tool.read": "#00e5e5",
        "tool.write": "#ffd166",
        "tool.shell": "#c678dd",
        "tool.network": "#61afef",
        "tool.memory": "#98c379",
        "tool.mcp": "#00e5e5",
        "status.info.badge": "#00e5e5",
        "status.success.badge": "#00e5e5",
        "status.warning.badge": "#ffd166",
        "status.error.badge": "#ff6b6b",
        "code": "#c9c9c9",
    }

    if high_contrast:
        base.update(
            {
                "muted": "#bfbfbf",
                "dim": "#4d4d4d",
                "panel.border": "#3a3a3a",
                "panel.title": "bold #7ef7f7",
                "panel.subtitle": "#8a8a8a",
                "assistant": "#f0f0f0",
                "assistant.panel.border": "#4fb3c2",
                "assistant.panel.title": "bold #e7ffff",
                "assistant.panel.subtitle": "#a1d9e4",
                "meta.value": "#e0e0e0",
                "tool": "bold #7ef7f7",
                "code": "#ededed",
            }
        )

    return Theme(base)


AGENT_THEME = _build_theme(high_contrast=False)
HIGH_CONTRAST_THEME = _build_theme(high_contrast=True)

_console: Console | None = None
_console_high_contrast = False


def _is_high_contrast_enabled() -> bool:
    return _truthy_env(os.environ.get("VORTEX_HIGH_CONTRAST"))


def get_console() -> Console:
    global _console
    global _console_high_contrast
    requested_contrast = _is_high_contrast_enabled()
    if _console is None:
        _console = Console(
            theme=HIGH_CONTRAST_THEME if requested_contrast else AGENT_THEME,
            highlight=False,
        )
        _console_high_contrast = requested_contrast
    else:
        if _console_high_contrast != requested_contrast:
            _console_high_contrast = requested_contrast
            try:
                _console.push_theme(
                    HIGH_CONTRAST_THEME if requested_contrast else AGENT_THEME
                )
            except Exception:
                _console.theme = (
                    HIGH_CONTRAST_THEME if requested_contrast else AGENT_THEME
                )
    return _console


class TUI:
    _PROMPT_MARKER = "╰─"
    _WORKSPACE_PROMPT_LABEL = "workspace"
    _STATUS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _WORKSPACE_ART = "\n".join(
        [
            "▗▖ ▗▖ ▗▄▖ ▗▄▄▖ ▗▖ ▗▖ ▗▄▄▖▗▄▄▖  ▗▄▖  ▗▄▄▖▗▄▄▄▖",
            "▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌▗▞▘▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ",
            "▐▌ ▐▌▐▌ ▐▌▐▛▀▚▖▐▛▚▖  ▝▀▚▖▐▛▀▘ ▐▛▀▜▌▐▌   ▐▛▀▀▘",
            "▐▙█▟▌▝▚▄▞▘▐▌ ▐▌▐▌ ▐▌▗▄▄▞▘▐▌   ▐▌ ▐▌▝▚▄▄▖▐▙▄▄▖",
            "                                             ",
            
        ]
    )
    _API_ART = "\n".join(
        [
            " ▗▄▖ ▗▄▄▖▗▄▄▄▖     ▗▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖▗▄▄▖ ",
            "▐▌ ▐▌▐▌ ▐▌ █      ▐▌   ▐▌     █  ▐▌ ▐▌▐▌ ▐▌",
            "▐▛▀▜▌▐▛▀▘  █       ▝▀▚▖▐▛▀▀▘  █  ▐▌ ▐▌▐▛▀▘ ",
            "▐▌ ▐▌▐▌  ▗▄█▄▖    ▗▄▄▞▘▐▙▄▄▖  █  ▝▚▄▞▘▐▌   ",
            "                                           ",
        ]
    )
    _LOGO = "\n".join(
        [
            "                                                   ",
            "██╗   ██╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗",
            "██║   ██║██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝ ",
            "██║   ██║██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ ",
            "╚██╗ ██╔╝██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ ",
            " ╚████╔╝ ╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗",
            "  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝",
            "                                                   ",
            
        ]
    )

    def __init__(
        self,
        config: Config,
        console: Console | None = None,
    ) -> None:
        self.console = console or get_console()
        self.config = config
        self.cwd = self.config.cwd
        self._assistant_stream_open = False
        self._assistant_buffer = ""
        self._assistant_animation_phase = 0
        self._assistant_live: Live | None = None
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        self._max_block_tokens = 2500
        self._last_status: str | None = None
        self._status_label = "thinking"
        self._status_detail = ""
        self._status_frame_index = 0
        self._status_live: Live | None = None
        self._supports_live_status = sys.stdout.isatty()
        self._status_suspended = False
        self._last_status_signature: tuple[Any, ...] | None = None
        self._last_error: str | None = None
        self._last_run_elapsed: float | None = None
        self._last_run_usage: TokenUsage | None = None
        self._prompt_notice_shown = False
        self._high_contrast = _is_high_contrast_enabled()
        self._setup_screen_depth = 0
        self._use_temporary_setup_screen = _truthy_env(
            os.environ.get("VORTEX_ALT_SETUP_SCREEN", "1")
        )
        self._prompt_session = None
        requested_prompt_toolkit = os.environ.get("VORTEX_USE_PROMPT_TOOLKIT", "")
        self._use_prompt_toolkit = requested_prompt_toolkit.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not requested_prompt_toolkit:
            self._use_prompt_toolkit = sys.version_info < (3, 14)

        if (
            self._use_prompt_toolkit
            and
            PromptSession is not None
            and InMemoryHistory is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            self._prompt_session = PromptSession(
                history=InMemoryHistory(),
            )
        else:
            self._maybe_warn_prompt_toolkit()

        self._apply_theme()

    def set_config(self, config: Config) -> None:
        self.config = config
        self.cwd = config.cwd

    def _apply_theme(self) -> None:
        theme = HIGH_CONTRAST_THEME if self._high_contrast else AGENT_THEME
        try:
            self.console.push_theme(theme)
        except Exception:
            try:
                self.console.theme = theme
            except Exception:
                pass

    def _maybe_warn_prompt_toolkit(self) -> None:
        if self._prompt_notice_shown:
            return
        if not self._use_prompt_toolkit:
            return
        if self._prompt_session is not None:
            return

        reason = "prompt_toolkit unavailable"
        if PromptSession is not None and InMemoryHistory is not None:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                reason = "non-interactive terminal"
            else:
                reason = "prompt_toolkit disabled"

        self.console.print(
            Text(
                f"Path completion/history disabled ({reason}); using basic input.",
                style="warning",
            )
        )
        self._prompt_notice_shown = True

    def _pause_live_status(self) -> None:
        if self._status_live is not None:
            try:
                self._status_live.stop()
            except Exception:
                pass
            self._status_live = None

    def _stop_assistant_live(self, *, keep_output: bool) -> None:
        if self._assistant_live is not None:
            try:
                self._assistant_live.stop()
            except Exception:
                pass
            self._assistant_live = None

        if keep_output and self._assistant_stream_open:
            self.console.print(self._render_assistant_panel(streaming=False))
            self.console.print()

        self._assistant_stream_open = False
        self._assistant_buffer = ""
        self._assistant_animation_phase = 0

    def clear_screen(self) -> None:
        self._stop_assistant_live(keep_output=False)
        self._pause_live_status()
        self.console.clear(home=True)

    def _can_use_temporary_setup_screen(self) -> bool:
        return (
            self._use_temporary_setup_screen
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )

    @contextlib.contextmanager
    def _temporary_setup_screen(self):
        entered = False
        should_enter = (
            self._can_use_temporary_setup_screen() and self._setup_screen_depth == 0
        )
        if should_enter:
            with contextlib.suppress(Exception):
                entered = self.console.set_alt_screen(True)
        self._setup_screen_depth += 1
        try:
            yield
        finally:
            self._setup_screen_depth = max(0, self._setup_screen_depth - 1)
            if entered and self._setup_screen_depth == 0:
                self._stop_assistant_live(keep_output=False)
                self._pause_live_status()
                with contextlib.suppress(Exception):
                    self.console.clear(home=True)
                with contextlib.suppress(Exception):
                    self.console.set_alt_screen(False)

    def _badge(self, label: str, style: str) -> Text:
        return Text(f" {label.upper()} ", style=style)

    def _home_relative(self, path: str | Path) -> str:
        text = str(path)
        home = str(Path.home())
        if text.startswith(home):
            return text.replace(home, "~", 1)
        return text

    def _command_pills(self, items: Iterable[str]) -> Columns:
        return Columns(
            [
                Panel.fit(
                    Text(item, style="chip"),
                    border_style="chip.border",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                for item in items
            ],
            expand=False,
            equal=False,
            padding=(0, 1),
        )

    def _action_table(
        self,
        rows: Iterable[tuple[str, str]],
        *,
        key_style: str = "meta.value",
    ) -> Table:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style=key_style, no_wrap=True)
        table.add_column(style="muted", overflow="fold")

        for key, value in rows:
            table.add_row(str(key), str(value))

        return table

    def _section_title(self, label: str, badge_style: str, title_style: str) -> Text:
        title = Text()
        title.append_text(self._badge(label, badge_style))
        title.append(" ")
        title.append(label.title(), style=title_style)
        return title

    def _panel(
        self,
        body,
        *,
        title: Text | str | None = None,
        subtitle: Text | str | None = None,
        border_style: str = "panel.border",
        padding: tuple[int, int] = (1, 2),
    ) -> Panel:
        return Panel(
            body,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=padding,
        )

    def _print_startup_art(self, art: str, label: str) -> None:
        self.console.print()
        self.console.print(Align.center(Text(art, style="brand")))
        self.console.print(
            Rule(
                Text(label.upper(), style="muted"),
                style="panel.border",
            )
        )

    def _kv_table(
        self,
        rows: Iterable[tuple[str, Any]],
        *,
        wrap_values: bool = True,
    ) -> Table:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="meta.label", no_wrap=True)
        table.add_column(style="meta.value", overflow="fold" if wrap_values else "ellipsis")

        for key, value in rows:
            table.add_row(str(key), str(value))

        return table

    def _summary_text(self, parts: Iterable[Any], style: str = "muted") -> Text:
        clean_parts = [str(part) for part in parts if part not in (None, "", [])]
        return Text(" • ".join(clean_parts), style=style)

    def _short_text(self, value: str | None, limit: int = 72) -> str:
        if not value:
            return ""
        clean = " ".join(str(value).split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."

    def _model_status_text(self, status: str | None) -> Text:
        label = status or "unknown"
        style_map = {
            "working": "success",
            "limited": "warning",
            "cached": "warning",
            "unknown": "muted",
            "missing-key": "warning",
            "quota": "warning",
            "rate-limited": "warning",
            "auth-error": "error",
            "unavailable": "error",
            "offline": "warning",
            "error": "error",
        }
        return Text(label, style=style_map.get(label, "muted"))

    def _model_bucket_text(self, bucket: str | None) -> Text:
        label = bucket or "not-working"
        style_map = {
            "working": "success",
            "limited": "warning",
            "quota": "warning",
            "not-working": "error",
        }
        text_map = {
            "working": "working",
            "limited": "limited",
            "quota": "quota",
            "not-working": "not working",
        }
        return Text(text_map.get(label, label), style=style_map.get(label, "muted"))

    def _format_checked_at(self, checked_at: str | None) -> str:
        if not checked_at:
            return "-"
        try:
            dt = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        except ValueError:
            return checked_at
        return dt.strftime("%Y-%m-%d %H:%M")

    def _tool_context_summary(self, tool_name: str, args: dict[str, Any]) -> Text | None:
        parts: list[str] = []

        path_value = args.get("path")
        if isinstance(path_value, str) and path_value:
            parts.append(str(display_path_rel_to_cwd(path_value, self.cwd)))

        command_value = args.get("command")
        if isinstance(command_value, str) and command_value.strip():
            parts.append(self._short_text(f"$ {command_value.strip()}", limit=96))

        query_value = args.get("query")
        if isinstance(query_value, str) and query_value.strip():
            parts.append(f"query: {self._short_text(query_value.strip(), limit=48)}")

        pattern_value = args.get("pattern")
        if isinstance(pattern_value, str) and pattern_value.strip():
            parts.append(f"pattern: {self._short_text(pattern_value.strip(), limit=40)}")

        url_value = args.get("url")
        if isinstance(url_value, str) and url_value.strip():
            parts.append(self._short_text(url_value.strip(), limit=72))

        action_value = args.get("action")
        if isinstance(action_value, str) and action_value.strip():
            parts.append(f"action: {action_value.strip()}")

        if not parts:
            return None

        return self._summary_text(parts)

    def _tool_subtitle(
        self,
        state: str | None = None,
        *,
        call_id: str = "",
        tool_kind: str | None = None,
    ) -> Text:
        parts: list[str] = []
        if state:
            parts.append(state)
        if tool_kind:
            parts.append(tool_kind)
        if call_id:
            parts.append(f"#{call_id[:8]}")
        return Text(" · ".join(parts), style="panel.subtitle")

    def prompt(self) -> str:
        return "you > "

    def _prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", f"{self._PROMPT_MARKER} "),
                ("fg:#555555", "you › "),
            ]
        )

    def _workspace_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", f"{self._WORKSPACE_PROMPT_LABEL} "),
                ("fg:#555555", "› "),
            ]
        )

    def _path_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", "path "),
                ("fg:#555555", "› "),
            ]
        )

    def _provider_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", "provider "),
                ("fg:#555555", "› "),
            ]
        )

    def _local_model_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", "setup "),
                ("fg:#555555", "› "),
            ]
        )

    def _model_name_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", "model "),
                ("fg:#555555", "› "),
            ]
        )

    def _api_key_prompt_message(self):
        if FormattedText is None:
            return None

        return FormattedText(
            [
                ("", "\n"),
                ("fg:#00e5e5", "api key "),
                ("fg:#555555", "› "),
            ]
        )

    async def read_prompt(self) -> str:
        self._pause_live_status()

        # Render the main chat prompt through Rich instead of relying on the
        # terminal's built-in input prompt. This keeps the `you >` prompt
        # visible in terminals where plain `input(prompt)` can disappear after
        # the welcome screen or Rich output.
        return self.console.input(
            f"\n[prompt]{self._PROMPT_MARKER} [/prompt][prompt.hint]you › [/prompt.hint]"
        )

    async def prompt_workspace_selection(
        self,
        *,
        current_dir: Path,
        fallback_dir: Path,
        recent_workspaces: list[dict[str, str]],
        current_label: str,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            table = Table(expand=True, box=None, padding=(0, 1))
            table.add_column("#", style="meta.label", no_wrap=True)
            table.add_column("Directory", style="meta.value")
            table.add_column("Notes", style="muted")

            options: list[tuple[str, str, str]] = [
                ("1", self._home_relative(current_dir), current_label),
            ]

            if fallback_dir.resolve() != current_dir.resolve():
                options.append(("2", self._home_relative(fallback_dir), "Default workspace"))

            next_index = len(options) + 1
            for entry in recent_workspaces[:5]:
                path = entry.get("path", "")
                if not path:
                    continue
                if Path(path).resolve() in {current_dir.resolve(), fallback_dir.resolve()}:
                    continue
                last_used = entry.get("last_used", "") or "recent"
                options.append((str(next_index), self._home_relative(path), last_used))
                next_index += 1

            options.append(
                (
                    str(next_index),
                    "Custom path...",
                    "Enter a project directory manually",
                )
            )

            for index, directory, note in options:
                table.add_row(index, directory, note)

            self.clear_screen()
            self._print_startup_art(self._WORKSPACE_ART, "workspace setup")
            self.console.print(Rule(style="panel.border"))
            messages: list[Any] = [
                Text("Choose a working directory for this session.", style="muted"),
                Text(
                    f"Current shell directory: {self._home_relative(current_dir)}",
                    style="meta.value",
                ),
            ]

            if error_message:
                messages.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                messages.extend([Text(), Text(info_message, style="warning")])

            messages.extend(
                [
                    Text(),
                    Text(
                        "Choose a number or type any directory path on this machine.",
                        style="muted",
                    ),
                    Text(
                        "Absolute paths, ~/ paths, and paths relative to the current shell directory are all supported.",
                        style="muted",
                    ),
                    Text(),
                    table,
                ]
            )

            self.console.print(
                self._panel(
                    Group(*messages),
                    title=Text.assemble(
                        self._badge("CWD", "status.info.badge"),
                        (" Workspace setup", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            default_choice = "1"
            if self._prompt_session is not None and PathCompleter is not None:
                return await self._prompt_session.prompt_async(
                    self._workspace_prompt_message(),
                    default=default_choice,
                    completer=PathCompleter(expanduser=True, only_directories=True),
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]workspace/path[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default=default_choice,
            )

    async def prompt_custom_workspace_path(
        self,
        *,
        base_dir: Path,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._WORKSPACE_ART, "custom workspace")
            self.console.print(Rule(style="panel.border"))
            body: list[Any] = [
                Text("Enter a project directory to use as the active workspace.", style="muted"),
                Text(
                    f"Relative paths resolve from: {self._home_relative(base_dir)}",
                    style="meta.value",
                ),
            ]

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            body.extend(
                [
                    Text(),
                    Text("Examples:", style="meta.label"),
                    Text("~/projects/my-app", style="meta.value"),
                    Text("/Users/you/src/my-app", style="meta.value"),
                    Text("../shared-workspace", style="meta.value"),
                    Text(),
                    Text("Press Enter with no value to go back.", style="muted"),
                ]
            )

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge("Path", "status.info.badge"),
                        (" Custom workspace", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None and PathCompleter is not None:
                return await self._prompt_session.prompt_async(
                    self._path_prompt_message(),
                    completer=PathCompleter(expanduser=True, only_directories=True),
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]path[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default="",
            )

    async def prompt_api_provider_url(
        self,
        *,
        workspace_dir: Path,
        default_value: str,
        api_key_env_name: str,
        docs_path: Path | None = None,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._API_ART, "provider setup")
            self.console.print(Rule(style="panel.border"))
            options = Table.grid(expand=True)
            options.add_column(style="meta.label", width=4)
            options.add_column(style="meta.value", ratio=2)
            options.add_column(style="muted", ratio=4)
            options.add_row(
                "1",
                "Local models",
                "Use Ollama on this device and choose or install a local model.",
            )
            options.add_row(
                "2",
                "OpenRouter",
                "Use https://openrouter.ai/api/v1",
            )
            options.add_row(
                "3",
                "OpenAI",
                "Use https://api.openai.com/v1",
            )
            options.add_row(
                "4",
                "Gemini",
                "Use https://generativelanguage.googleapis.com/v1beta/openai",
            )
            body: list[Any] = [
                Text(
                    "VORTEX needs an OpenAI-compatible provider URL before it can start.",
                    style="muted",
                ),
                Text(
                    f"Workspace: {self._home_relative(workspace_dir)}",
                    style="meta.value",
                ),
                Text(
                    f"Saved to: {self._home_relative(workspace_dir / '.env')}",
                    style="meta.value",
                ),
                Text(
                    f"API key variable: {api_key_env_name}",
                    style="meta.value",
                ),
            ]

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            body.extend(
                [
                    Text(),
                    Text("Quick choices:", style="meta.label"),
                    options,
                    Text(),
                    Text(
                        "You can also paste any custom OpenAI-compatible base URL directly.",
                        style="muted",
                    ),
                    Text(
                        "Choosing Local models will walk through Ollama install, startup, and model setup for this device.",
                        style="muted",
                    ),
                    Text(
                        "Local Ollama setups usually do not need an API key.",
                        style="muted",
                    ),
                ]
            )

            if docs_path is not None:
                body.extend(
                    [
                        Text(),
                        Text(
                            f"Need help choosing one? See {self._home_relative(docs_path)}",
                            style="muted",
                        ),
                    ]
                )

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge("API", "status.info.badge"),
                        (" Provider setup", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None:
                return await self._prompt_session.prompt_async(
                    self._provider_prompt_message(),
                    default=default_value,
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]provider[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default=default_value,
            )

    async def prompt_local_model_name(
        self,
        *,
        workspace_dir: Path,
        current_model_name: str,
        installed_models: list[str],
        recommended_models: list[tuple[str, str, str]],
        free_space_label: str,
        memory_label: str | None,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._API_ART, "local model change")
            self.console.print(Rule(style="panel.border"))

            installed_table = Table.grid(expand=True)
            installed_table.add_column(style="meta.label", width=4)
            installed_table.add_column(style="meta.value", ratio=4)
            if installed_models:
                for index, model_name in enumerate(installed_models, start=1):
                    installed_table.add_row(str(index), model_name)
            else:
                installed_table.add_row("-", "No local Ollama models are installed yet.")

            recommended_table = Table.grid(expand=True)
            recommended_table.add_column(style="meta.value", ratio=3)
            recommended_table.add_column(style="muted", ratio=5)
            if recommended_models:
                for model_name, summary, note in recommended_models:
                    label = model_name if not summary else f"{model_name} ({summary})"
                    recommended_table.add_row(label, note)
            else:
                recommended_table.add_row(
                    "No built-in recommendations fit the current machine checks.",
                    "You can still type any Ollama model name to try manually.",
                )

            body: list[Any] = [
                Text(
                    f"Current model: {current_model_name}",
                    style="meta.value",
                ),
                Text(
                    f"Workspace: {self._home_relative(workspace_dir)}",
                    style="meta.value",
                ),
                Text(
                    f"Free model storage: {free_space_label}",
                    style="meta.value",
                ),
            ]

            if memory_label is not None:
                body.append(
                    Text(
                        f"System memory: {memory_label}",
                        style="meta.value",
                    )
                )

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            body.extend(
                [
                    Text(),
                    Text("Installed local models on this machine", style="meta.label"),
                    installed_table,
                    Text(),
                    Text("Recommended models that fit these machine checks", style="meta.label"),
                    recommended_table,
                    Text(),
                    Text(
                        "Type an installed model number to switch immediately, or type any Ollama model name to install and use it.",
                        style="muted",
                    ),
                    Text(
                        "Press Enter with no value to cancel.",
                        style="muted",
                    ),
                ]
            )

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge("Local", "status.info.badge"),
                        (" Change model", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None:
                return await self._prompt_session.prompt_async(
                    self._model_name_prompt_message(),
                    default="",
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]model[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default="",
            )

    async def prompt_local_model_choice(
        self,
        *,
        workspace_dir: Path,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._API_ART, "local or external")
            self.console.print(Rule(style="panel.border"))

            options = Table.grid(expand=True)
            options.add_column(style="meta.label", width=4)
            options.add_column(style="meta.value", ratio=2)
            options.add_column(style="muted", ratio=4)
            options.add_row(
                "1",
                "Fast + light",
                "Use Ollama with qwen2.5-coder:1.5b. Smaller download, quicker startup, easier on laptops.",
            )
            options.add_row(
                "2",
                "Better coding quality",
                "Use Ollama with qwen2.5-coder:3b. Larger download, slower than 1.5b, but usually better edits and code fixes.",
            )
            options.add_row(
                "3",
                "External API",
                "Use OpenAI, OpenRouter, Gemini, or another OpenAI-compatible provider instead.",
            )

            body: list[Any] = [
                Text(
                    "Choose how this workspace should get its model.",
                    style="muted",
                ),
                Text(
                    f"Workspace: {self._home_relative(workspace_dir)}",
                    style="meta.value",
                ),
                Text(
                    f"Saved to: {self._home_relative(workspace_dir / '.env')}",
                    style="meta.value",
                ),
            ]

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            body.extend(
                [
                    Text(),
                    Text(
                        "If Ollama is missing, VORTEX can optionally help install it with permission.",
                        style="muted",
                    ),
                    Text(
                        "You can still switch local models later with /change-model, or switch providers with /api-change.",
                        style="muted",
                    ),
                    Text(),
                    options,
                ]
            )

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge("Setup", "status.info.badge"),
                        (" Model source", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None:
                return await self._prompt_session.prompt_async(
                    self._local_model_prompt_message(),
                    default="1",
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]setup[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default="1",
            )

    async def prompt_setup_decision(
        self,
        *,
        badge_label: str,
        title: str,
        workspace_dir: Path,
        body_lines: list[str],
        options: list[tuple[str, str, str]],
        default_choice: str = "1",
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._API_ART, "local setup")
            self.console.print(Rule(style="panel.border"))

            option_table = Table.grid(expand=True)
            option_table.add_column(style="meta.label", width=4)
            option_table.add_column(style="meta.value", ratio=2)
            option_table.add_column(style="muted", ratio=4)
            for key, label, note in options:
                option_table.add_row(key, label, note)

            body: list[Any] = [
                Text(
                    f"Workspace: {self._home_relative(workspace_dir)}",
                    style="meta.value",
                ),
                Text(
                    f"Saved to: {self._home_relative(workspace_dir / '.env')}",
                    style="meta.value",
                ),
            ]

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            for line in body_lines:
                body.extend([Text(), Text(line, style="muted")])

            body.extend([Text(), option_table])

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge(badge_label, "status.info.badge"),
                        (f" {title}", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None:
                return await self._prompt_session.prompt_async(
                    self._local_model_prompt_message(),
                    default=default_choice,
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]setup[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default=default_choice,
            )

    async def prompt_api_key(
        self,
        *,
        workspace_dir: Path,
        provider_url: str,
        api_key_env_name: str,
        docs_path: Path | None = None,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        with self._temporary_setup_screen():
            self.clear_screen()
            self._print_startup_art(self._API_ART, "api key setup")
            self.console.print(Rule(style="panel.border"))
            body: list[Any] = [
                Text(
                    "Enter the API key for the provider you want this workspace to use.",
                    style="muted",
                ),
                Text(
                    f"Provider URL: {provider_url}",
                    style="meta.value",
                ),
                Text(
                    f"Saved to: {self._home_relative(workspace_dir / '.env')} as {api_key_env_name}",
                    style="meta.value",
                ),
                Text("The key is hidden while you type.", style="muted"),
            ]

            if error_message:
                body.extend([Text(), Text(error_message, style="error")])
            elif info_message:
                body.extend([Text(), Text(info_message, style="warning")])

            if docs_path is not None:
                body.extend(
                    [
                        Text(),
                        Text(
                            f"Need help getting a key? See {self._home_relative(docs_path)}",
                            style="muted",
                        ),
                    ]
                )

            self.console.print(
                self._panel(
                    Group(*body),
                    title=Text.assemble(
                        self._badge("API", "status.info.badge"),
                        (" Key setup", "panel.title"),
                    ),
                    subtitle=Text("startup", style="panel.subtitle"),
                )
            )

            if self._prompt_session is not None:
                return await self._prompt_session.prompt_async(
                    self._api_key_prompt_message(),
                    default="",
                    is_password=True,
                    mouse_support=False,
                )

            return Prompt.ask(
                "[meta.label]api key[/meta.label] [prompt.hint]›[/prompt.hint]",
                console=self.console,
                default="",
                password=True,
            )

    def print_welcome(self, *, release_info: ReleaseInfo | None = None) -> None:
        self._pause_live_status()
        workspace = self._home_relative(self.config.cwd)
        rows = [
            (
                "Version",
                release_info.current_version if release_info is not None else "unknown",
            ),
            ("Model", self.config.model_name),
            ("Workspace", workspace),
            ("Approval", self.config.approval.value),
            ("Turns", f"{self.config.max_turns} max"),
        ]
        if release_info is not None and release_info.update_available:
            assert release_info.latest_version is not None
            rows.insert(1, ("Update", f"{release_info.latest_version} available"))
        quick_actions = [
            ("/help", "Show the full command reference"),
            ("/scan", "Refresh workspace context"),
            ("/index", "Rebuild the symbol index"),
            ("/cwd", "Switch to another project"),
            ("/models", "Inspect or refresh provider models"),
            ("/config", "Review the current session settings"),
        ]

        self.console.print()
        self.console.print(Align.center(Text(self._LOGO, style="brand")))
        self.console.print(
            Rule(
                Text.assemble(
                    ("VORTEX TERMINAL", "muted"),
                    (
                        f" · v{release_info.current_version}" if release_info is not None else "",
                        "muted",
                    ),
                    (" · LOCAL AGENT · LIVE TOOLS", "muted"),
                    (
                        f" · workspace {Path(self.config.cwd).name}",
                        "muted",
                    ),
                    (
                        f" · profile {self.config.active_model_profile or 'default'}",
                        "muted",
                    ),
                ),
                style="panel.border",
            )
        )
        self.console.print()
        self.console.print(
            Columns(
                [
                    self._panel(
                        Group(
                            Text(
                                "Session is ready for workspace-aware coding tasks.",
                                style="muted",
                            ),
                            Text(),
                            self._kv_table(rows),
                        ),
                        title=Text.assemble(
                            self._badge("Session", "status.info.badge"),
                            (" Current context", "panel.title"),
                        ),
                        subtitle=Text("ready", style="panel.subtitle"),
                    ),
                    self._panel(
                        Group(
                            Text(
                                "Start with a prompt or one of the shortcuts below.",
                                style="muted",
                            ),
                            Text(),
                            self._action_table(quick_actions),
                        ),
                        title=Text.assemble(
                            self._badge("Start", "status.info.badge"),
                            (" Quick actions", "panel.title"),
                        ),
                        subtitle=Text("slash commands", style="panel.subtitle"),
                    ),
                ],
                expand=True,
                equal=True,
            )
        )
        self.console.print()
        self.console.print(
            self._summary_text(
                [
                    "Ctrl+C stops the current run and keeps the session open",
                    "type a message below to begin",
                ]
            )
        )

    def begin_assistant(self) -> None:
        self._pause_live_status()
        self.console.print()
        self._assistant_stream_open = True
        self._assistant_buffer = ""
        self._assistant_animation_phase = 0
        self._assistant_live = Live(
            self._render_assistant_panel(streaming=True),
            console=self.console,
            auto_refresh=False,
            transient=True,
        )
        self._assistant_live.start()

    def end_assistant(self) -> None:
        self._stop_assistant_live(keep_output=True)

    def _assistant_text_style_for_mode(self, mode: str) -> str:
        if mode == "inline_code":
            return "assistant.inline_code"
        if mode == "fenced_code":
            return "assistant.code_block"
        return "assistant"

    def _render_assistant_text(self, content: str) -> Text:
        rendered = Text()
        mode = "text"
        fence_header_pending = False
        line_index = 0
        column = 0
        base_styles = {
            "text": self.console.get_style(self._assistant_text_style_for_mode("text")),
            "inline_code": self.console.get_style(
                self._assistant_text_style_for_mode("inline_code")
            ),
            "fenced_code": self.console.get_style(
                self._assistant_text_style_for_mode("fenced_code")
            ),
        }

        def append_char(char: str) -> None:
            nonlocal line_index, column

            if char == "\n":
                rendered.append(char, style=base_styles[mode])
                line_index += 1
                column = 0
                return

            rendered.append(char, style=base_styles[mode])
            column += 1

        i = 0
        while i < len(content):
            char = content[i]

            if fence_header_pending:
                if char == "\n":
                    fence_header_pending = False
                    append_char(char)
                i += 1
                continue

            if char == "`":
                run_end = i
                while run_end < len(content) and content[run_end] == "`":
                    run_end += 1

                tick_run = content[i:run_end]
                if run_end == len(content):
                    if len(tick_run) == 1 and mode == "inline_code":
                        mode = "text"
                        i = run_end
                        continue
                    if len(tick_run) >= 3 and mode == "fenced_code":
                        mode = "text"
                        fence_header_pending = False
                        i = run_end
                        continue
                    for tick in tick_run:
                        append_char(tick)
                    i = run_end
                    continue

                if len(tick_run) >= 3 and mode != "inline_code":
                    if mode == "fenced_code":
                        mode = "text"
                    else:
                        mode = "fenced_code"
                        fence_header_pending = True
                elif len(tick_run) == 1 and mode != "fenced_code":
                    mode = "text" if mode == "inline_code" else "inline_code"
                else:
                    for tick in tick_run:
                        append_char(tick)

                i = run_end
                continue

            append_char(char)
            i += 1

        return rendered

    def _render_assistant_panel(self, *, streaming: bool) -> Panel:
        body = self._render_assistant_text(self._assistant_buffer)
        if not body.plain:
            body = Text(
                "Thinking through the reply...",
                style="assistant",
            )

        title = Text.assemble(
            self._badge("Agent", "assistant.badge"),
            (" VORTEX", "assistant.panel.title"),
            (" Response", "assistant.panel.title"),
        )
        subtitle = Text(
            "streaming" if streaming else "complete",
            style="assistant.panel.subtitle",
        )
        return Panel(
            body,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="assistant.panel.border",
            box=box.ROUNDED,
            padding=(1, 2),
            style=Style(bgcolor="#0f1115"),
        )

    def stream_assistant_delta(self, content: str) -> None:
        if not content:
            return

        self._assistant_buffer += content
        if self._assistant_live is not None:
            self._assistant_live.update(
                self._render_assistant_panel(streaming=True),
                refresh=True,
            )

    def clear_error(self) -> None:
        self._last_error = None

    def record_error(self, message: str | None) -> None:
        if not message:
            return
        self._last_error = self._short_text(str(message), limit=120)

    def reset_run_metrics(self) -> None:
        self._last_run_elapsed = None
        self._last_run_usage = None

    def set_run_metrics(self, elapsed_seconds: float | None, usage: TokenUsage | None) -> None:
        self._last_run_elapsed = elapsed_seconds
        self._last_run_usage = usage

    def _format_tokens(self, value: int) -> str:
        return f"{value:,}"

    def show_status(self, message: str) -> None:
        if self._status_suspended:
            return

        label = "status"
        detail = message
        if ":" in message:
            prefix, suffix = message.split(":", 1)
            if prefix.strip():
                label = prefix.strip()
                detail = suffix.strip() or prefix.strip()

        signature = (
            label.lower(),
            detail,
            round(self._last_run_elapsed or 0.0, 1) if self._last_run_elapsed else None,
            (
                (self._last_run_usage.total_tokens)
                if self._last_run_usage
                else None
            ),
            (
                self._last_run_usage.cached_tokens
                if self._last_run_usage
                else None
            ),
            self._last_error,
        )

        changed = signature != self._last_status_signature
        self._last_status_signature = signature
        self._last_status = message
        self._status_label = label.lower()
        self._status_detail = detail

        if changed:
            self._status_frame_index = 0

        if self._supports_live_status:
            if self._status_live is None:
                self.console.print()
                self._status_live = Live(
                    self._render_status_line(),
                    console=self.console,
                    auto_refresh=False,
                    transient=True,
                )
                self._status_live.start()

            self._refresh_status_line()
            return

        if changed:
            self.console.print()
            self.console.print(self._render_status_line())

    def advance_status_frame(self) -> None:
        if (
            self._status_suspended
            or not self._last_status
            or not self._supports_live_status
            or self._status_live is None
        ):
            return

        self._status_frame_index = (self._status_frame_index + 1) % len(
            self._STATUS_FRAMES
        )
        self._refresh_status_line()

    def suspend_status(self) -> None:
        self._status_suspended = True
        self.clear_status()

    def resume_status(self) -> None:
        self._status_suspended = False

    def _render_status_line(self) -> Text:
        line = Text()
        frame = self._STATUS_FRAMES[self._status_frame_index]
        line.append(f"{frame} ", style="thinking.badge")
        line.append(f"{self._status_label.upper()} ", style="meta.label")
        line.append("· ", style="muted")
        line.append(self._status_detail, style="thinking")

        if self._last_run_elapsed is not None:
            line.append(" · ", style="muted")
            line.append(f"{self._last_run_elapsed:.1f}s", style="meta.value")

        if self._last_run_usage:
            tokens = self._last_run_usage.total_tokens or (
                self._last_run_usage.prompt_tokens
                + self._last_run_usage.completion_tokens
            )
            token_parts = [f"tok {self._format_tokens(tokens)}"]
            if self._last_run_usage.cached_tokens:
                token_parts.append(
                    f"cached {self._format_tokens(self._last_run_usage.cached_tokens)}"
                )
            line.append(" · ", style="muted")
            line.append(" ".join(token_parts), style="meta.value")
        elif self._last_run_elapsed is not None:
            line.append(" · ", style="muted")
            line.append("tok n/a", style="muted")

        if self._last_error:
            line.append(" · ", style="muted")
            line.append("ERR ", style="status.error.badge")
            line.append(self._last_error, style="error")
        return line

    def _refresh_status_line(self) -> None:
        if self._status_live is None:
            return

        self._status_live.update(self._render_status_line(), refresh=True)

    def clear_status(self) -> None:
        self._last_status = None
        self._status_label = "thinking"
        self._status_detail = ""
        self._status_frame_index = 0

        if self._status_live is not None:
            self._status_live.stop()
            self._status_live = None

    def show_notice(self, message: str, level: str = "info") -> None:
        self._pause_live_status()
        text_style = {
            "info": "info",
            "success": "success",
            "warning": "warning",
            "error": "error",
        }.get(level, "info")
        markers = {
            "info": "i",
            "success": "ok",
            "warning": "!",
            "error": "x",
        }

        line = Text()
        line.append(f"{markers.get(level, 'i')} ", style=f"status.{level}.badge")
        line.append(level.upper(), style=text_style)
        line.append("  ", style="muted")
        line.append(message, style=text_style)
        self.console.print()
        self.console.print(line)

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        preferred_order = {
            "read_file": ["path", "offset", "limit"],
            "write_file": ["path", "create_directories", "content"],
            "edit": ["path", "replace_all", "old_string", "new_string"],
            "find_symbol": ["query", "kind", "language", "limit"],
            "shell": ["command", "timeout", "cwd"],
            "list_dir": ["path", "include_hidden"],
            "grep": ["path", "case_insensitive", "pattern"],
            "glob": ["path", "pattern"],
            "todos": ["id", "action", "content"],
            "memory": ["action", "key", "value"],
        }

        preferred = preferred_order.get(tool_name, [])
        ordered: list[tuple[str, Any]] = []
        seen: set[str] = set()

        for key in preferred:
            if key in args:
                ordered.append((key, args[key]))
                seen.add(key)

        for key in args:
            if key not in seen:
                ordered.append((key, args[key]))

        return ordered

    def _render_args_table(self, tool_name: str, args: dict[str, Any]) -> Table:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="meta.label", justify="right", no_wrap=True)
        table.add_column(style="code", overflow="fold")

        for key, value in self._ordered_args(tool_name, args):
            if isinstance(value, str) and key in {"content", "old_string", "new_string"}:
                line_count = len(value.splitlines()) or 0
                byte_count = len(value.encode("utf-8", errors="replace"))
                value = f"<{line_count} lines • {byte_count} bytes>"
            elif isinstance(value, bool):
                value = str(value).lower()

            table.add_row(key, str(value))

        return table

    def _tool_title(self, label: str, name: str, call_id: str, style: str) -> Text:
        title = Text()
        title.append_text(self._badge(label, style))
        title.append(" ")
        title.append(name, style="tool")
        return title

    def tool_call_start(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        arguments: dict[str, Any],
    ) -> None:
        self._pause_live_status()
        self._tool_args_by_call_id[call_id] = arguments
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"

        display_args = dict(arguments)
        for key in ("path", "cwd"):
            value = display_args.get(key)
            if isinstance(value, str) and self.cwd:
                display_args[key] = str(display_path_rel_to_cwd(value, self.cwd))

        body = (
            self._render_args_table(name, display_args)
            if display_args
            else Text("No arguments provided", style="muted")
        )
        summary = self._tool_context_summary(name, display_args)
        if summary:
            body = Group(summary, Text(), body)

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=self._tool_title("Tool", name, call_id, "tool.badge"),
                subtitle=self._tool_subtitle(
                    "running",
                    call_id=call_id,
                    tool_kind=tool_kind,
                ),
                border_style=border_style,
            )
        )

    def _extract_read_file_code(self, text: str) -> tuple[int, str] | None:
        body = text
        header_match = re.match(r"^Showing lines (\d+)-(\d+) of (\d+)\n\n", text)

        if header_match:
            body = text[header_match.end() :]

        code_lines: list[str] = []
        start_line: int | None = None

        for line in body.splitlines():
            match = re.match(r"^\s*(\d+)\|(.*)$", line)
            if not match:
                return None
            line_no = int(match.group(1))
            if start_line is None:
                start_line = line_no
            code_lines.append(match.group(2))

        if start_line is None:
            return None

        return start_line, "\n".join(code_lines)

    def _guess_language(self, path: str | None) -> str:
        if not path:
            return "text"

        suffix = Path(path).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "jsx",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".json": "json",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".css": "css",
            ".html": "html",
            ".xml": "xml",
            ".sql": "sql",
            ".asm": "nasm",
        }.get(suffix, "text")

    def _render_output_block(
        self,
        content: str,
        language: str = "text",
        *,
        word_wrap: bool = True,
    ) -> Syntax:
        return Syntax(
            content,
            language,
            theme="monokai",
            word_wrap=word_wrap,
            line_numbers=False,
        )

    def tool_call_complete(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        success: bool,
        output: str,
        error: str | None,
        metadata: dict[str, Any] | None,
        diff: str | None,
        truncated: bool,
        exit_code: int | None,
    ) -> None:
        self._pause_live_status()
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"
        state = "done" if success else "failed"
        state_style = "status.success.badge" if success else "status.error.badge"

        if not success:
            self.record_error(error or output or f"{name} failed")

        args = self._tool_args_by_call_id.get(call_id, {})
        blocks: list[Any] = []
        primary_path = (
            metadata.get("path")
            if isinstance(metadata, dict) and isinstance(metadata.get("path"), str)
            else None
        )

        if name == "read_file" and success:
            if primary_path:
                parsed = self._extract_read_file_code(output)
                shown_start = metadata.get("shown_start") if metadata else None
                shown_end = metadata.get("shown_end") if metadata else None
                total_lines = metadata.get("total_lines") if metadata else None
                header = self._summary_text(
                    [
                        display_path_rel_to_cwd(primary_path, self.cwd),
                        (
                            f"lines {shown_start}-{shown_end} of {total_lines}"
                            if shown_start and shown_end and total_lines
                            else None
                        ),
                    ]
                )
                blocks.append(header)

                if parsed:
                    start_line, code = parsed
                    blocks.append(
                        Syntax(
                            code,
                            self._guess_language(primary_path),
                            theme="monokai",
                            line_numbers=True,
                            start_line=start_line,
                            word_wrap=False,
                        )
                    )
                else:
                    blocks.append(
                        self._render_output_block(
                            truncate_text(output, "", self._max_block_tokens),
                            self._guess_language(primary_path),
                            word_wrap=False,
                        )
                    )
            else:
                blocks.append(
                    self._render_output_block(
                        truncate_text(output, "", self._max_block_tokens),
                        "text",
                        word_wrap=False,
                    )
                )
        elif name in {"write_file", "edit"} and success and diff:
            if output.strip():
                blocks.append(Text(output.strip(), style="muted"))
            blocks.append(
                Syntax(
                    truncate_text(diff, self.config.model_name, self._max_block_tokens),
                    "diff",
                    theme="monokai",
                    word_wrap=True,
                )
            )
        elif name == "shell" and success:
            command = args.get("command")
            if isinstance(command, str) and command.strip():
                blocks.append(Text(f"$ {command.strip()}", style="muted"))
            if exit_code is not None:
                blocks.append(Text(f"exit_code={exit_code}", style="muted"))
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "list_dir" and success:
            blocks.append(
                self._summary_text(
                    [
                        metadata.get("path") if metadata else None,
                        (
                            f"{metadata.get('entries')} entries"
                            if metadata and isinstance(metadata.get("entries"), int)
                            else None
                        ),
                    ]
                )
            )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "grep" and success:
            blocks.append(
                self._summary_text(
                    [
                        (
                            f"{metadata.get('matches')} matches"
                            if metadata and isinstance(metadata.get("matches"), int)
                            else None
                        ),
                        (
                            f"searched {metadata.get('files_searched')} files"
                            if metadata and isinstance(metadata.get("files_searched"), int)
                            else None
                        ),
                    ]
                )
            )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "find_symbol" and success:
            blocks.append(
                self._summary_text(
                    [
                        args.get("query"),
                        (
                            f"{metadata.get('matches')} matches"
                            if metadata and isinstance(metadata.get("matches"), int)
                            else None
                        ),
                        args.get("language"),
                        args.get("kind"),
                    ]
                )
            )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "glob" and success:
            if metadata and isinstance(metadata.get("matches"), int):
                blocks.append(Text(f"{metadata['matches']} matches", style="muted"))
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "web_search" and success:
            blocks.append(
                self._summary_text(
                    [
                        args.get("query"),
                        (
                            f"{metadata.get('results')} results"
                            if metadata and isinstance(metadata.get("results"), int)
                            else None
                        ),
                    ]
                )
            )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name == "web_fetch" and success:
            blocks.append(
                self._summary_text(
                    [
                        metadata.get("status_code") if metadata else None,
                        (
                            f"{metadata.get('content_length')} bytes"
                            if metadata and isinstance(metadata.get("content_length"), int)
                            else None
                        ),
                        args.get("url"),
                    ]
                )
            )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        elif name in {"todos", "memory"} and success:
            if name == "memory":
                blocks.append(
                    self._summary_text(
                        [
                            args.get("action"),
                            args.get("key"),
                            (
                                "found"
                                if metadata and metadata.get("found") is True
                                else "missing"
                                if metadata and metadata.get("found") is False
                                else None
                            ),
                        ]
                    )
                )
            blocks.append(
                self._render_output_block(
                    truncate_text(output, self.config.model_name, self._max_block_tokens)
                )
            )
        else:
            if error and not success:
                blocks.append(Text(error, style="error"))

            output_display = truncate_text(output, self.config.model_name, self._max_block_tokens)
            if output_display.strip():
                blocks.append(self._render_output_block(output_display))
            elif not error:
                blocks.append(Text("No output", style="muted"))

        if truncated:
            blocks.append(Text("Tool output was truncated to keep the interface readable.", style="warning"))

        self.console.print()
        self.console.print(
            self._panel(
                Group(*blocks),
                title=self._tool_title("Tool", name, call_id, "tool.badge"),
                subtitle=Text.assemble(
                    self._badge(state, state_style),
                    (
                        f"  {self._tool_subtitle(call_id=call_id, tool_kind=tool_kind).plain}",
                        "panel.subtitle",
                    ),
                ),
                border_style=border_style,
            )
        )


    """
        if u are this deep into this repo first of all thankyou for looking at my work and secondly i'm sorry for the state of this code, 
        it is very much a first draft and i expect to refactor it heavily as i continue development. 
        that being said, if you have any suggestions or want to contribute please feel free to open an issue or a pull request!
    """

    def handle_confirmation(self, confirmation: ToolConfirmation) -> bool:
        self.suspend_status()

        summary_parts = [confirmation.description]
        if confirmation.affected_paths:
            summary_parts.append(f"{len(confirmation.affected_paths)} path(s)")
        if confirmation.is_dangerous:
            summary_parts.append("higher risk")

        blocks: list[Any] = [
            self._summary_text(summary_parts, style="warning"),
            Text(confirmation.tool_name, style="tool"),
        ]

        if confirmation.command:
            blocks.append(Text(f"$ {confirmation.command}", style="warning"))

        if confirmation.diff:
            blocks.append(
                Syntax(
                    confirmation.diff.to_diff(),
                    "diff",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        self.console.print()
        self.console.print(
            self._panel(
                Group(*blocks),
                title=Text.assemble(
                    self._badge("Approval", "status.warning.badge"),
                    (" Tool confirmation required", "warning"),
                ),
                subtitle=Text("confirm with y / n", style="panel.subtitle"),
                border_style="warning",
            )
        )

        try:
            response = Prompt.ask(
                "\nApprove?",
                choices=["y", "n", "yes", "no"],
                default="n",
                console=self.console,
            )
            return response.lower() in {"y", "yes"}
        finally:
            self.resume_status()

    def show_env_change(
        self,
        previous: dict[str, str] | None,
        new: dict[str, str] | None,
    ) -> None:
        prev = previous or {}
        new_values = new or {}

        added = sorted(set(new_values) - set(prev))
        removed = sorted(set(prev) - set(new_values))
        changed = sorted(
            key
            for key in new_values
            if key in prev and new_values.get(key) != prev.get(key)
        )

        summary = []
        if added:
            summary.append(f"{len(added)} added")
        if changed:
            summary.append(f"{len(changed)} updated")
        if removed:
            summary.append(f"{len(removed)} removed")

        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Change", style="meta.label", no_wrap=True)
        table.add_column("Key", style="meta.value")

        def add_rows(keys: list[str], label: str):
            for key in keys[:12]:
                table.add_row(label, key)
            if len(keys) > 12:
                table.add_row(label, f"… {len(keys) - 12} more")

        add_rows(added, "added")
        add_rows(changed, "updated")
        add_rows(removed, "removed")

        body: Any
        if not summary:
            body = Text("No .env changes for this workspace.", style="muted")
        else:
            body = Group(
                self._summary_text(summary),
                Text("Keys only; values are hidden for safety.", style="muted"),
                table,
            )

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Env", "status.info.badge"),
                    (" Workspace .env delta", "panel.title"),
                ),
                subtitle=Text(self._home_relative(self.config.cwd), style="panel.subtitle"),
            )
        )

    def set_high_contrast(self, enabled: bool, *, persist: bool = True) -> None:
        if self._high_contrast == enabled:
            return
        self._high_contrast = enabled
        self._apply_theme()
        if persist:
            try:
                upsert_env_file(
                    self.config.cwd / ".env",
                    {"VORTEX_HIGH_CONTRAST": "1" if enabled else "0"},
                )
            except Exception:
                # Avoid breaking the session if writing fails
                pass

    def show_config(self) -> None:
        base_url = self.config.base_url or "default"
        rows: list[tuple[str, object]] = [
            ("Model", self.config.model_name),
            (
                "Active profile",
                self.config.active_model_profile or "default",
            ),
            ("Base URL", base_url),
            ("API key source", self.config.api_key_source_label),
            ("Temperature", self.config.temperature),
            ("Max output tokens", self.config.max_output_tokens),
            ("Approval", self.config.approval.value),
            ("Working dir", self.config.cwd),
            ("Max turns", self.config.max_turns),
            ("Configured profiles", len(self.config.models)),
            ("Hooks enabled", self.config.hooks_enabled),
        ]

        if self.config.profile_uses_gemini_openai_compat(
            self.config.active_model_profile
        ):
            gemini_overrides = self.config.request_overrides
            if gemini_overrides.get("reasoning_effort"):
                rows.append(
                    ("Gemini reasoning", gemini_overrides["reasoning_effort"])
                )

            google = gemini_overrides.get("extra_body", {}).get("google", {})
            thinking = google.get("thinking_config", {})
            if thinking:
                thinking_bits: list[str] = []
                if "include_thoughts" in thinking:
                    thinking_bits.append(
                        f"include_thoughts={thinking['include_thoughts']}"
                    )
                if "thinking_level" in thinking:
                    thinking_bits.append(
                        f"thinking_level={thinking['thinking_level']}"
                    )
                if "thinking_budget" in thinking:
                    thinking_bits.append(
                        f"thinking_budget={thinking['thinking_budget']}"
                    )
                rows.append(("Gemini thinking", ", ".join(thinking_bits)))

            if google.get("cached_content"):
                rows.append(("Gemini cached content", google["cached_content"]))

        self.console.print()
        self.console.print(
            self._panel(
                Group(
                    Text(
                        "Resolved settings for the current workspace and active model profile.",
                        style="muted",
                    ),
                    Text(),
                    self._kv_table(rows),
                ),
                title=Text.assemble(
                    self._badge("Config", "status.info.badge"),
                    (" Current settings", "panel.title"),
                ),
                subtitle=Text(self.config.active_model_label, style="panel.subtitle"),
            )
        )

    def show_model_profiles(
        self,
        config: Config,
        *,
        model_catalog: list[ModelCatalogResult] | None = None,
        model_health: dict[tuple[str, str], ModelHealthRecord] | None = None,
    ) -> None:
        profile_table = Table(expand=True, box=None, padding=(0, 1))
        profile_table.add_column("Name", style="meta.value")
        profile_table.add_column("Model", style="meta.value")
        profile_table.add_column("Base URL", style="muted")
        profile_table.add_column("Key source", style="muted")
        profile_table.add_column("Active", style="meta.value", no_wrap=True)

        profiles = config.list_model_profiles()
        for name, profile in profiles:
            profile_table.add_row(
                name,
                profile.model.name,
                config.resolve_profile_base_url(name),
                config.resolve_profile_key_source_label(name),
                "yes" if config.active_model_profile == name else "",
            )

        profile_section = (
            profile_table
            if profiles
            else Group(
                Text("No custom model profiles configured yet.", style="muted"),
                Text(
                    "Add them in .ai-agent/config.toml under [models.<name>].",
                    style="muted",
                ),
            )
        )

        catalog_results = model_catalog or []
        catalog_models = order_model_catalog_entries(
            flatten_model_catalog(
                catalog_results,
                health_records=model_health,
            ),
            active_profile_name=config.active_model_profile,
            active_model_name=config.model_name,
        )
        bucket_counts = Counter(model_status_bucket(entry.status) for entry in catalog_models)
        grouped_models = group_model_catalog_entries_by_bucket(catalog_models)

        model_sections: list[Any] = []
        for bucket, entries in grouped_models:
            status_title = self._model_bucket_text(bucket)
            status_title.append(f" ({len(entries)})", style="muted")
            show_note_column = any(entry.note for entry in entries)

            section_table = Table(expand=True, box=None, padding=(0, 1))
            section_table.add_column("#", style="meta.label", no_wrap=True)
            section_table.add_column("Model ID", style="meta.value")
            section_table.add_column("Profile", style="muted", no_wrap=True)
            section_table.add_column("Reason", no_wrap=True)
            if show_note_column:
                section_table.add_column("Note", style="muted")
            section_table.add_column("Base URL", style="muted")
            section_table.add_column("Checked", style="muted", no_wrap=True)
            section_table.add_column("Active", style="meta.value", no_wrap=True)

            for entry in entries:
                row = [
                    str(entry.index),
                    entry.model_name,
                    entry.display_name,
                    self._model_status_text(entry.status),
                ]
                if show_note_column:
                    row.append(self._short_text(entry.note, 72))
                row.extend(
                    [
                        entry.base_url,
                        self._format_checked_at(entry.checked_at),
                        (
                        "yes"
                        if config.active_model_profile == entry.profile_name
                        and config.model_name == entry.model_name
                        else ""
                        ),
                    ]
                )
                section_table.add_row(*row)

            model_sections.extend([status_title, section_table, Text()])

        provider_issue_rows: list[tuple[str, str, Text, Text, str]] = []
        for result in catalog_results:
            if result.models:
                continue

            lowered_error = (result.error or "").lower()
            status = "missing-key" if "no api key" in lowered_error else "error"
            bucket = model_status_bucket(status)
            provider_issue_rows.append(
                (
                    result.source.display_name,
                    result.source.base_url,
                    self._model_bucket_text(bucket),
                    self._model_status_text(status),
                    self._format_checked_at(result.checked_at),
                )
            )

        if provider_issue_rows:
            issues_table = Table(expand=True, box=None, padding=(0, 1))
            issues_table.add_column("Provider", style="meta.value")
            issues_table.add_column("Base URL", style="muted")
            issues_table.add_column("Bucket", no_wrap=True)
            issues_table.add_column("Reason", no_wrap=True)
            issues_table.add_column("Checked", style="muted", no_wrap=True)

            for provider_name, base_url, bucket_text, status_text, checked_at in provider_issue_rows:
                issues_table.add_row(
                    provider_name,
                    base_url,
                    bucket_text,
                    status_text,
                    checked_at,
                )

            model_sections.extend(
                [
                    Text("provider issues", style="warning"),
                    issues_table,
                ]
            )

        catalog_section = (
            Group(*model_sections)
            if model_sections
            else Text("No provider-backed model listings are available.", style="muted")
        )

        summary_parts: list[str] = []
        for bucket in MODEL_BUCKET_ORDER:
            count = bucket_counts.get(bucket)
            if count:
                label = "not working" if bucket == "not-working" else bucket
                summary_parts.append(f"{count} {label}")

        summary_block = Group(
            self._summary_text(summary_parts)
            if summary_parts
            else Text("Refresh to check live model availability.", style="muted"),
            Text(
                "Models are grouped from ready to degraded so the best options stay at the top.",
                style="muted",
            ),
        )

        body = Group(
            Text("Profiles", style="meta.label"),
            profile_section,
            Text(),
            Text("Discovered models", style="meta.label"),
            summary_block,
            Text(
                "Use /models refresh to update live status, then /model <number> to switch using the grouped order shown here.",
                style="muted",
            ),
            catalog_section,
        )

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Models", "status.info.badge"),
                    (f" Profiles {len(profiles)}", "panel.title"),
                    (" · ", "muted"),
                    (f"Models {len(catalog_models)}", "panel.title"),
                ),
                subtitle=Text(
                    f"active: {config.active_model_label}",
                    style="panel.subtitle",
                ),
            )
        )

    def show_workspace_snapshot(self, summary: str | None) -> None:
        self.console.print()

        if not summary:
            body = Text("No workspace snapshot is available yet.", style="muted")
        else:
            body = Syntax(
                summary,
                "markdown",
                theme="monokai",
                word_wrap=True,
                line_numbers=False,
            )

        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Scan", "status.info.badge"),
                    (" Workspace snapshot", "panel.title"),
                ),
                subtitle=Text(self._home_relative(self.config.cwd), style="panel.subtitle"),
            )
        )

    def show_code_index(self, summary: str | None) -> None:
        self.console.print()

        if not summary:
            body = Text("No code index is available yet.", style="muted")
        else:
            body = Syntax(
                summary,
                "markdown",
                theme="monokai",
                word_wrap=True,
                line_numbers=False,
            )

        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Index", "status.info.badge"),
                    (" Codebase index", "panel.title"),
                ),
                subtitle=Text(self._home_relative(self.config.cwd), style="panel.subtitle"),
            )
        )

    def show_recent_workspaces(self, entries: list[dict[str, str]]) -> None:
        self.console.print()

        if not entries:
            body = Text("No recent workspaces recorded yet.", style="muted")
        else:
            table = Table(expand=True, box=None, padding=(0, 1))
            table.add_column("#", style="meta.label", no_wrap=True)
            table.add_column("Directory", style="meta.value")
            table.add_column("Last used", style="muted", no_wrap=True)

            for index, entry in enumerate(entries, start=1):
                table.add_row(
                    str(index),
                    self._home_relative(entry.get("path", "")),
                    entry.get("last_used", ""),
                )
            body = table

        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Recent", "status.info.badge"),
                    (" Recent workspaces", "panel.title"),
                ),
                subtitle=Text(f"{len(entries)} remembered", style="panel.subtitle"),
            )
        )

    def show_stats(self, stats: dict[str, Any]) -> None:
        self.console.print()
        self.console.print(
            self._panel(
                self._kv_table(stats.items()),
                title=Text.assemble(
                    self._badge("Stats", "status.info.badge"),
                    (" Session statistics", "panel.title"),
                ),
                subtitle=Text(self.config.active_model_label, style="panel.subtitle"),
            )
        )

    def show_tools(self, tools: list[Any]) -> None:
        rows = []
        for tool in tools:
            rows.append((tool.name, getattr(tool.kind, "value", "tool")))

        self.console.print()
        self.console.print(
            self._panel(
                self._kv_table(rows) if rows else Text("No tools available", style="muted"),
                title=Text.assemble(
                    self._badge("Tools", "tool.badge"),
                    (f" Available tools ({len(tools)})", "panel.title"),
                ),
                subtitle=Text(self._home_relative(self.config.cwd), style="panel.subtitle"),
            )
        )

    def show_mcp_servers(self, servers: list[dict[str, Any]]) -> None:
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Server", style="meta.value")
        table.add_column("Status", style="meta.value", no_wrap=True)
        table.add_column("Tools", style="meta.value", justify="right")

        for server in servers:
            status = str(server.get("status", "unknown"))
            status_style = "success" if status == "connected" else "error"
            table.add_row(
                str(server.get("name", "unknown")),
                Text(status, style=status_style),
                str(server.get("tools", 0)),
            )

        self.console.print()
        self.console.print(
            self._panel(
                table if servers else Text("No MCP servers configured", style="muted"),
                title=Text.assemble(
                    self._badge("MCP", "status.info.badge"),
                    (f" Servers ({len(servers)})", "panel.title"),
                ),
                subtitle=Text("model context protocol", style="panel.subtitle"),
            )
        )

    def show_saved_sessions(self, sessions: list[dict[str, Any]], title_text: str) -> None:
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("ID", style="meta.value")
        table.add_column("Turns", style="meta.value", justify="right")
        table.add_column("Updated", style="meta.value")

        for session in sessions:
            table.add_row(
                str(session.get("session_id", "")),
                str(session.get("turn_count", "")),
                str(session.get("updated_at", "")),
            )

        self.console.print()
        self.console.print(
            self._panel(
                table if sessions else Text("Nothing saved yet", style="muted"),
                title=Text.assemble(
                    self._badge("Sessions", "status.info.badge"),
                    (f" {title_text}", "panel.title"),
                ),
                subtitle=Text(f"{len(sessions)} item(s)", style="panel.subtitle"),
            )
        )

    def show_key_help(self) -> None:
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Key / Command", style="meta.value", no_wrap=True)
        table.add_column("Action", style="muted")
        table.add_row("Enter", "Send the prompt to the agent")
        table.add_row("Ctrl+C", "Stop the current run; press again to quit")
        table.add_row("/clear", "Reset conversation and loop detector")
        table.add_row("/cwd", "Switch workspace and reload tools/context")
        table.add_row("?", "Show this keyboard cheat sheet")
        table.add_row("/help", "Full command reference")

        self.console.print()
        self.console.print(
            self._panel(
                table,
                title=Text.assemble(
                    self._badge("Keys", "status.info.badge"),
                    (" Shortcuts", "panel.title"),
                ),
                subtitle=Text("navigation", style="panel.subtitle"),
            )
        )

    def show_help(self) -> None:
        commands = Table(expand=True, box=None, padding=(0, 1))
        commands.add_column("Command", style="meta.value", no_wrap=True)
        commands.add_column("What it does", style="muted")
        commands.add_row(Text("/help"), "Show this help screen")
        commands.add_row(Text("/exit or /quit"), "Exit the agent")
        commands.add_row(Text("/clear"), "Clear conversation history")
        commands.add_row(Text("/scan"), "Refresh and show workspace context")
        commands.add_row(Text("/index"), "Refresh and show the codebase symbol index")
        commands.add_row(Text("/cwd [path|index]"), "Switch to another project directory")
        commands.add_row(Text("/recent"), "Show remembered workspaces")
        commands.add_row(Text("/config"), "Show current configuration")
        commands.add_row(Text("/api-change"), "Re-enter provider URL and API key (restarts session)")
        commands.add_row(Text("/models [refresh]"), "List model profiles and discover models for each configured API key")
        commands.add_row(Text("/model <name|number>"), "Switch profile, pick a discovered model, or change the model name")
        commands.add_row(Text("/model force <name|number>"), "Change the model name directly even if it is not in the discovered list")
        commands.add_row(Text("/change-model"), "When using local Ollama, switch installed models or install another local model")
        commands.add_row(Text("/approval <mode>"), "Change approval mode")
        commands.add_row(Text("/stats"), "Show session statistics")
        commands.add_row(Text("/tools"), "List available tools")
        commands.add_row(Text("/mcp"), "Show MCP server status")
        commands.add_row(Text("/mcp attach <name> <url|command>"), "Attach an MCP server at runtime")
        commands.add_row(Text("/contrast [on|off|toggle]"), "Switch between standard and high-contrast themes")
        commands.add_row(Text("/save"), "Save the current session")
        commands.add_row(Text("/checkpoint [name]"), "Create a checkpoint")
        commands.add_row(Text("/sessions"), "List saved sessions")
        commands.add_row(Text("/resume <session_id>"), "Resume a saved session")
        commands.add_row(Text("/restore <checkpoint_id>"), "Restore a checkpoint")
        commands.add_row(Text("? or /?"), "Show keyboard shortcuts")

        tips = self._panel(
            Group(
                Text("Type a normal message to start an agent run.", style="muted"),
                Text("Press Ctrl+C to stop the current run and return to the prompt.", style="muted"),
                Text("Tool calls, approvals, and outputs are shown as structured cards.", style="muted"),
                Text("Use /cwd to switch projects; the session reloads for the new directory.", style="muted"),
                Text("Use /index or the find_symbol tool to jump to definitions faster.", style="muted"),
                Text("Use vortex --update outside the app to install the latest published release.", style="muted"),
                Text(
                    "Add custom providers in .ai-agent/config.toml with [models.<name>].",
                    style="muted",
                ),
            ),
            title=Text("Workflow", style="panel.title"),
            subtitle=Text("recommended flow", style="panel.subtitle"),
        )

        self.console.print()
        self.console.print(
            Columns(
                [
                    self._panel(
                        commands,
                        title=Text.assemble(
                            self._badge("Help", "status.info.badge"),
                            (" Commands", "panel.title"),
                        ),
                    ),
                    tips,
                ],
                expand=True,
                equal=True,
            )
        )
