from __future__ import annotations

import os
import re
import sys
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

from config.config import Config
from tools.base import ToolConfirmation
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

AGENT_THEME = Theme(
    {
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
)

_console: Console | None = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class TUI:
    _PROMPT_MARKER = "╰─"
    _WORKSPACE_PROMPT_LABEL = "workspace"
    _STATUS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _ASSISTANT_AURORA = (
        "#102636",
        "#153244",
        "#1a3e52",
        "#1f4a60",
        "#25566d",
        "#2c627b",
    )
    _ASSISTANT_INLINE_AURORA = (
        "#1e3944",
        "#244552",
        "#2b5260",
        "#325f70",
    )
    _ASSISTANT_CODE_AURORA = (
        "#11181f",
        "#152029",
        "#192733",
        "#1d2f3c",
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

    def set_config(self, config: Config) -> None:
        self.config = config
        self.cwd = config.cwd

    def _pause_live_status(self) -> None:
        if self._status_live is not None:
            self._status_live.stop()
            self._status_live = None

    def clear_screen(self) -> None:
        self._pause_live_status()
        self.console.clear(home=True)

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
            "quota": "warning",
            "not-working": "error",
        }
        text_map = {
            "working": "working",
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
        self.clear_screen()
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
        default_url: str,
        api_key_env_name: str,
        docs_path: Path | None = None,
        error_message: str | None = None,
        info_message: str | None = None,
    ) -> str:
        self.clear_screen()
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
                Text("Common provider URLs:", style="meta.label"),
                Text("https://api.openai.com/v1", style="meta.value"),
                Text("https://openrouter.ai/api/v1", style="meta.value"),
                Text("http://localhost:11434/v1", style="meta.value"),
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
                default=default_url,
                mouse_support=False,
            )

        return Prompt.ask(
            "[meta.label]provider[/meta.label] [prompt.hint]›[/prompt.hint]",
            console=self.console,
            default=default_url,
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
        self.clear_screen()
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

    def print_welcome(self) -> None:
        self._pause_live_status()
        workspace = self._home_relative(self.config.cwd)
        rows = [
            ("Model", self.config.model_name),
            ("Workspace", workspace),
            ("Approval", self.config.approval.value),
            ("Turns", f"{self.config.max_turns} max"),
        ]
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
                Text("VORTEX TERMINAL · LOCAL AGENT · LIVE TOOLS", style="muted"),
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
        self.console.print(
            Rule(
                Text.assemble(
                    self._badge("Agent", "assistant.badge"),
                    (" VORTEX", "assistant.badge"),
                ),
                style="panel.border",
            )
        )
        self._assistant_stream_open = True
        self._assistant_mode = "text"
        self._assistant_pending_backticks = ""
        self._assistant_fence_header_pending = False

    def end_assistant(self) -> None:
        self._finalize_assistant_pending_backticks()
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False
        self._assistant_mode = "text"
        self._assistant_pending_backticks = ""
        self._assistant_fence_header_pending = False

    def _assistant_text_style(self) -> str:
        if self._assistant_mode == "inline_code":
            return "assistant.inline_code"
        if self._assistant_mode == "fenced_code":
            return "assistant.code_block"
        return "assistant"

    def _flush_assistant_pending_backticks(self) -> None:
        if not self._assistant_pending_backticks:
            return

        self.console.print(
            Text(self._assistant_pending_backticks, style=self._assistant_text_style()),
            end="",
        )
        self._assistant_pending_backticks = ""

    def _finalize_assistant_pending_backticks(self) -> None:
        if not self._assistant_pending_backticks:
            return

        tick_count = len(self._assistant_pending_backticks)

        if tick_count == 1 and self._assistant_mode == "inline_code":
            self._assistant_pending_backticks = ""
            self._assistant_mode = "text"
            return

        if tick_count >= 3 and self._assistant_mode == "fenced_code":
            self._assistant_pending_backticks = ""
            self._assistant_mode = "text"
            self._assistant_fence_header_pending = False
            return

        self._flush_assistant_pending_backticks()

    def stream_assistant_delta(self, content: str) -> None:
        text = f"{self._assistant_pending_backticks}{content}"
        self._assistant_pending_backticks = ""

        rendered = Text()
        segment_buffer: list[str] = []
        segment_style = self._assistant_text_style()

        def flush_segment() -> None:
            nonlocal segment_buffer
            if segment_buffer:
                rendered.append("".join(segment_buffer), style=segment_style)
                segment_buffer = []

        i = 0
        while i < len(text):
            char = text[i]

            if self._assistant_fence_header_pending:
                if char == "\n":
                    self._assistant_fence_header_pending = False
                    new_style = self._assistant_text_style()
                    if new_style != segment_style:
                        flush_segment()
                    segment_style = new_style
                    segment_buffer.append(char)
                i += 1
                continue

            if char == "`":
                run_end = i
                while run_end < len(text) and text[run_end] == "`":
                    run_end += 1

                tick_run = text[i:run_end]
                if run_end == len(text):
                    self._assistant_pending_backticks = tick_run
                    break

                flush_segment()

                if len(tick_run) >= 3 and self._assistant_mode != "inline_code":
                    if self._assistant_mode == "fenced_code":
                        self._assistant_mode = "text"
                    else:
                        self._assistant_mode = "fenced_code"
                        self._assistant_fence_header_pending = True
                    segment_style = self._assistant_text_style()
                elif len(tick_run) == 1 and self._assistant_mode != "fenced_code":
                    self._assistant_mode = (
                        "text" if self._assistant_mode == "inline_code" else "inline_code"
                    )
                    segment_style = self._assistant_text_style()
                else:
                    rendered.append(tick_run, style=self._assistant_text_style())
                    segment_style = self._assistant_text_style()

                i = run_end
                continue

            new_style = self._assistant_text_style()
            if new_style != segment_style:
                flush_segment()
                segment_style = new_style
            segment_buffer.append(char)
            i += 1

        flush_segment()

        if rendered:
            self.console.print(rendered, end="")

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

        changed = (
            message != self._last_status
            or label.lower() != self._status_label
            or detail != self._status_detail
        )
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

        if not changed:
            return

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

    def show_config(self) -> None:
        base_url = self.config.base_url or "default"
        self.console.print()
        self.console.print(
            self._panel(
                Group(
                    Text(
                        "Resolved settings for the current workspace and active model profile.",
                        style="muted",
                    ),
                    Text(),
                    self._kv_table(
                        [
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
                    ),
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

            section_table = Table(expand=True, box=None, padding=(0, 1))
            section_table.add_column("#", style="meta.label", no_wrap=True)
            section_table.add_column("Model ID", style="meta.value")
            section_table.add_column("Profile", style="muted", no_wrap=True)
            section_table.add_column("Reason", no_wrap=True)
            section_table.add_column("Base URL", style="muted")
            section_table.add_column("Checked", style="muted", no_wrap=True)
            section_table.add_column("Active", style="meta.value", no_wrap=True)

            for entry in entries:
                section_table.add_row(
                    str(entry.index),
                    entry.model_name,
                    entry.display_name,
                    self._model_status_text(entry.status),
                    entry.base_url,
                    self._format_checked_at(entry.checked_at),
                    (
                        "yes"
                        if config.active_model_profile == entry.profile_name
                        and config.model_name == entry.model_name
                        else ""
                    ),
                )

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
        commands.add_row(Text("/models [refresh]"), "List model profiles and discover models for each configured API key")
        commands.add_row(Text("/model <name|number>"), "Switch profile, pick a discovered model, or change the model name")
        commands.add_row(Text("/model force <name|number>"), "Change the model name directly even if it is not in the discovered list")
        commands.add_row(Text("/approval <mode>"), "Change approval mode")
        commands.add_row(Text("/stats"), "Show session statistics")
        commands.add_row(Text("/tools"), "List available tools")
        commands.add_row(Text("/mcp"), "Show MCP server status")
        commands.add_row(Text("/save"), "Save the current session")
        commands.add_row(Text("/checkpoint [name]"), "Create a checkpoint")
        commands.add_row(Text("/sessions"), "List saved sessions")
        commands.add_row(Text("/resume <session_id>"), "Resume a saved session")
        commands.add_row(Text("/restore <checkpoint_id>"), "Restore a checkpoint")

        tips = self._panel(
            Group(
                Text("Type a normal message to start an agent run.", style="muted"),
                Text("Press Ctrl+C to stop the current run and return to the prompt.", style="muted"),
                Text("Tool calls, approvals, and outputs are shown as structured cards.", style="muted"),
                Text("Use /cwd to switch projects; the session reloads for the new directory.", style="muted"),
                Text("Use /index or the find_symbol tool to jump to definitions faster.", style="muted"),
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
