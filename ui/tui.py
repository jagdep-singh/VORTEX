from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from config.config import Config
from tools.base import ToolConfirmation
from utils.paths import display_path_rel_to_cwd
from utils.text import truncate_text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
except Exception:
    PromptSession = None
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
        "assistant.code_block": "#c9c9c9 on #121212",
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
    _LOGO = "\n".join(
        [
            "     ██╗ █████╗ ███████╗███████╗       ██████╗ ██████╗ ██████╗ ███████╗",
            "     ██║██╔══██╗╚══███╔╝╚══███╔╝      ██╔════╝██╔═══██╗██╔══██╗██╔════╝",
            "     ██║███████║  ███╔╝   ███╔╝ █████╗██║     ██║   ██║██║  ██║█████╗",
            "██   ██║██╔══██║ ███╔╝   ███╔╝  ╚════╝██║     ██║   ██║██║  ██║██╔══╝",
            "╚█████╔╝██║  ██║███████╗███████╗      ╚██████╗╚██████╔╝██████╔╝███████╗",
            " ╚════╝ ╚═╝  ╚═╝╚══════╝╚══════╝       ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
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
        self._assistant_mode = "text"
        self._assistant_pending_backticks = ""
        self._assistant_fence_header_pending = False
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        self._max_block_tokens = 2500
        self._last_status: str | None = None
        self._prompt_session = None

        if (
            PromptSession is not None
            and InMemoryHistory is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            self._prompt_session = PromptSession(
                history=InMemoryHistory(),
            )

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

    def prompt(self) -> str:
        return (
            f"\n[prompt]{self._PROMPT_MARKER}[/prompt] "
            "[prompt.hint]you › [/prompt.hint]"
        )

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

    async def read_prompt(self) -> str:
        if self._prompt_session is not None:
            return await self._prompt_session.prompt_async(
                self._prompt_message(),
                mouse_support=False,
            )

        return self.console.input(self.prompt())

    def print_welcome(self) -> None:
        workspace = self._home_relative(self.config.cwd)
        rows = [
            ("model", self.config.model_name),
            ("workspace", workspace),
            ("approval", self.config.approval.value),
            ("turns", f"{self.config.max_turns} max"),
        ]

        value_width = max(len(value) for _, value in rows)
        inner_width = max(62, 13 + value_width)
        top = Text(f"┌─ session {'─' * (inner_width - 12)}┐", style="dim")
        bottom = Text(f"└{'─' * inner_width}┘", style="dim")

        session_lines: list[Text] = [top]
        for label, value in rows:
            line = Text("│  ", style="dim")
            line.append(f"{label:<9}", style="meta.label")
            line.append(f" {value:<{inner_width - 13}}", style="meta.value")
            line.append("│", style="dim")
            session_lines.append(line)
        session_lines.append(bottom)

        self.console.print()
        self.console.print(Text(self._LOGO, style="brand"))
        self.console.print(
            Text(
                "─── A LOCAL CODING AGENT  ·  TOOLS  ·  APPROVALS ───",
                style="muted",
            )
        )
        self.console.print()
        for line in session_lines:
            self.console.print(line)
        self.console.print()
        self.console.print(Text("commands", style="muted"))
        self.console.print(
            self._command_pills(
                ["/help", "/scan", "/index", "/config", "/models", "/approval", "/model", "/exit"]
            )
        )
        self.console.print()
        self.console.print(
            Text("Ctrl+C to stop current run · stay in session", style="muted")
        )

    def begin_assistant(self) -> None:
        title = Text()
        title.append("assistant", style="muted")
        title.append(" · ", style="dim")
        title.append("Jazz-Code", style="assistant.badge")

        self.console.print()
        self.console.print(Rule(title, style="panel.border"))
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
        if message == self._last_status:
            return

        label = "status"
        detail = message
        if ":" in message:
            prefix, suffix = message.split(":", 1)
            if prefix.strip():
                label = prefix.strip()
                detail = suffix.strip() or prefix.strip()

        line = Text()
        line.append(f"{self._PROMPT_MARKER} ", style="thinking.badge")
        line.append(f"{label.lower()} › ", style="muted")
        line.append(detail, style="thinking")

        self.console.print()
        self.console.print(line)
        self._last_status = message

    def clear_status(self) -> None:
        self._last_status = None

    def show_notice(self, message: str, level: str = "info") -> None:
        text_style = {
            "info": "info",
            "success": "success",
            "warning": "warning",
            "error": "error",
        }.get(level, "info")

        line = Text()
        line.append(f"{self._PROMPT_MARKER} ", style="thinking.badge")
        line.append(f"{level} › ", style="muted")
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
        if call_id:
            title.append("  ")
            title.append(f"#{call_id[:8]}", style="muted")
        return title

    def tool_call_start(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        arguments: dict[str, Any],
    ) -> None:
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

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=self._tool_title("Tool", name, call_id, "tool.badge"),
                subtitle=Text("running", style="panel.subtitle"),
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
                subtitle=self._badge(state, state_style),
                border_style=border_style,
            )
        )

    def handle_confirmation(self, confirmation: ToolConfirmation) -> bool:
        blocks: list[Any] = [
            Text(confirmation.tool_name, style="tool"),
            Text(confirmation.description, style="meta.value"),
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
                subtitle=Text("y / n", style="panel.subtitle"),
                border_style="warning",
            )
        )

        response = Prompt.ask(
            "\nApprove?",
            choices=["y", "n", "yes", "no"],
            default="n",
            console=self.console,
        )
        return response.lower() in {"y", "yes"}

    def show_config(self) -> None:
        base_url = self.config.base_url or "default"
        self.console.print()
        self.console.print(
            self._panel(
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
                        ("Approval", self.config.approval.value),
                        ("Working dir", self.config.cwd),
                        ("Max turns", self.config.max_turns),
                        ("Configured profiles", len(self.config.models)),
                        ("Hooks enabled", self.config.hooks_enabled),
                    ]
                ),
                title=Text.assemble(
                    self._badge("Config", "status.info.badge"),
                    (" Current settings", "panel.title"),
                ),
            )
        )

    def show_model_profiles(self, config: Config) -> None:
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Name", style="meta.value")
        table.add_column("Model", style="meta.value")
        table.add_column("Base URL", style="muted")
        table.add_column("Key source", style="muted")
        table.add_column("Active", style="meta.value", no_wrap=True)

        profiles = config.list_model_profiles()
        for name, profile in profiles:
            table.add_row(
                name,
                profile.model.name,
                profile.base_url or "default",
                profile.key_source_label,
                "yes" if config.active_model_profile == name else "",
            )

        if not profiles:
            body = Group(
                Text("No custom model profiles configured yet.", style="muted"),
                Text(
                    "Add them in .ai-agent/config.toml under [models.<name>].",
                    style="muted",
                ),
            )
        else:
            body = table

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=Text.assemble(
                    self._badge("Models", "status.info.badge"),
                    (f" Configured profiles ({len(profiles)})", "panel.title"),
                ),
                subtitle=Text(
                    f"active: {config.active_model_profile or 'default'}",
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
        commands.add_row(Text("/config"), "Show current configuration")
        commands.add_row(Text("/models"), "List configured model profiles")
        commands.add_row(Text("/model <name>"), "Switch profile or change model name")
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
                Text("Use /index or the find_symbol tool to jump to definitions faster.", style="muted"),
                Text(
                    "Add custom providers in .ai-agent/config.toml with [models.<name>].",
                    style="muted",
                ),
            ),
            title=Text("Workflow", style="panel.title"),
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
