from __future__ import annotations

import re
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

AGENT_THEME = Theme(
    {
        "info": "bright_cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "muted": "grey58",
        "dim": "grey42",
        "surface": "grey11",
        "surface.alt": "grey15",
        "panel.border": "grey27",
        "panel.title": "bold bright_cyan",
        "panel.subtitle": "grey58",
        "brand": "bold bright_cyan",
        "brand.badge": "bold black on bright_cyan",
        "chip": "bold black on grey70",
        "prompt": "bold bright_cyan",
        "prompt.hint": "grey58",
        "user": "bright_blue bold",
        "user.badge": "bold black on bright_blue",
        "assistant": "bright_white",
        "assistant.badge": "bold black on bright_cyan",
        "assistant.inline_code": "bold bright_white on grey23",
        "assistant.code_block": "white on grey11",
        "thinking": "italic grey70",
        "thinking.badge": "bold black on grey78",
        "meta.label": "grey58",
        "meta.value": "bright_white",
        "tool": "bold bright_magenta",
        "tool.badge": "bold black on bright_magenta",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        "status.info.badge": "bold black on bright_cyan",
        "status.success.badge": "bold black on green",
        "status.warning.badge": "bold black on yellow",
        "status.error.badge": "bold white on bright_red",
        "code": "bright_white",
    }
)

_console: Console | None = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class TUI:
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

    def _badge(self, label: str, style: str) -> Text:
        return Text(f" {label.upper()} ", style=style)

    def _chip_row(self, items: Iterable[str]) -> Text:
        text = Text()
        for index, item in enumerate(items):
            if index:
                text.append(" ")
            text.append_text(self._badge(item, "chip"))
        return text

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
        return "\n[user.badge] YOU [/user.badge] [prompt]›[/prompt] "

    def print_welcome(self) -> None:
        header = Text()
        header.append_text(self._badge("Jazz-Code", "brand.badge"))
        header.append(" ")
        header.append("Agent Console", style="brand")

        subtitle = Text(
            "A local coding shell with streaming replies, tool execution, approvals, and session controls.",
            style="muted",
        )

        overview = self._panel(
            self._kv_table(
                [
                    ("Model", self.config.model_name),
                    ("Workspace", self.config.cwd),
                    ("Approval", self.config.approval.value),
                    ("Max turns", self.config.max_turns),
                ]
            ),
            title=Text("Session", style="panel.title"),
        )

        controls = self._panel(
            Group(
                self._chip_row(["/help", "/config", "/approval", "/model", "/exit"]),
                Text(""),
                Text("Ctrl+C stops the current run without quitting.", style="muted"),
                Text("Type a prompt to start working in the current workspace.", style="muted"),
            ),
            title=Text("Controls", style="panel.title"),
        )

        body = Group(
            header,
            subtitle,
            Text(""),
            Columns([overview, controls], expand=True, equal=True),
        )

        self.console.print()
        self.console.print(
            self._panel(
                body,
                title=Text("Terminal UI", style="panel.title"),
                subtitle=Text("ready", style="panel.subtitle"),
            )
        )

    def begin_assistant(self) -> None:
        title = Text()
        title.append_text(self._badge("Assistant", "assistant.badge"))
        title.append(" ")
        title.append("Jazz-Code", style="assistant")

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
        line.append_text(self._badge(label, "thinking.badge"))
        line.append(" ")
        line.append(detail, style="thinking")

        self.console.print()
        self.console.print(line)
        self._last_status = message

    def clear_status(self) -> None:
        self._last_status = None

    def show_notice(self, message: str, level: str = "info") -> None:
        badge_style = {
            "info": "status.info.badge",
            "success": "status.success.badge",
            "warning": "status.warning.badge",
            "error": "status.error.badge",
        }.get(level, "status.info.badge")
        text_style = {
            "info": "info",
            "success": "success",
            "warning": "warning",
            "error": "error",
        }.get(level, "info")

        line = Text()
        line.append_text(self._badge(level, badge_style))
        line.append(" ")
        line.append(message, style=text_style)
        self.console.print()
        self.console.print(line)

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        preferred_order = {
            "read_file": ["path", "offset", "limit"],
            "write_file": ["path", "create_directories", "content"],
            "edit": ["path", "replace_all", "old_string", "new_string"],
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
        self.console.print()
        self.console.print(
            self._panel(
                self._kv_table(
                    [
                        ("Model", self.config.model_name),
                        ("Temperature", self.config.temperature),
                        ("Approval", self.config.approval.value),
                        ("Working dir", self.config.cwd),
                        ("Max turns", self.config.max_turns),
                        ("Hooks enabled", self.config.hooks_enabled),
                    ]
                ),
                title=Text.assemble(
                    self._badge("Config", "status.info.badge"),
                    (" Current settings", "panel.title"),
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
        commands.add_row(Text("/config"), "Show current configuration")
        commands.add_row(Text("/model <name>"), "Switch models")
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
