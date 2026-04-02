import asyncio
import contextlib
import hashlib
import os
from pathlib import Path
import sys

import click
from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)


from agent.agent import Agent
from agent.events import AgentEventType
from agent.persistence import PersistenceManager, SessionSnapshot
from agent.session import Session
from config.config import ApprovalPolicy, Config
from config.loader import load_config
from ui.tui import TUI, get_console
from utils.model_discovery import (
    ModelCatalogEntry,
    ModelCatalogResult,
    ModelCatalogStore,
    build_model_catalog_sources,
    discover_model_catalog,
    flatten_model_catalog,
)
from utils.workspace_history import WorkspaceHistoryManager


def _load_env_file(path: Path | None) -> None:
    if path is None:
        return
    load_dotenv(path, override=False)


_load_env_file(PROJECT_ROOT / ".env")
_load_env_file(Path.cwd() / ".env")

console = get_console()


class CLI:
    def __init__(
        self,
        config: Config,
        workspace_history: WorkspaceHistoryManager | None = None,
    ):
        self.agent: Agent | None = None
        self.config = config
        self.tui = TUI(config, console)
        self._active_message_task: asyncio.Task[str | None] | None = None
        self.workspace_history = workspace_history or WorkspaceHistoryManager()
        self._model_catalog_store = ModelCatalogStore()
        self._model_catalog_results: list[ModelCatalogResult] | None = None
        self._model_catalog_signature: tuple[tuple[str, str, str, str], ...] | None = None

    async def _reset_model_client(self) -> None:
        if self.agent and self.agent.session and self.agent.session.client:
            await self.agent.session.client.close()

    async def _load_config_for_workspace(self, target: Path) -> Config:
        target = _validate_workspace_directory(target)
        config = load_config(cwd=target)
        config.cwd = target

        errors = config.validate()
        if errors:
            raise ValueError("\n".join(errors))

        return config

    async def _switch_workspace(self, target: Path) -> None:
        if not self.agent:
            return

        target = target.expanduser().resolve()
        if target == self.config.cwd.resolve():
            self.tui.show_notice(
                f"Already using working directory: {target}",
                level="info",
            )
            return

        new_config = await self._load_config_for_workspace(target)
        new_session = Session(config=new_config)
        new_session.approval_manager.confirmation_callback = self.tui.handle_confirmation

        try:
            await new_session.initialize()
        except Exception:
            with contextlib.suppress(Exception):
                await new_session.client.close()
            with contextlib.suppress(Exception):
                await new_session.mcp_manager.shutdown()
            raise

        old_session = self.agent.session
        self.config = new_config
        self.agent.config = new_config
        self.agent.session = new_session
        self.tui.set_config(new_config)
        self.workspace_history.record(new_config.cwd)
        self._invalidate_model_catalog()

        if old_session:
            with contextlib.suppress(Exception):
                await old_session.client.close()
            with contextlib.suppress(Exception):
                await old_session.mcp_manager.shutdown()

        self.tui.show_notice(
            f"Switched working directory to: {new_config.cwd}",
            level="success",
        )
        self.tui.show_workspace_snapshot(new_session.workspace_snapshot)
        self.tui.show_code_index(new_session.code_index_summary)

    async def _prompt_custom_workspace(self, *, base_dir: Path) -> Path | None:
        while True:
            custom_input = (
                await self.tui.prompt_custom_workspace_path(base_dir=base_dir)
            ).strip()
            if not custom_input:
                return None

            try:
                return _validate_workspace_directory(
                    _resolve_workspace_path(custom_input, base_dir)
                )
            except ValueError as exc:
                self.tui.show_notice(str(exc), level="error")
                self.tui.show_notice(
                    "Enter another path or press Enter to cancel.",
                    level="info",
                )

    async def _prompt_for_workspace(
        self,
        *,
        current_dir: Path,
        fallback_dir: Path,
        current_label: str,
    ) -> Path | None:
        recent = self.workspace_history.list_recent()
        choice = (
            await self.tui.prompt_workspace_selection(
                current_dir=current_dir,
                fallback_dir=fallback_dir,
                recent_workspaces=recent,
                current_label=current_label,
            )
        ).strip()

        if not choice:
            choice = "1"

        option_map, custom_index = _build_workspace_option_map(
            current_dir=current_dir,
            fallback_dir=fallback_dir,
            recent_workspaces=recent,
        )

        if choice.isdigit():
            selected_index = int(choice) - 1
            if selected_index == custom_index - 1:
                return await self._prompt_custom_workspace(base_dir=current_dir)
            if 0 <= selected_index < len(option_map):
                return option_map[selected_index]
            return None

        try:
            return _validate_workspace_directory(
                _resolve_workspace_path(choice, current_dir)
            )
        except ValueError:
            return None

    async def _stop_active_run(self) -> bool:
        had_active_task = self._active_message_task is not None

        if self._active_message_task and not self._active_message_task.done():
            self._active_message_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._active_message_task

        if had_active_task:
            self._active_message_task = None
            self.tui.clear_status()
            self.tui.show_notice(
                "Stopped current run. Ready for the next prompt.",
                level="warning",
            )
            return True

        return False

    def _invalidate_model_catalog(self) -> None:
        self._model_catalog_results = None
        self._model_catalog_signature = None

    def _current_catalog_signature(self) -> tuple[tuple[str, str, str, str], ...]:
        signature: list[tuple[str, str, str, str]] = []

        for source in build_model_catalog_sources(self.config):
            signature.append(
                (
                    source.profile_name or "",
                    source.base_url,
                    source.key_source_label,
                    hashlib.sha256((source.api_key or "").encode("utf-8")).hexdigest(),
                )
            )

        return tuple(signature)

    def _cached_model_catalog_entries(self) -> list[ModelCatalogEntry]:
        if not self._model_catalog_results:
            return []
        return flatten_model_catalog(self._model_catalog_results)

    async def _get_model_catalog(
        self,
        *,
        refresh: bool = False,
    ) -> list[ModelCatalogResult]:
        signature = self._current_catalog_signature()
        if (
            not refresh
            and self._model_catalog_results is not None
            and self._model_catalog_signature == signature
        ):
            return self._model_catalog_results

        self.tui.show_status("Loading models from configured providers...")
        try:
            live_results = await discover_model_catalog(self.config)
        finally:
            self.tui.clear_status()

        results: list[ModelCatalogResult] = []
        cached_sources: list[str] = []
        for result in live_results:
            if result.error:
                cached_result = self._model_catalog_store.get(result.source)
                if cached_result is not None:
                    cached_result.error = result.error
                    results.append(cached_result)
                    cached_sources.append(result.source.display_name)
                    continue

                results.append(result)
                continue

            self._model_catalog_store.save(result)
            results.append(result)

        self._model_catalog_results = results
        self._model_catalog_signature = signature

        if cached_sources:
            self.tui.show_notice(
                "Using cached model list for: " + ", ".join(cached_sources),
                level="warning",
            )

        return results

    def _select_preferred_catalog_entry(
        self,
        matches: list[ModelCatalogEntry],
    ) -> ModelCatalogEntry:
        for entry in matches:
            if entry.profile_name == self.config.active_model_profile:
                return entry

        for entry in matches:
            if entry.profile_name is None and self.config.active_model_profile is None:
                return entry

        return matches[0]

    async def run_single(self, message: str) -> str | None:
        try:
            async with Agent(self.config) as agent:
                self.agent = agent
                return await self._process_message(message)
        finally:
            self.tui.clear_status()

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome()

        try:
            async with Agent(
                self.config,
                confirmation_callback=self.tui.handle_confirmation,
            ) as agent:
                self.agent = agent

                while True:
                    try:
                        user_input = (await self.tui.read_prompt()).strip()
                        if not user_input:
                            continue

                        if user_input.isdigit():
                            catalog_entries = self._cached_model_catalog_entries()
                            selected_index = int(user_input)
                            if 1 <= selected_index <= len(catalog_entries):
                                should_continue = await self._handle_command(
                                    f"/model {user_input}"
                                )
                                if not should_continue:
                                    break
                                continue

                        if user_input.startswith("/"):
                            should_continue = await self._handle_command(user_input)
                            if not should_continue:
                                break
                            continue

                        self._active_message_task = asyncio.create_task(
                            self._process_message(user_input)
                        )
                        try:
                            await self._active_message_task
                        except asyncio.CancelledError:
                            current_task = asyncio.current_task()
                            if current_task is not None:
                                current_task.uncancel()
                            await self._stop_active_run()
                        finally:
                            self._active_message_task = None
                    except KeyboardInterrupt:
                        stopped = await self._stop_active_run()
                        if not stopped:
                            self.tui.show_notice(
                                "Press Ctrl+C to stop a run or use /exit to quit.",
                                level="info",
                            )
                    except EOFError:
                        break
        finally:
            self.tui.clear_status()

        self.tui.show_notice("Goodbye!", level="info")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool = self.agent.session.tool_registry.get(tool_name)
        if not tool:
            return None
        return tool.kind.value

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None
        status_animation_task = asyncio.create_task(self._animate_status())

        try:
            async for event in self.agent.run(message):
                if event.type == AgentEventType.AGENT_START:
                    self.tui.show_status("Thinking: understanding your request...")
                elif event.type == AgentEventType.STATUS:
                    thinking_message = event.data.get("message")
                    if thinking_message:
                        self.tui.show_status(thinking_message)
                elif event.type == AgentEventType.TEXT_DELTA:
                    content = event.data.get("content", "")
                    if not assistant_streaming:
                        self.tui.clear_status()
                        self.tui.begin_assistant()
                        assistant_streaming = True
                    self.tui.stream_assistant_delta(content)
                elif event.type == AgentEventType.TEXT_COMPLETE:
                    final_response = event.data.get("content")
                    if assistant_streaming:
                        self.tui.end_assistant()
                        assistant_streaming = False
                    self.tui.clear_status()
                elif event.type == AgentEventType.AGENT_ERROR:
                    error = event.data.get("error", "Unknown error")
                    self.tui.clear_status()
                    self.tui.show_notice(f"Error: {error}", level="error")
                elif event.type == AgentEventType.TOOL_CALL_START:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    self.tui.tool_call_start(
                        event.data.get("call_id", ""),
                        tool_name,
                        tool_kind,
                        event.data.get("arguments", {}),
                    )
                elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    self.tui.tool_call_complete(
                        event.data.get("call_id", ""),
                        tool_name,
                        tool_kind,
                        event.data.get("success", False),
                        event.data.get("output", ""),
                        event.data.get("error"),
                        event.data.get("metadata"),
                        event.data.get("diff"),
                        event.data.get("truncated", False),
                        event.data.get("exit_code"),
                    )
        except asyncio.CancelledError:
            if assistant_streaming:
                self.tui.end_assistant()
            self.tui.clear_status()
            raise
        finally:
            status_animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status_animation_task

        return final_response

    async def _animate_status(self) -> None:
        while True:
            await asyncio.sleep(0.12)
            self.tui.advance_status_frame()

    async def _handle_command(self, command: str) -> bool:
        raw_command = command.strip()
        parts = raw_command.split(maxsplit=1)
        cmd_name = parts[0]
        cmd_args = parts[1].strip() if len(parts) > 1 else ""
        cmd_name_lower = cmd_name.lower()
        if cmd_name_lower == "/exit" or cmd_name_lower == "/quit":
            return False
        elif cmd_name_lower == "/help":
            self.tui.show_help()
        elif cmd_name_lower == "/clear":
            self.agent.session.context_manager.clear()
            self.agent.session.loop_detector.clear()
            self.tui.show_notice("Conversation cleared.", level="success")
        elif cmd_name_lower == "/scan":
            snapshot, code_index = self.agent.session.refresh_workspace_context()
            self.tui.show_workspace_snapshot(snapshot)
            self.tui.show_code_index(code_index)
        elif cmd_name_lower == "/index":
            summary = self.agent.session.refresh_code_index()
            self.tui.show_code_index(summary)
        elif cmd_name_lower == "/cwd":
            if cmd_args:
                target = cmd_args
                recent = self.workspace_history.list_recent()
                if target.isdigit():
                    index = int(target) - 1
                    if 0 <= index < len(recent):
                        target_path = Path(recent[index]["path"])
                    else:
                        self.tui.show_notice(
                            f"No recent workspace at index {target}.",
                            level="error",
                        )
                        return True
                else:
                    target_path = Path(target).expanduser()
                    if not target_path.is_absolute():
                        target_path = (self.config.cwd / target_path).resolve()

                try:
                    await self._switch_workspace(target_path)
                except Exception as exc:
                    self.tui.show_notice(f"Failed to switch workspace: {exc}", level="error")
            else:
                selected = await self._prompt_for_workspace(
                    current_dir=self.config.cwd,
                    fallback_dir=WORKSPACE,
                    current_label="Current project",
                )
                if not selected:
                    self.tui.show_notice("Invalid workspace selection.", level="error")
                else:
                    try:
                        await self._switch_workspace(selected)
                    except Exception as exc:
                        self.tui.show_notice(
                            f"Failed to switch workspace: {exc}",
                            level="error",
                        )
        elif cmd_name_lower == "/recent":
            self.tui.show_recent_workspaces(self.workspace_history.list_recent())
        elif cmd_name_lower == "/config":
            self.tui.show_config()
        elif cmd_name_lower == "/models":
            action = cmd_args.strip().lower()
            if action == "refresh":
                model_catalog = await self._get_model_catalog(refresh=True)
            elif action:
                self.tui.show_notice(
                    "Usage: /models or /models refresh",
                    level="error",
                )
                return True
            else:
                model_catalog = await self._get_model_catalog()
            self.tui.show_model_profiles(
                self.config,
                model_catalog=model_catalog,
            )
        elif cmd_name_lower == "/model":
            if cmd_args:
                target = cmd_args.strip()
                profile_name = None
                force_switch = False

                if target.startswith("force "):
                    force_switch = True
                    target = target[6:].strip()

                if target.startswith("use "):
                    profile_name = target[4:].strip()
                elif target in self.config.models:
                    profile_name = target

                if profile_name:
                    try:
                        self.config.switch_model_profile(profile_name)
                        await self._reset_model_client()
                        if self.config.api_key:
                            self.tui.show_notice(
                                f"Switched to model profile: {profile_name} "
                                f"({self.config.model_name})",
                                level="success",
                            )
                        else:
                            self.tui.show_notice(
                                f"Switched to profile '{profile_name}', but no API key "
                                "is currently resolved for it.",
                                level="warning",
                            )
                    except ValueError as exc:
                        self.tui.show_notice(str(exc), level="error")
                elif target.isdigit():
                    catalog_models = flatten_model_catalog(
                        await self._get_model_catalog()
                    )
                    selected_index = int(target) - 1
                    if 0 <= selected_index < len(catalog_models):
                        await self._switch_to_catalog_entry(
                            catalog_models[selected_index]
                        )
                    else:
                        self.tui.show_notice(
                            f"No discovered model at index {target}.",
                            level="error",
                        )
                        self.tui.show_notice(
                            "Use /models to inspect the discovered model list.",
                            level="info",
                        )
                else:
                    matches = [
                        entry
                        for entry in self._cached_model_catalog_entries()
                        if entry.model_name == target
                    ]
                    if matches and not force_switch:
                        await self._switch_to_catalog_entry(
                            self._select_preferred_catalog_entry(matches)
                        )
                    else:
                        self.config.model_name = target
                        await self._reset_model_client()
                        self.tui.show_notice(
                            f"Model changed to: {target}",
                            level="success",
                        )
            else:
                self.tui.show_notice(
                    "Current model: "
                    f"{self.config.model_name}"
                    + (
                        f" via profile '{self.config.active_model_profile}'"
                        if self.config.active_model_profile
                        else ""
                    ),
                    level="info",
                )
        elif cmd_name_lower == "/approval":
            if cmd_args:
                try:
                    approval = ApprovalPolicy(cmd_args)
                    self.config.approval = approval
                    self.tui.show_notice(
                        f"Approval policy changed to: {cmd_args}",
                        level="success",
                    )
                except:
                    self.tui.show_notice(
                        f"Incorrect approval policy: {cmd_args}",
                        level="error",
                    )
                    self.tui.show_notice(
                        "Valid options: "
                        + ", ".join(policy.value for policy in ApprovalPolicy),
                        level="info",
                    )
            else:
                self.tui.show_notice(
                    f"Current approval policy: {self.config.approval.value}",
                    level="info",
                )
        elif cmd_name_lower == "/stats":
            stats = self.agent.session.get_stats()
            self.tui.show_stats(stats)
        elif cmd_name_lower == "/tools":
            tools = self.agent.session.tool_registry.get_tools()
            self.tui.show_tools(tools)
        elif cmd_name_lower == "/mcp":
            mcp_servers = self.agent.session.mcp_manager.get_all_servers()
            self.tui.show_mcp_servers(mcp_servers)
        elif cmd_name_lower == "/save":
            persistence_manager = PersistenceManager()
            session_snapshot = SessionSnapshot(
                session_id=self.agent.session.session_id,
                created_at=self.agent.session.created_at,
                updated_at=self.agent.session.updated_at,
                turn_count=self.agent.session.turn_count,
                messages=self.agent.session.context_manager.get_messages(),
                total_usage=self.agent.session.context_manager.total_usage,
            )
            persistence_manager.save_session(session_snapshot)
            self.tui.show_notice(
                f"Session saved: {self.agent.session.session_id}",
                level="success",
            )
        elif cmd_name_lower == "/sessions":
            persistence_manager = PersistenceManager()
            sessions = persistence_manager.list_sessions()
            self.tui.show_saved_sessions(sessions, "Saved sessions")
        elif cmd_name_lower == "/resume":
            if not cmd_args:
                self.tui.show_notice(
                    "Usage: /resume <session_id>",
                    level="error",
                )
            else:
                persistence_manager = PersistenceManager()
                snapshot = persistence_manager.load_session(cmd_args)
                if not snapshot:
                    self.tui.show_notice("Session does not exist.", level="error")
                else:
                    session = Session(
                        config=self.config,
                    )
                    await session.initialize()
                    session.session_id = snapshot.session_id
                    session.created_at = snapshot.created_at
                    session.updated_at = snapshot.updated_at
                    session.turn_count = snapshot.turn_count
                    session.context_manager.total_usage = snapshot.total_usage

                    for msg in snapshot.messages:
                        if msg.get("role") == "system":
                            continue
                        elif msg["role"] == "user":
                            session.context_manager.add_user_message(
                                msg.get("content", "")
                            )
                        elif msg["role"] == "assistant":
                            session.context_manager.add_assistant_message(
                                msg.get("content", ""), msg.get("tool_calls")
                            )
                        elif msg["role"] == "tool":
                            session.context_manager.add_tool_result(
                                msg.get("tool_call_id", ""), msg.get("content", "")
                            )

                    await self.agent.session.client.close()
                    await self.agent.session.mcp_manager.shutdown()

                    self.agent.session = session
                    self.tui.show_notice(
                        f"Resumed session: {session.session_id}",
                        level="success",
                    )
        elif cmd_name_lower == "/checkpoint":
            persistence_manager = PersistenceManager()
            session_snapshot = SessionSnapshot(
                session_id=self.agent.session.session_id,
                created_at=self.agent.session.created_at,
                updated_at=self.agent.session.updated_at,
                turn_count=self.agent.session.turn_count,
                messages=self.agent.session.context_manager.get_messages(),
                total_usage=self.agent.session.context_manager.total_usage,
            )
            checkpoint_id = persistence_manager.save_checkpoint(session_snapshot)
            self.tui.show_notice(
                f"Checkpoint created: {checkpoint_id}",
                level="success",
            )
        elif cmd_name_lower == "/restore":
            if not cmd_args:
                self.tui.show_notice(
                    "Usage: /restore <checkpoint_id>",
                    level="error",
                )
            else:
                persistence_manager = PersistenceManager()
                snapshot = persistence_manager.load_checkpoint(cmd_args)
                if not snapshot:
                    self.tui.show_notice("Checkpoint does not exist.", level="error")
                else:
                    session = Session(
                        config=self.config,
                    )
                    await session.initialize()
                    session.session_id = snapshot.session_id
                    session.created_at = snapshot.created_at
                    session.updated_at = snapshot.updated_at
                    session.turn_count = snapshot.turn_count
                    session.context_manager.total_usage = snapshot.total_usage

                    for msg in snapshot.messages:
                        if msg.get("role") == "system":
                            continue
                        elif msg["role"] == "user":
                            session.context_manager.add_user_message(
                                msg.get("content", "")
                            )
                        elif msg["role"] == "assistant":
                            session.context_manager.add_assistant_message(
                                msg.get("content", ""), msg.get("tool_calls")
                            )
                        elif msg["role"] == "tool":
                            session.context_manager.add_tool_result(
                                msg.get("tool_call_id", ""), msg.get("content", "")
                            )

                    await self.agent.session.client.close()
                    await self.agent.session.mcp_manager.shutdown()

                    self.agent.session = session
                    self.tui.show_notice(
                        f"Restored checkpoint into session: {session.session_id}",
                        level="success",
                    )
        else:
            self.tui.show_notice(f"Unknown command: {cmd_name}", level="error")

        return True

    async def _switch_to_catalog_entry(self, entry: ModelCatalogEntry) -> None:
        self.config.active_model_profile = entry.profile_name
        self.config.model_name = entry.model_name
        await self._reset_model_client()

        if self.config.api_key:
            self.tui.show_notice(
                "Model changed to: "
                f"{entry.model_name}"
                + (
                    f" via profile '{entry.display_name}'"
                    if entry.profile_name is not None
                    else ""
                ),
                level="success",
            )
        else:
            self.tui.show_notice(
                "Switched to "
                f"'{entry.model_name}'"
                + (
                    f" via profile '{entry.display_name}'"
                    if entry.profile_name is not None
                    else ""
                )
                + ", but no API key is currently resolved for it.",
                level="warning",
            )


def _load_validated_config(requested_cwd: Path) -> Config:
    validated_cwd = _validate_workspace_directory(requested_cwd)
    config = load_config(cwd=validated_cwd)
    config.cwd = validated_cwd
    errors = config.validate()
    if errors:
        raise ValueError("\n".join(errors))
    return config


@contextlib.contextmanager
def _temporary_workspace_env(cwd: Path):
    env_path = cwd / ".env"
    if not env_path.is_file():
        yield
        return

    env_values = {
        key: value
        for key, value in dotenv_values(env_path).items()
        if value is not None and key not in os.environ
    }

    if not env_values:
        yield
        return

    os.environ.update(env_values)
    try:
        yield
    finally:
        for key in env_values:
            os.environ.pop(key, None)


def _validate_startup_workspace_choice(candidate: Path) -> str | None:
    try:
        validated_cwd = _validate_workspace_directory(candidate)
    except ValueError as exc:
        return str(exc)

    try:
        with _temporary_workspace_env(validated_cwd):
            _load_validated_config(validated_cwd)
    except Exception as exc:
        return str(exc)

    return None


def _prompt_startup_custom_workspace(
    *,
    selector_tui: TUI,
    base_dir: Path,
) -> Path | None:
    error_message: str | None = None
    info_message = (
        "Enter any absolute path, ~/ path, or path relative to the current shell directory."
    )

    while True:
        custom_input = asyncio.run(
            selector_tui.prompt_custom_workspace_path(
                base_dir=base_dir,
                error_message=error_message,
                info_message=info_message,
            )
        ).strip()

        if not custom_input:
            return None

        try:
            return _validate_workspace_directory(
                _resolve_workspace_path(custom_input, base_dir)
            )
        except ValueError as exc:
            error_message = str(exc)
            info_message = "Enter another directory path or press Enter to go back."


def _resolve_workspace_path(path_input: str, base_dir: Path) -> Path:
    candidate = Path(path_input).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _validate_workspace_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Path is not a directory: {resolved}")
    return resolved


def _build_workspace_option_map(
    *,
    current_dir: Path,
    fallback_dir: Path,
    recent_workspaces: list[dict[str, str]],
) -> tuple[list[Path], int]:
    option_map: list[Path] = [current_dir.resolve()]
    if fallback_dir.resolve() != current_dir.resolve():
        option_map.append(fallback_dir.resolve())

    for entry in recent_workspaces[:5]:
        path_text = entry.get("path")
        if not path_text:
            continue
        path = Path(path_text).resolve()
        if path in option_map:
            continue
        option_map.append(path)

    custom_index = len(option_map) + 1
    return option_map, custom_index


def _choose_startup_workspace(
    *,
    prompt: str | None,
    requested_cwd: Path | None,
    workspace_history: WorkspaceHistoryManager,
) -> Path:
    if requested_cwd is not None:
        return _validate_workspace_directory(requested_cwd)

    latest_workspace = workspace_history.latest()

    if prompt is not None:
        return (latest_workspace or WORKSPACE).resolve()

    temp_config = Config(cwd=latest_workspace or Path.cwd().resolve())
    selector_tui = TUI(temp_config, console)
    current_dir = Path.cwd().resolve()
    recent_workspaces = workspace_history.list_recent()
    option_map, custom_index = _build_workspace_option_map(
        current_dir=current_dir,
        fallback_dir=WORKSPACE.resolve(),
        recent_workspaces=recent_workspaces,
    )
    error_message: str | None = None
    info_message = (
        "Choose one of the listed workspaces or enter a directory path directly."
    )

    while True:
        selected = asyncio.run(
            selector_tui.prompt_workspace_selection(
                current_dir=current_dir,
                fallback_dir=WORKSPACE.resolve(),
                recent_workspaces=recent_workspaces,
                current_label="Current shell directory",
                error_message=error_message,
                info_message=info_message,
            )
        ).strip()
        error_message = None
        info_message = None

        if not selected:
            selected = "1"

        if selected.isdigit():
            index = int(selected) - 1
            if index == custom_index - 1:
                custom_path = _prompt_startup_custom_workspace(
                    selector_tui=selector_tui,
                    base_dir=current_dir,
                )
                if custom_path is None:
                    info_message = (
                        "Choose one of the listed workspaces or enter a directory path directly."
                    )
                    continue

                validation_error = _validate_startup_workspace_choice(custom_path)
                if validation_error:
                    error_message = validation_error
                    info_message = (
                        "Fix that workspace configuration or choose another directory."
                    )
                    continue

                selector_tui.clear_screen()
                return custom_path.resolve()

            if 0 <= index < len(option_map):
                selected_path = option_map[index]
                validation_error = _validate_startup_workspace_choice(selected_path)
                if validation_error:
                    error_message = validation_error
                    info_message = (
                        "Fix that workspace configuration or choose another directory."
                    )
                    continue

                selector_tui.clear_screen()
                return selected_path.resolve()

            error_message = f"No workspace option at index {selected}."
            info_message = "Choose a listed number or enter a directory path."
            continue

        try:
            selected_path = _validate_workspace_directory(
                _resolve_workspace_path(selected, current_dir)
            )
        except ValueError as exc:
            error_message = str(exc)
            info_message = "Choose a listed number or enter a valid directory path."
            continue

        validation_error = _validate_startup_workspace_choice(selected_path)
        if validation_error:
            error_message = validation_error
            info_message = (
                "Fix that workspace configuration or choose another directory."
            )
            continue

        selector_tui.clear_screen()
        return selected_path.resolve()


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
def main(
    prompt: str | None,
    cwd: Path | None,
):
    workspace_history = WorkspaceHistoryManager()

    try:
        requested_cwd = _choose_startup_workspace(
            prompt=prompt,
            requested_cwd=cwd,
            workspace_history=workspace_history,
        )
        _load_env_file(requested_cwd / ".env")
        config = _load_validated_config(requested_cwd)
    except Exception as e:
        console.print(f"[error]Configuration Error: {e}[/error]")
        sys.exit(1)

    workspace_history.record(requested_cwd)
    cli = CLI(config, workspace_history=workspace_history)

    # messages = [{"role": "user", "content": prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
