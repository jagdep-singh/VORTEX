import asyncio
from collections import Counter
import contextlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import shlex

import click
from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)
INITIAL_ENV = dict(os.environ)
ACTIVE_WORKSPACE_ENV: dict[str, str] = {}


from agent.agent import Agent
from agent.events import AgentEventType
from agent.persistence import PersistenceManager, SessionSnapshot
from agent.session import Session
from client.response import TokenUsage
from config.config import ApprovalPolicy, Config
from config.loader import load_config
from ui.tui import TUI, get_console
from utils.credential_setup import (
    normalize_base_url,
    resolve_api_key_env_name,
    should_prompt_for_base_url,
    split_config_errors,
    suggested_base_url,
    upsert_env_file,
    validate_base_url,
)
from utils.model_discovery import (
    ModelCatalogEntry,
    ModelCatalogResult,
    ModelCatalogStore,
    build_model_catalog_sources,
    discover_model_catalog,
    flatten_model_catalog,
    order_model_catalog_entries,
    select_best_working_model,
)
from utils.model_health import ModelHealthChecker, ModelHealthRecord, ModelHealthStore
from utils.ollama_setup import (
    OLLAMA_OPENAI_BASE_URL,
    LocalModelOption,
    build_ollama_install_shell_command,
    format_bytes,
    get_free_space_bytes,
    get_local_model_option,
    get_local_model_option_by_name,
    has_enough_space,
    is_ollama_base_url,
    is_ollama_installed,
    list_installed_models,
    ollama_install_docs_url,
    pull_model,
    resolve_ollama_models_dir,
    supports_automatic_ollama_install,
)
from utils.versioning import (
    ReleaseInfo,
    VersionManager,
    get_current_version,
    recommended_update_instruction,
)
from utils.provider_auth import base_url_allows_missing_api_key
from utils.workspace_history import WorkspaceHistoryManager


def _clean_error_message(error: object) -> str:
    if isinstance(error, list):
        cleaned = [_clean_error_message(item) for item in error]
        return "; ".join(part for part in cleaned if part)

    if isinstance(error, dict):
        err = error.get("error") if isinstance(error, dict) and "error" in error else error
        if isinstance(err, dict):
            parts: list[str] = []
            for key in ("code", "status", "message"):
                value = err.get(key)
                if value:
                    parts.append(str(value))
            if parts:
                return " ".join(parts)
        return str(err)

    return str(error).strip()


def _load_env_file(path: Path | None, *, override: bool = False) -> None:
    if path is None:
        return
    load_dotenv(path, override=override)


def _read_env_values(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}

    return {
        key: value
        for key, value in dotenv_values(path).items()
        if value is not None
    }


def _activate_workspace_env(cwd: Path) -> None:
    global ACTIVE_WORKSPACE_ENV

    for key in ACTIVE_WORKSPACE_ENV:
        original_value = INITIAL_ENV.get(key)
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    env_values = _read_env_values(cwd / ".env")
    os.environ.update(env_values)
    ACTIVE_WORKSPACE_ENV = env_values

console = get_console()


class CLI:
    def __init__(
        self,
        config: Config,
        workspace_history: WorkspaceHistoryManager | None = None,
        version_manager: VersionManager | None = None,
    ):
        self.agent: Agent | None = None
        self.config = config
        self.tui = TUI(config, console)
        self._active_message_task: asyncio.Task[str | None] | None = None
        self.workspace_history = workspace_history or WorkspaceHistoryManager()
        self.version_manager = version_manager or VersionManager(project_root=PROJECT_ROOT)
        self.release_info: ReleaseInfo | None = None
        self._model_catalog_store = ModelCatalogStore()
        self._model_health_store = ModelHealthStore()
        self._model_catalog_results: list[ModelCatalogResult] | None = None
        self._model_catalog_signature: tuple[tuple[str, str, str, str], ...] | None = None
        self._tool_progress_task: asyncio.Task | None = None
        self._current_tool_label: str | None = None
        self._current_tool_started_at: float | None = None

    async def _reset_model_client(self) -> None:
        if self.agent and self.agent.session and self.agent.session.client:
            await self.agent.session.client.close()

    async def _load_config_for_workspace(self, target: Path) -> Config:
        target = _validate_workspace_directory(target)
        with _temporary_workspace_env(target):
            config, errors = _load_config_with_errors(target)
        credential_errors, other_errors = split_config_errors(errors)
        if other_errors:
            raise ValueError("\n".join(other_errors))
        if credential_errors:
            config = await _prompt_for_missing_api_credentials(
                requested_cwd=target,
                config=config,
                tui=self.tui,
            )
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
        previous_config = self.config
        auto_model_notice: tuple[str, str] | None = None
        new_session: Session | None = None
        previous_env = dict(ACTIVE_WORKSPACE_ENV)
        new_env_values = _read_env_values(target / ".env")
        try:
            _activate_workspace_env(target)
            self.config = new_config
            self.tui.set_config(new_config)
            auto_model_notice = await self._ensure_working_model(reason="Workspace")

            new_session = Session(config=new_config)
            new_session.approval_manager.confirmation_callback = self.tui.handle_confirmation
            await new_session.initialize()
        except Exception:
            _activate_workspace_env(previous_config.cwd)
            self.config = previous_config
            self.tui.set_config(previous_config)
            with contextlib.suppress(Exception):
                if new_session is not None:
                    await new_session.client.close()
            with contextlib.suppress(Exception):
                if new_session is not None:
                    await new_session.mcp_manager.shutdown()
            raise

        old_session = self.agent.session
        self.agent.config = new_config
        self.agent.session = new_session
        self.tui.set_config(new_config)
        self.workspace_history.record(new_config.cwd)
        self._invalidate_model_catalog()

        self.tui.show_env_change(previous_env, new_env_values)

        if old_session:
            with contextlib.suppress(Exception):
                await old_session.client.close()
            with contextlib.suppress(Exception):
                await old_session.mcp_manager.shutdown()

        self.tui.show_notice(
            f"Switched working directory to: {new_config.cwd}",
            level="success",
        )
        if auto_model_notice:
            self.tui.show_notice(auto_model_notice[0], level=auto_model_notice[1])
        self.tui.show_workspace_snapshot(new_session.workspace_snapshot)
        self.tui.show_code_index(new_session.code_index_summary)

    async def _restart_session_with_config(self, new_config: Config, *, reason: str) -> None:
        previous_config = self.config
        new_session: Session | None = None
        try:
            new_session = Session(config=new_config)
            new_session.approval_manager.confirmation_callback = self.tui.handle_confirmation
            await new_session.initialize()

            old_session = self.agent.session if self.agent else None
            if self.agent:
                self.agent.config = new_config
                self.agent.session = new_session
            self.config = new_config
            self.tui.set_config(new_config)
            self._invalidate_model_catalog()

            if old_session:
                with contextlib.suppress(Exception):
                    await old_session.client.close()
                with contextlib.suppress(Exception):
                    await old_session.mcp_manager.shutdown()

            self.tui.show_notice(
                f"Started a fresh session after {reason.lower()}.",
                level="success",
            )
            self.tui.show_workspace_snapshot(new_session.workspace_snapshot)
            self.tui.show_code_index(new_session.code_index_summary)
        except Exception:
            self.config = previous_config
            self.tui.set_config(previous_config)
            raise

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

    async def _restart_session_with_config(self, new_config: Config, *, reason: str) -> None:
        previous_config = self.config
        new_session: Session | None = None
        try:
            new_session = Session(config=new_config)
            new_session.approval_manager.confirmation_callback = self.tui.handle_confirmation
            await new_session.initialize()

            old_session = self.agent.session if self.agent else None
            if self.agent:
                self.agent.config = new_config
                self.agent.session = new_session
            self.config = new_config
            self.tui.set_config(new_config)
            self._invalidate_model_catalog()

            if old_session:
                with contextlib.suppress(Exception):
                    await old_session.client.close()
                with contextlib.suppress(Exception):
                    await old_session.mcp_manager.shutdown()

            self.tui.show_notice(
                f"Started a fresh session after {reason.lower()}.",
                level="success",
            )
            self.tui.show_workspace_snapshot(new_session.workspace_snapshot)
            self.tui.show_code_index(new_session.code_index_summary)
        except Exception:
            self.config = previous_config
            self.tui.set_config(previous_config)
            raise

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
        entries = flatten_model_catalog(
            self._model_catalog_results,
            health_records=self._get_model_health_index(self._model_catalog_results),
        )
        return order_model_catalog_entries(
            entries,
            active_profile_name=self.config.active_model_profile,
            active_model_name=self.config.model_name,
        )

    def _get_model_health_index(
        self,
        catalog_results: list[ModelCatalogResult] | None = None,
    ) -> dict[tuple[str, str], ModelHealthRecord]:
        results = catalog_results or self._model_catalog_results or []
        records: dict[tuple[str, str], ModelHealthRecord] = {}

        for result in results:
            provider = result.source.display_name
            stored_records = self._model_health_store.get_many(provider, result.models)
            for model_name, record in stored_records.items():
                records[(provider, model_name)] = record

        return records

    def _summarize_model_health_records(
        self,
        records: dict[tuple[str, str], ModelHealthRecord],
    ) -> str:
        if not records:
            return "No models were checked."

        status_counts = Counter(record.status for record in records.values())
        parts = [f"{len(records)} checked"]

        for status in (
            "working",
            "quota",
            "rate-limited",
            "auth-error",
            "unavailable",
            "offline",
            "error",
            "missing-key",
            "cached",
            "unknown",
        ):
            count = status_counts.get(status)
            if count:
                parts.append(f"{count} {status}")

        remaining_statuses = sorted(
            status
            for status in status_counts
            if status
            not in {
                "working",
                "quota",
                "rate-limited",
                "auth-error",
                "unavailable",
                "offline",
                "error",
                "missing-key",
                "cached",
                "unknown",
            }
        )
        for status in remaining_statuses:
            parts.append(f"{status_counts[status]} {status}")

        return " • ".join(parts)

    async def _probe_model_catalog(
        self,
        catalog_results: list[ModelCatalogResult],
        *,
        status_prefix: str = "Refresh",
    ) -> dict[tuple[str, str], ModelHealthRecord]:
        probe_targets = [result for result in catalog_results if result.models]
        if not probe_targets:
            return {}

        total_models = sum(len(result.models) for result in probe_targets)
        completed_models = 0
        records: dict[tuple[str, str], ModelHealthRecord] = {}

        for provider_index, result in enumerate(probe_targets, start=1):
            provider = result.source.display_name
            checker = ModelHealthChecker(
                provider=provider,
                base_url=result.source.base_url,
                api_key=result.source.api_key,
                store=self._model_health_store,
            )

            self.tui.show_status(
                f"{status_prefix}: "
                f"checking provider {provider_index}/{len(probe_targets)}"
                f" • {provider}"
                f" • {completed_models}/{total_models} models"
            )

            async def _progress(
                _done: int,
                _total: int,
                record: ModelHealthRecord,
            ) -> None:
                nonlocal completed_models
                completed_models += 1
                records[(record.provider, record.model)] = record
                self.tui.show_status(
                    f"{status_prefix}: "
                    f"checking models {completed_models}/{total_models}"
                    f" • {record.provider}"
                )

            try:
                provider_records = await checker.probe_models(
                    result.models,
                    progress_callback=_progress,
                )
            except Exception as exc:
                self.tui.show_notice(
                    f"Failed to probe models for {provider}: {exc}",
                    level="warning",
                )
                continue

            for record in provider_records:
                records[(record.provider, record.model)] = record

        return records

    async def _get_model_catalog(
        self,
        *,
        refresh: bool = False,
        probe: bool = False,
        status_prefix: str = "Models",
        show_cached_notice: bool = True,
        completion_notice: str | None = None,
    ) -> list[ModelCatalogResult]:
        signature = self._current_catalog_signature()
        if (
            not refresh
            and self._model_catalog_results is not None
            and self._model_catalog_signature == signature
        ):
            return self._model_catalog_results

        self.tui.show_status(f"{status_prefix}: loading configured providers")
        try:
            live_results = await discover_model_catalog(self.config)
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

            probe_records: dict[tuple[str, str], ModelHealthRecord] = {}
            if probe:
                probe_records = await self._probe_model_catalog(
                    results,
                    status_prefix=status_prefix,
                )

            self._model_catalog_results = results
            self._model_catalog_signature = signature

            if cached_sources and show_cached_notice:
                self.tui.show_notice(
                    "Using cached model list for: " + ", ".join(cached_sources),
                    level="warning",
                )

            if probe and completion_notice:
                self.tui.show_notice(
                    completion_notice + ": "
                    + self._summarize_model_health_records(probe_records),
                    level="success",
                )

            return results
        finally:
            self.tui.clear_status()

    async def _probe_current_model_if_needed(
        self,
        health_records: dict[tuple[str, str], ModelHealthRecord],
        *,
        status_prefix: str,
    ) -> ModelHealthRecord | None:
        provider = self.config.active_model_profile or "default"
        model_name = self.config.model_name
        record_key = (provider, model_name)

        existing = health_records.get(record_key)
        if existing is not None:
            return existing

        checker = ModelHealthChecker(
            provider=provider,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            store=self._model_health_store,
        )
        self.tui.show_status(
            f"{status_prefix}: checking current model"
            f" • {provider}"
        )
        record = await checker.probe_model(model_name)
        health_records[record_key] = record
        return record

    async def _repair_api_credentials(
        self,
        *,
        current_record: ModelHealthRecord,
    ) -> None:
        detail = (current_record.detail or "").strip()
        if current_record.status == "auth-error":
            prompt_error = (
                "Current API credentials were rejected by the provider."
                + (f" {detail}" if detail else "")
            )
        elif current_record.status == "missing-key":
            prompt_error = "No API key is configured for this workspace."
        else:
            prompt_error = detail or "Update the API credentials for this workspace."

        self.tui.clear_status()
        updated_config = await _prompt_for_missing_api_credentials(
            requested_cwd=self.config.cwd,
            config=self.config,
            tui=self.tui,
            initial_api_key_error=prompt_error,
        )
        self.config = updated_config
        self.tui.set_config(updated_config)
        self._invalidate_model_catalog()
        await self._reset_model_client()

    async def _ensure_working_model(
        self,
        *,
        reason: str,
        allow_credential_repair: bool = True,
        allow_local_repair: bool = True,
    ) -> tuple[str, str] | None:
        catalog_results = await self._get_model_catalog(
            refresh=True,
            probe=True,
            status_prefix=reason,
            show_cached_notice=False,
            completion_notice=None,
        )
        health_records = self._get_model_health_index(catalog_results)
        current_record = await self._probe_current_model_if_needed(
            health_records,
            status_prefix=reason,
        )

        current_profile_name = self.config.active_model_profile
        current_model_name = self.config.model_name

        if current_record and current_record.status == "working":
            return None

        if allow_local_repair and _should_repair_local_ollama_setup(
            config=self.config,
            record=current_record,
        ):
            self.tui.clear_status()
            updated_config = await _prompt_for_local_ollama_setup(
                requested_cwd=self.config.cwd,
                tui=self.tui,
                preferred_model_name=self.config.model_name,
            )
            if updated_config is None:
                _clear_local_ollama_workspace_env(self.config.cwd)
                cleared_config, errors = _load_config_with_errors(self.config.cwd)
                credential_errors, other_errors = split_config_errors(errors)
                if other_errors:
                    raise ValueError("\n".join(other_errors))
                if credential_errors:
                    updated_config = await _prompt_for_missing_api_credentials(
                        requested_cwd=self.config.cwd,
                        config=cleared_config,
                        tui=self.tui,
                        force_prompt_base_url=True,
                    )
                else:
                    updated_config = cleared_config

            self.config = updated_config
            self.tui.set_config(updated_config)
            self._invalidate_model_catalog()
            await self._reset_model_client()
            return await self._ensure_working_model(
                reason=reason,
                allow_credential_repair=allow_credential_repair,
                allow_local_repair=False,
            )

        if allow_credential_repair and _should_repair_api_credentials(current_record):
            await self._repair_api_credentials(current_record=current_record)
            return await self._ensure_working_model(
                reason=reason,
                allow_credential_repair=False,
                allow_local_repair=allow_local_repair,
            )

        selected_entry = select_best_working_model(
            catalog_results,
            health_records=health_records,
            preferred_profile_name=current_profile_name,
            preferred_model_name=current_model_name,
        )
        if selected_entry is None:
            return (
                f"No working model was found during {reason.lower()} checks. "
                f"Staying on {self.config.active_model_label}.",
                "warning",
            )

        if (
            selected_entry.profile_name == current_profile_name
            and selected_entry.model_name == current_model_name
        ):
            return None

        previous_label = self.config.active_model_label
        self.config.active_model_profile = selected_entry.profile_name
        self.config.model_name = selected_entry.model_name
        await self._reset_model_client()

        selected_label = (
            f"{selected_entry.display_name} ({selected_entry.model_name})"
            if selected_entry.profile_name is not None
            else selected_entry.model_name
        )
        return (
            f"Auto-selected a working model: {selected_label} "
            f"(previously {previous_label}).",
            "success",
        )

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
            startup_notice = await self._ensure_working_model(reason="Startup")
            if startup_notice:
                self.tui.show_notice(startup_notice[0], level=startup_notice[1])
            async with Agent(self.config) as agent:
                self.agent = agent
                return await self._process_message(message)
        finally:
            self.tui.clear_status()

    async def run_interactive(self) -> str | None:
        startup_notice = await self._ensure_working_model(reason="Startup")
        self.tui.show_status("Startup: checking for new VORTEX releases")
        self.release_info = await asyncio.to_thread(self.version_manager.get_release_info)
        self.tui.clear_status()
        self.tui.clear_screen()
        self.tui.print_welcome(release_info=self.release_info)
        if startup_notice:
            self.tui.show_notice(startup_notice[0], level=startup_notice[1])
        if self.release_info and self.release_info.update_available:
            assert self.release_info.latest_version is not None
            self.tui.show_notice(
                "Update available: "
                f"{self.release_info.current_version} -> {self.release_info.latest_version}. "
                f"{recommended_update_instruction(self.release_info.install_mode).capitalize()} when you're ready.",
                level="warning",
            )

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

                        if user_input.strip() in {"?", "/?"}:
                            self.tui.show_key_help()
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
        start_time = time.monotonic()
        self.tui.reset_run_metrics()

        try:
            async for event in self.agent.run(message):
                if event.type == AgentEventType.AGENT_START:
                    self.tui.clear_error()
                    self.tui.show_status("Thinking: understanding your request...")
                elif event.type == AgentEventType.STATUS:
                    thinking_message = event.data.get("message")
                    if thinking_message:
                        self.tui.show_status(thinking_message)
                elif event.type == AgentEventType.TEXT_DELTA:
                    content = event.data.get("content", "")
                    if not assistant_streaming:
                        self.tui.show_status("Streaming reply…")
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
                    error = _clean_error_message(event.data.get("error", "Unknown error"))
                    self.tui.clear_status()
                    self.tui.record_error(error)
                    self.tui.show_notice(f"Error: {error}", level="error")
                    lower_error = str(error).lower()
                    if "invalid argument" in lower_error or "code: 400" in lower_error:
                        self.tui.show_notice(
                            "The model rejected the request (invalid argument). Try /models refresh or /api-change, then rerun the task.",
                            level="warning",
                        )
                elif event.type == AgentEventType.AGENT_END:
                    usage_data = event.data.get("usage")
                    usage = None
                    if isinstance(usage_data, dict):
                        try:
                            usage = TokenUsage(**usage_data)
                        except Exception:
                            usage = None
                    elapsed = time.monotonic() - start_time
                    self.tui.set_run_metrics(elapsed, usage)
                    self.tui.show_status("Ready")
                elif event.type == AgentEventType.TOOL_CALL_START:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    args = event.data.get("arguments", {}) or {}
                    cmd_preview = ""
                    if tool_name == "shell":
                        cmd_text = str(args.get("command", "")).strip()
                        if cmd_text:
                            cmd_preview = f": {cmd_text[:80]}"
                    self._start_tool_progress(f"{tool_name}{cmd_preview}")
                    self.tui.tool_call_start(
                        event.data.get("call_id", ""),
                        tool_name,
                        tool_kind,
                        event.data.get("arguments", {}),
                    )
                elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    self._stop_tool_progress()
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
        elif cmd_name_lower in {"/?", "?"}:
            self.tui.show_key_help()
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
                model_catalog = await self._get_model_catalog(
                    refresh=True,
                    probe=True,
                    status_prefix="Refresh",
                    completion_notice="Model refresh complete",
                )
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
                model_health=self._get_model_health_index(model_catalog),
            )
        elif cmd_name_lower == "/model":
            if cmd_args:
                target = cmd_args.strip()
                if target.lower() == "refresh":
                    model_catalog = await self._get_model_catalog(
                        refresh=True,
                        probe=True,
                        status_prefix="Refresh",
                        completion_notice="Model refresh complete",
                    )
                    self.tui.show_model_profiles(
                        self.config,
                        model_catalog=model_catalog,
                        model_health=self._get_model_health_index(model_catalog),
                    )
                    return True

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
                        if self.config.client_api_key:
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
                    catalog_models = order_model_catalog_entries(
                        flatten_model_catalog(
                            await self._get_model_catalog(),
                            health_records=self._get_model_health_index(),
                        ),
                        active_profile_name=self.config.active_model_profile,
                        active_model_name=self.config.model_name,
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
        elif cmd_name_lower == "/api-change":
            self.tui.show_notice(
                "API change will prompt for a provider URL and key, then restart the session.",
                level="warning",
            )
            updated_config = await _prompt_for_missing_api_credentials(
                requested_cwd=self.config.cwd,
                config=self.config,
                tui=self.tui,
                force_prompt_base_url=True,
            )
            await self._restart_session_with_config(updated_config, reason="API change")
        elif cmd_name_lower == "/stats":
            stats = self.agent.session.get_stats()
            self.tui.show_stats(stats)
        elif cmd_name_lower == "/tools":
            tools = self.agent.session.tool_registry.get_tools()
            self.tui.show_tools(tools)
        elif cmd_name_lower == "/mcp":
            sub = cmd_args.strip()
            if sub.startswith("attach"):
                parts = sub.split(maxsplit=2)
                if len(parts) < 3:
                    self.tui.show_notice(
                        "Usage: /mcp attach <name> <url|command...>",
                        level="error",
                    )
                    return True
                name = parts[1]
                target = parts[2].strip()
                server_config = None
                if "://" in target:
                    server_config = MCPServerConfig(url=target)
                else:
                    tokens = shlex.split(target)
                    if not tokens:
                        self.tui.show_notice(
                            "Provide a command to launch the MCP server.",
                            level="error",
                        )
                        return True
                    server_config = MCPServerConfig(
                        command=tokens[0],
                        args=tokens[1:],
                    )

                try:
                    added = await self.agent.session.mcp_manager.add_server(
                        name,
                        server_config,
                        self.agent.session.tool_registry,
                    )
                    self.tui.show_notice(
                        f"Connected MCP '{name}' with {added} tool(s).",
                        level="success",
                    )
                except Exception as exc:
                    self.tui.show_notice(
                        f"Failed to attach MCP '{name}': {exc}",
                        level="error",
                    )
                return True

            mcp_servers = self.agent.session.mcp_manager.get_all_servers()
            self.tui.show_mcp_servers(mcp_servers)
        elif cmd_name_lower == "/contrast":
            mode = cmd_args.strip().lower()
            if mode in {"on", "enable", "true", "yes", "1"}:
                enabled = True
            elif mode in {"off", "disable", "false", "no", "0"}:
                enabled = False
            else:
                enabled = not getattr(self.tui, "_high_contrast", False)

            self.tui.set_high_contrast(enabled)
            self.tui.show_notice(
                f"High-contrast theme {'enabled' if enabled else 'disabled'}.",
                level="success",
            )
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

    def _start_tool_progress(self, label: str) -> None:
        self._current_tool_label = label
        self._current_tool_started_at = time.monotonic()
        if self._tool_progress_task is None or self._tool_progress_task.done():
            self._tool_progress_task = asyncio.create_task(self._animate_tool_progress())

    def _stop_tool_progress(self) -> None:
        self._current_tool_label = None
        self._current_tool_started_at = None
        if self._tool_progress_task is not None:
            self._tool_progress_task.cancel()
            self._tool_progress_task = None
        self.tui.clear_status()

    async def _animate_tool_progress(self) -> None:
        try:
            while self._current_tool_label:
                elapsed = 0
                if self._current_tool_started_at is not None:
                    elapsed = int(time.monotonic() - self._current_tool_started_at)
                self.tui.show_status(
                    f"Running {self._current_tool_label} • {elapsed}s elapsed"
                )
                await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            return

    async def _switch_to_catalog_entry(self, entry: ModelCatalogEntry) -> None:
        self.config.active_model_profile = entry.profile_name
        self.config.model_name = entry.model_name
        await self._reset_model_client()

        if self.config.client_api_key:
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


def _load_config_with_errors(requested_cwd: Path) -> tuple[Config, list[str]]:
    validated_cwd = _validate_workspace_directory(requested_cwd)
    config = load_config(cwd=validated_cwd)
    config.cwd = validated_cwd
    return config, config.validate()


def _load_validated_config(requested_cwd: Path) -> Config:
    config, errors = _load_config_with_errors(requested_cwd)
    if errors:
        raise ValueError("\n".join(errors))
    return config


def _run_update_flow() -> int:
    version_manager = VersionManager(project_root=PROJECT_ROOT)
    current_version = version_manager.current_version
    release_info = version_manager.get_release_info()

    console.print(
        f"[info]VORTEX {current_version}[/info]",
    )
    if release_info.latest_version and release_info.update_available:
        console.print(
            "[warning]"
            f"New version available: {release_info.latest_version}"
            "[/warning]"
        )

    update_command = version_manager.resolve_update_command()
    if update_command is not None:
        console.print(
            f"[muted]Running update command: {update_command.display}[/muted]"
        )

    result = version_manager.perform_self_update()
    level = "success" if result.success else "error"
    console.print(f"[{level}]{result.message}[/{level}]")
    return 0 if result.success else 1


async def _prompt_for_missing_api_credentials(
    *,
    requested_cwd: Path,
    config: Config,
    tui: TUI,
    initial_api_key_error: str | None = None,
    force_prompt_base_url: bool = False,
) -> Config:
    env_path = requested_cwd / ".env"
    api_key_env_name = resolve_api_key_env_name(config)
    provider_url = suggested_base_url(config)
    provider_error: str | None = None
    api_key_error: str | None = initial_api_key_error
    docs_path = PROJECT_ROOT / "docs" / "api-keys.md"
    if not docs_path.is_file():
        docs_path = None
    prompt_for_base_url = force_prompt_base_url or should_prompt_for_base_url(config)

    if _should_offer_local_model_choice(
        config=config,
        initial_api_key_error=initial_api_key_error,
        force_prompt_base_url=force_prompt_base_url,
    ):
        local_config = await _prompt_for_local_ollama_setup(
            requested_cwd=requested_cwd,
            tui=tui,
        )
        if local_config is not None:
            return local_config

    while True:
        if prompt_for_base_url:
            provider_input = (
                await tui.prompt_api_provider_url(
                    workspace_dir=requested_cwd,
                    default_url=provider_url,
                    api_key_env_name=api_key_env_name,
                    docs_path=docs_path,
                    error_message=provider_error,
                    info_message=(
                        "This gets saved into the workspace .env file so future launches reuse it."
                    ),
                )
            ).strip()
            provider_url = normalize_base_url(provider_input or provider_url)
            provider_error = validate_base_url(provider_url)
            if provider_error:
                continue
        else:
            provider_url = suggested_base_url(config)

        if base_url_allows_missing_api_key(provider_url):
            env_updates: dict[str, str] = {}
            if prompt_for_base_url:
                env_updates["BASE_URL"] = provider_url
            if env_updates:
                upsert_env_file(env_path, env_updates)
                _activate_workspace_env(requested_cwd)

            reloaded_config, errors = _load_config_with_errors(requested_cwd)
            credential_errors, other_errors = split_config_errors(errors)
            if other_errors:
                raise ValueError("\n".join(other_errors))
            if not credential_errors:
                return reloaded_config

            api_key_error = credential_errors[0]
            continue

        api_key = (
            await tui.prompt_api_key(
                workspace_dir=requested_cwd,
                provider_url=provider_url,
                api_key_env_name=api_key_env_name,
                docs_path=docs_path,
                error_message=api_key_error,
                info_message=(
                    "Your key is hidden while you type and is saved only in this workspace's .env file."
                ),
            )
        ).strip()

        if not api_key:
            api_key_error = (
                f"Enter a non-empty API key for {api_key_env_name} to continue."
            )
            continue

        env_updates = {api_key_env_name: api_key}
        if prompt_for_base_url:
            env_updates["BASE_URL"] = provider_url

        upsert_env_file(env_path, env_updates)
        _activate_workspace_env(requested_cwd)

        reloaded_config, errors = _load_config_with_errors(requested_cwd)
        credential_errors, other_errors = split_config_errors(errors)
        if other_errors:
            raise ValueError("\n".join(other_errors))
        if not credential_errors:
            return reloaded_config

        api_key_error = credential_errors[0]


def _should_offer_local_model_choice(
    *,
    config: Config,
    initial_api_key_error: str | None,
    force_prompt_base_url: bool,
) -> bool:
    if initial_api_key_error is not None:
        return False
    if force_prompt_base_url:
        return False
    if config.active_model_profile is not None:
        return False
    if config.models:
        return False
    if os.environ.get("BASE_URL"):
        return False
    if os.environ.get("API_KEY"):
        return False
    if os.environ.get("MODEL_NAME"):
        return False
    return True


async def _run_interactive_shell_command(command: str) -> int:
    process = await asyncio.create_subprocess_shell(command)
    return await process.wait()


def _start_ollama_background_process() -> None:
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["open", "-a", "Ollama", "--args", "hidden"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "macOS could not open the Ollama app automatically."
            )
        return

    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return

    subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


async def _install_ollama_with_permission(
    *,
    requested_cwd: Path,
    tui: TUI,
    allow_external: bool,
) -> tuple[bool, str | None]:
    install_command = build_ollama_install_shell_command()
    if install_command is None:
        return (
            False,
            f"Automatic Ollama install is not supported here. Install it manually from {ollama_install_docs_url()}.",
        )

    if os.name == "nt":
        install_detail = (
            "This downloads and launches Ollama's official Windows installer for your account."
        )
        permission_detail = (
            "It may open installer windows and prompt you for normal Windows approval."
        )
    elif sys.platform == "darwin":
        install_detail = (
            "This downloads and runs Ollama's official macOS installer for this device."
        )
        permission_detail = (
            "It may move files into Applications and ask for your system password."
        )
    else:
        install_detail = (
            "This downloads and runs Ollama's official Linux installer for this device."
        )
        permission_detail = (
            "It may install system files and ask for sudo permissions."
        )

    choice = await _prompt_validated_setup_decision(
        tui=tui,
        badge_label="Install",
        title="Install Ollama",
        workspace_dir=requested_cwd,
        body_lines=[
            "VORTEX will run Ollama's official installer for this device.",
            install_detail,
            permission_detail,
            f"Source: {ollama_install_docs_url()}",
        ],
        options=[
            (
                "1",
                "Install now",
                "Run the official Ollama installer in this terminal.",
            ),
            ("2", "Back", "Return to local setup without installing."),
            ("3", "External API", "Skip local setup for now."),
        ]
        if allow_external
        else [
            (
                "1",
                "Install now",
                "Run the official Ollama installer in this terminal.",
            ),
            ("2", "Back", "Return to local setup without installing."),
        ],
        default_choice="1",
    )
    if choice == "2":
        return False, None
    if allow_external and choice == "3":
        return False, "__external__"

    tui.clear_status()
    tui.show_notice(
        "Running the official Ollama installer in this terminal.",
        level="warning",
    )
    tui.show_notice(
        permission_detail,
        level="info",
    )
    exit_code = await _run_interactive_shell_command(install_command)
    if exit_code != 0:
        return False, f"The Ollama installer exited with status {exit_code}."
    if not is_ollama_installed():
        return (
            False,
            "The installer finished, but `ollama` is still not available in PATH.",
        )
    return True, None


async def _start_ollama_with_permission(
    *,
    requested_cwd: Path,
    tui: TUI,
    allow_external: bool,
) -> tuple[bool, str | None]:
    choice = await _prompt_validated_setup_decision(
        tui=tui,
        badge_label="Local",
        title="Start Ollama",
        workspace_dir=requested_cwd,
        body_lines=[
            "VORTEX can try to start the local Ollama service for this session.",
            "On macOS this opens the Ollama app. On Linux or Windows it starts `ollama serve` in the background.",
        ],
        options=[
            ("1", "Start now", "Try to launch the local Ollama server."),
            ("2", "Back", "Return to local setup without starting it."),
            ("3", "External API", "Skip local setup for now."),
        ]
        if allow_external
        else [
            ("1", "Start now", "Try to launch the local Ollama server."),
            ("2", "Back", "Return to local setup without starting it."),
        ],
        default_choice="1",
    )
    if choice == "2":
        return False, None
    if allow_external and choice == "3":
        return False, "__external__"

    try:
        await asyncio.to_thread(_start_ollama_background_process)
    except Exception as exc:
        return False, _clean_error_message(exc)

    await asyncio.sleep(2.0)
    return True, None


async def _prompt_for_local_ollama_setup(
    *,
    requested_cwd: Path,
    tui: TUI,
    preferred_model_name: str | None = None,
) -> Config | None:
    preferred_option = get_local_model_option_by_name(preferred_model_name or "")
    chooser_error: str | None = None
    chooser_info = (
        "Choose a small local Ollama model or continue with an external API."
    )

    if preferred_option is not None:
        status, reloaded_config = await _complete_local_ollama_setup(
            requested_cwd=requested_cwd,
            tui=tui,
            option=preferred_option,
        )
        if status == "configured":
            return reloaded_config
        if status == "external":
            return None
        chooser_info = "Choose another local model or continue with an external API."

    while True:
        local_choice = (
            await tui.prompt_local_model_choice(
                workspace_dir=requested_cwd,
                error_message=chooser_error,
                info_message=chooser_info,
            )
        ).strip() or "1"

        if local_choice == "3":
            return None

        option = get_local_model_option(local_choice)
        if option is None:
            chooser_error = _format_choice_error(["1", "2", "3"])
            chooser_info = "Pick a local Ollama model or continue with an external API."
            continue

        status, reloaded_config = await _complete_local_ollama_setup(
            requested_cwd=requested_cwd,
            tui=tui,
            option=option,
        )
        if status == "configured":
            return reloaded_config
        if status == "external":
            return None

        chooser_error = None
        chooser_info = "Choose another local model or continue with an external API."


async def _complete_local_ollama_setup(
    *,
    requested_cwd: Path,
    tui: TUI,
    option: LocalModelOption,
    allow_external: bool = True,
) -> tuple[str, Config | None]:
    env_path = requested_cwd / ".env"
    install_error: str | None = None
    start_error: str | None = None

    while True:
        if not is_ollama_installed():
            if supports_automatic_ollama_install():
                choice = await _prompt_validated_setup_decision(
                    tui=tui,
                    badge_label="Local",
                    title="Ollama missing",
                    workspace_dir=requested_cwd,
                    body_lines=[
                        "Ollama is not installed on this device yet.",
                        "VORTEX can run Ollama's official installer if you want.",
                        f"Manual docs: {ollama_install_docs_url()}",
                    ],
                    options=[
                        (
                            "1",
                            "Install Ollama",
                            "Run the official Ollama installer with your permission.",
                        ),
                        ("2", "Retry", "Check again after installing Ollama."),
                        ("3", "Back", "Pick a different local model."),
                        ("4", "External API", "Skip local setup for now."),
                    ]
                    if allow_external
                    else [
                        (
                            "1",
                            "Install Ollama",
                            "Run the official Ollama installer with your permission.",
                        ),
                        ("2", "Retry", "Check again after installing Ollama."),
                        ("3", "Back", "Pick a different local model."),
                    ],
                    error_message=install_error,
                    info_message=(
                        f"Local setup targets {option.model_name} once Ollama is available."
                    ),
                )
                if choice == "1":
                    installed, result = await _install_ollama_with_permission(
                        requested_cwd=requested_cwd,
                        tui=tui,
                        allow_external=allow_external,
                    )
                    if installed:
                        install_error = None
                        continue
                    if result == "__external__":
                        return "external", None
                    if result:
                        install_error = result
                    continue
                if choice == "2":
                    install_error = None
                    continue
                if choice == "3":
                    return "back", None
                return "external", None

            choice = await _prompt_validated_setup_decision(
                tui=tui,
                badge_label="Local",
                title="Ollama missing",
                workspace_dir=requested_cwd,
                body_lines=[
                    "Ollama is not installed on this device yet.",
                    "Install Ollama first, then come back and choose Retry.",
                    f"Manual docs: {ollama_install_docs_url()}",
                ],
                options=_local_setup_options(
                    primary=("1", "Retry", "Check again after installing Ollama."),
                    secondary=("2", "Back", "Pick a different local model."),
                    allow_external=allow_external,
                ),
                error_message=install_error,
                info_message=(
                    f"Local setup targets {option.model_name} once Ollama is available."
                ),
            )
            if choice == "1":
                install_error = None
                continue
            if choice == "2":
                return "back", None
            return "external", None

        try:
            installed_models = await list_installed_models()
        except Exception as exc:
            choice = await _prompt_validated_setup_decision(
                tui=tui,
                badge_label="Local",
                title="Ollama not running",
                workspace_dir=requested_cwd,
                body_lines=[
                    "Ollama is installed, but the local server did not respond.",
                    "You can let VORTEX try to start Ollama, or start it yourself and retry.",
                    f"Expected endpoint: {OLLAMA_OPENAI_BASE_URL}",
                ],
                options=[
                    ("1", "Start Ollama", "Try to launch the local Ollama server now."),
                    ("2", "Retry", "Check the local Ollama server again."),
                    ("3", "Back", "Pick a different local model."),
                    ("4", "External API", "Skip local setup for now."),
                ]
                if allow_external
                else [
                    ("1", "Start Ollama", "Try to launch the local Ollama server now."),
                    ("2", "Retry", "Check the local Ollama server again."),
                    ("3", "Back", "Pick a different local model."),
                ],
                error_message=start_error or _clean_error_message(exc),
            )
            if choice == "1":
                started, result = await _start_ollama_with_permission(
                    requested_cwd=requested_cwd,
                    tui=tui,
                    allow_external=allow_external,
                )
                if started:
                    start_error = None
                    continue
                if result == "__external__":
                    return "external", None
                if result:
                    start_error = result
                continue
            if choice == "2":
                start_error = None
                continue
            if choice == "3":
                return "back", None
            return "external", None

        if option.model_name not in installed_models:
            models_dir = resolve_ollama_models_dir()
            free_bytes = get_free_space_bytes(models_dir)
            required_bytes = option.size_bytes

            if not has_enough_space(
                required_bytes=required_bytes,
                free_bytes=free_bytes,
            ):
                choice = await _prompt_validated_setup_decision(
                    tui=tui,
                    badge_label="Space",
                    title="Not enough space",
                    workspace_dir=requested_cwd,
                    body_lines=[
                        f"{option.model_name} needs about {format_bytes(required_bytes)}.",
                        f"Only {format_bytes(free_bytes)} is free in {models_dir}.",
                        "Free up more disk space, then choose Retry.",
                    ],
                    options=[
                        ("1", "Back", "Pick a different local model."),
                        ("2", "External API", "Skip local setup for now."),
                        ("3", "Retry", "Check disk space again."),
                    ]
                    if allow_external
                    else [
                        ("1", "Back", "Pick a different local model."),
                        ("2", "Retry", "Check disk space again."),
                    ],
                    default_choice="1",
                )
                if choice == "1":
                    return "back", None
                if allow_external and choice == "2":
                    return "external", None
                continue

            choice = await _prompt_validated_setup_decision(
                tui=tui,
                badge_label="Download",
                title="Download local model",
                workspace_dir=requested_cwd,
                body_lines=[
                    f"Model: {option.model_name}",
                    f"Approx download: {format_bytes(required_bytes)}",
                    f"Free space available: {format_bytes(free_bytes)}",
                ],
                options=_local_setup_options(
                    primary=("1", "Download now", "Pull the model with Ollama now."),
                    secondary=("2", "Back", "Pick a different local model."),
                    allow_external=allow_external,
                ),
            )
            if choice == "2":
                return "back", None
            if allow_external and choice == "3":
                return "external", None

            try:
                tui.show_status(f"Local setup: pulling {option.model_name}")
                await pull_model(
                    option.model_name,
                    progress_callback=lambda message: tui.show_status(
                        f"Local setup: {message}"
                    ),
                )
            except Exception as exc:
                tui.clear_status()
                choice = await _prompt_validated_setup_decision(
                    tui=tui,
                    badge_label="Download",
                    title="Download failed",
                    workspace_dir=requested_cwd,
                    body_lines=[
                        "The local model download did not finish.",
                        "You can retry, go back, or switch to an external API.",
                    ],
                    options=_local_setup_options(
                        primary=("1", "Retry", "Try downloading the model again."),
                        secondary=("2", "Back", "Pick a different local model."),
                        allow_external=allow_external,
                    ),
                    error_message=_clean_error_message(exc),
                )
                if choice == "1":
                    continue
                if choice == "2":
                    return "back", None
                return "external", None
            finally:
                tui.clear_status()

            continue

        upsert_env_file(
            env_path,
            {
                "BASE_URL": OLLAMA_OPENAI_BASE_URL,
                "MODEL_NAME": option.model_name,
                "API_KEY": "",
            },
        )
        _activate_workspace_env(requested_cwd)

        reloaded_config, errors = _load_config_with_errors(requested_cwd)
        credential_errors, other_errors = split_config_errors(errors)
        if other_errors:
            raise ValueError("\n".join(other_errors))
        if credential_errors:
            raise ValueError("\n".join(credential_errors))
        return "configured", reloaded_config


def _clear_local_ollama_workspace_env(requested_cwd: Path) -> None:
    upsert_env_file(
        requested_cwd / ".env",
        {
            "BASE_URL": "",
            "MODEL_NAME": "",
            "API_KEY": "",
        },
    )
    _activate_workspace_env(requested_cwd)


async def _prompt_validated_setup_decision(
    *,
    tui: TUI,
    badge_label: str,
    title: str,
    workspace_dir: Path,
    body_lines: list[str],
    options: list[tuple[str, str, str]],
    default_choice: str = "1",
    error_message: str | None = None,
    info_message: str | None = None,
) -> str:
    valid_choices = [key for key, _, _ in options]
    current_error = error_message
    current_info = info_message

    while True:
        choice = (
            await tui.prompt_setup_decision(
                badge_label=badge_label,
                title=title,
                workspace_dir=workspace_dir,
                body_lines=body_lines,
                options=options,
                default_choice=default_choice,
                error_message=current_error,
                info_message=current_info,
            )
        ).strip() or default_choice

        if choice in valid_choices:
            return choice

        current_error = _format_choice_error(valid_choices)
        current_info = None


def _local_setup_options(
    *,
    primary: tuple[str, str, str],
    secondary: tuple[str, str, str],
    allow_external: bool,
) -> list[tuple[str, str, str]]:
    options = [primary, secondary]
    if allow_external:
        options.append(
            ("3", "External API", "Skip local setup and use a hosted provider.")
        )
    return options


def _format_choice_error(valid_choices: list[str]) -> str:
    if not valid_choices:
        return "Choose a valid option to continue."
    if len(valid_choices) == 1:
        return f"Choose {valid_choices[0]} to continue."
    if len(valid_choices) == 2:
        return f"Choose {valid_choices[0]} or {valid_choices[1]} to continue."
    return (
        f"Choose {', '.join(valid_choices[:-1])}, or {valid_choices[-1]} to continue."
    )


def _should_repair_local_ollama_setup(
    *,
    config: Config,
    record: ModelHealthRecord | None,
) -> bool:
    if config.active_model_profile is not None:
        return False
    if not is_ollama_base_url(config.base_url):
        return False
    if record is None:
        return True
    return record.status != "working"


def _should_repair_api_credentials(record: ModelHealthRecord | None) -> bool:
    if record is None:
        return False
    return record.status in {"missing-key", "auth-error"}


@contextlib.contextmanager
def _temporary_workspace_env(cwd: Path):
    env_path = cwd / ".env"
    env_values = _read_env_values(env_path)
    impacted_keys = set(ACTIVE_WORKSPACE_ENV) | set(env_values)
    if not impacted_keys:
        yield
        return

    previous_values = {key: os.environ.get(key) for key in impacted_keys}

    for key in ACTIVE_WORKSPACE_ENV:
        original_value = INITIAL_ENV.get(key)
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    os.environ.update(env_values)
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _validate_startup_workspace_choice(candidate: Path) -> str | None:
    try:
        validated_cwd = _validate_workspace_directory(candidate)
    except ValueError as exc:
        return str(exc)

    try:
        with _temporary_workspace_env(validated_cwd):
            _, errors = _load_config_with_errors(validated_cwd)
    except Exception as exc:
        return str(exc)

    _, other_errors = split_config_errors(errors)
    if other_errors:
        return "\n".join(other_errors)

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
@click.version_option(
    version=get_current_version(PROJECT_ROOT),
    prog_name="vortex",
)
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
@click.option(
    "--update",
    is_flag=True,
    help="Upgrade the installed vortex-agent-cli package and exit.",
)
def main(
    prompt: str | None,
    cwd: Path | None,
    update: bool,
):
    if update:
        sys.exit(_run_update_flow())

    workspace_history = WorkspaceHistoryManager()

    try:
        previous_env = dict(ACTIVE_WORKSPACE_ENV)
        requested_cwd = _choose_startup_workspace(
            prompt=prompt,
            requested_cwd=cwd,
            workspace_history=workspace_history,
        )
        _activate_workspace_env(requested_cwd)
        config, errors = _load_config_with_errors(requested_cwd)
        credential_errors, other_errors = split_config_errors(errors)
        if other_errors:
            raise ValueError("\n".join(other_errors))
        if credential_errors:
            selector_tui = TUI(config, console)
            config = asyncio.run(
                _prompt_for_missing_api_credentials(
                    requested_cwd=requested_cwd,
                    config=config,
                    tui=selector_tui,
                )
            )
    except Exception as e:
        console.print(f"[error]Configuration Error: {e}[/error]")
        sys.exit(1)

    workspace_history.record(requested_cwd)
    cli = CLI(config, workspace_history=workspace_history)
    try:
        new_env_values = dict(ACTIVE_WORKSPACE_ENV)
        cli.tui.show_env_change(previous_env, new_env_values)
    except Exception:
        pass

    # messages = [{"role": "user", "content": prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
