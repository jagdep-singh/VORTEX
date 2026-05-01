from __future__ import annotations
import re
from typing import AsyncGenerator, Callable
from agent.events import AgentEvent, AgentEventType
from agent.session import Session
from client.response import StreamEventType, TokenUsage, ToolCall, ToolResultMessage
from config.config import Config
from prompts.system import create_loop_breaker_prompt
from tools.base import ToolConfirmation
from utils.ollama_setup import is_lightweight_local_model

_TOOL_REQUIRED_PATTERNS = (
    "build",
    "create",
    "make",
    "implement",
    "fix",
    "edit",
    "update",
    "refactor",
    "change",
    "write",
    "add",
    "remove",
    "delete",
    "rename",
    "scaffold",
    "generate",
    "nextjs",
    "react",
    "page",
    "component",
    "repo",
    "codebase",
    "file",
    "project",
    "three.js",
)
_GENERIC_REPLY_PATTERNS = (
    "how can i assist you today",
    "as a large language model",
    "feel free to ask",
    "provide as much context as possible",
)


def _normalize_response_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _message_likely_requires_tools(message: str) -> bool:
    lowered = message.lower()
    return any(pattern in lowered for pattern in _TOOL_REQUIRED_PATTERNS)


def _response_looks_like_task_slug(response_text: str) -> bool:
    stripped = response_text.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return False
    if " " in stripped:
        return False
    if len(stripped) < 20:
        return False
    return bool(re.fullmatch(r"[a-z0-9_]+", stripped))


def _response_is_low_effort_for_workspace_task(response_text: str) -> bool:
    normalized = _normalize_response_text(response_text)
    lowered = normalized.lower()
    if not normalized:
        return True
    if _response_looks_like_task_slug(normalized):
        return True
    if any(pattern in lowered for pattern in _GENERIC_REPLY_PATTERNS):
        return True
    if len(normalized) < 80:
        return True
    return False


def _max_text_only_retries(model_name: str, base_url: str | None) -> int:
    if is_lightweight_local_model(model_name, base_url):
        return 2
    return 1


def _should_retry_text_only_workspace_response(
    *,
    user_message: str,
    response_text: str,
    tool_calls_present: bool,
    retry_count: int,
    model_name: str,
    base_url: str | None,
) -> bool:
    if tool_calls_present:
        return False
    if not _message_likely_requires_tools(user_message):
        return False
    if retry_count >= _max_text_only_retries(model_name, base_url):
        return False
    if _response_is_low_effort_for_workspace_task(response_text):
        return True
    if is_lightweight_local_model(model_name, base_url):
        return len(_normalize_response_text(response_text)) < 240
    return False


def _should_fail_text_only_workspace_response(
    *,
    user_message: str,
    response_text: str,
    tool_calls_present: bool,
) -> bool:
    if tool_calls_present:
        return False
    if not _message_likely_requires_tools(user_message):
        return False
    return _response_is_low_effort_for_workspace_task(response_text)


def _build_tool_retry_message(model_name: str, base_url: str | None) -> str:
    base = (
        "Your previous reply did not make progress on the workspace task. "
        "This request requires using the available tools. "
        "Inspect files, run commands, or edit the workspace now instead of answering with a greeting, disclaimer, summary, or task label."
    )
    if is_lightweight_local_model(model_name, base_url):
        return (
            base
            + " The active model is small, so an imperfect attempt is still better than a text-only reply."
        )
    return base


class Agent:
    def __init__(
        self,
        config: Config,
        confirmation_callback: Callable[[ToolConfirmation], bool] | None = None,
    ):
        self.config = config
        self.session: Session | None = Session(self.config)
        self.session.approval_manager.confirmation_callback = confirmation_callback

    async def run(self, message: str):
        await self.session.hook_system.trigger_before_agent(message)
        self.session.refresh_workspace_context()
        yield AgentEvent.agent_start(message)
        self.session.context_manager.add_user_message(message)

        final_response: str | None = None

        async for event in self._agentic_loop(message):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        await self.session.hook_system.trigger_after_agent(message, final_response)
        usage = None
        if self.session and self.session.context_manager:
            usage = self.session.context_manager.latest_usage

        yield AgentEvent.agent_end(final_response, usage)

    async def _agentic_loop(
        self,
        initial_user_message: str,
    ) -> AsyncGenerator[AgentEvent, None]:
        max_turns = self.config.max_turns
        text_only_retry_count = 0

        for turn_num in range(max_turns):
            self.session.increment_turn()
            response_text = ""

            if self.session.context_manager.needs_compression():
                yield AgentEvent.status("Thinking: compacting earlier context...")
                summary, usage = await self.session.chat_compactor.compress(
                    self.session.context_manager
                )

                if summary:
                    self.session.context_manager.replace_with_summary(summary)
                    self.session.context_manager.set_latest_usage(usage)
                    self.session.context_manager.add_usage(usage)

            tool_schemas = self.session.tool_registry.get_schemas()

            tool_calls: list[ToolCall] = []
            usage: TokenUsage | None = None
            defer_text_output = (
                turn_num == 0 and _message_likely_requires_tools(initial_user_message)
            )

            if turn_num == 0:
                yield AgentEvent.status("Thinking: planning the next step...")
            else:
                yield AgentEvent.status("Thinking: reviewing the latest results...")

            async for event in self.session.client.chat_completion(
                self.session.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
            ):
                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        if not defer_text_output:
                            yield AgentEvent.text_delta(content)
                elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                    if event.tool_call:
                        tool_calls.append(event.tool_call)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(
                        event.error or "Unknown error occurred.",
                    )
                elif event.type == StreamEventType.MESSAGE_COMPLETE:
                    usage = event.usage

            self.session.context_manager.add_assistant_message(
                response_text or None,
                ([tc.to_openai_dict() for tc in tool_calls] if tool_calls else None),
            )

            if usage:
                self.session.context_manager.set_latest_usage(usage)
                self.session.context_manager.add_usage(usage)

            if _should_retry_text_only_workspace_response(
                user_message=initial_user_message,
                response_text=response_text,
                tool_calls_present=bool(tool_calls),
                retry_count=text_only_retry_count,
                model_name=self.config.model_name,
                base_url=self.config.base_url,
            ):
                text_only_retry_count += 1
                self.session.context_manager.add_user_message(
                    _build_tool_retry_message(
                        self.config.model_name,
                        self.config.base_url,
                    )
                )
                yield AgentEvent.status(
                    "Thinking: retrying with a stronger tool-use nudge..."
                )
                continue

            if response_text:
                yield AgentEvent.text_complete(response_text)
                self.session.loop_detector.record_action(
                    "response",
                    text=response_text,
                )
                loop_description = self.session.loop_detector.check_for_loop()
                if loop_description:
                    yield AgentEvent.agent_error(
                        f"Stopped to avoid a loop: {loop_description}"
                    )
                    return

            if not tool_calls:
                if _should_fail_text_only_workspace_response(
                    user_message=initial_user_message,
                    response_text=response_text,
                    tool_calls_present=bool(tool_calls),
                ):
                    yield AgentEvent.agent_error(
                        "The current model replied in plain text instead of using workspace tools for the task. "
                        "VORTEX retried automatically, but this model is still too weak for reliable agentic work. "
                        "Try qwen2.5-coder:3b, another stronger local model, or a hosted provider."
                    )
                    return
                self.session.context_manager.prune_tool_outputs()
                return

            yield AgentEvent.status("Thinking: choosing a tool to move forward...")
            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                )

                self.session.loop_detector.record_action(
                    "tool_call",
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )
                loop_description = self.session.loop_detector.check_for_loop()
                if loop_description:
                    yield AgentEvent.agent_error(
                        f"Stopped to avoid a loop: {loop_description}"
                    )
                    return

                result = await self.session.tool_registry.invoke(
                    tool_call.name,
                    tool_call.arguments,
                    self.config.cwd,
                    self.session.hook_system,
                    self.session.approval_manager,
                )

                yield AgentEvent.tool_call_complete(
                    tool_call.call_id,
                    tool_call.name,
                    result,
                )

                tool_result_message = ToolResultMessage(
                    tool_call_id=tool_call.call_id,
                    content=result.to_model_output(),
                    name=tool_call.name,
                    is_error=not result.success,
                )
                self.session.context_manager.add_tool_result(
                    tool_result_message.tool_call_id,
                    tool_result_message.content,
                    name=tool_result_message.name,
                )

            yield AgentEvent.status("Thinking: reviewing tool output...")
            continue

        yield AgentEvent.agent_error(f"Maximum turns ({max_turns}) reached")

    async def __aenter__(self) -> Agent:
        await self.session.initialize()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        if self.session and self.session.client and self.session.mcp_manager:
            await self.session.client.close()
            await self.session.mcp_manager.shutdown()
            self.session = None
