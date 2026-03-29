from __future__ import annotations
import json
from typing import AsyncGenerator, Callable
from agent.events import AgentEvent, AgentEventType
from agent.session import Session
from client.response import StreamEventType, TokenUsage, ToolCall, ToolResultMessage
from config.config import Config
from prompts.system import create_loop_breaker_prompt
from tools.base import ToolConfirmation


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

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        await self.session.hook_system.trigger_after_agent(message, final_response)
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        max_turns = self.config.max_turns

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
                (
                    [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
            )

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

            if usage:
                self.session.context_manager.set_latest_usage(usage)
                self.session.context_manager.add_usage(usage)

            if not tool_calls:
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
                    is_error=not result.success,
                )
                self.session.context_manager.add_tool_result(
                    tool_result_message.tool_call_id,
                    tool_result_message.content,
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
