import asyncio
import contextlib
from pathlib import Path
import sys

import click
from dotenv import load_dotenv

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

load_dotenv(PROJECT_ROOT / ".env")

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Agent | None = None
        self.config = config
        self.tui = TUI(config, console)
        self._active_message_task: asyncio.Task[str | None] | None = None

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

    async def run_single(self, message: str) -> str | None:
        async with Agent(self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome()

        async with Agent(
            self.config,
            confirmation_callback=self.tui.handle_confirmation,
        ) as agent:
            self.agent = agent

            while True:
                try:
                    user_input = console.input(self.tui.prompt()).strip()
                    if not user_input:
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

        return final_response

    async def _handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        parts = cmd.split(maxsplit=1)
        cmd_name = parts[0]
        cmd_args = parts[1] if len(parts) > 1 else ""
        if cmd_name == "/exit" or cmd_name == "/quit":
            return False
        elif command == "/help":
            self.tui.show_help()
        elif command == "/clear":
            self.agent.session.context_manager.clear()
            self.agent.session.loop_detector.clear()
            self.tui.show_notice("Conversation cleared.", level="success")
        elif command == "/config":
            self.tui.show_config()
        elif cmd_name == "/model":
            if cmd_args:
                self.config.model_name = cmd_args
                self.tui.show_notice(
                    f"Model changed to: {cmd_args}",
                    level="success",
                )
            else:
                self.tui.show_notice(
                    f"Current model: {self.config.model_name}",
                    level="info",
                )
        elif cmd_name == "/approval":
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
        elif cmd_name == "/stats":
            stats = self.agent.session.get_stats()
            self.tui.show_stats(stats)
        elif cmd_name == "/tools":
            tools = self.agent.session.tool_registry.get_tools()
            self.tui.show_tools(tools)
        elif cmd_name == "/mcp":
            mcp_servers = self.agent.session.mcp_manager.get_all_servers()
            self.tui.show_mcp_servers(mcp_servers)
        elif cmd_name == "/save":
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
        elif cmd_name == "/sessions":
            persistence_manager = PersistenceManager()
            sessions = persistence_manager.list_sessions()
            self.tui.show_saved_sessions(sessions, "Saved sessions")
        elif cmd_name == "/resume":
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
        elif cmd_name == "/checkpoint":
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
        elif cmd_name == "/restore":
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
    requested_cwd = (cwd or WORKSPACE).resolve()

    try:
        config = load_config(cwd=requested_cwd)
    except Exception as e:
        console.print(f"[error]Configuration Error: {e}[/error]")
        sys.exit(1)

    config.cwd = requested_cwd

    errors = config.validate()

    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")

        sys.exit(1)

    cli = CLI(config)

    # messages = [{"role": "user", "content": prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
