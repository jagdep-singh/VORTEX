from datetime import datetime
import json
from typing import Any
import uuid
from client.llm_client import LLMClient
from config.config import Config
from config.loader import get_data_dir
from context.compaction import ChatCompactor
from context.code_index import WorkspaceCodeIndex, build_workspace_code_index
from context.loop_detector import LoopDetector
from context.manager import ContextManager
from context.workspace_snapshot import build_workspace_snapshot
from hooks.hook_system import HookSystem
from safety.approval import ApprovalManager
from tools.discovery import ToolDiscoveryManager
from tools.mcp.mcp_manager import MCPManager
from tools.registry import create_default_registry


class Session:
    def __init__(self, config: Config):
        self.config = config
        self.client = LLMClient(config=config)
        self.tool_registry = create_default_registry(config)
        self.context_manager: ContextManager | None = None
        self.discovery_manager = ToolDiscoveryManager(
            self.config,
            self.tool_registry,
        )
        self.mcp_manager = MCPManager(self.config)
        self.chat_compactor = ChatCompactor(self.client)
        self.approval_manager = ApprovalManager(
            self.config.approval,
            self.config.cwd,
        )
        self.loop_detector = LoopDetector()
        self.hook_system = HookSystem(config)
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        self.turn_count = 0
        self.workspace_snapshot: str | None = None
        self.code_index: WorkspaceCodeIndex | None = None
        self.code_index_summary: str | None = None

    async def initialize(self) -> None:
        await self.mcp_manager.initialize()
        self.mcp_manager.register_tools(self.tool_registry)

        self.discovery_manager.discover_all()
        self.refresh_workspace_context()
        self.context_manager = ContextManager(
            config=self.config,
            user_memory=self._load_memory(),
            tools=self.tool_registry.get_tools(),
            workspace_snapshot=self.workspace_snapshot,
            code_index_summary=self.code_index_summary,
        )

    def refresh_workspace_snapshot(self) -> str | None:
        self.workspace_snapshot = build_workspace_snapshot(
            self.config.cwd,
            self.config.model_name,
        )

        if self.context_manager:
            self.context_manager.set_workspace_snapshot(self.workspace_snapshot)

        return self.workspace_snapshot

    def refresh_code_index(self) -> str | None:
        self.code_index = build_workspace_code_index(self.config.cwd)
        self.code_index_summary = (
            self.code_index.render_summary(self.config.model_name)
            if self.code_index
            else None
        )

        if self.context_manager:
            self.context_manager.set_code_index_summary(self.code_index_summary)

        return self.code_index_summary

    def refresh_workspace_context(self) -> tuple[str | None, str | None]:
        snapshot = self.refresh_workspace_snapshot()
        code_index_summary = self.refresh_code_index()
        return snapshot, code_index_summary

    def _load_memory(self) -> str | None:
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "user_memory.json"

        if not path.exists():
            return None

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            entries = data.get("entries")
            if not entries:
                return None

            lines = ["User preferences and notes:"]
            for key, value in entries.items():
                lines.append(f"- {key}: {value}")

            return "\n".join(lines)
        except Exception:
            return None

    def increment_turn(self) -> int:
        self.turn_count += 1
        self.updated_at = datetime.now()

        return self.turn_count

    def get_stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": self.turn_count,
            "message_count": self.context_manager.message_count,
            "token_usage": self.context_manager.total_usage,
            "tools_count": len(self.tool_registry.get_tools()),
            "mcp_servers": len(self.tool_registry.connected_mcp_servers),
            "indexed_source_files": self.code_index.indexed_files if self.code_index else 0,
            "indexed_symbols": len(self.code_index.symbols) if self.code_index else 0,
        }
