from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from config.loader import get_data_dir


class WorkspaceHistoryManager:
    FILE_NAME = "recent_workspaces.json"
    MAX_ENTRIES = 8

    def __init__(self) -> None:
        self.data_dir = get_data_dir()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.file_path = self.data_dir / self.FILE_NAME

    def list_recent(self) -> list[dict[str, str]]:
        entries = self._load()
        recent: list[dict[str, str]] = []

        for entry in entries:
            path = entry.get("path")
            if not path:
                continue

            resolved = Path(path).expanduser()
            if not resolved.exists() or not resolved.is_dir():
                continue

            recent.append(
                {
                    "path": str(resolved.resolve()),
                    "last_used": str(entry.get("last_used", "")),
                }
            )

        return recent

    def latest(self) -> Path | None:
        recent = self.list_recent()
        if not recent:
            return None
        return Path(recent[0]["path"])

    def record(self, workspace: Path) -> None:
        resolved = workspace.expanduser().resolve()
        entries = [
            entry
            for entry in self.list_recent()
            if entry.get("path") != str(resolved)
        ]
        entries.insert(
            0,
            {
                "path": str(resolved),
                "last_used": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self._save(entries[: self.MAX_ENTRIES])

    def _load(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []

        try:
            with open(self.file_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        result: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                result.append(item)
        return result

    def _save(self, entries: list[dict[str, str]]) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as fp:
                json.dump(entries, fp, indent=2)

            try:
                os.chmod(self.file_path, 0o600)
            except OSError:
                pass
        except OSError:
            return
