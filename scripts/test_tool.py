#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    trigger = os.environ.get("AI_AGENT_TRIGGER")
    cwd = os.environ.get("AI_AGENT_CWD")
    tool_name = os.environ.get("AI_AGENT_TOOL_NAME")
    user_message = os.environ.get("AI_AGENT_USER_MESSAGE")
    error = os.environ.get("AI_AGENT_ERROR")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "trigger": trigger,
        "cwd": cwd,
        "tool_name": tool_name,
        "user_message": user_message,
        "error": error,
    }

    default_log_path = Path(__file__).resolve().parent.parent / ".ai-agent" / "hook.log"
    log_path = Path(os.environ.get("AI_AGENT_HOOK_LOG", str(default_log_path))).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[HOOK] {json.dumps(log_data)}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
