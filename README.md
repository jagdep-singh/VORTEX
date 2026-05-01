# VORTEX

VORTEX is an art-forward terminal AI pair‑programmer: OpenAI‑compatible brain, aurora TUI skin, live tools, and a workspace‑aware memory so it actually helps you ship.

## Highlights
- Streamed replies in a sculpted Rich TUI with animated gradients and structured tool cards.
- Real tools: read/write/edit files, shell, search, symbol index, MCP servers, checkpoints, and sessions.
- Workspace‑aware: picks a working directory first, reloads config/tools per project, remembers recents.
- Model aware: provider profiles, live model discovery, health probes, and buckets (working/quota/not working).
- Safe by default: approval modes, loop detection, compact workspace snapshot, and a lightweight code index.

## Install
- Best: `pipx install vortex-agent-cli`
- Local checkout: `python3 -m pip install . --no-build-isolation`
- One-shot dev env: `./scripts/install.sh`
- Optional for MCP servers: `python3 -m pip install fastmcp`

## Update
- Standard install: `vortex --update`
- Editable/local checkout: pull latest git instead (the app will tell you).

## Run
- Interactive: `vortex`
- Single prompt: `vortex "write a hello world program in c"`
- Choose project up front: `vortex --cwd /path/to/project`
- Inside the app, `/cwd` switches projects and rebuilds context.

## Configure
Put `.ai-agent/config.toml` in your project:
```toml
active_model_profile = "openrouter"

[models.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[models.openrouter.model]
name = "openrouter/free"
temperature = 0
max_output_tokens = 8192
```

First run can now offer a local-or-external setup choice. For a local Ollama path, VORTEX can guide the workspace through a small starter model setup on macOS, Linux, and Windows: it checks whether Ollama is installed and running, can optionally launch Ollama's official installer with your permission, checks local disk space, offers to pull the selected model, and saves the local base URL plus model choice into the workspace `.env`. If you choose an external provider instead, it will prompt for the provider URL and API key and store them in the same workspace `.env`. Use `/config` anytime to see the resolved profile, base URL, key source, and model.

Gemini via Google’s OpenAI-compatible endpoint:
```toml
[models.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"

[models.gemini.model]
name = "gemini-2.0-flash"
temperature = 0.2
max_output_tokens = 8192

[models.gemini.gemini]
reasoning_effort = "low"
cached_content = "cachedContents/abc123"

[models.gemini.gemini.thinking_config]
include_thoughts = true
# Use only one of these:
# thinking_level = "medium"
# thinking_budget = 8192
```

Small local Ollama recommendations:
- `qwen2.5-coder:1.5b` is the better low-storage default. It is smaller, starts faster, and is the safer pick for laptops that are tight on RAM or disk.
- `qwen2.5-coder:3b` is the better quality local option if you can spare more space. It is slower and heavier, but usually gives more reliable coding and editing output.
- Suggested default chooser copy:
  - `Fast + light`: `qwen2.5-coder:1.5b`
  - `Better coding quality`: `qwen2.5-coder:3b`
  - `Use external API instead`

## Core commands
- `/models [refresh]` – list or probe models for all profiles.
- `/model <name|number>` – switch profile or pick a discovered model.
- `/config` – show resolved settings.
- `/api-change` – re-enter provider URL and API key (restarts the session).
- `/scan` and `/index` – refresh workspace snapshot and symbol index.
- `/save`, `/sessions`, `/resume`, `/checkpoint`, `/restore` – persistence.
- `/tools`, `/mcp` – inspect tools and MCP servers.
- `/mcp attach <name> <url|command>` – connect an MCP server at runtime (SSE via URL, or stdio via command + args).
- `/help` – full reference.

## MCP servers
- Declare static servers in `.ai-agent/config.toml` under `[mcp_servers.<name>]` with either `command/args` (stdio) or `url` (SSE).
- Attach on the fly with `/mcp attach demo http://localhost:3000/mcp` or `/mcp attach ollama ollama serve`.
- Requires the `fastmcp` Python package (install once per environment). Tools are registered as `server__toolname` inside the agent.

## Docker (optional)
```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/workspace \
  -v vortex-data:/data \
  vortex
```
Add `--cwd /workspace/subdir` if you want a different project inside the container.

## Release
- Version is in `pyproject.toml`.
- CI builds/tests/publishes via `.github/workflows/publish-pypi.yml`.
- Create a GitHub release after bumping the version to publish to PyPI.
- Current version: 1.0.0.

## Progress log (local)
- Working notes for handoff live in `progress_report.txt` (ignored by git).

## Shape of the repo
- `main.py` – CLI entry.
- `ui/tui.py` – aurora terminal UI.
- `agent/` – agent loop, events, persistence.
- `tools/` – builtin tools, discovery, registry, MCP.
- `context/` – workspace snapshot, code index, compaction.
- `utils/` – credentials, versioning, discovery helpers.
- `workspace/` – default scratch project.
