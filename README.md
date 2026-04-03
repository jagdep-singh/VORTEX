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

First run will prompt for provider URL and API key and store them in your workspace `.env`. Use `/config` anytime to see the resolved profile, base URL, key source, and model.

## Core commands
- `/models [refresh]` – list or probe models for all profiles.
- `/model <name|number>` – switch profile or pick a discovered model.
- `/config` – show resolved settings.
- `/scan` and `/index` – refresh workspace snapshot and symbol index.
- `/save`, `/sessions`, `/resume`, `/checkpoint`, `/restore` – persistence.
- `/tools`, `/mcp` – inspect tools and MCP servers.
- `/help` – full reference.

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

## Shape of the repo
- `main.py` – CLI entry.
- `ui/tui.py` – aurora terminal UI.
- `agent/` – agent loop, events, persistence.
- `tools/` – builtin tools, discovery, registry, MCP.
- `context/` – workspace snapshot, code index, compaction.
- `utils/` – credentials, versioning, discovery helpers.
- `workspace/` – default scratch project.
