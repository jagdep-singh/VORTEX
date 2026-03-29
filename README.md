# VORTEX

VORTEX is a local terminal coding agent built around an OpenAI-compatible chat API, a Rich-based TUI, and a small tool system for reading files, editing code, running shell commands, and managing sessions.

It is designed to run against a working directory on your machine, stream model output live, show tool calls as they happen, and keep enough session state around to support multi-step coding tasks.

## What This Project Does

- Runs in interactive mode or single-prompt mode
- Streams VORTEX agent output and tool execution in a custom terminal UI
- Lets the model call local tools such as file read/write/edit, search, shell, and memory
- Builds a compact workspace snapshot so the agent starts with project context
- Builds a lightweight symbol index so the agent can reason about functions, classes, structs, and types
- Supports approval policies for risky or mutating actions
- Saves sessions and checkpoints
- Loads extra tools from `.ai-agent/tools`
- Can connect to MCP servers and register their tools
- Supports custom model/provider profiles with different API keys and base URLs

## How It Runs

The entrypoint is [main.py](./main.py).

By default:

- the app starts in interactive mode if you run `python3 main.py`
- if you do not pass `--cwd`, VORTEX shows a workspace picker before the session starts
- the picker includes a `Custom path...` option for manually entering a project directory
- the selected working directory is remembered for later sessions
- the terminal UI is rendered by [ui/tui.py](./ui/tui.py)
- the agent runtime is driven by [agent/agent.py](./agent/agent.py)

Single prompt mode:

```bash
python3 main.py "write a hello world program in c"
```

Interactive mode:

```bash
python3 main.py
```

Use a different working directory:

```bash
python3 main.py --cwd .
```

Switch projects inside the app:

```text
/cwd
/cwd .
/cwd ~/some/project
/cwd 2
```

## Working Directory Behavior

Config and tool discovery are tied to the active `cwd`.

- If you run `python3 main.py --cwd <path>`, that directory is used directly
- If you run `python3 main.py` interactively, VORTEX asks you to choose a working directory first
- Custom paths are validated as existing directories before the session switches to them
- Recent working directories are remembered and can be reopened with `/cwd <index>` or viewed with `/recent`
- If you want project-local config from the repo root, run `python3 main.py --cwd .`
- `.ai-agent/config.toml` is loaded from the active working directory
- `AGENT.MD` is also loaded from the active working directory
- `.ai-agent/tools/*.py` is discovered from the active working directory

Changing `cwd` is treated as a project switch:

- the app reloads config and `AGENT.MD` from the new directory
- project-local tools are rediscovered
- the workspace snapshot and code index are rebuilt
- the live session context is reset to avoid mixing projects

## Requirements

This repo does not currently ship with a packaged installer or lockfile, so setup is still manual.

Core Python dependencies used by the codebase include:

- `openai`
- `rich`
- `click`
- `python-dotenv`
- `pydantic`
- `platformdirs`
- `prompt_toolkit`
- `tomli` for Python versions without `tomllib`

Optional integrations:

- `ddgs` for web search
- `fastmcp` for MCP usage

The bundled `requirements.txt` installs the core runtime used by the app and the Docker image. Optional integrations remain opt-in.

## Environment Variables

Legacy single-provider setup:

```bash
API_KEY=your_key_here
BASE_URL=https://openrouter.ai/api/v1
```

You can place these in a local `.env` file. If you store secrets in `.env`, keep that file out of version control.

## Docker

This repo now includes:

- [Dockerfile](./Dockerfile)
- [.dockerignore](./.dockerignore)
- [docker/entrypoint.sh](./docker/entrypoint.sh)
- [requirements.txt](./requirements.txt)

The container is built for interactive terminal usage.

Build the image:

```bash
docker build -t vortex .
```

Compose build:

```bash
docker compose build
```

Start with Compose:

```bash
docker compose up --build
```

For this specific project, `docker compose run --rm vortex` is usually the better choice because VORTEX is an interactive terminal app rather than a background service.

Interactive Compose run:

```bash
docker compose run --rm vortex
```

Run it against the current directory as the working project:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/workspace \
  -v vortex-data:/data \
  vortex
```

What that does:

- mounts your current directory into `/workspace`
- stores app data, sessions, and platformdirs state in `/data`
- starts the agent with `/workspace` as the default `--cwd`
- keeps the TUI interactive because of `-it`

Single prompt mode:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/workspace \
  -v vortex-data:/data \
  vortex "write a hello world program in c"
```

Use a different mounted project:

```bash
docker run --rm -it \
  --env-file .env \
  -v /path/to/project:/workspace \
  -v vortex-data:/data \
  vortex
```

Override the working directory manually:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/workspace \
  -v vortex-data:/data \
  vortex --cwd /workspace/subdir
```

Notes:

- the container defaults to `/workspace` unless you pass your own `--cwd`
- if you want custom `.ai-agent/config.toml`, put it inside the mounted project
- if you want to avoid root-owned files on Linux, add `--user "$(id -u):$(id -g)"`
- the base image includes the app and core Python dependencies, but not optional `ddgs` or `fastmcp`

## Custom Model Profiles

VORTEX now supports named provider profiles, so users are not locked to one model or one API key.

You can define them in `.ai-agent/config.toml` for the active working directory:

```toml
active_model_profile = "openrouter"

[models.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[models.openrouter.model]
name = "nvidia/nemotron-3-super-120b-a12b:free"
temperature = 0

[models.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[models.openai.model]
name = "gpt-4.1-mini"
temperature = 0.2
```

You can also inline a key directly:

```toml
[models.local_gateway]
base_url = "https://example.com/v1"
api_key = "your_key_here"

[models.local_gateway.model]
name = "some/model"
temperature = 0
```

Runtime commands:

- `/models` shows configured profiles
- `/model openrouter` switches to a named profile
- `/model use openrouter` also switches to a named profile
- `/model gpt-4.1-mini` changes the current model name directly
- `/config` shows the active model, base URL, and API key source

If the active profile has no resolved API key, the app will warn you.

## Interactive Commands

The terminal command set currently includes:

- `/help`
- `/exit`
- `/quit`
- `/clear`
- `/scan`
- `/index`
- `/cwd [path|index]`
- `/recent`
- `/config`
- `/models`
- `/model <name>`
- `/approval <mode>`
- `/stats`
- `/tools`
- `/mcp`
- `/save`
- `/checkpoint [name]`
- `/sessions`
- `/resume <session_id>`
- `/restore <checkpoint_id>`

Useful interaction behavior:

- `Ctrl+C` stops the current run without exiting the app
- arrow keys work in the input prompt through `prompt_toolkit`
- VORTEX agent output streams live
- tool calls are rendered as structured terminal cards
- the agent refreshes a compact workspace snapshot at the start of each run
- the agent also refreshes a lightweight codebase index and exposes it through `/index` and `find_symbol`
- `/cwd` switches the whole active project and rebuilds session context

## Codebase Index

VORTEX now includes a lightweight source-code index that extracts symbols from common languages such as Python, C, C++, JavaScript, TypeScript, Go, Rust, and Java.

This gives the agent a faster way to orient itself around a project:

- a compact symbol summary is injected into the system prompt
- `/scan` refreshes and shows the current workspace snapshot
- `/index` shows the current symbol index in the TUI
- the built-in `find_symbol` tool can look up functions, classes, structs, enums, interfaces, types, traits, and macros by name

This is intentionally heuristic and lightweight. It is not a full language server, but it gives the agent a much better structural map of the workspace than plain file listing alone.

## Approval Modes

Approval policy values come from [config/config.py](./config/config.py):

- `on-request`
- `on-failure`
- `auto`
- `auto-edit`
- `never`
- `yolo`

These control when the agent pauses before mutating files or executing risky actions.

## Built-in Tools

The built-in tool registry is created in [tools/registry.py](./tools/registry.py).

Current built-ins:

- `read_file`
- `write_file`
- `edit`
- `find_symbol`
- `shell`
- `list_dir`
- `grep`
- `glob`
- `web_search`
- `web_fetch`
- `todos`
- `memory`

The project also registers two built-in subagent tools from [tools/subagents.py](./tools/subagents.py):

- `subagent_codebase_investigator`
- `subagent_code_reviewer`

## Tool Discovery

Custom tools can be dropped into:

- `<cwd>/.ai-agent/tools/*.py`
- the user config directory returned by `platformdirs.user_config_dir("ai-agent")`

Each discovered file is imported dynamically and any `Tool` subclasses are registered.

## MCP Support

MCP servers are configured in `.ai-agent/config.toml` and initialized through:

- [tools/mcp/mcp_manager.py](./tools/mcp/mcp_manager.py)
- [tools/mcp/client.py](./tools/mcp/client.py)
- [tools/mcp/mcp_tool.py](./tools/mcp/mcp_tool.py)

Both stdio-style servers and HTTP/SSE-style servers are supported by config.

## Sessions, Checkpoints, and Memory

Session persistence lives in [agent/persistence.py](./agent/persistence.py).

The app can:

- save a session
- list sessions
- resume a session
- create a checkpoint
- restore a checkpoint

User memory is loaded from the app data directory and injected into context when available.

## Hooks

Hooks are configured in `.ai-agent/config.toml` and executed by [hooks/hook_system.py](./hooks/hook_system.py).

Available trigger points:

- `before_agent`
- `after_agent`
- `before_tool`
- `after_tool`
- `on_error`

## Terminal UI

The current UI is a custom Rich interface with:

- a cyan-on-dark shell theme
- a VORTEX ASCII header
- streaming VORTEX agent output
- inline styling for backticked text
- structured tool call panels
- approval prompts
- status lines for model thinking/progress

The input prompt uses `prompt_toolkit` so cursor movement and history work properly in a real terminal.

## Project Layout

High-level structure:

```text
main.py                  CLI entrypoint
agent/                   agent loop, events, session, persistence
client/                  OpenAI-compatible streaming client and response models
config/                  config schema and loader
context/                 message history, compression, loop detection
hooks/                   hook execution
prompts/                 system prompt construction
safety/                  approval policy logic
tools/                   builtin tools, discovery, registry, MCP, subagents
ui/                      terminal user interface
utils/                   misc helpers
workspace/               default working directory
```

## Current State

This project is already usable for local coding tasks, but it is still a hand-rolled agent app rather than a fully packaged product.

A few practical things to know:

- setup is still manual
- secrets/config handling is flexible but not yet polished into a full settings flow
- optional integrations are lazy-loaded and only matter if you use those features
- the default experience is centered around the local `workspace/` folder

## Quick Start

1. Install the Python dependencies you need for this repo.
2. Set `API_KEY` and optionally `BASE_URL`, or configure a model profile.
3. Run `python3 main.py`.
4. Type a request at the `╰─ you ›` prompt.
5. Use `/help` if you want to inspect commands from inside the app.
