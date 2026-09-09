# VORTEX

**A terminal AI pair-programmer.** OpenAI-compatible brain, an aurora-themed TUI skin, a real tool belt, and workspace-aware memory — so it actually helps you ship code instead of just chatting about it.

```bash
pipx install vortex-agent-cli
vortex
```

---

## Why VORTEX

Most terminal AI agents are either bare-bones chat wrappers or heavyweight IDE plugins. VORTEX sits in between: a single `vortex` command that drops you into a polished, animated terminal UI backed by a real agent loop — one that can read and edit your files, run shell commands, search the web, index your codebase, and remember what it did last time, all while asking permission before anything risky happens.

## Highlights

- **A terminal UI that doesn't look like a terminal UI** — streamed replies in a sculpted Rich interface with animated gradients and structured tool cards.
- **Real tools, not toy demos** — read/write/edit files, shell execution, grep/glob search, a symbol index, MCP servers, checkpoints, and session persistence.
- **Workspace-aware** — picks a working directory up front, reloads config and tools per project, and remembers recent projects.
- **Model-aware** — provider profiles, live model discovery, health probes, and buckets for working / rate-limited / broken models.
- **Safe by default** — approval modes for mutating actions, loop detection, a compact workspace snapshot, and a lightweight code index instead of dumping your whole repo into context.

## Install

| Method | Command |
|---|---|
| Recommended | `pipx install vortex-agent-cli` |
| Local checkout | `python3 -m pip install . --no-build-isolation` |
| One-shot dev environment | `./scripts/install.sh` |
| Optional, for MCP servers | `python3 -m pip install fastmcp` |

**Requires Python 3.10+.**

### Updating

- Standard install: `vortex --update`
- Editable/local checkout: pull the latest git changes instead — the app will tell you if it can't self-update.

## Quick start

```bash
# Interactive session in the current directory
vortex

# One-shot prompt, no session
vortex "write a hello world program in c"

# Choose the project up front
vortex --cwd /path/to/project
```

Inside a session, `/cwd` switches projects and rebuilds context on the fly.

On first run, VORTEX offers a local-or-external model setup:

- **Local (Ollama):** VORTEX checks whether Ollama is installed and running, can optionally launch its official installer with your permission, checks available disk space, offers to pull a starter model, and saves the base URL and model choice into your workspace's `.env`.
- **External provider:** VORTEX prompts for a provider URL and API key and stores them the same way.

Run `/config` anytime to see the resolved profile, base URL, key source, and active model. See [`docs/api-keys.md`](docs/api-keys.md) for how to get keys from OpenAI, OpenRouter, or a local gateway.

## Configuration

Drop a `.ai-agent/config.toml` into your project to control which model VORTEX talks to:

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

Gemini, via Google's OpenAI-compatible endpoint, is also supported:

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

### Running fully local with Ollama

If disk or RAM is tight, pick a smaller model; if you can spare the space, go bigger for better coding output:

| Model | Best for |
|---|---|
| `qwen2.5-coder:1.5b` | Fast + light — the safer default on constrained laptops |
| `qwen2.5-coder:3b` | Better coding quality — slower and heavier, more reliable edits |

## Core commands

| Command | Description |
|---|---|
| `/models [refresh]` | List or probe models across all profiles |
| `/model <name\|number>` | Switch profile or pick a discovered model |
| `/config` | Show resolved settings |
| `/api-change` | Re-enter provider URL and API key (restarts the session) |
| `/scan`, `/index` | Refresh the workspace snapshot and symbol index |
| `/save`, `/sessions`, `/resume` | Session persistence |
| `/checkpoint`, `/restore` | Rewind to a previous checkpoint |
| `/tools`, `/mcp` | Inspect available tools and MCP servers |
| `/mcp attach <name> <url\|command>` | Connect an MCP server at runtime |
| `/help` | Full command reference |

## Built-in tools

VORTEX ships with a working tool belt out of the box: reading, writing, and editing files; running shell commands; listing directories; `grep`/`glob` search; symbol lookup; a scratch-space memory tool; a todo tracker; and web search/fetch. Every mutating action goes through an approval layer, and a set of pattern-matched dangerous commands (destructive `rm`, disk formatting, piping remote scripts into a shell, fork bombs, and similar) are flagged before they ever run.

## MCP servers

VORTEX speaks [MCP](https://modelcontextprotocol.io/) as a client, so it can pull in external tool servers:

- Declare static servers in `.ai-agent/config.toml` under `[mcp_servers.<name>]`, using either `command`/`args` (stdio) or `url` (SSE).
- Attach one on the fly: `/mcp attach demo http://localhost:3000/mcp` or `/mcp attach ollama ollama serve`.
- Requires the `fastmcp` package (install once per environment). Attached tools show up in the agent as `server__toolname`.

## Docker

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD":/workspace \
  -v vortex-data:/data \
  vortex
```

Add `--cwd /workspace/subdir` to target a different project inside the container.

## Project layout

```
main.py            CLI entry point
ui/tui.py           Aurora terminal UI
agent/              Agent loop, events, persistence
tools/               Builtin tools, discovery, registry, MCP client
context/            Workspace snapshot, code index, compaction
safety/             Approval and dangerous-command detection
utils/               Credentials, versioning, model/provider discovery
workspace/          Default scratch project
```

## Release process

- Version lives in `pyproject.toml` (currently `1.0.0`).
- CI builds, tests, and publishes via `.github/workflows/publish-pypi.yml`.
- Bump the version, then cut a GitHub release to publish to PyPI.

## Links

- [Repository](https://github.com/jagdep-singh/VORTEX)
- [Issues](https://github.com/jagdep-singh/VORTEX/issues)
- [Getting an API key](docs/api-keys.md)
