# AI Agent

An AI agent that can execute tasks using tools and manage conversations.

## Features

### Core Functionality

- Interactive and single-run modes
- Streaming text responses
- Multi-turn conversations with tool calling
- Configurable model settings and temperature

### Built-in Tools

- File operations: read, write, edit files
- Directory operations: list directories, search with glob patterns
- Text search: grep for pattern matching
- Shell execution: run shell commands
- Web access: search and fetch web content
- Memory: store and retrieve information
- Todo: manage task lists

### Context Management

- Automatic context compression when approaching token limits
- Tool output pruning to manage context size
- Token usage tracking

### Safety and Approval

- Multiple approval policies: on-request, auto, never, yolo
- Dangerous command detection and blocking
- Path-based safety checks
- User confirmation prompts for mutating operations

### Session Management

- Save and resume sessions
- Create checkpoints
- Persistent session storage

### MCP Integration

- Connect to Model Context Protocol servers
- Use tools from MCP servers
- Support for stdio and HTTP/SSE transports

### Subagents

- Specialized subagents for specific tasks
- Built-in subagents: codebase investigator, code reviewer
- Configurable subagent definitions with custom tools and limits

### Loop Detection

- Detects repeating actions
- Prevents infinite loops in agent execution

### Hooks System

- Execute scripts before/after agent runs
- Execute scripts before/after tool calls
- Error handling hooks
- Custom commands and scripts

### Configuration

- Configurable working directory
- Tool allowlisting
- Developer and user instructions
- Shell environment policies
- MCP server configuration
- Named model profiles with custom `base_url` and API key sources

### User Interface

- Terminal UI with formatted output
- Command interface: /help, /config, /models, /model, /tools, /mcp, /stats, /save, /resume, /checkpoint, /restore
- Real-time tool call visualization

## Custom Model Profiles

You can define your own provider-backed models in `.ai-agent/config.toml`:

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

Then use `/models` to list them and `/model <profile-name>` to switch between them at runtime.
