# Getting Your Own API Key

VORTEX works with providers that expose an OpenAI-compatible chat API.

At minimum, you need:

- a provider base URL
- an API key for that provider

Common setups:

## OpenAI

1. Create or sign in to your OpenAI account.
2. Go to the API platform and create an API key.
3. Use this provider URL:

```text
https://api.openai.com/v1
```

4. Paste that URL into the VORTEX setup prompt.
5. Paste your API key when VORTEX asks for it.

## OpenRouter

1. Create or sign in to your OpenRouter account.
2. Create an API key from your account settings.
3. Use this provider URL:

```text
https://openrouter.ai/api/v1
```

4. Paste that URL into the VORTEX setup prompt.
5. Paste your API key when VORTEX asks for it.

## Local gateways

If you run a local OpenAI-compatible gateway such as Ollama or LiteLLM, use that server's `/v1` endpoint. For example:

```text
http://localhost:11434/v1
```

Make sure the server is already running before starting VORTEX.

## Where VORTEX saves these values

When you complete the startup prompt, VORTEX writes the values into the selected workspace's `.env` file.

Typical entries look like this:

```bash
BASE_URL=https://api.openai.com/v1
API_KEY=your_key_here
```

If your active model profile expects a named environment variable such as `OPENAI_API_KEY`, VORTEX saves the key under that variable name instead.

## Tips

- Keep `.env` out of version control.
- If your provider already has a base URL configured in `.ai-agent/config.toml`, VORTEX may only ask for the missing API key.
- You can still manage advanced multi-provider setups manually in `.ai-agent/config.toml`.
