# Ollama MCP

MCP server that exposes your **local Ollama API** as tools so agents (Cursor, Claude Desktop, etc.) can use local models for chat, completion, and embeddings.

## Prerequisites

- [Ollama](https://ollama.com) installed and running (default: `http://localhost:11434`)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Install & run

```bash
uv sync
uv run server.py
```

The server uses **stdio** transport (no HTTP server). MCP clients start it as a subprocess.

## Environment and custom .env

The server reads `OLLAMA_BASE_URL` from the environment. You can use a **`.env` file** in the project root so you don’t have to set it in your shell or MCP config.

1. Copy the example file and edit as needed:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` (it is gitignored). Example:

   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   ```

   Use a different URL if Ollama runs elsewhere (e.g. `http://192.168.1.10:11434` or a Docker host).

3. When you run the server (`uv run server.py` or via your MCP client), it loads `.env` from the same directory as `server.py` before reading env vars.

| Variable | Default | Description |
|----------|--------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL (no trailing slash). |

You can still override these in the MCP client config (e.g. Cursor’s `env` block) or in your shell; those take precedence over `.env`.

## MCP configuration

### Cursor

In this repo you can copy the example and point it at your path:

```bash
mkdir -p .cursor
cp .cursor/mcp.json.example .cursor/mcp.json
# Edit .cursor/mcp.json and set the path in "args" to your absolute path (e.g. pwd)
```

Or add manually to `~/.cursor/mcp.json` or your project’s `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/ollama-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

Replace `/ABSOLUTE/PATH/TO/ollama-mcp` with the real path (e.g. output of `pwd` in the repo).

To use a custom Ollama URL without a `.env` file, add an `env` block:

```json
"ollama": {
  "command": "uv",
  "args": ["--directory", "/ABSOLUTE/PATH/TO/ollama-mcp", "run", "server.py"],
  "env": {
    "OLLAMA_BASE_URL": "http://192.168.1.10:11434"
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent config path on your OS:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/ollama-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `list_models` | List installed Ollama models |
| `list_running_models` | List models currently loaded in memory |
| `show_model` | Get details for a model (params, size, etc.) |
| `chat` | Multi-turn chat (model + messages array). Optional `stream: true` to use Ollama streaming (full reply still returned). |
| `generate` | Single-prompt completion. Optional `stream: true` to use Ollama streaming (full reply still returned). |
| `embed` | Get embeddings (model + text or list of strings) |
| `pull_model` | Pull a model from the registry |
| `delete_model` | Delete an installed model |

## Testing

**Unit tests** (mocked Ollama API; no server required):

```bash
uv run pytest tests/ -v
```

**Integration tests** (run the MCP server as subprocess and send JSON-RPC; optional Ollama for real tool results):

```bash
uv run python scripts/run_integration_tests.py
```

- Checks: initialize handshake, `tools/list` (all 8 tools), `tools/call list_models`, `tools/call generate`.
- If Ollama is not running, `list_models` and `generate` still return (error message or timeout); the script verifies the protocol and tool wiring.

## Docs and memory

Project design and notes live in `.docs/` (gitignored). See `.docs/DESIGN.md` and `.docs/MCP_CONFIG.md` for details.

## License

MIT
