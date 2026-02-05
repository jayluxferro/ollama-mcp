# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-05

### Added

- MCP server exposing local Ollama API as tools for Cursor, Claude Desktop, and other agents.
- **Tools**: `ollama_version` (health check), `list_models`, `list_running_models`, `show_model`, `chat`, `generate`, `embed`, `copy_model`, `pull_model`, `delete_model`.
- **Streaming**: `chat` and `generate` support `stream: true` (Ollama streams; full reply still returned).
- **Config**: `.env` support; `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`, `OLLAMA_STREAM_READ_TIMEOUT`, `OLLAMA_MCP_LOG_LEVEL`.
- **HTTP**: shared async client, configurable timeouts, longer read timeout for streaming.
- **Pull**: `pull_model` consumes full NDJSON stream and returns final status.
- **Tests**: Unit tests (mocked), integration script (MCP protocol over stdio).
- **CI**: GitHub Actions workflow runs unit tests and integration script on push/PR; no Ollama service required.
- **Docs**: README, `.env.example`, Cursor and Claude Desktop config examples.
