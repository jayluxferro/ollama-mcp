"""
Ollama MCP server: exposes local Ollama API as MCP tools for Cursor, Claude, etc.
Run with: uv run server.py
Use stderr for logging (stdio is used for MCP protocol).
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Load .env from project root so OLLAMA_BASE_URL etc. can be set there
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# Log to stderr only (stdio is used for MCP)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=__import__("sys").stderr,
)
logger = logging.getLogger("ollama-mcp")

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API = f"{OLLAMA_BASE.rstrip('/')}/api"

mcp = FastMCP("ollama")


def _api_url(path: str) -> str:
    return f"{OLLAMA_API.rstrip('/')}/{path.lstrip('/')}"


async def _request(
    method: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any] | list[Any]:
    url = _api_url(path)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(method, url, **kwargs)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}


async def _request_stream(
    path: str,
    payload: dict[str, Any],
    content_key: str,
) -> str:
    """Call a streaming endpoint; accumulate content from each NDJSON chunk and return full text."""
    url = _api_url(path)
    payload = {**payload, "stream": True}
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Thinking: chat has message.thinking, generate has top-level thinking (avoid double-count)
                msg = chunk.get("message") or {}
                if isinstance(msg, dict) and msg.get("thinking"):
                    thinking_parts.append(msg["thinking"])
                elif chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])
                # Content: chat has message.content, generate has response
                if isinstance(msg, dict) and content_key in msg:
                    content_parts.append(msg[content_key] or "")
                elif content_key in chunk:
                    content_parts.append(chunk[content_key] or "")
    content = "".join(content_parts)
    if thinking_parts:
        content = f"[Thinking] {' '.join(thinking_parts).strip()}\n\n{content}"
    return content


# ---- Models ----

@mcp.tool()
async def list_models() -> str:
    """List all installed Ollama models (name, size, modified)."""
    try:
        data = await _request("GET", "tags")
        models = data.get("models") or []
        if not models:
            return "No models installed. Use pull_model to pull a model (e.g. llama3.2)."
        lines = []
        for m in models:
            name = m.get("name", "?")
            size = m.get("size")
            size_mb = f"{size / (1024**2):.0f} MB" if size else "?"
            lines.append(f"- {name} ({size_mb})")
        return "\n".join(lines)
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}. Is Ollama running at {OLLAMA_BASE}?"
    except Exception as e:
        logger.exception("list_models")
        return f"Error: {e}"


@mcp.tool()
async def list_running_models() -> str:
    """List models currently loaded in Ollama (running)."""
    try:
        data = await _request("GET", "ps")
        models = data.get("models") or []
        if not models:
            return "No models currently loaded."
        return "\n".join(f"- {m.get('name', '?')}" for m in models)
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("list_running_models")
        return f"Error: {e}"


@mcp.tool()
async def show_model(model: str) -> str:
    """Get details for an installed model (parameters, family, size, etc.).
    Args:
        model: Model name (e.g. llama3.2, gemma3).
    """
    try:
        data = await _request("POST", "show", json={"name": model})
        return json.dumps(data, indent=2)
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("show_model")
        return f"Error: {e}"


# ---- Chat & Generate ----

@mcp.tool()
async def chat(
    model: str,
    messages: list[dict[str, str]],
    stream: bool = False,
) -> str:
    """Generate the next assistant message for a conversation (multi-turn).
    Args:
        model: Model name (e.g. llama3.2, gemma3).
        messages: List of message objects with 'role' and 'content', e.g. [{"role":"user","content":"Hello"}].
        stream: If true, Ollama streams tokens (we accumulate and return the full reply). If false, Ollama returns one JSON response.
    """
    try:
        if stream:
            return await _request_stream(
                "chat",
                {"model": model, "messages": messages},
                content_key="content",
            )
        payload = {"model": model, "messages": messages, "stream": False}
        data = await _request("POST", "chat", json=payload)
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        if msg.get("thinking"):
            content = f"[Thinking] {msg.get('thinking')}\n\n{content}"
        return content
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("chat")
        return f"Error: {e}"


@mcp.tool()
async def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    stream: bool = False,
) -> str:
    """Generate a completion for a single prompt (no conversation history).
    Args:
        model: Model name (e.g. llama3.2, gemma3).
        prompt: The user prompt text.
        system: Optional system prompt.
        stream: If true, Ollama streams tokens (we accumulate and return the full reply). If false, Ollama returns one JSON response.
    """
    try:
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        if system:
            payload["system"] = system
        if stream:
            return await _request_stream("generate", payload, content_key="response")
        payload["stream"] = False
        data = await _request("POST", "generate", json=payload)
        response = data.get("response") or ""
        if data.get("thinking"):
            response = f"[Thinking] {data.get('thinking')}\n\n{response}"
        return response
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("generate")
        return f"Error: {e}"


# ---- Embeddings ----

@mcp.tool()
async def embed(model: str, text: str | list[str]) -> str:
    """Get embeddings for text. Text can be a string or list of strings.
    Args:
        model: Embedding model name (e.g. nomic-embed-text).
        text: Single string or list of strings to embed.
    """
    try:
        inputs = [text] if isinstance(text, str) else text
        payload = {"model": model, "input": inputs}
        data = await _request("POST", "embed", json=payload)
        embeddings = data.get("embeddings") or []
        return json.dumps({"embeddings": embeddings, "count": len(embeddings)})
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("embed")
        return f"Error: {e}"


# ---- Model lifecycle ----

@mcp.tool()
async def pull_model(name: str, insecure: bool = False) -> str:
    """Pull a model from the registry (e.g. llama3.2, gemma3). May take a while.
    Args:
        name: Model name to pull (e.g. llama3.2, mistral).
        insecure: Allow insecure connections to the registry.
    """
    try:
        payload: dict[str, Any] = {"name": name}
        if insecure:
            payload["insecure"] = True
        data = await _request("POST", "pull", json=payload)
        status = data.get("status", "unknown")
        return f"Pull status: {status}. Check Ollama for progress."
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("pull_model")
        return f"Error: {e}"


@mcp.tool()
async def delete_model(name: str) -> str:
    """Delete an installed model from Ollama.
    Args:
        name: Exact model name to delete.
    """
    try:
        await _request("DELETE", "delete", json={"name": name})
        return f"Deleted model: {name}"
    except httpx.HTTPError as e:
        return f"Ollama request failed: {e}"
    except Exception as e:
        logger.exception("delete_model")
        return f"Error: {e}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
