"""Unit tests for MCP tools with mocked Ollama API (_request / _request_stream)."""
import json
from unittest.mock import AsyncMock, patch

import pytest

# Import after conftest may have set env; we patch _request so no real I/O.
import server


@pytest.mark.asyncio
async def test_ollama_version_success() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"version": "0.12.6"},
    ):
        out = await server.ollama_version()
    assert "0.12.6" in out and "reachable" in out


@pytest.mark.asyncio
async def test_ollama_version_http_error() -> None:
    import httpx
    with patch.object(server, "_request", new_callable=AsyncMock, side_effect=httpx.ConnectError("nope")):
        out = await server.ollama_version()
    assert "Ollama request failed" in out


@pytest.mark.asyncio
async def test_list_models_empty() -> None:
    with patch.object(server, "_request", new_callable=AsyncMock, return_value={"models": []}):
        out = await server.list_models()
    assert "No models installed" in out


@pytest.mark.asyncio
async def test_list_models_success() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={
            "models": [
                {"name": "llama3.2", "size": 100 * 1024 * 1024},
                {"name": "gemma3", "size": None},
            ]
        },
    ):
        out = await server.list_models()
    assert "llama3.2" in out and "100 MB" in out
    assert "gemma3" in out and "?" in out


@pytest.mark.asyncio
async def test_list_models_http_error() -> None:
    import httpx
    with patch.object(server, "_request", new_callable=AsyncMock, side_effect=httpx.ConnectError("nope")):
        out = await server.list_models()
    assert "Ollama request failed" in out


@pytest.mark.asyncio
async def test_list_running_models_empty() -> None:
    with patch.object(server, "_request", new_callable=AsyncMock, return_value={"models": []}):
        out = await server.list_running_models()
    assert "No models currently loaded" in out


@pytest.mark.asyncio
async def test_list_running_models_success() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"models": [{"name": "llama3.2"}, {"name": "gemma3"}]},
    ):
        out = await server.list_running_models()
    assert "llama3.2" in out and "gemma3" in out


@pytest.mark.asyncio
async def test_show_model_success() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"modelfile": "# test", "parameters": {"num_ctx": 4096}},
    ):
        out = await server.show_model("llama3.2")
    data = json.loads(out)
    assert data["parameters"]["num_ctx"] == 4096


@pytest.mark.asyncio
async def test_chat_non_stream() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"message": {"role": "assistant", "content": "Hello back!"}},
    ):
        out = await server.chat("llama3.2", [{"role": "user", "content": "Hello"}], stream=False)
    assert out == "Hello back!"


@pytest.mark.asyncio
async def test_chat_non_stream_with_thinking() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={
            "message": {"role": "assistant", "content": "Yes.", "thinking": "Brief thought."},
        },
    ):
        out = await server.chat("llama3.2", [{"role": "user", "content": "OK?"}], stream=False)
    assert "[Thinking]" in out and "Brief thought." in out and "Yes." in out


@pytest.mark.asyncio
async def test_chat_stream() -> None:
    async def mock_stream(*args: object, **kwargs: object) -> str:
        return "Streamed reply"

    with patch.object(server, "_request_stream", new_callable=AsyncMock, side_effect=mock_stream):
        out = await server.chat("llama3.2", [{"role": "user", "content": "Hi"}], stream=True)
    assert out == "Streamed reply"


@pytest.mark.asyncio
async def test_generate_non_stream() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"response": "Generated text."},
    ):
        out = await server.generate("llama3.2", "Say hi", stream=False)
    assert out == "Generated text."


@pytest.mark.asyncio
async def test_generate_with_system() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"response": "OK."},
    ) as m:
        await server.generate("llama3.2", "Go", system="You are helpful.", stream=False)
    call = m.call_args
    assert call.kwargs.get("json", {}).get("system") == "You are helpful."


@pytest.mark.asyncio
async def test_generate_stream() -> None:
    async def mock_stream(*args: object, **kwargs: object) -> str:
        return "Streamed generate"

    with patch.object(server, "_request_stream", new_callable=AsyncMock, side_effect=mock_stream):
        out = await server.generate("llama3.2", "Hi", stream=True)
    assert out == "Streamed generate"


@pytest.mark.asyncio
async def test_embed_single_string() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"embeddings": [[0.1, 0.2]]},
    ):
        out = await server.embed("nomic-embed-text", "hello")
    data = json.loads(out)
    assert data["count"] == 1
    assert data["embeddings"] == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_embed_list_of_strings() -> None:
    with patch.object(
        server,
        "_request",
        new_callable=AsyncMock,
        return_value={"embeddings": [[0.1], [0.2]]},
    ) as m:
        await server.embed("nomic-embed-text", ["a", "b"])
    call = m.call_args
    assert call.kwargs.get("json", {}).get("input") == ["a", "b"]


@pytest.mark.asyncio
async def test_copy_model() -> None:
    with patch.object(server, "_request", new_callable=AsyncMock, return_value={}):
        out = await server.copy_model("llama3.2", "llama3.2-backup")
    assert "Copied" in out and "llama3.2" in out and "llama3.2-backup" in out


@pytest.mark.asyncio
async def test_pull_model() -> None:
    with patch.object(
        server,
        "_request_pull_stream",
        new_callable=AsyncMock,
        return_value={"status": "success"},
    ):
        out = await server.pull_model("llama3.2")
    assert "Pull finished" in out and "success" in out


@pytest.mark.asyncio
async def test_pull_model_insecure() -> None:
    with patch.object(
        server, "_request_pull_stream", new_callable=AsyncMock, return_value={"status": "ok"}
    ) as m:
        await server.pull_model("x", insecure=True)
    assert m.call_args.args[1].get("insecure") is True


@pytest.mark.asyncio
async def test_delete_model() -> None:
    with patch.object(server, "_request", new_callable=AsyncMock, return_value={}):
        out = await server.delete_model("old-model")
    assert "Deleted model: old-model" in out
