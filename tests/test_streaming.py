"""Unit tests for streaming Ollama responses (_request_stream)."""
from unittest.mock import patch

import pytest

import server


class _StreamResponse:
    """Minimal stream response with aiter_lines and raise_for_status."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self) -> list[str]:
        for line in self._lines:
            yield line


class _StreamCM:
    def __init__(self, resp: _StreamResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> _StreamResponse:
        return self._resp

    async def __aexit__(self, *args: object) -> None:
        pass


class _MockClient:
    def __init__(self, stream_resp: _StreamResponse) -> None:
        self._stream_resp = stream_resp

    def stream(self, *args: object, **kwargs: object) -> _StreamCM:
        return _StreamCM(self._stream_resp)

    async def __aenter__(self) -> "_MockClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_client(stream_lines: list[str]) -> _MockClient:
    return _MockClient(_StreamResponse(stream_lines))


@pytest.mark.asyncio
async def test_request_stream_accumulates_content() -> None:
    """Streaming generate-style NDJSON: each chunk has 'response'."""
    lines = [
        '{"model":"m","response":"Hello","done":false}',
        '{"model":"m","response":" world","done":false}',
        '{"model":"m","response":"!","done":true}',
    ]
    with patch.object(server, "_api_url", return_value="http://test/api/generate"):
        with patch("httpx.AsyncClient", return_value=_make_client(lines)):
            out = await server._request_stream(
                "generate",
                {"model": "m", "prompt": "Hi"},
                content_key="response",
            )
    assert out == "Hello world!"


@pytest.mark.asyncio
async def test_request_stream_chat_message_content() -> None:
    """Streaming chat: each chunk has message.content."""
    lines = [
        '{"model":"m","message":{"content":"Hi","role":"assistant"},"done":false}',
        '{"model":"m","message":{"content":" there","role":"assistant"},"done":true}',
    ]
    with patch.object(server, "_api_url", return_value="http://test/api/chat"):
        with patch("httpx.AsyncClient", return_value=_make_client(lines)):
            out = await server._request_stream(
                "chat",
                {"model": "m", "messages": [{"role": "user", "content": "Say hi"}]},
                content_key="content",
            )
    assert out == "Hi there"


@pytest.mark.asyncio
async def test_request_stream_includes_thinking() -> None:
    """Thinking from generate (top-level) is prepended."""
    lines = [
        '{"model":"m","thinking":"Hmm","response":"","done":false}',
        '{"model":"m","response":"Yes.","done":true}',
    ]
    with patch.object(server, "_api_url", return_value="http://test/api/generate"):
        with patch("httpx.AsyncClient", return_value=_make_client(lines)):
            out = await server._request_stream(
                "generate",
                {"model": "m", "prompt": "?"},
                content_key="response",
            )
    assert "[Thinking]" in out and "Hmm" in out and "Yes." in out
