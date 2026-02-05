"""Pytest configuration and shared fixtures for ollama-mcp tests."""
import sys
from pathlib import Path

import pytest

# Ensure project root is on path so "import server" works
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


@pytest.fixture(autouse=True)
def set_ollama_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point OLLAMA_BASE_URL at a dummy host so tests don't hit real Ollama."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:11434")


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for pytest-asyncio."""
    return "asyncio"
