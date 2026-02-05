#!/usr/bin/env python3
"""
Integration tests: run the MCP server as a subprocess and exercise the protocol.
Requires: project root with uv and server.py (run from repo root).
Optional: Ollama running at OLLAMA_BASE_URL for real tool results; otherwise
list_models returns an error message string (still a valid tool result).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = ROOT / "server.py"
TIMEOUT_SEC = 15


def read_line_with_timeout(pipe, timeout: float) -> str | None:
    """Read a single line from pipe; return None on timeout."""
    result: list[str | None] = [None]

    def read():
        line = pipe.readline()
        result[0] = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0] if t.is_alive() is False else None


def send_request(proc: subprocess.Popen, request: dict, timeout: float = TIMEOUT_SEC) -> dict | None:
    """Send one JSON-RPC request and read one response line."""
    out = proc.stdin
    if out is None:
        return None
    line = (json.dumps(request) + "\n").encode("utf-8")
    out.write(line)
    out.flush()
    raw = read_line_with_timeout(proc.stdout, timeout) if proc.stdout else None
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"message": f"Invalid JSON: {raw[:200]}"}}


def main() -> int:
    if not SERVER_PY.exists():
        print(f"Not found: {SERVER_PY}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

    cmd = [sys.executable, str(SERVER_PY)]
    # Prefer uv run so deps are correct
    uv = os.environ.get("UV", "uv")
    if Path(ROOT / "pyproject.toml").exists():
        cmd = [uv, "run", "--directory", str(ROOT), "server.py"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=False,
    )
    assert proc.stdin and proc.stdout

    errors: list[str] = []

    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "integration-test", "version": "0.1.0"},
        },
    }
    init_resp = send_request(proc, init_req)
    if not init_resp:
        errors.append("initialize: no response (timeout or crash)")
    elif "result" not in init_resp and "error" in init_resp:
        errors.append(f"initialize error: {init_resp['error']}")
    else:
        print("ok initialize")

    # 2. Initialized notification (no id)
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode("utf-8"))
    proc.stdin.flush()

    # 3. tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    list_resp = send_request(proc, list_req)
    if not list_resp:
        errors.append("tools/list: no response")
    elif "error" in list_resp:
        errors.append(f"tools/list error: {list_resp['error']}")
    else:
        tools = list_resp.get("result", {}).get("tools", [])
        names = [t.get("name") for t in tools if t.get("name")]
        expected = {"list_models", "list_running_models", "show_model", "chat", "generate", "embed", "pull_model", "delete_model"}
        missing = expected - set(names)
        if missing:
            errors.append(f"tools/list missing: {missing}")
        else:
            print(f"ok tools/list ({len(names)} tools)")

    # 4. tools/call list_models (safe, no side effects)
    call_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}}
    call_resp = send_request(proc, call_req, timeout=20)
    if not call_resp:
        errors.append("tools/call list_models: no response")
    elif "error" in call_resp:
        errors.append(f"tools/call list_models error: {call_resp['error']}")
    else:
        content = call_resp.get("result", {}).get("content", [])
        text = content[0].get("text", "") if content else ""
        if "Ollama request failed" in text or "No models installed" in text or "llama" in text.lower() or "gemma" in text.lower():
            print("ok tools/call list_models (got result)")
        else:
            print(f"ok tools/call list_models: {text[:80]}...")

    # 5. tools/call generate with stream=false (minimal)
    gen_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "generate", "arguments": {"model": "llama3.2", "prompt": "Say 1", "stream": False}},
    }
    gen_resp = send_request(proc, gen_req, timeout=60)
    if not gen_resp:
        errors.append("tools/call generate: no response (timeout?)")
    elif "error" in gen_resp:
        # Expected if Ollama down or model missing
        print("tools/call generate: server error (Ollama down or model missing is ok)")
    else:
        print("ok tools/call generate")

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("All integration checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
