"""Quick smoke-test for the RKLLM chat server (dummy engine).

Run::

    python services/llm/test_server.py

Uses only stdlib (urllib, no pip dependencies).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request


BASE = "http://127.0.0.1:18001"


def _post(path: str, body: dict) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _get(path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_health():
    code, data = _get("/health")
    assert code == 200, f"health: {code}"
    assert data["model_loaded"], f"model not loaded: {data}"
    print("  [OK] GET /health")


def test_models():
    code, data = _get("/v1/models")
    assert code == 200
    assert len(data["data"]) >= 1
    print("  [OK] GET /v1/models")


def test_non_streaming():
    code, raw = _post("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "stream": False,
    })
    assert code == 200, f"non-stream: {code} body={raw[:200]}"
    data = json.loads(raw)
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert len(choice["message"]["content"]) > 0
    print(f"  [OK] POST /v1/chat/completions (non-streaming): {choice['message']['content']!r}")


def test_streaming():
    code, raw = _post("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "stream": True,
    })
    assert code == 200, f"stream: {code}"
    tokens = []
    finish_reason = None
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {})
            tok = delta.get("content", "")
            if tok:
                tokens.append(tok)
            fr = chunk["choices"][0].get("finish_reason")
            if fr:
                finish_reason = fr
    assert finish_reason == "stop", f"finish_reason={finish_reason}"
    assert len(tokens) > 0
    print(f"  [OK] POST /v1/chat/completions (streaming): {len(tokens)} tokens -> {''.join(tokens)!r}")


if __name__ == "__main__":
    print("Starting server...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "services.llm.server:app",
            "--host", "127.0.0.1",
            "--port", "18001",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    try:
        test_health()
        test_models()
        test_non_streaming()
        test_streaming()
        print("\nAll tests passed.")
    finally:
        proc.terminate()
        proc.wait()
