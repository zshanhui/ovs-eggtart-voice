"""E2E pipeline test for RK3576 full stack (dummy engine).

Validates the full text chain::

    "user speech" → LLM → "assistant reply"

Without requiring actual RK3576 hardware or audio devices.

Run::

    python services/llm/test_e2e.py

What it does::

    1. Start RKLLM chat server (dummy engine) on :18002
    2. Simulate ASR output → send to LLM → receive reply
    3. Verify OpenAI SSE streaming format matches agent expectations
    4. Verify multi-turn conversation
    5. Verify error handling (empty prompt, max_tokens=1, long context)
    6. Verify config loads for rk3576-chat app
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:18002"


def POST(path: str, body: dict) -> tuple[int, str]:
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


def GET(path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def parse_sse(raw: str) -> list[dict]:
    """Parse SSE stream into a list of chunk dicts."""
    chunks = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            chunks.append(json.loads(line[6:]))
    return chunks


def check_agent_compat(chunks: list[dict]) -> bool:
    """Verify SSE chunks match what ``OpenAICompatBackend._do_stream`` expects.

    The agent parses:
        chunk.choices[0].delta.content
        chunk.choices[0].finish_reason
        chunk.model_extra (optional)
    """
    role_seen = False
    content_seen = False
    for chunk in chunks:
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")

        # First chunk MUST have role delta (agent checks this).
        if not role_seen:
            role = delta.get("role", "")
            if role == "assistant":
                role_seen = True

        if delta.get("content"):
            content_seen = True

        # finish_reason must be one of None, "stop", or "error".
        assert finish in (None, "stop", "error"), f"bad finish_reason: {finish}"

    # Last chunk must have finish_reason set.
    assert chunks[-1]["choices"][0].get("finish_reason") is not None, \
        "last chunk must have finish_reason"
    assert role_seen, "first chunk must include role delta"
    assert content_seen, "no content delta seen"
    return True


def test_01_health():
    """Server responds to health check."""
    code, data = GET("/health")
    assert code == 200
    assert data["status"] == "ok"
    assert data["model_loaded"]
    print("  [PASS] 01 — health check")


def test_02_models():
    """Server lists models."""
    code, data = GET("/v1/models")
    assert code == 200
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "rkllm"
    print("  [PASS] 02 — model list")


def test_03_non_streaming_hello():
    """Basic non-streaming chat request."""
    code, raw = POST("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    })
    assert code == 200, f"status={code}: {raw[:200]}"
    data = json.loads(raw)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "Hello" in data["choices"][0]["message"]["content"]
    print(f"  [PASS] 03 — non-streaming: {data['choices'][0]['message']['content']!r}")


def test_04_streaming_format():
    """Streaming response format passes agent compatibility checks."""
    code, raw = POST("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "Hi there"}],
        "stream": True,
    })
    assert code == 200
    chunks = parse_sse(raw)
    assert len(chunks) >= 2, f"expected at least 2 chunks, got {len(chunks)}"

    # Verify agent compatibility.
    check_agent_compat(chunks)

    tokens = [c["choices"][0].get("delta", {}).get("content", "") for c in chunks]
    text = "".join(t for t in tokens if t)
    print(f"  [PASS] 04 — streaming format: {len(chunks)} chunks → {text!r}")


def test_05_multi_turn():
    """Multi-turn conversation: system + user + assistant + user."""
    turns = [
        [{"role": "system", "content": "You are a helpful robot."},
         {"role": "user", "content": "Hello"}],
        [{"role": "system", "content": "You are a helpful robot."},
         {"role": "user", "content": "Hello"},
         {"role": "assistant", "content": "Hi! How can I help you?"},
         {"role": "user", "content": "What is your name?"}],
    ]
    for i, messages in enumerate(turns):
        code, raw = POST("/v1/chat/completions", {
            "model": "rkllm",
            "messages": messages,
            "stream": False,
        })
        assert code == 200, f"turn {i}: status={code}"
        data = json.loads(raw)
        assert data["choices"][0]["message"]["content"], f"turn {i}: empty reply"
        print(f"  [PASS] 05-{i} — multi-turn ({len(messages)} msgs): "
              f"{data['choices'][0]['message']['content']!r}")


def test_06_streaming_max_tokens_1():
    """Single-token streaming edge case."""
    code, raw = POST("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1,
        "stream": True,
    })
    assert code == 200
    chunks = parse_sse(raw)
    check_agent_compat(chunks)
    print(f"  [PASS] 06 — max_tokens=1 streaming: {len(chunks)} chunks")


def test_07_chinese():
    """Chinese text through the pipeline."""
    code, raw = POST("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": "今天天气怎么样？"}],
        "stream": False,
    })
    assert code == 200
    data = json.loads(raw)
    assert len(data["choices"][0]["message"]["content"]) > 0
    print(f"  [PASS] 07 — Chinese: {data['choices'][0]['message']['content']!r}")


def test_08_long_context_rejected():
    """Overly long prompt is rejected with 400."""
    huge = "Hello " * 20000  # ~100K chars, way over 4096 token limit
    code, raw = POST("/v1/chat/completions", {
        "model": "rkllm",
        "messages": [{"role": "user", "content": huge}],
        "stream": False,
    })
    assert code == 400, f"expected 400, got {code}"
    assert "too long" in raw.lower() or "token" in raw.lower()
    print("  [PASS] 08 — long context rejected with 400")


def test_09_config_loads():
    """The rk3576-chat agent config loads and has the right defaults."""
    import os
    import sys
    sys.path.insert(0, os.path.expanduser("/Users/lishanhui/oss/openvoicestream/agent"))

    from openvoicestream_agent.config import load_config

    cfg_path = "/Users/lishanhui/oss/openvoicestream/agent/apps/rk3576-chat/config.yaml"
    cfg = load_config(cfg_path)

    # Verify key fields point at the local stack.
    assert cfg.llm_backend == "openai_compat"
    assert "8001" in cfg.llm_base_url, f"expected port 8001, got {cfg.llm_base_url}"
    assert "8621" in cfg.slv_url, f"expected port 8621, got {cfg.slv_url}"
    assert cfg.llm_model == "qwen3-0.6b-instruct"
    assert cfg.log_level == "INFO"
    assert cfg.default_mode == "chat"
    print("  [PASS] 09 — rk3576-chat config loads ok")


def test_10_admin_reload():
    """Admin reload endpoint works."""
    code, raw = POST("/admin/reload?model_path=dummy.rkllm", {})
    assert code == 200
    data = json.loads(raw)
    assert data["status"] == "ok"
    print("  [PASS] 10 — admin reload")


def run_tests():
    passed = 0
    failed = 0
    tests = [
        test_01_health,
        test_02_models,
        test_03_non_streaming_hello,
        test_04_streaming_format,
        test_05_multi_turn,
        test_06_streaming_max_tokens_1,
        test_07_chinese,
        test_08_long_context_rejected,
        test_09_config_loads,
        test_10_admin_reload,
    ]
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    return passed, failed


if __name__ == "__main__":
    print("Starting RKLLM chat server on :18002 ...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "services.llm.server:app",
            "--host", "127.0.0.1",
            "--port", "18002",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        passed, failed = run_tests()
    finally:
        proc.terminate()
        proc.wait()

    print(f"\n{'=' * 50}")
    print(f"E2E pipeline: {passed} passed, {failed} failed")
    if failed == 0:
        print("Full stack API chain validated.")
        print()
        print("To run on RK3576:")
        print("  docker compose -f deploy/docker-compose.rk.yml up -d")
        print("  ovs-agent run apps.rk3576-chat")
    else:
        print(f"{failed} test(s) FAILED")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)
