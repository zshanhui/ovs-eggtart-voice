"""RKLLM chat server — OpenAI-compatible ``/v1/chat/completions`` endpoint.

Launch::

    uvicorn services.llm.server:app --host 0.0.0.0 --port 8001

Env vars::

    RKLLM_MODEL_PATH         path to .rkllm chat model (required)
    RKLLM_ENGINE             "real" or "dummy" (default: "dummy")
    RKLLM_LIB_PATH           override librkllmrt.so location
    RKLLM_MAX_TOKENS         default max_tokens (default: 512)
    RKLLM_MAX_CONTEXT_TOKENS hard cap on prompt tokens (default: 4096)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .chat_template import apply_chat_template, estimate_tokens
from .rkllm_engine import DummyRKLLMEngine, RKLLMConfig, RKLLMEngine, RealRKLLMEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_MODEL_PATH = os.environ.get("RKLLM_MODEL_PATH", "")
_ENGINE_KIND = os.environ.get("RKLLM_ENGINE", "dummy").lower()
_MAX_TOKENS = int(os.environ.get("RKLLM_MAX_TOKENS", "512"))
_MAX_CONTEXT = int(os.environ.get("RKLLM_MAX_CONTEXT_TOKENS", "4096"))
_SYSTEM_PROMPT = os.environ.get("RKLLM_SYSTEM_PROMPT", "You are a helpful assistant.")

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine: RKLLMEngine | None = None


def _create_engine() -> RKLLMEngine:
    if _ENGINE_KIND == "real":
        return RealRKLLMEngine()
    return DummyRKLLMEngine()


def get_engine() -> RKLLMEngine:
    """Return the singleton engine (must be called after startup)."""
    if _engine is None or not _engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    logger.info("RKLLM chat server starting (engine=%s)", _ENGINE_KIND)
    _engine = _create_engine()
    model_path = _MODEL_PATH or "dummy.rkllm"
    if not _MODEL_PATH:
        logger.warning("RKLLM_MODEL_PATH not set — loading dummy engine")
    logger.info("Loading model: %s", model_path)
    _engine.load(model_path)
    logger.info("Model loaded.")
    yield
    if _engine is not None:
        _engine.unload()
        logger.info("Model unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RKLLM Chat Server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "rkllm"
    messages: list[Message]
    max_tokens: int = _MAX_TOKENS
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness check."""
    ready = _engine is not None and _engine.is_ready()
    return {"status": "ok" if ready else "loading", "model_loaded": ready}


@app.get("/v1/models")
async def list_models():
    """Model list (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rkllm",
                "object": "model",
                "owned_by": "rockchip",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    engine: RKLLMEngine
    try:
        engine = get_engine()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Engine error")

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    messages_dicts: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    prompt = apply_chat_template(
        messages_dicts,
        add_generation_prompt=True,
        system_prompt=_SYSTEM_PROMPT,
    )
    token_est = estimate_tokens(prompt)
    if token_est > _MAX_CONTEXT:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long: ~{token_est} tokens (max {_MAX_CONTEXT})",
        )

    logger.info(
        "chat: model=%r messages=%d prompt_est=%d stream=%s",
        req.model,
        len(req.messages),
        token_est,
        req.stream,
    )

    # ------------------------------------------------------------------
    # RKLLM inference config
    # ------------------------------------------------------------------
    llm_config = RKLLMConfig(
        model_path=_MODEL_PATH,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name = req.model or "rkllm"

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    if not req.stream:
        text, stats = engine.generate(prompt, llm_config)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.prompt_tokens + stats.completion_tokens,
            },
            "cache_metrics": None,
        }

    # ------------------------------------------------------------------
    # Streaming (SSE)
    # ------------------------------------------------------------------
    async def _stream() -> AsyncGenerator[str, None]:
        tokens: list[str] = []
        pending_text: list[str] = []
        start_time = time.monotonic()

        def _on_token(token: str) -> None:
            tokens.append(token)
            pending_text.append(token)

        # Run RLKLM inference in a thread so we don't block the async loop.
        import asyncio

        loop = asyncio.get_running_loop()

        def _run_inference() -> tuple[str, Any]:
            return engine.generate(prompt, llm_config, on_token=_on_token)

        # Start inference in a background thread.
        future = loop.run_in_executor(None, _run_inference)

        # Stream tokens as they arrive (polling the pending_text buffer).
        # In the real RKLLM engine the callback fires from a C thread; we
        # poll at ~20 Hz which adds < 50 ms latency jitter.
        done = False
        idx = 0
        sent_role = False
        while not done:
            # Drain pending tokens
            while idx < len(tokens):
                tok = tokens[idx]
                idx += 1
                chunk: dict[str, Any] = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }
                    ],
                }
                if not sent_role:
                    chunk["choices"][0]["delta"]["role"] = "assistant"
                    sent_role = True
                chunk["choices"][0]["delta"]["content"] = tok
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Check if inference completed
            if future.done():
                done = True
                try:
                    _full_text, _stats = future.result()
                except Exception:
                    # Emit error and bail.
                    err_chunk: dict[str, Any] = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "error",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    return
            else:
                await asyncio.sleep(0.05)  # 20 Hz poll

        # Drain any stragglers
        while idx < len(tokens):
            tok = tokens[idx]
            idx += 1
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": tok},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": token_est,
                "completion_tokens": len(tokens),
                "total_tokens": token_est + len(tokens),
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        elapsed = time.monotonic() - start_time
        logger.info(
            "stream done: tokens=%d elapsed=%.0fms tok/s=%.1f",
            len(tokens),
            elapsed * 1000,
            len(tokens) / elapsed if elapsed > 0 else 0,
        )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Admin (reload model without restarting container)
# ---------------------------------------------------------------------------


@app.post("/admin/reload")
async def admin_reload(model_path: str | None = None):
    """Hot-reload the RKLLM model."""
    global _engine
    path = model_path or _MODEL_PATH
    if not path:
        raise HTTPException(status_code=400, detail="No model path provided")

    if _engine is not None:
        _engine.unload()

    _engine = _create_engine()
    _engine.load(path)
    return {"status": "ok", "model_path": path}
