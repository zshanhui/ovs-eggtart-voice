"""RKLLM inference engine abstraction.

Provides a clean interface over ``librkllmrt.so`` so the chat server never
touches ctypes directly. A DummyEngine is included for testing without NPU
hardware.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class RKLLMConfig:
    """Parameters for one RKLLM inference run."""

    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class RKLLMStats:
    """Collected after a generation run completes."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


# ---------------------------------------------------------------------------
# Abstract engine
# ---------------------------------------------------------------------------


class RKLLMEngine(ABC):
    """Abstract base for RKLLM-backed text generation."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load a ``.rkllm`` model onto the NPU. Must be called once before
        ``generate``."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release NPU resources."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: RKLLMConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, RKLLMStats]:
        """Run inference and return ``(full_text, stats)``.

        If *on_token* is provided it is called for each decoded token as it
        arrives (used for SSE streaming).
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return ``True`` when a model is loaded and ready."""
        ...


# ---------------------------------------------------------------------------
# Dummy engine (for testing without NPU)
# ---------------------------------------------------------------------------

_DUMMY_RESPONSES: dict[str, str] = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "goodbye": "Goodbye! Have a great day!",
    "天气": "今天天气不错，适合出门走走。",
    "你好": "你好！有什么我可以帮忙的吗？",
    "翻译": "Translation: This is a test response from the dummy RKLLM engine.",
}


class DummyRKLLMEngine(RKLLMEngine):
    """Toy engine for testing the chat server without NPU hardware.

    Returns canned responses with simulated streaming delay (~30ms per token).
    """

    def __init__(self) -> None:
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        self._model_path = model_path
        logger.info("DummyRKLLM: loaded %s", model_path)

    def unload(self) -> None:
        self._model_path = None
        logger.info("DummyRKLLM: unloaded")

    def is_ready(self) -> bool:
        return self._model_path is not None

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        config: RKLLMConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, RKLLMStats]:
        if not self._model_path:
            raise RuntimeError("Model not loaded")

        cfg = config or RKLLMConfig()

        # Pick a canned response
        text = "I'm a helpful assistant running on a Rockchip NPU."
        for key, resp in _DUMMY_RESPONSES.items():
            if key in prompt.lower():
                text = resp
                break

        # Simulate token-by-token streaming
        t0 = time.monotonic()
        tokens = list(text)
        for i, ch in enumerate(tokens):
            time.sleep(0.03)
            if on_token:
                on_token(ch)
            if i == 0:
                first_token_elapsed = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - t0) * 1000
        stats = RKLLMStats(
            prompt_tokens=len(prompt) // 2,
            completion_tokens=len(tokens),
            first_token_ms=first_token_elapsed,
            total_time_ms=total_ms,
            tokens_per_second=len(tokens) / (total_ms / 1000) if total_ms > 0 else 0,
        )
        return text, stats


# ---------------------------------------------------------------------------
# Real RKLLM engine — production path via librkllmrt.so
# ---------------------------------------------------------------------------
# Uses the ctypes binding in services.llm.rkllm_binding which was
# reverse-engineered from Rockchip's public rkllm_api_demo at
# airockchip/rknn-llm on GitHub.
#
# The binding handles:
#   - rkllm_createDefaultParam()  → returns RKLLMParam by value
#   - rkllm_init(handle*, param*, callback, userdata)  → int
#   - rkllm_run(handle, input*, infer*, userdata)       → int (BLOCKING)
#   - rkllm_set_chat_template(handle, system, user_pfx, asst_pfx) → int
#   - rkllm_abort(handle) / rkllm_destroy(handle)
#
# rkllm_run is a synchronous (blocking) call.  The callback fires in-thread
# during the call, so the engine MUST run on a background thread.


class RealRKLLMEngine(RKLLMEngine):
    """Production engine that calls ``librkllmrt.so`` via ctypes.

    Requires a compiled ``.rkllm`` chat model (Step 3 — model conversion
    on a CUDA PC with the RKLLM-Toolkit).
    """

    def __init__(self, lib_path: str | None = None) -> None:
        self._lib_path = lib_path
        self._llm: Any = None  # services.llm.rkllm_binding.RKLLM
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        from services.llm.rkllm_binding import RKLLM, RKLLMError

        logger.info("Loading RKLLM model: %s", model_path)
        try:
            self._llm = RKLLM(
                model_path,
                max_tokens=512,
                max_context_len=4096,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.05,
                lib_path=self._lib_path,
            )
        except RKLLMError as exc:
            raise RuntimeError(f"Failed to load RKLLM model: {exc}") from exc
        self._model_path = model_path
        logger.info("RKLLM model loaded.")

    def unload(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
        self._model_path = None

    def is_ready(self) -> bool:
        return self._llm is not None and self._llm.is_loaded

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        config: RKLLMConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, RKLLMStats]:
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        cfg = config or RKLLMConfig()

        tokens: list[str] = []
        t0 = time.monotonic()
        first_token_ms = 0.0

        # rkllm_run is BLOCKING — the Python callback fires in-thread.
        for i, tok in enumerate(self._llm.chat(prompt)):
            tokens.append(tok)
            if i == 0:
                first_token_ms = (time.monotonic() - t0) * 1000
            if on_token:
                on_token(tok)

        total_ms = (time.monotonic() - t0) * 1000
        full_text = "".join(tokens)

        stats = RKLLMStats(
            prompt_tokens=len(prompt) // 2,
            completion_tokens=len(tokens),
            first_token_ms=first_token_ms,
            total_time_ms=total_ms,
            tokens_per_second=len(tokens) / (total_ms / 1000) if total_ms > 0 else 0,
        )
        return full_text, stats
