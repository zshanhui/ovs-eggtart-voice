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
# Real RKLLM engine (skeleton — integrate with librkllmrt.so)
# ---------------------------------------------------------------------------
#
# The RKLLM C runtime provides these functions (from rkllm.h):
#
#   RKLLMHandle  rkllm_init(RKLLMParam *param, RKLLMResultCallback callback, void *userdata)
#   int           rkllm_run(RKLLMHandle handle, RKLLMInput *input, RKLLMRunParam *run_param, void *userdata)
#   int           rkllm_abort(RKLLMHandle handle)
#   int           rkllm_destroy(RKLLMHandle handle)
#
# The callback signature is:
#   void (*callback)(RKLLMResult *result, void *userdata, int state)
#
# Where RKLLMResult contains token text and state indicates:
#   RKLLM_RUN_NORMAL  = 1  (token in progress)
#   RKLLM_RUN_FINISH  = 0  (generation complete)
#   RKLLM_RUN_ERROR   = -1 (error)
#
# To complete this implementation:
#   1. Obtain the RKLLM SDK from Rockchip (rknn-llm v1.2.3+)
#   2. Define ctypes structs matching RKLLMParam, RKLLMInput, RKLLMRunParam
#   3. Set LIB_PATH = "/opt/asr/lib/librkllmrt.so" (or env RKLLM_LIB_PATH)
#   4. Wire the C callback to a Python callable that collects tokens
#   5. For chat models, use RKLLM_INPUT_TEXT (not RKLLM_INPUT_EMBED)
#
# Reference: third_party/qwen3-edgellm-jetson/ (Jetson equivalent pattern)


class RealRKLLMEngine(RKLLMEngine):
    """Production engine that calls ``librkllmrt.so`` via ctypes.

    Currently a skeleton — the ctypes definitions and C-callback wiring
    need to be filled in once the RKLLM SDK headers are available.
    """

    LIB_PATH = "/opt/asr/lib/librkllmrt.so"

    def __init__(self, lib_path: str | None = None) -> None:
        self._lib_path = lib_path or self.LIB_PATH
        self._handle: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        # TODO: call rkllm_init with:
        #   - model_path pointing to the .rkllm file
        #   - RKLLM_INPUT_TEXT input type (chat model, not embed)
        #   - max_context_len, max_new_tokens from config
        #   - NPU core mask (0x7 for all 3 cores on rk3588, 0x3 on rk3576)
        #
        # from ctypes import CDLL, c_char_p, c_void_p, c_int, CFUNCTYPE, Structure, POINTER
        #
        # _lib = CDLL(self._lib_path)
        # _lib.rkllm_init.restype = c_void_p
        # _lib.rkllm_init.argtypes = [POINTER(RKLLMParam), ...]
        #
        # See the comment block above for the full C API reference.
        raise NotImplementedError(
            "RealRKLLMEngine requires the RKLLM SDK headers and ctypes wiring. "
            "Use DummyRKLLMEngine for testing the server logic."
        )

    def unload(self) -> None:
        # TODO: call rkllm_destroy(handle)
        self._handle = None
        self._loaded = False

    def is_ready(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        config: RKLLMConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, RKLLMStats]:
        # TODO:
        #   1. Set up RKLLMRunParam (max_tokens, temperature, top_p, top_k)
        #   2. Set up RKLLMInput with prompt text
        #   3. Register a Python callback that:
        #        - Collects token text into a buffer
        #        - Calls on_token(token) for each token
        #        - On RKLLM_RUN_FINISH, signals completion
        #   4. Call rkllm_run(handle, input, run_param)
        #   5. Return (full_text, stats)
        raise NotImplementedError(
            "RealRKLLMEngine requires the RKLLM SDK headers and ctypes wiring."
        )
