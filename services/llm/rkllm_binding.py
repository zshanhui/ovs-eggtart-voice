"""RKLLM C API binding via ctypes — production-ready wrapper for ``librkllmrt.so``.

API source: Rockchip rknn-llm SDK, reverse-engineered from the official
``rkllm_api_demo/deploy/src/llm_demo.cpp`` at airockchip/rknn-llm on GitHub.

Struct layouts and function signatures match the public SDK headers.  This
file replaces the skeleton in ``rkllm_engine.py`` and is used by
``RealRKLLMEngine`` (which will be updated to import from here).
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
    cast,
    memset,
    pointer,
)
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from rkllm.h)
# ---------------------------------------------------------------------------

# LLMCallState (callback state)
RKLLM_RUN_NORMAL = 1
RKLLM_RUN_FINISH = 0
RKLLM_RUN_ERROR = -1

# Input types
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_EMBED = 1
RKLLM_INPUT_TOKEN = 2

# Inference modes
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LOGITS = 1

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


class _ExtendParam(Structure):
    _fields_ = [
        ("base_domain_id", c_int),
        ("embed_flash", c_int),
    ]


class RKLLMParam(Structure):
    """Model init parameters.  Returned BY VALUE by ``rkllm_createDefaultParam``."""

    _fields_ = [
        ("model_path", c_char_p),
        ("top_k", c_int),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("max_new_tokens", c_int),
        ("max_context_len", c_int),
        ("skip_special_token", c_bool),
        ("extend_param", _ExtendParam),
        # Reserve padding for future SDK additions — the struct may grow.
    ]


class RKLLMInput(Structure):
    """Per-request input.  Zero-initialise before filling."""

    _fields_ = [
        ("input_type", c_int),
        ("role", c_char_p),
        ("prompt_input", c_char_p),
    ]


class _LastHiddenLayer(Structure):
    _fields_ = [
        ("embd_size", c_size_t),
        ("num_tokens", c_size_t),
        ("hidden_states", POINTER(c_float)),
    ]


class RKLLMResult(Structure):
    """Delivered to the callback for each token (or on finish/error)."""

    _fields_ = [
        ("text", c_char_p),
        ("token_id", c_int),
        ("last_hidden_layer", _LastHiddenLayer),
    ]
    # NOTE: `text` is valid only for the duration of the callback invocation.
    # The SDK frees it on the next callback.  Copy it immediately.


class RKLLMLoraAdapter(Structure):
    _fields_ = [
        ("lora_adapter_path", c_char_p),
        ("lora_adapter_name", c_char_p),
        ("scale", c_float),
    ]


class RKLLMLoraParam(Structure):
    _fields_ = [
        ("lora_adapter_name", c_char_p),
    ]


class RKLLMPromptCacheParam(Structure):
    _fields_ = [
        ("save_prompt_cache", c_bool),
        ("prompt_cache_path", c_char_p),
    ]


class RKLLMInferParam(Structure):
    """Per-request inference parameters.  Zero-initialise before filling."""

    _fields_ = [
        ("mode", c_int),
        ("keep_history", c_int),  # 0 = single-turn, 1 = multi-turn
        ("lora_params", POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", POINTER(RKLLMPromptCacheParam)),
    ]


# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------

# Signature: int callback(RKLLMResult *result, void *userdata, LLMCallState state)
RKLMLCallback = CFUNCTYPE(c_int, POINTER(RKLLMResult), c_void_p, c_int)

# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

_DEFAULT_LIB = "/opt/asr/lib/librkllmrt.so"


def _load_lib(path: str | None = None) -> ctypes.CDLL:
    lib_path = path or os.environ.get("RKLLM_LIB_PATH", _DEFAULT_LIB)
    lib = ctypes.CDLL(lib_path)

    # ── rkllm_createDefaultParam ──────────────────────────────────────
    # Returns RKLLMParam BY VALUE.  ctypes can handle this if we set the
    # correct restype — the caller receives a Python RKLLMParam instance.
    lib.rkllm_createDefaultParam.restype = RKLLMParam

    # ── rkllm_init ────────────────────────────────────────────────────
    # int rkllm_init(LLMHandle *handle, RKLLMParam *param, callback, void *userdata)
    lib.rkllm_init.argtypes = [
        POINTER(c_void_p),         # handle (output)
        POINTER(RKLLMParam),       # param
        RKLMLCallback,             # callback
        c_void_p,                  # userdata
    ]
    lib.rkllm_init.restype = c_int

    # ── rkllm_run ─────────────────────────────────────────────────────
    # int rkllm_run(LLMHandle handle, RKLLMInput *input,
    #               RKLLMInferParam *infer, void *userdata)
    lib.rkllm_run.argtypes = [
        c_void_p,                  # handle
        POINTER(RKLLMInput),       # input
        POINTER(RKLLMInferParam),  # infer_params
        c_void_p,                  # userdata
    ]
    lib.rkllm_run.restype = c_int

    # ── rkllm_abort ──────────────────────────────────────────────────
    lib.rkllm_abort.argtypes = [c_void_p]
    lib.rkllm_abort.restype = c_int

    # ── rkllm_destroy ────────────────────────────────────────────────
    lib.rkllm_destroy.argtypes = [c_void_p]
    lib.rkllm_destroy.restype = c_int

    # ── rkllm_set_chat_template ──────────────────────────────────────
    # int rkllm_set_chat_template(LLMHandle handle, char *system_prompt,
    #                             char *user_prefix, char *assistant_prefix)
    lib.rkllm_set_chat_template.argtypes = [
        c_void_p, c_char_p, c_char_p, c_char_p,
    ]
    lib.rkllm_set_chat_template.restype = c_int

    # ── rkllm_clear_kv_cache ─────────────────────────────────────────
    # int rkllm_clear_kv_cache(LLMHandle handle, int flag,
    #                          void *param1, void *param2)
    lib.rkllm_clear_kv_cache.argtypes = [
        c_void_p, c_int, c_void_p, c_void_p,
    ]
    lib.rkllm_clear_kv_cache.restype = c_int

    # ── Optional: lora / prompt cache ─────────────────────────────────
    try:
        lib.rkllm_load_lora.argtypes = [c_void_p, POINTER(RKLLMLoraAdapter)]
        lib.rkllm_load_lora.restype = c_int

        lib.rkllm_load_prompt_cache.argtypes = [c_void_p, c_char_p]
        lib.rkllm_load_prompt_cache.restype = c_int
    except AttributeError:
        pass  # Older SDK releases ship without these.

    return lib


# ---------------------------------------------------------------------------
# High-level Python wrapper
# ---------------------------------------------------------------------------


class RKLLMError(Exception):
    """Raised when an RKLLM C API call returns non-zero."""


class RKLLM:
    """A single RKLLM chat model loaded on the NPU.

    Usage::

        llm = RKLLM("/opt/models/chat.rkllm")
        for token in llm.chat("Hello"):
            print(token, end="", flush=True)
        llm.close()
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_tokens: int = 512,
        max_context_len: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repeat_penalty: float = 1.05,
        lib_path: str | None = None,
    ) -> None:
        self._lib = _load_lib(lib_path)
        self._handle: c_void_p | None = None
        self._tokens: list[str] = []
        self._error: str | None = None
        self._finished: bool = False
        self._callback_ref = RKLMLCallback(self._on_result)  # prevent GC

        # ── Build RKLLMParam ──────────────────────────────────────────
        param = self._lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode("utf-8")
        param.max_new_tokens = max_tokens
        param.max_context_len = max_context_len
        param.temperature = temperature
        param.top_p = top_p
        param.top_k = top_k
        param.repeat_penalty = repeat_penalty

        # ── Init ──────────────────────────────────────────────────────
        handle_ptr = c_void_p()
        ret = self._lib.rkllm_init(
            pointer(handle_ptr),
            pointer(param),
            self._callback_ref,
            None,
        )
        if ret != 0:
            raise RKLLMError(f"rkllm_init failed with code {ret}")
        self._handle = handle_ptr

        # ── Set chat template ─────────────────────────────────────────
        self._set_default_template()

        logger.info(
            "RKLLM loaded: model=%s max_tokens=%d ctx=%d",
            model_path, max_tokens, max_context_len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        keep_history: bool = False,
    ):
        """Stream tokens from a single turn of chat.  Call ``close()`` when done.

        Yields each decoded token as a Python ``str``.  Blocks the calling
        thread — run in a background thread for use with async code.
        """
        self._tokens = []
        self._error = None
        self._finished = False

        # ── Build input ───────────────────────────────────────────────
        inp = RKLLMInput()
        memset(pointer(inp), 0, ctypes.sizeof(RKLLMInput))
        inp.input_type = RKLLM_INPUT_PROMPT
        inp.prompt_input = prompt.encode("utf-8")

        # ── Build infer params ────────────────────────────────────────
        infer = RKLLMInferParam()
        memset(pointer(infer), 0, ctypes.sizeof(RKLLMInferParam))
        infer.mode = RKLLM_INFER_GENERATE
        infer.keep_history = 1 if keep_history else 0

        # ── Run (BLOCKING — the callback fires from within this call) ─
        ret = self._lib.rkllm_run(
            self._handle,
            pointer(inp),
            pointer(infer),
            None,
        )
        if ret != 0:
            raise RKLLMError(f"rkllm_run failed with code {ret}")
        if self._error:
            raise RKLLMError(self._error)

        # Yield collected tokens (the callback stored them in self._tokens).
        for tok in self._tokens:
            yield tok

    def abort(self) -> None:
        """Interrupt an in-progress ``rkllm_run`` call."""
        if self._handle is not None:
            self._lib.rkllm_abort(self._handle)

    def clear_history(self) -> None:
        """Reset KV cache (drop conversation history)."""
        if self._handle is not None:
            self._lib.rkllm_clear_kv_cache(self._handle, 1, None, None)

    def close(self) -> None:
        """Unload model and release NPU resources."""
        if self._handle is not None:
            self._lib.rkllm_destroy(self._handle)
            self._handle = None
            logger.info("RKLLM unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._handle is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_default_template(self) -> None:
        """Apply a minimal ChatML template so the model understands turns."""
        if self._handle is None:
            return
        system = b"You are a helpful assistant."
        user_prefix = b"<|im_start|>user\n"
        assistant_prefix = b"<|im_start|>assistant\n"
        self._lib.rkllm_set_chat_template(
            self._handle, system, user_prefix, assistant_prefix,
        )

    def _on_result(
        self,
        result_ptr: POINTER(RKLLMResult),
        _userdata: c_void_p,
        state: c_int,
    ) -> c_int:
        """C callback — fires once per token (or on finish / error)."""
        result = result_ptr.contents

        if state == RKLLM_RUN_ERROR:
            err_text = result.text.decode("utf-8", errors="replace") if result.text else "unknown"
            self._error = err_text
            return 0

        if state == RKLLM_RUN_FINISH:
            self._finished = True
            return 0

        # RKLLM_RUN_NORMAL: a token arrived.
        if result.text:
            tok = result.text.decode("utf-8", errors="replace")
            self._tokens.append(tok)

        return 0
