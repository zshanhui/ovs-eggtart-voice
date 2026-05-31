"""Kokoro TTS backend for Jetson TensorRT.

Supports: BASIC_TTS, STREAMING (chunked PCM from synthesized audio),
MULTI_SPEAKER.

The hot path keeps text normalization/tokenization and voice lookup in Python,
then runs the Kokoro acoustic model with either a prebuilt TensorRT engine or
the CPU ONNX Runtime fallback. We deliberately do not depend on ORT GPU /
TensorRT Execution Provider here; the production acceleration path should be a
hand-written TensorRT+CPU hybrid, not provider-level graph partitioning.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import re
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.backends.jetson.matcha_trt import CudaMemoryPool, _read_arena_size_bytes


@dataclass
class _DeviceTensor:
    """Handle to a tensor that lives in device memory between TRT stages.

    Lifetime is bounded by the per-synthesize CudaMemoryPool — these tensors
    are valid until the synthesize call's outer try/finally calls
    slot.reset_per_request() (which calls pool.free_all()). Never escape this
    scope; pass by value within one _synthesize_one() invocation only.
    """
    ptr: int
    shape: tuple[int, ...]
    dtype: type
    nbytes: int


@dataclass(frozen=True)
class _OrtIoNames:
    """Cached input/output name set for an ORT InferenceSession.

    Built once at session load time so the per-call _run_cpu_onnx() helper
    can skip re-querying session metadata on every request. Immutable
    (frozenset + tuple), safe to share across slots / threads.
    """
    input_names: frozenset
    output_names: tuple


@dataclass(frozen=True)
class _TrtOutputMeta:
    """Cached output tensor name + numpy dtype for one TRT engine output."""
    name: str
    dtype: object


@dataclass(frozen=True)
class _TrtEngineMeta:
    """Cached output metadata for a TRT engine.

    Only names + dtypes are cached; output shapes still must be queried per
    call via ``ctx.get_tensor_shape(name)`` because they are dynamic.
    Immutable (tuple), safe to share across slots / threads.
    """
    outputs: tuple
from app.core.tts_backend import TTSBackend, TTSCapability
from app.core.tts_speakers import resolve_speaker_kwargs

logger = logging.getLogger(__name__)

# Paths — captured at module import for back-compat. Instance code reads
# self._<path> attrs set in __init__ via _resolve_kokoro_paths(), so a backend
# built after profile hot-reload sees the new artifact locations even though
# these module constants stay frozen at first import.
_MODEL_BASE = os.environ.get("KOKORO_MODEL_BASE", "/opt/models/kokoro-multi-lang-v1_0")
_MODEL_ONNX = os.environ.get("KOKORO_ONNX", os.path.join(_MODEL_BASE, "model.onnx"))
_ENGINE_PATH = os.environ.get(
    "KOKORO_TRT_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_fp16.engine"),
)
_HYBRID_DIR = os.environ.get("KOKORO_HYBRID_DIR", os.path.join(_MODEL_BASE, "hybrid"))
_HYBRID_PREFIX_ENGINE_ENV = os.environ.get("KOKORO_HYBRID_PREFIX_ENGINE")
_HYBRID_PREFIX_ENGINE_DYN = os.path.join(_HYBRID_DIR, "kokoro_prefix_encoder_dyn4_128_fp16.engine")
_HYBRID_PREFIX_ENGINE_FIXED = os.path.join(_HYBRID_DIR, "kokoro_prefix_encoder_s96_fp16.engine")
_HYBRID_SUFFIX_ONNX = os.environ.get(
    "KOKORO_HYBRID_SUFFIX_ONNX",
    os.path.join(_HYBRID_DIR, "kokoro_suffix_encoder.onnx"),
)
_SPLIT_ENCODER_ENGINE = os.environ.get(
    "KOKORO_SPLIT_ENCODER_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_prefix_encoder_dyn4_128_fp16.engine"),
)
_SPLIT_LENGTH_ONNX = os.environ.get(
    "KOKORO_SPLIT_LENGTH_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_length_regulator.onnx"),
)
_SPLIT_DECODER_ENGINE = os.environ.get(
    "KOKORO_SPLIT_DECODER_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_decoder_backbone_dyn64_256_fp16.engine"),
)
_SPLIT_DECODER_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_DECODER_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_decoder_backbone_dyn256_512_fp16.engine"),
)
_SPLIT_SOURCE_ENGINE = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_source_dyn128_512_bf16.engine"),
)
_SPLIT_SOURCE_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_source_dyn512_1024_bf16.engine"),
)
_SPLIT_SOURCE_ONNX = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_generator_source.onnx"),
)
_SPLIT_GENERATOR_ENGINE = os.environ.get(
    "KOKORO_SPLIT_GENERATOR_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_rest_preexp_dyn64_256_fp16.engine"),
)
_SPLIT_GENERATOR_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_GENERATOR_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_rest_preexp_dyn256_512_fp16.engine"),
)
_SPLIT_ISTFT_ONNX = os.environ.get(
    "KOKORO_SPLIT_ISTFT_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_postspec_istft.onnx"),
)
_VOICES_BIN = os.environ.get("KOKORO_VOICES", os.path.join(_MODEL_BASE, "voices.bin"))
_TOKENS_PATH = os.environ.get("KOKORO_TOKENS", os.path.join(_MODEL_BASE, "tokens.txt"))

SAMPLE_RATE = 24000
MAX_TOKENS = int(os.environ.get("KOKORO_MAX_TOKENS", "510"))
DEFAULT_SPEAKER_ID = int(os.environ.get("KOKORO_DEFAULT_SID", os.environ.get("TTS_DEFAULT_SID", "52")))
DEFAULT_SPEED = float(os.environ.get("TTS_DEFAULT_SPEED", "1.0"))
VOICE_STYLES = 510
STYLE_DIM = 256
STYLE_BYTES = VOICE_STYLES * STYLE_DIM * 4
STREAM_SEGMENT_TOKENS = int(os.environ.get("KOKORO_STREAM_MAX_SEGMENT_TOKENS", "64"))
STREAM_SEGMENT_TEXT = os.environ.get("KOKORO_STREAM_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")
SYNTH_SEGMENT_TEXT = os.environ.get("KOKORO_SYNTH_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")


def _hybrid_prefix_engine_path(paths: dict[str, str] | None = None) -> str:
    """Resolve the hybrid prefix engine path.

    If a ``paths`` dict (from :func:`_resolve_kokoro_paths`) is supplied, use
    its values so the resolution matches what the backend instance captured
    at __init__. Falling back to module-level state preserves cold-boot
    behaviour for any external caller that still imports this helper.
    """
    if paths is not None:
        env_explicit = paths["hybrid_prefix_engine_env"]
        dyn = paths["hybrid_prefix_engine_dyn"]
        fixed = paths["hybrid_prefix_engine_fixed"]
    else:
        env_explicit = _HYBRID_PREFIX_ENGINE_ENV
        dyn = _HYBRID_PREFIX_ENGINE_DYN
        fixed = _HYBRID_PREFIX_ENGINE_FIXED
    if env_explicit:
        return env_explicit
    if os.path.exists(dyn):
        return dyn
    return fixed


def _resolve_kokoro_paths() -> dict[str, str | None]:
    """Resolve all Kokoro artifact paths from the *current* os.environ.

    Called from KokoroTRTBackend.__init__ on each construction so the backend
    sees the latest profile-applied env (BackendManager rebuilds the backend
    after each apply_profile()).
    """
    model_base = os.environ.get(
        "KOKORO_MODEL_BASE", "/opt/models/kokoro-multi-lang-v1_0"
    )
    hybrid_dir = os.environ.get("KOKORO_HYBRID_DIR", os.path.join(model_base, "hybrid"))
    return {
        "model_base": model_base,
        "model_onnx": os.environ.get("KOKORO_ONNX", os.path.join(model_base, "model.onnx")),
        "engine_path": os.environ.get(
            "KOKORO_TRT_ENGINE",
            os.path.join(model_base, "engines", "kokoro_fp16.engine"),
        ),
        "hybrid_dir": hybrid_dir,
        "hybrid_prefix_engine_env": os.environ.get("KOKORO_HYBRID_PREFIX_ENGINE"),
        "hybrid_prefix_engine_dyn": os.path.join(hybrid_dir, "kokoro_prefix_encoder_dyn4_128_fp16.engine"),
        "hybrid_prefix_engine_fixed": os.path.join(hybrid_dir, "kokoro_prefix_encoder_s96_fp16.engine"),
        "hybrid_suffix_onnx": os.environ.get(
            "KOKORO_HYBRID_SUFFIX_ONNX",
            os.path.join(hybrid_dir, "kokoro_suffix_encoder.onnx"),
        ),
        "split_encoder_engine": os.environ.get(
            "KOKORO_SPLIT_ENCODER_ENGINE",
            os.path.join(model_base, "engines", "kokoro_prefix_encoder_dyn4_128_fp16.engine"),
        ),
        "split_length_onnx": os.environ.get(
            "KOKORO_SPLIT_LENGTH_ONNX",
            os.path.join(model_base, "engines", "cpu_length_regulator.onnx"),
        ),
        "split_decoder_engine": os.environ.get(
            "KOKORO_SPLIT_DECODER_ENGINE",
            os.path.join(model_base, "engines", "kokoro_decoder_backbone_dyn64_256_fp16.engine"),
        ),
        "split_decoder_engine_long": os.environ.get(
            "KOKORO_SPLIT_DECODER_ENGINE_LONG",
            os.path.join(model_base, "engines", "kokoro_decoder_backbone_dyn256_512_fp16.engine"),
        ),
        "split_source_engine": os.environ.get(
            "KOKORO_SPLIT_SOURCE_ENGINE",
            os.path.join(model_base, "engines", "kokoro_generator_source_dyn128_512_bf16.engine"),
        ),
        "split_source_engine_long": os.environ.get(
            "KOKORO_SPLIT_SOURCE_ENGINE_LONG",
            os.path.join(model_base, "engines", "kokoro_generator_source_dyn512_1024_bf16.engine"),
        ),
        "split_source_onnx": os.environ.get(
            "KOKORO_SPLIT_SOURCE_ONNX",
            os.path.join(model_base, "engines", "cpu_generator_source.onnx"),
        ),
        "split_generator_engine": os.environ.get(
            "KOKORO_SPLIT_GENERATOR_ENGINE",
            os.path.join(model_base, "engines", "kokoro_generator_rest_preexp_dyn64_256_fp16.engine"),
        ),
        "split_generator_engine_long": os.environ.get(
            "KOKORO_SPLIT_GENERATOR_ENGINE_LONG",
            os.path.join(model_base, "engines", "kokoro_generator_rest_preexp_dyn256_512_fp16.engine"),
        ),
        "split_istft_onnx": os.environ.get(
            "KOKORO_SPLIT_ISTFT_ONNX",
            os.path.join(model_base, "engines", "cpu_postspec_istft.onnx"),
        ),
        "voices_bin": os.environ.get("KOKORO_VOICES", os.path.join(model_base, "voices.bin")),
        "tokens_path": os.environ.get("KOKORO_TOKENS", os.path.join(model_base, "tokens.txt")),
    }


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    np.clip(arr, -1.0, 1.0, out=arr)
    pcm = (arr * 32767).astype(np.int16)
    data_size = pcm.nbytes
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class _KokoroCtxSlot:
    """Pre-allocated TRT context + pool slot for one concurrent Kokoro request.

    Three runtime modes — engine / hybrid / split_generator. A slot holds the
    contexts the current mode needs; other-mode ctx fields are empty.

    A pool of K slots is built at preload() time. _synthesize_one() acquires
    one slot, runs the inference, then releases it back. This amortizes the
    ~900ms-per-context create_execution_context() cost on Jetson Orin Nano
    across the service lifetime instead of paying it per request.
    """

    def __init__(
        self,
        runtime_mode: str,
        engine,
        split_engines: dict,
        split_long_engines: dict,
    ):
        # Arena-backed pool — eliminates per-stage cudaMalloc driver-lock
        # cost on the kokoro hot path. Size tunable via
        # OVS_KOKORO_ARENA_SIZE_MB (or the shared OVS_CUDA_ARENA_SIZE_MB).
        self.pool = CudaMemoryPool(
            arena_size_bytes=_read_arena_size_bytes("OVS_KOKORO_ARENA_SIZE_MB"),
        )
        self.ctx = None
        self.split_ctxs: dict[str, object] = {}
        self.split_long_ctxs: dict[str, object] = {}
        if runtime_mode in ("engine", "hybrid"):
            if engine is not None:
                self.ctx = engine.create_execution_context()
        elif runtime_mode == "split_generator":
            self.split_ctxs = {
                name: eng.create_execution_context()
                for name, eng in split_engines.items()
            }
            self.split_long_ctxs = {
                name: eng.create_execution_context()
                for name, eng in split_long_engines.items()
            }
        elif runtime_mode in ("ort_cpu", "cpu", "ort", "onnxruntime"):
            # CPU ORT path needs no TRT context. We still keep an (unused)
            # pool so destroy() is symmetric across slots.
            pass
        else:
            raise ValueError(f"Unknown kokoro runtime_mode: {runtime_mode}")

    def reset_per_request(self):
        # Defensive sync: device-resident chains in _run_split_generator skip
        # per-stage pool.synchronize() to keep kernels back-to-back on the
        # CUDA stream. On the success path the final host-output stage already
        # syncs, so this is a no-op (GPU is idle). On exception paths, in-flight
        # kernels may still be writing arena buffers; we must wait before
        # free_all() returns those bytes to the arena for reuse by the next
        # request — otherwise the next request can see stale/corrupt writes.
        try:
            self.pool.synchronize()
        except Exception:
            logger.exception("KokoroCtxSlot.reset_per_request: pool.synchronize raised; continuing free_all")
        try:
            self.pool.free_all()
        except Exception:
            logger.exception("KokoroCtxSlot.reset_per_request: pool.free_all raised")

    def destroy(self):
        try:
            self.pool.synchronize()
        except Exception:
            pass
        try:
            self.pool.destroy()
        except Exception:
            pass
        self.ctx = None
        self.split_ctxs = {}
        self.split_long_ctxs = {}


class KokoroTRTBackend(TTSBackend):
    """English Kokoro v1.0 TTS accelerated with TensorRT on Jetson."""

    # Hot-reload enabled 2026-05-21 after applying matcha's release-order fix
    # (commit 2967688): execution contexts dropped before engines, CUDA stream
    # synchronized and destroyed, ORT sessions cleared, gc.collect() x2 to
    # reap pybind cycles. ORT sessions are all CPU EP (no CUDA EP allocator
    # leak surface). See unload() docstring for the full ordering rationale.
    # Hardware (VRAM) verification on orin-nano will be a separate dispatch.
    supports_hot_reload: bool = True

    @classmethod
    def concurrency_capability(cls, profile=None):
        from app.core.concurrency_capability import ConcurrencyCapability

        # K = OVS_TTS_STREAM_MAX_WORKERS (default 2), matching _build_ctx_pool().
        # Profile may override via tts_stream_max_workers (or nested
        # tts_backend_config.stream_max_workers). Engines (weights) shared;
        # each _KokoroCtxSlot holds its own TRT execution contexts.
        env_val = os.environ.get("OVS_TTS_STREAM_MAX_WORKERS")
        profile_val = None
        if isinstance(profile, dict):
            profile_val = profile.get("tts_stream_max_workers")
            if profile_val is None:
                cfg = profile.get("tts_backend_config")
                if isinstance(cfg, dict):
                    profile_val = cfg.get("stream_max_workers")
        try:
            k = int(env_val) if env_val is not None else (
                int(profile_val) if profile_val is not None else 2
            )
        except (TypeError, ValueError):
            k = 2
        k = max(1, k)
        return ConcurrencyCapability(
            supports_parallel=k > 1,
            max_concurrent=k,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._runtime_mode = os.environ.get("KOKORO_TRT_RUNTIME", "auto").strip().lower()
        self._engine = None
        # Per-call concurrency rework (N>=2 safety): TRT IExecutionContext is
        # not thread-safe. Shared self._ctx / self._pool / self._split_ctxs
        # are no longer populated by load paths — each _synthesize_one() call
        # builds its own context set + pool and tears them down in a finally
        # block. Engines (weights) remain shared and immutable.
        #
        # These attrs stay declared (None / empty dict) so unload() and other
        # introspection code paths keep working with no signature change.
        self._ctx = None
        self._pool: CudaMemoryPool | None = None
        self._ort_sess = None
        self._suffix_sess = None
        self._split_length_sess = None
        self._split_source_sess = None
        self._split_istft_sess = None
        self._split_engines = {}
        self._split_ctxs = {}
        self._split_long_engines = {}
        self._split_long_ctxs = {}
        self._token_input_name = "input_ids"
        self._output_name = None
        # IO name cache for ORT sessions (built when session is loaded).
        # Keyed by role: "split_length" / "split_source" / "split_istft" /
        # "suffix" / "ort_main". Immutable values, safe across slots.
        self._ort_io: dict[str, _OrtIoNames] = {}
        # Engine output-metadata cache (names + dtypes). Shapes still
        # per-call because they are dynamic. Keyed by role:
        # "engine" / "suffix_prefix" /
        # "split_encoder" / "split_decoder" / "split_source" /
        # "split_generator" / "split_decoder_long" /
        # "split_source_long" / "split_generator_long".
        self._trt_meta: dict[str, _TrtEngineMeta] = {}
        self._hybrid_fixed_seq_len: int | None = None
        self._hybrid_max_seq_len: int | None = None
        self._hybrid_min_seq_len: int | None = None
        self._ready = False
        # Pre-allocated context pool (built in preload() once we know the
        # final runtime_mode). N slots, each owning one CUDA stream + per-mode
        # execution contexts. _synthesize_one() borrows + returns. See
        # _MatchaCtxSlot docstring for the latency rationale.
        self._ctx_pool: "queue.Queue[_KokoroCtxSlot]" = queue.Queue()
        self._slots: list[_KokoroCtxSlot] = []
        self._pool_size: int = 0
        # Snapshot artifact paths from the *current* env at construction.
        # BackendManager rebuilds the backend after each apply_profile() so
        # __init__ sees the latest profile-applied env. Module-level _*_PATH
        # constants are kept frozen for back-compat (no external imports use
        # them; verified at the time of this refactor 2026-05-21).
        self._paths = _resolve_kokoro_paths()

    @property
    def name(self) -> str:
        return "kokoro_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_SPEAKER,
        }

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def unload(self) -> None:
        """Release TRT engines + execution contexts + ORT sessions + CUDA pool.

        Mirrors :meth:`MatchaTRTBackend.unload` (commit 2967688) so kokoro
        participates in cross-implementation hot reload. Kokoro has more
        engine/context pairs than matcha because of the split-generator
        architecture (encoder/decoder/source/generator plus an optional
        "long" 256-512 bucket) and the hybrid prefix engine, so each pair
        gets the same context-before-engine treatment.

        Ordering:
            1. Sync the CUDA stream — pending kernels must finish before we
               pull TRT contexts out from under them.
            2. Drop execution contexts BEFORE engines. The TRT engine
               destructor may skip workspace cleanup if execution contexts
               are still attached, leaking activation memory. We loop the
               split (and long-bucket) ctx dicts before the engine dicts,
               then handle the optional ``_ctx``/``_engine`` (hybrid or
               full-engine mode).
            3. Drop engines (each holds tens-hundreds of MB of device
               weights).
            4. Drop ORT sessions (all CPU EP for kokoro — no CUDA EP
               allocator surface).
            5. Destroy the CUDA pool (cudaStreamDestroy + free remaining
               allocations).
            6. gc.collect() twice — first pass clears acyclic, second pass
               walks reference cycles produced by TRT Python bindings.

        Idempotent. Safe to call from BackendManager rollback.
        """
        if (
            not self._ready
            and self._engine is None
            and self._ctx is None
            and self._ort_sess is None
            and self._suffix_sess is None
            and self._split_length_sess is None
            and self._split_source_sess is None
            and self._split_istft_sess is None
            and not self._split_engines
            and not self._split_ctxs
            and not self._split_long_engines
            and not self._split_long_ctxs
            and self._pool is None
            and not self._slots
        ):
            return

        try:
            # 1. Sync stream
            if self._pool is not None:
                try:
                    self._pool.synchronize()
                except Exception:
                    logger.exception("Kokoro unload: pool.synchronize failed; continuing")

            # 2. Execution contexts before engines.
            # 2a. Drain per-slot ctxs + streams + remaining device buffers.
            self._teardown_ctx_pool()

            # 2b. Back-compat: legacy shared ctx dicts (always empty after
            # pre-allocated pool refactor) are still cleared so tests that
            # populate them for unload-ordering assertions still pass.
            for name, ctx in list(self._split_ctxs.items()):
                try:
                    del ctx
                except Exception:
                    logger.exception("Kokoro unload: split ctx[%s] del raised", name)
            self._split_ctxs = {}

            for name, ctx in list(self._split_long_ctxs.items()):
                try:
                    del ctx
                except Exception:
                    logger.exception("Kokoro unload: split long ctx[%s] del raised", name)
            self._split_long_ctxs = {}

            if self._ctx is not None:
                try:
                    del self._ctx
                except Exception:
                    logger.exception("Kokoro unload: main ctx del raised")
                self._ctx = None

            # 3. Engines
            for name, eng in list(self._split_engines.items()):
                try:
                    del eng
                except Exception:
                    logger.exception("Kokoro unload: split engine[%s] del raised", name)
            self._split_engines = {}

            for name, eng in list(self._split_long_engines.items()):
                try:
                    del eng
                except Exception:
                    logger.exception("Kokoro unload: split long engine[%s] del raised", name)
            self._split_long_engines = {}

            if self._engine is not None:
                try:
                    del self._engine
                except Exception:
                    logger.exception("Kokoro unload: main engine del raised")
                self._engine = None

            # 4. ORT sessions (all CPU EP).
            self._ort_sess = None
            self._suffix_sess = None
            self._split_length_sess = None
            self._split_source_sess = None
            self._split_istft_sess = None

            # 5. CUDA pool teardown
            if self._pool is not None:
                try:
                    self._pool.destroy()
                except Exception:
                    logger.exception("Kokoro unload: pool.destroy failed; continuing")
                self._pool = None

            # 6. Force finalizers
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            logger.exception("KokoroTRTBackend.unload outer-try failed; continuing")
        finally:
            self._token_to_id = {}
            self._output_name = None
            self._hybrid_fixed_seq_len = None
            self._hybrid_max_seq_len = None
            self._hybrid_min_seq_len = None
            # Clear cached IO + engine metadata to prevent stale references
            # from holding on to released sessions / engines after reload.
            self._ort_io = {}
            self._trt_meta = {}
            self._ready = False

    def preload(self) -> None:
        self._load_tokens()
        if self._runtime_mode in ("cpu", "ort", "ort_cpu", "onnxruntime"):
            self._load_ort()
        elif self._runtime_mode in ("split", "split_generator", "trt_split", "trt_cpu_split"):
            self._load_split_generator()
        elif self._runtime_mode in ("hybrid", "trt_cpu", "trt_prefix"):
            self._load_hybrid()
        elif os.path.exists(self._paths['engine_path']):
            self._load_engine()
        elif self._split_generator_assets_exist():
            self._load_split_generator()
        elif os.path.exists(_hybrid_prefix_engine_path(self._paths)) and os.path.exists(self._paths['hybrid_suffix_onnx']):
            self._load_hybrid()
        else:
            logger.warning("Kokoro TRT engine missing at %s; using CPU ORT fallback", self._paths['engine_path'])
            self._load_ort()
        self._build_ctx_pool()
        try:
            self._warmup()
        except Exception as exc:
            if self._runtime_mode == "engine":
                logger.warning(
                    "Kokoro direct TensorRT warmup failed (%s); falling back to CPU ORT",
                    exc,
                )
                self._engine = None
                self._ctx = None
                self._pool = None
                # Stale "engine" TRT meta would point at the released engine;
                # purge before falling back so subsequent calls see only the
                # ORT path. _load_ort() repopulates ORT IO caches.
                self._trt_meta.pop("engine", None)
                self._teardown_ctx_pool()
                self._load_ort()
                self._build_ctx_pool()
                self._warmup()
            else:
                raise
        self._ready = True

    def _build_ctx_pool(self) -> None:
        """Pre-allocate K context slots and seed the queue.

        K = OVS_TTS_STREAM_MAX_WORKERS (default 2). Slot type depends on the
        already-resolved runtime mode. Logs the slot count so deploy
        verification can grep for it.
        """
        try:
            k = int(os.environ.get("OVS_TTS_STREAM_MAX_WORKERS", "2"))
        except ValueError:
            k = 2
        k = max(1, k)
        self._pool_size = k
        self._slots = []
        # Drain any prior queue contents (rebuild path after warmup fallback).
        while True:
            try:
                self._ctx_pool.get_nowait()
            except queue.Empty:
                break
        t0 = time.time()
        for _ in range(k):
            slot = _KokoroCtxSlot(
                self._runtime_mode,
                self._engine,
                self._split_engines,
                self._split_long_engines,
            )
            self._slots.append(slot)
            self._ctx_pool.put(slot)
        logger.info(
            "Kokoro ctx pool: %d slots pre-allocated (mode=%s, %.2fs)",
            k, self._runtime_mode, time.time() - t0,
        )

    def _teardown_ctx_pool(self) -> None:
        """Drain queue and destroy all slots. Safe to call multiple times."""
        while True:
            try:
                self._ctx_pool.get_nowait()
            except queue.Empty:
                break
        for i, slot in enumerate(self._slots):
            try:
                slot.destroy()
            except Exception:
                logger.exception("Kokoro teardown: slot[%d] destroy raised", i)
        self._slots = []
        self._pool_size = 0

    def _load_tokens(self) -> None:
        if not os.path.exists(self._paths['tokens_path']):
            raise FileNotFoundError(f"Kokoro tokens not found: {self._paths['tokens_path']}")
        with open(self._paths['tokens_path'], "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n").rstrip("\r")
                if not raw:
                    continue
                rsep = max(raw.rfind(" "), raw.rfind("\t"))
                if rsep < 0:
                    continue
                tok = raw[:rsep] or " "
                try:
                    self._token_to_id[tok] = int(raw[rsep + 1:])
                except ValueError:
                    continue
        logger.info("Loaded %d Kokoro tokens from %s", len(self._token_to_id), self._paths['tokens_path'])

    def _build_ort_io_cache(self, role: str, sess) -> None:
        """Snapshot ORT session input/output names into ``self._ort_io[role]``.

        Called from every load path so per-call ``_run_cpu_onnx`` can skip
        the live ``get_inputs()`` / ``get_outputs()`` round-trip.
        """
        inputs = frozenset(item.name for item in sess.get_inputs())
        outputs = tuple(item.name for item in sess.get_outputs())
        self._ort_io[role] = _OrtIoNames(inputs, outputs)

    def _build_trt_meta_cache(self, role: str, engine) -> None:
        """Snapshot TRT engine output (name, dtype) pairs into self._trt_meta.

        Shapes are dynamic and still queried per call from the execution
        context. Only the immutable name + dtype tuples are cached here so
        hot-path callers avoid the C++ <-> Python enumeration overhead.
        """
        import tensorrt as trt
        outputs = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            dtype = _trt_dtype_to_np(engine.get_tensor_dtype(name))
            outputs.append(_TrtOutputMeta(name=name, dtype=dtype))
        self._trt_meta[role] = _TrtEngineMeta(outputs=tuple(outputs))

    def _load_engine(self) -> None:
        import tensorrt as trt

        t0 = time.time()
        with open(self._paths['engine_path'], "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize Kokoro engine: {self._paths['engine_path']}")
        # Per-call concurrency rework: ctx + pool are per-call now.
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        if "tokens" in names:
            self._token_input_name = "tokens"
        elif "input_ids" in names:
            self._token_input_name = "input_ids"
        self._build_trt_meta_cache("engine", self._engine)
        self._runtime_mode = "engine"
        logger.info("Kokoro TRT engine loaded: %s (%.1fs)", self._paths['engine_path'], time.time() - t0)

    def _load_hybrid(self) -> None:
        import onnxruntime as ort
        import tensorrt as trt

        prefix_engine = _hybrid_prefix_engine_path(self._paths)
        if not os.path.exists(prefix_engine):
            raise FileNotFoundError(f"Kokoro hybrid prefix engine not found: {prefix_engine}")
        if not os.path.exists(self._paths['hybrid_suffix_onnx']):
            raise FileNotFoundError(f"Kokoro hybrid suffix ONNX not found: {self._paths['hybrid_suffix_onnx']}")

        t0 = time.time()
        with open(prefix_engine, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize Kokoro hybrid prefix engine: {prefix_engine}")
        # Per-call concurrency rework: ctx + pool are per-call now.
        self._configure_hybrid_token_profile()
        self._build_trt_meta_cache("engine", self._engine)
        self._suffix_sess = ort.InferenceSession(self._paths['hybrid_suffix_onnx'], providers=["CPUExecutionProvider"])
        self._build_ort_io_cache("suffix", self._suffix_sess)
        self._token_input_name = "tokens"
        self._runtime_mode = "hybrid"
        logger.info(
            "Kokoro hybrid loaded: prefix=%s suffix=%s token_profile=fixed:%s max:%s (%.1fs)",
            prefix_engine,
            self._paths['hybrid_suffix_onnx'],
            self._hybrid_fixed_seq_len,
            self._hybrid_max_seq_len,
            time.time() - t0,
        )

    def _split_generator_assets_exist(self) -> bool:
        required = (
            self._paths['split_encoder_engine'],
            self._paths['split_length_onnx'],
            self._paths['split_decoder_engine'],
            self._paths['split_generator_engine'],
            self._paths['split_istft_onnx'],
        )
        if not all(os.path.exists(path) for path in required):
            return False
        return os.path.exists(self._paths['split_source_engine']) or os.path.exists(self._paths['split_source_onnx'])

    def _load_split_generator(self) -> None:
        import onnxruntime as ort
        import tensorrt as trt

        required = {
            "encoder": self._paths['split_encoder_engine'],
            "decoder": self._paths['split_decoder_engine'],
            "generator": self._paths['split_generator_engine'],
        }
        if os.path.exists(self._paths['split_source_engine']):
            required["source"] = self._paths['split_source_engine']
        for name, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro split {name} engine not found: {path}")
        for name, path in {
            "length regulator": self._paths['split_length_onnx'],
            "ISTFT": self._paths['split_istft_onnx'],
        }.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro split {name} ONNX not found: {path}")
        if "source" not in required and not os.path.exists(self._paths['split_source_onnx']):
            raise FileNotFoundError(
                f"Kokoro split source engine/ONNX not found: {self._paths['split_source_engine']} / {self._paths['split_source_onnx']}"
            )

        t0 = time.time()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._split_engines = {}
        self._split_long_engines = {}
        # Per-call concurrency rework: execution contexts + pool created
        # per synthesize call. Only engines (weights) are loaded here.
        for name, path in required.items():
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize Kokoro split {name} engine: {path}")
            self._split_engines[name] = engine
            self._build_trt_meta_cache(f"split_{name}", engine)
        long_required = {
            "decoder": self._paths['split_decoder_engine_long'],
            "source": self._paths['split_source_engine_long'],
            "generator": self._paths['split_generator_engine_long'],
        }
        if all(os.path.exists(path) for path in long_required.values()):
            for name, path in long_required.items():
                with open(path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize Kokoro split long {name} engine: {path}")
                self._split_long_engines[name] = engine
                self._build_trt_meta_cache(f"split_{name}_long", engine)
        elif any(os.path.exists(path) for path in long_required.values()):
            missing = [path for path in long_required.values() if not os.path.exists(path)]
            logger.warning("Ignoring incomplete Kokoro 256-512 bucket; missing: %s", missing)
        self._configure_split_token_profile()
        self._split_length_sess = ort.InferenceSession(self._paths['split_length_onnx'], providers=["CPUExecutionProvider"])
        self._build_ort_io_cache("split_length", self._split_length_sess)
        self._split_istft_sess = ort.InferenceSession(self._paths['split_istft_onnx'], providers=["CPUExecutionProvider"])
        self._build_ort_io_cache("split_istft", self._split_istft_sess)
        if "source" not in required:
            self._split_source_sess = ort.InferenceSession(self._paths['split_source_onnx'], providers=["CPUExecutionProvider"])
            self._build_ort_io_cache("split_source", self._split_source_sess)
        self._token_input_name = "tokens"
        self._runtime_mode = "split_generator"
        logger.info(
            "Kokoro split-generator loaded: encoder=%s decoder=%s source=%s generator=%s "
            "long_bucket=%s length=%s istft=%s token_profile=fixed:%s max:%s (%.1fs)",
            self._paths['split_encoder_engine'],
            self._paths['split_decoder_engine'],
            self._paths['split_source_engine'] if "source" in required else self._paths['split_source_onnx'],
            self._paths['split_generator_engine'],
            bool(self._split_long_engines),
            self._paths['split_length_onnx'],
            self._paths['split_istft_onnx'],
            self._hybrid_fixed_seq_len,
            self._hybrid_max_seq_len,
            time.time() - t0,
        )

    def _configure_split_token_profile(self) -> None:
        engine = self._split_engines.get("encoder")
        if engine is None:
            return
        try:
            min_shape, _opt_shape, max_shape = engine.get_tensor_profile_shape("tokens", 0)
            min_seq = int(tuple(min_shape)[1])
            max_seq = int(tuple(max_shape)[1])
            self._hybrid_min_seq_len = min_seq
            self._hybrid_max_seq_len = max_seq
            self._hybrid_fixed_seq_len = max_seq if min_seq == max_seq else None
        except Exception:
            self._hybrid_min_seq_len = None
            self._hybrid_fixed_seq_len = None
            self._hybrid_max_seq_len = int(os.environ.get("KOKORO_SPLIT_MAX_SEQ_LEN", "128"))

    def _configure_hybrid_token_profile(self) -> None:
        assert self._engine is not None
        try:
            min_shape, _opt_shape, max_shape = self._engine.get_tensor_profile_shape("tokens", 0)
            min_seq = int(tuple(min_shape)[1])
            max_seq = int(tuple(max_shape)[1])
            self._hybrid_min_seq_len = min_seq
            self._hybrid_max_seq_len = max_seq
            self._hybrid_fixed_seq_len = max_seq if min_seq == max_seq else None
        except Exception:
            fixed = int(os.environ.get("KOKORO_HYBRID_TOKEN_LEN", "0"))
            self._hybrid_min_seq_len = None
            self._hybrid_fixed_seq_len = fixed or None
            self._hybrid_max_seq_len = fixed or int(os.environ.get("KOKORO_HYBRID_MAX_SEQ_LEN", "128"))

    def _load_ort(self) -> None:
        import onnxruntime as ort

        if not os.path.exists(self._paths['model_onnx']):
            raise FileNotFoundError(f"Kokoro ONNX not found: {self._paths['model_onnx']}")
        providers = ["CPUExecutionProvider"]
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_sess = ort.InferenceSession(self._paths['model_onnx'], sess_opt, providers=providers)
        self._build_ort_io_cache("ort_main", self._ort_sess)
        input_names = {item.name for item in self._ort_sess.get_inputs()}
        if "tokens" in input_names:
            self._token_input_name = "tokens"
        elif "input_ids" in input_names:
            self._token_input_name = "input_ids"
        else:
            raise RuntimeError(f"Kokoro ONNX missing token input; inputs={sorted(input_names)}")
        outputs = self._ort_sess.get_outputs()
        self._output_name = outputs[0].name if outputs else None
        active = self._ort_sess.get_providers()
        self._runtime_mode = "ort_cpu"
        logger.info("Kokoro ORT providers: %s", active)

    def _warmup(self) -> None:
        start = time.time()
        for text in ("OK.", "Hello."):
            self.synthesize(text)
        logger.info("Kokoro warmup: %.1fs", time.time() - start)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        del pitch_shift, language
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs)
        sid = voice.get("speaker_id", DEFAULT_SPEAKER_ID)
        if SYNTH_SEGMENT_TEXT and self._runtime_mode in ("hybrid", "split_generator"):
            max_tokens = max(1, (self._hybrid_max_seq_len or 128) - 2)
            token_count = len(self._text_to_token_ids(text))
            if token_count > max_tokens:
                segment_limit = int(os.environ.get("KOKORO_SYNTH_MAX_SEGMENT_TOKENS", str(STREAM_SEGMENT_TOKENS)))
                return self._synthesize_segments(text, segment_limit, speaker_id=sid, speed=speed)
        return self._synthesize_one(text, speaker_id=sid, speed=speed)

    def _synthesize_segments(
        self,
        text: str,
        max_tokens: int,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, dict]:
        t_start = time.time()
        segments = self._split_stream_text(text, max_tokens)
        pcm_parts: list[bytes] = []
        metas: list[dict] = []
        for segment in segments:
            wav, meta = self._synthesize_one(segment, speaker_id=speaker_id, speed=speed)
            metas.append(meta)
            if len(wav) > 44:
                pcm_parts.append(wav[44:])
        pcm = b"".join(pcm_parts)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        wav = _samples_to_wav(samples, SAMPLE_RATE)
        duration = len(samples) / SAMPLE_RATE
        elapsed = time.time() - t_start
        return wav, {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": SAMPLE_RATE,
            "num_tokens": sum(int(meta.get("num_tokens", 0)) for meta in metas),
            "infer_ms": round(sum(float(meta.get("infer_ms", 0.0)) for meta in metas), 1),
            "language": "en",
            "runtime": self._runtime_mode,
            "segments": len(segments),
            "truncated": False,
        }

    def _synthesize_one(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, dict]:
        sid = DEFAULT_SPEAKER_ID if speaker_id is None else int(speaker_id)
        spd = DEFAULT_SPEED if speed is None else float(speed)
        t_start = time.time()
        token_ids = self._text_to_token_ids(text)
        if not token_ids:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            return _samples_to_wav(silence, SAMPLE_RATE), {
                "duration": 0.1,
                "inference_time": 0.0,
                "sample_rate": SAMPLE_RATE,
                "language": "en",
                "runtime": self._runtime_mode,
            }

        max_tokens = MAX_TOKENS
        if self._runtime_mode in ("hybrid", "split_generator"):
            max_tokens = max(1, (self._hybrid_max_seq_len or 128) - 2)
        truncated = len(token_ids) > max_tokens
        token_ids = token_ids[:max_tokens]
        ids = [0, *token_ids, 0]
        # Padding policy:
        #   1) Fixed-shape engine -> pad to fixed len (legacy behavior)
        #   2) Dynamic engine with known min -> pad up to min if below
        #      (defensive: avoids triggering the CPU ORT fallback below which
        #      leaks an untracked ORT session per short input).
        if self._runtime_mode in ("hybrid", "split_generator"):
            if self._hybrid_fixed_seq_len:
                ids = ids + [0] * max(0, self._hybrid_fixed_seq_len - len(ids))
            elif self._hybrid_min_seq_len and len(ids) < self._hybrid_min_seq_len:
                ids = ids + [0] * (self._hybrid_min_seq_len - len(ids))
        input_ids = np.array([ids], dtype=np.int64)
        style = self._load_style(sid, len(token_ids))
        speed_arr = np.array([spd], dtype=np.float32)

        t_infer = time.time()
        # Borrow a pre-allocated context slot if the pool has been built
        # (production path). free_all() between requests releases device
        # buffers while keeping the stream + ctxs warm.
        #
        # When no slot is available (unit-test bypass path: tests build the
        # backend with __new__ + a runtime_mode attr but never preload, so
        # _slots is empty), the run helpers are mocked by the test and the
        # pool / ctx kwargs are ignored — pass None.
        slot: _KokoroCtxSlot | None = None
        if self._slots:
            slot = self._ctx_pool.get()
        pool_arg = slot.pool if slot is not None else None
        ctx_arg = slot.ctx if slot is not None else None
        split_ctxs_arg = slot.split_ctxs if slot is not None else {}
        split_long_ctxs_arg = slot.split_long_ctxs if slot is not None else {}
        try:
            if self._runtime_mode == "engine":
                audio = self._run_engine(
                    input_ids, style, speed_arr,
                    pool=pool_arg, ctx=ctx_arg,
                )
            elif self._runtime_mode == "hybrid":
                audio = self._run_hybrid(
                    input_ids, style, speed_arr,
                    pool=pool_arg, ctx=ctx_arg,
                )
            elif self._runtime_mode == "split_generator":
                try:
                    audio = self._run_split_generator(
                        input_ids, style, speed_arr,
                        pool=pool_arg,
                        split_ctxs=split_ctxs_arg,
                        split_long_ctxs=split_long_ctxs_arg,
                    )
                except ValueError as exc:
                    if os.environ.get("KOKORO_SPLIT_CPU_FALLBACK", "1").lower() in ("0", "false", "no"):
                        raise
                    logger.warning("Kokoro split-generator shape mismatch; falling back to CPU ORT: %s", exc)
                    self._load_ort()
                    audio = self._run_ort(input_ids, style, speed_arr)
            else:
                audio = self._run_ort(input_ids, style, speed_arr)
        finally:
            if slot is not None:
                try:
                    slot.reset_per_request()
                finally:
                    self._ctx_pool.put(slot)
        infer_ms = (time.time() - t_infer) * 1000

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        wav = _samples_to_wav(audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.time() - t_start
        return wav, {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": SAMPLE_RATE,
            "num_tokens": len(token_ids),
            "infer_ms": round(infer_ms, 1),
            "language": "en",
            "runtime": self._runtime_mode,
            "truncated": truncated,
        }

    def generate_streaming(self, text: str, **kwargs):
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, **kwargs)
        sid = voice.get("speaker_id", DEFAULT_SPEAKER_ID)
        segments = [text]
        if STREAM_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = self._split_stream_text(text, kwargs.get("segment_max_tokens"))
        try:
            chunk_ms = int(os.environ.get("KOKORO_STREAM_CHUNK_MS", "40"))
        except ValueError:
            chunk_ms = 40
        chunk_ms = max(10, min(200, chunk_ms))
        chunk_bytes = max(2, int(SAMPLE_RATE * chunk_ms / 1000) * 2)
        for segment in segments:
            wav, _meta = self.synthesize(
                segment,
                speaker_id=sid,
                speed=kwargs.get("speed"),
            )
            if len(wav) <= 44:
                continue
            pcm = wav[44:]
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset:offset + chunk_bytes]
                if chunk:
                    yield chunk

    def _split_stream_text(self, text: str, max_tokens: Optional[int] = None) -> list[str]:
        text = " ".join((text or "").split())
        if not text:
            return []
        if max_tokens is None:
            max_tokens = STREAM_SEGMENT_TOKENS
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = STREAM_SEGMENT_TOKENS
        if max_tokens <= 0:
            return [text]
        max_tokens = max(16, max_tokens)

        parts = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", text) if part.strip()]
        if not parts:
            parts = [text]
        segments: list[str] = []
        for part in parts:
            segments.extend(self._split_text_by_token_count(part, max_tokens))
        return segments or [text]

    def _split_text_by_token_count(self, text: str, max_tokens: int) -> list[str]:
        if len(self._text_to_token_ids(text)) <= max_tokens:
            return [text]
        words = text.split()
        if not words:
            return [text]
        segments: list[str] = []
        current_words: list[str] = []
        for word in words:
            candidate_words = [*current_words, word]
            candidate = " ".join(candidate_words)
            if current_words and len(self._text_to_token_ids(candidate)) > max_tokens:
                segments.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words
            current = " ".join(current_words)
            if current and len(self._text_to_token_ids(current)) > max_tokens:
                segments.extend(self._split_long_word(current, max_tokens))
                current_words = []
        if current_words:
            segments.append(" ".join(current_words))
        return segments

    def _split_long_word(self, text: str, max_tokens: int) -> list[str]:
        parts: list[str] = []
        current = ""
        for ch in text:
            candidate = f"{current}{ch}"
            if current and len(self._text_to_token_ids(candidate)) > max_tokens:
                parts.append(current)
                current = ch
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts or [text]

    def _text_to_token_ids(self, text: str) -> list[int]:
        import piper_phonemize

        text = text.strip()
        if not text:
            return []
        sentences = piper_phonemize.phonemize_espeak(text, "en-us")
        ids: list[int] = []
        for sent_idx, phonemes in enumerate(sentences or []):
            if sent_idx > 0:
                self._append_token(ids, " ")
            joined = "".join(p for p in phonemes if p)
            for ch in joined:
                self._append_token(ids, ch)
        if ids:
            return ids

        # Last-resort ASCII path keeps smoke tests debuggable if espeak data is
        # missing; quality is not expected to be acceptable in this branch.
        for ch in re.sub(r"\s+", " ", text.lower()):
            self._append_token(ids, ch)
        return ids

    def _append_token(self, ids: list[int], token: str) -> None:
        tid = self._token_to_id.get(token)
        if tid is not None:
            ids.append(tid)

    def _load_style(self, speaker_id: int, token_count: int) -> np.ndarray:
        if not os.path.exists(self._paths['voices_bin']):
            raise FileNotFoundError(f"Kokoro voices not found: {self._paths['voices_bin']}")
        style_idx = max(0, min(VOICE_STYLES - 1, int(token_count)))
        offset = speaker_id * STYLE_BYTES + style_idx * STYLE_DIM * 4
        size = os.path.getsize(self._paths['voices_bin'])
        if offset + STYLE_DIM * 4 > size:
            raise ValueError(
                f"Kokoro speaker_id {speaker_id} out of range for {self._paths['voices_bin']} "
                f"(file has about {size // STYLE_BYTES} speakers)"
            )
        with open(self._paths['voices_bin'], "rb") as f:
            f.seek(offset)
            data = f.read(STYLE_DIM * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, STYLE_DIM).copy()

    def _run_ort(self, input_ids: np.ndarray, style: np.ndarray, speed: np.ndarray) -> np.ndarray:
        return self._ort_sess.run(
            None,
            {self._token_input_name: input_ids, "style": style, "speed": speed},
        )[0]

    def _run_engine(
        self,
        input_ids: np.ndarray,
        style: np.ndarray,
        speed: np.ndarray,
        *,
        pool: "CudaMemoryPool",
        ctx,
    ) -> np.ndarray:
        assert pool is not None and ctx is not None and self._engine is not None

        def bind_input(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            self._set_or_validate_input_shape(ctx, name, arr)

        bind_input(self._token_input_name, input_ids)
        bind_input("style", style.astype(np.float32, copy=False))
        bind_input("speed", speed.astype(np.float32, copy=False))

        output_name = self._output_tensor_name()
        out_shape = tuple(int(d) for d in ctx.get_tensor_shape(output_name))
        if any(d < 0 for d in out_shape):
            pool.free_all()
            raise RuntimeError(
                "Kokoro TRT engine produced a dynamic output shape that the "
                "direct full-engine backend cannot allocate. Use "
                "KOKORO_TRT_RUNTIME=hybrid with the TensorRT prefix engine."
            )
        output = np.empty(out_shape, dtype=np.float32)
        d_out = pool.allocate(output.nbytes)
        ctx.set_tensor_address(output_name, d_out)
        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            pool.free_all()
            raise RuntimeError("Kokoro TRT execute_async_v3 returned False")
        pool.synchronize()
        pool.copy_dtoh(d_out, output)
        pool.free_all()
        return output

    def _run_hybrid(
        self,
        input_ids: np.ndarray,
        style: np.ndarray,
        speed: np.ndarray,
        *,
        pool: "CudaMemoryPool",
        ctx,
    ) -> np.ndarray:
        assert self._suffix_sess is not None and self._engine is not None and ctx is not None
        prefix_outputs = self._run_trt_context(
            self._engine,
            ctx,
            {"tokens": input_ids, "style": style.astype(np.float32, copy=False), "speed": speed.astype(np.float32, copy=False)},
            pool=pool,
            meta=self._trt_meta.get("engine"),
        )
        suffix_io = self._ort_io.get("suffix")
        if suffix_io is not None:
            suffix_input_names = suffix_io.input_names
        else:
            suffix_input_names = {item.name for item in self._suffix_sess.get_inputs()}
        feeds = {}
        for name, arr in {"tokens": input_ids, "style": style, "speed": speed}.items():
            if name in suffix_input_names:
                feeds[name] = arr
        for name, arr in prefix_outputs.items():
            feeds[name] = arr
        return self._suffix_sess.run(None, feeds)[0]

    def _run_split_generator(
        self,
        input_ids: np.ndarray,
        style: np.ndarray,
        speed: np.ndarray,
        *,
        pool: "CudaMemoryPool",
        split_ctxs: dict[str, object],
        split_long_ctxs: dict[str, object],
    ) -> np.ndarray:
        assert self._split_length_sess is not None and self._split_istft_sess is not None

        stage: dict[str, np.ndarray] = {
            "tokens": input_ids,
            "style": style.astype(np.float32, copy=False),
            "speed": speed.astype(np.float32, copy=False),
        }
        stage.update(self._run_named_trt_engine("encoder", stage, pool=pool, ctx=split_ctxs["encoder"]))
        stage.update(_run_cpu_onnx(self._split_length_sess, stage, io_names=self._ort_io.get("split_length")))
        frame_t = int(stage["/encoder/MatMul_1_output_0"].shape[2])
        bucket_engines, bucket_ctxs = self._select_split_bucket(
            frame_t, split_ctxs=split_ctxs, split_long_ctxs=split_long_ctxs,
        )

        # Stages 3 (decoder TRT) → 4 (source TRT or CPU) → 5 (generator TRT)
        # → 6 (iSTFT CPU). If source is TRT, we can keep decoder/source
        # outputs device-resident and skip 2 D2H + 2 H2D round trips per
        # synthesize. iSTFT (CPU) forces generator's output back to host.
        source_is_trt = "source" in bucket_engines
        device_chain_outputs: dict[str, _DeviceTensor] = {}

        # style is bound by generator (and originally by decoder) — for the
        # device path we still bind style via H2D each stage because it's tiny
        # (256 floats) and not produced by another TRT engine. Cheaper to keep
        # the host array around than thread the device pointer through.

        if source_is_trt:
            # Pure TRT chain: decoder → generator (with style + source-Div_4
            # rebound on generator). source's Div_4 output overrides decoder's.
            # Device-resident chain: decoder → source → generator all run on
            # the same CUDA stream. CUDA stream ordering guarantees the next
            # ctx.execute_async_v3() waits for prior kernels, so we skip the
            # per-stage pool.synchronize() that would otherwise block the CPU
            # waiting for the GPU. The final generator stage (host return)
            # forces a sync before copy_dtoh, and reset_per_request() also
            # sync's defensively before free_all() on exception paths.
            decoder_dev = self._run_split_bucket_engine(
                bucket_engines, bucket_ctxs, "decoder", {
                    "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
                    "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
                    "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
                    "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
                    "style": stage["style"],
                },
                pool=pool, return_device=True, sync=False,
            )
            source_dev = self._run_split_bucket_engine(
                bucket_engines, bucket_ctxs, "source", {
                    "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
                },
                pool=pool, return_device=True, sync=False,
            )
            # Merge — source's outputs override decoder's where keys collide
            # (specifically Div_4 in the kokoro split topology).
            device_chain_outputs = {**decoder_dev, **source_dev}

            # Generator needs Div_4 (from source) and Concat_3 (from decoder).
            # style is the only host-side input.
            needed = (
                "/decoder/decoder/decode.3/Div_4_output_0",
                "/decoder/decoder/generator/Concat_3_output_0",
            )
            gen_device_inputs = {k: device_chain_outputs[k] for k in needed if k in device_chain_outputs}
            gen = self._run_split_bucket_engine(
                bucket_engines, bucket_ctxs, "generator",
                {"style": stage["style"]},
                pool=pool,
                device_inputs=gen_device_inputs,
                return_device=False,  # iSTFT is CPU; output must be host
            )
        else:
            # CPU source path — cannot keep decoder output device-resident
            # because source ORT wants host arrays. Fall back to legacy host
            # I/O for all three stages.
            stage.update(self._run_split_bucket_engine(
                bucket_engines, bucket_ctxs, "decoder", {
                    "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
                    "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
                    "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
                    "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
                    "style": stage["style"],
                }, pool=pool,
            ))
            assert self._split_source_sess is not None
            stage.update(_run_cpu_onnx(self._split_source_sess, stage, io_names=self._ort_io.get("split_source")))
            gen = self._run_split_bucket_engine(
                bucket_engines, bucket_ctxs, "generator", {
                    "/decoder/decoder/decode.3/Div_4_output_0": stage["/decoder/decoder/decode.3/Div_4_output_0"],
                    "/decoder/decoder/generator/Concat_3_output_0": stage["/decoder/decoder/generator/Concat_3_output_0"],
                    "style": stage["style"],
                }, pool=pool,
            )
        return _run_cpu_onnx(self._split_istft_sess, gen, io_names=self._ort_io.get("split_istft"))["audio"]

    def _select_split_bucket(
        self,
        frame_t: int,
        *,
        split_ctxs: dict[str, object],
        split_long_ctxs: dict[str, object],
    ):
        if frame_t <= 256:
            return self._split_engines, split_ctxs
        if frame_t <= 512 and self._split_long_engines:
            return self._split_long_engines, split_long_ctxs
        raise ValueError(
            f"Kokoro split-generator frame length {frame_t} is outside available TRT buckets "
            f"(base<=256, long<=512 loaded={bool(self._split_long_engines)})"
        )

    def _run_named_trt_engine(
        self,
        name: str,
        inputs: dict[str, np.ndarray],
        *,
        pool: "CudaMemoryPool",
        ctx,
        device_inputs: dict[str, "_DeviceTensor"] | None = None,
        return_device: bool = False,
        sync: bool = True,
    ):
        engine = self._split_engines[name]
        return self._run_trt_context(
            engine, ctx, inputs, pool=pool,
            device_inputs=device_inputs, return_device=return_device,
            sync=sync,
            meta=self._trt_meta.get(f"split_{name}"),
        )

    def _run_split_bucket_engine(
        self,
        engines: dict[str, object],
        ctxs: dict[str, object],
        name: str,
        inputs: dict[str, np.ndarray],
        *,
        pool: "CudaMemoryPool",
        device_inputs: dict[str, "_DeviceTensor"] | None = None,
        return_device: bool = False,
        sync: bool = True,
    ):
        # Identify bucket family (long vs base) by identity to pick the right
        # cached engine meta. _split_long_engines and _split_engines are
        # distinct dicts loaded at preload; one of them owns ``engines``.
        if engines is self._split_long_engines:
            meta_key = f"split_{name}_long"
        else:
            meta_key = f"split_{name}"
        return self._run_trt_context(
            engines[name], ctxs[name], inputs, pool=pool,
            device_inputs=device_inputs, return_device=return_device,
            sync=sync,
            meta=self._trt_meta.get(meta_key),
        )

    def _run_trt_context(
        self,
        engine,
        ctx,
        inputs: dict[str, np.ndarray],
        *,
        pool: "CudaMemoryPool",
        device_inputs: dict[str, "_DeviceTensor"] | None = None,
        return_device: bool = False,
        sync: bool = True,
        meta: "_TrtEngineMeta | None" = None,
    ):
        """Run a TRT context with optional device-resident I/O.

        Args:
            inputs: host numpy arrays bound via H2D copy.
            device_inputs: pre-existing device pointers (from a previous TRT
                stage's ``return_device=True`` output) — bound directly with
                no H2D copy. Inputs in ``device_inputs`` override entries in
                ``inputs`` with the same name.
            return_device: if True, do not D2H output buffers; instead return
                ``dict[str, _DeviceTensor]`` of device pointers valid for the
                lifetime of ``pool``. The caller is responsible for ensuring
                ``pool.free_all()`` is not called until those tensors are no
                longer needed (synthesize-level finally handles this).

        Pool lifecycle: this method NEVER calls ``pool.free_all()``. The outer
        ``_synthesize_one()`` finally block (via ``slot.reset_per_request()``)
        is the single source of truth for releasing per-request device memory.
        That gives multi-stage callers a stable device-resident buffer chain.
        """
        assert pool is not None
        device_inputs = device_inputs or {}

        def bind_host_input(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            self._validate_engine_input_shape(engine, name, arr)
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            self._set_or_validate_input_shape(ctx, name, arr)

        def bind_device_input(name: str, dt: "_DeviceTensor") -> None:
            ctx.set_tensor_address(name, dt.ptr)
            # set_input_shape mirrors the dynamic-shape path used for host
            # arrays — emulate by constructing a tiny stand-in with the right
            # shape attribute. Using a class with .shape avoids a real
            # numpy alloc.
            class _ShapeOnly:
                __slots__ = ("shape",)
                def __init__(self, shape):
                    self.shape = shape
            self._set_or_validate_input_shape(ctx, name, _ShapeOnly(dt.shape))

        # First H2D-bind any host inputs not shadowed by device_inputs.
        for name, arr in inputs.items():
            if name in device_inputs:
                continue
            bind_host_input(name, arr)
        for name, dt in device_inputs.items():
            bind_device_input(name, dt)

        host_output_ptrs: list[tuple[str, int, np.ndarray]] = []
        device_output_handles: dict[str, _DeviceTensor] = {}

        if meta is not None:
            # Fast path: skip per-call enumeration of engine output tensors.
            output_iter = ((o.name, o.dtype) for o in meta.outputs)
        else:
            # Fallback: legacy callers / tests without a populated cache.
            import tensorrt as trt
            output_iter_list = []
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                    continue
                output_iter_list.append((name, _trt_dtype_to_np(engine.get_tensor_dtype(name))))
            output_iter = iter(output_iter_list)

        for name, dtype in output_iter:
            shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Kokoro hybrid prefix output has dynamic shape: {name} {shape}")
            if return_device:
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                ptr = pool.allocate(nbytes)
                ctx.set_tensor_address(name, ptr)
                device_output_handles[name] = _DeviceTensor(
                    ptr=ptr, shape=shape, dtype=dtype, nbytes=nbytes,
                )
            else:
                out = np.empty(shape, dtype=dtype)
                ptr = pool.allocate(out.nbytes)
                ctx.set_tensor_address(name, ptr)
                host_output_ptrs.append((name, ptr, out))

        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            raise RuntimeError("Kokoro hybrid prefix TRT execute_async_v3 returned False")
        # When sync=False AND return_device=True, skip the host-blocking
        # synchronize: subsequent TRT kernels enqueued on the same CUDA stream
        # serialize automatically, so device-resident chain hops don't need a
        # CPU round-trip. When we're returning host buffers (return_device
        # False) we MUST sync before copy_dtoh (regardless of sync flag).
        if sync or not return_device:
            pool.synchronize()

        if return_device:
            return device_output_handles
        outputs: dict[str, np.ndarray] = {}
        for name, ptr, out in host_output_ptrs:
            pool.copy_dtoh(ptr, out)
            outputs[name] = out
        return outputs

    def _validate_engine_input_shape(self, engine, name: str, arr: np.ndarray) -> None:
        shape = tuple(int(d) for d in engine.get_tensor_shape(name))
        if not shape or any(dim < 0 for dim in shape):
            return
        if shape != tuple(arr.shape):
            raise ValueError(f"{name} shape {tuple(arr.shape)} does not match fixed TRT shape {shape}")

    def _set_or_validate_input_shape(self, ctx, name: str, arr: np.ndarray) -> None:
        try:
            ok = ctx.set_input_shape(name, tuple(arr.shape))
        except Exception:
            return
        if ok is False:
            raise ValueError(f"{name} shape {tuple(arr.shape)} is outside the TRT optimization profile")

    def _output_tensor_name(self) -> str:
        import tensorrt as trt

        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        outputs = [
            name for name in names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if not outputs:
            raise RuntimeError("Kokoro TRT engine has no output tensors")
        return outputs[0]


def _trt_dtype_to_np(dtype):
    import tensorrt as trt

    if dtype == trt.float32:
        return np.float32
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.int32:
        return np.int32
    if dtype == trt.int64:
        return np.int64
    if dtype == trt.bool:
        return np.bool_
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _run_cpu_onnx(
    sess,
    feeds: dict[str, np.ndarray],
    io_names: "_OrtIoNames | None" = None,
) -> dict[str, np.ndarray]:
    """Run an ORT session, filtering feeds to only its declared inputs.

    Pass ``io_names`` (built once at session-load time) to skip the
    ``sess.get_inputs()`` / ``sess.get_outputs()`` round-trip; falls back to
    the live session metadata when the cache is unavailable (legacy callers
    + tests).
    """
    if io_names is None:
        input_names = {item.name for item in sess.get_inputs()}
        output_names = tuple(item.name for item in sess.get_outputs())
    else:
        input_names = io_names.input_names
        output_names = io_names.output_names
    actual = {name: value for name, value in feeds.items() if name in input_names}
    return dict(zip(output_names, sess.run(list(output_names), actual)))
