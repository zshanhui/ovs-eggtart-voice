"""FastAPI speech service: ASR + TTS with pluggable backends."""

from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
class _WSHandle:
    """Lightweight WS-session handle for BackendManager.register_ws().

    Replaces ``types.SimpleNamespace`` here because Python 3.10's
    SimpleNamespace lacks ``__weakref__`` (added in 3.11), and
    BackendManager._ws_handles is a WeakSet. The Jetson image still
    ships Python 3.10.12, so any handle stored in a WeakSet must be a
    plain class.
    """
    __slots__ = ("websocket", "task", "__weakref__")

    def __init__(self, websocket, task):
        self.websocket = websocket
        self.task = task
from typing import Literal, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jetson Speech Service", version="2.0.0")


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speaker_id: int | None = None
    speaker_embedding_b64: str | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None


class CloneRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None


class CloneStreamRequest(BaseModel):
    text: str
    speaker_embedding_b64: str  # base64-encoded speaker embedding
    language: str | None = None
    streaming_profile: str | None = None
    first_chunk_frames: int | None = None
    chunk_frames: int | None = None


_asr_backend = None

# Dedicated single-thread executor for streaming TTS (T3 fix).
# Default asyncio executor spawns multiple worker threads; each new thread
# observes a cold CUDA per-thread context for the C++ TRT engine, which
# inflates streaming prefill from ~16ms (warm) to 33-122ms (cold) under
# any concurrency. Pinning streaming TTS to a single worker keeps the
# CUDA context warm across all requests.
_tts_stream_executor: ThreadPoolExecutor | None = None

# Dedicated single-thread executor for streaming ASR.  Without this,
# concurrent WS connections dispatch ASR work to separate IO threads,
# each racing on _ASR_CUDA_STREAM (process-global singleton) leading to
# CUDA Graph capture failures.  One worker serialises all ASR ops on a
# consistent thread with a warm CUDA context.
_asr_executor: ThreadPoolExecutor | None = None


def _request_voice_kwargs(req: TTSRequest, *, backend=None) -> dict:
    """Resolve TTS kwargs for one synth call.

    Mixes (in priority order):
      request payload > runtime overrides > speaker-table default

    Returns a dict combining the backend-specific speaker kwargs (from
    :func:`speaker_kwargs_for_id`) plus ``speed`` / ``pitch_shift`` when the
    merge (request payload + runtime overrides) yields a value. Callers
    should ``**``-spread the result into ``synthesize`` / ``generate_streaming``
    and must NOT additionally pass ``speed`` / ``pitch_shift`` from the raw
    request, or runtime overrides will be silently discarded (FIX_2).

    ``backend`` is the live TTS backend (from BackendManager.acquire()); when
    omitted we fall back to ``tts_service`` / env so the helper still works
    if called outside an acquire() scope.
    """
    from app.core.tts_speakers import speaker_kwargs_for_id
    from app.core.tts_runtime import merge_tts_request_kwargs

    speaker_id = req.speaker_id if req.speaker_id is not None else req.sid
    if speaker_id is not None and req.speaker_embedding_b64:
        raise ValueError("speaker_id and speaker_embedding_b64 cannot be used together")
    if req.speaker_embedding_b64:
        try:
            return {"speaker_embedding": base64.b64decode(req.speaker_embedding_b64)}
        except Exception as exc:
            raise ValueError("Invalid base64 speaker_embedding_b64") from exc

    if backend is not None:
        model_id = backend.model_id
    else:
        from app.core import tts_service
        if tts_service.is_ready():
            model_id = tts_service.get_backend().model_id
        else:
            model_id = os.environ.get("OVS_TTS_MODEL_ID") or "qwen3-tts"

    merged = merge_tts_request_kwargs(
        request_speaker_id=speaker_id,
        request_speed=req.speed,
        request_pitch_shift=getattr(req, "pitch", None),
        model_id=model_id,
    )
    # Translate merged speaker_id into backend-specific kwargs (speaker_id
    # for preset, speaker_embedding for an embedding-typed entry, etc.).
    out: dict = speaker_kwargs_for_id(merged["speaker_id"], model_id)
    # FIX_2: thread merged speed / pitch_shift through so PATCH /admin/tts/runtime
    # actually takes effect. Only include keys that resolved to a non-None value
    # so backends keep using their intrinsic defaults when nothing was set.
    if merged.get("speed") is not None:
        out["speed"] = merged["speed"]
    if merged.get("pitch_shift") is not None:
        out["pitch_shift"] = merged["pitch_shift"]
    return out


def _get_asr_backend():
    return _asr_backend


_tts_lazy_start_lock = None  # asyncio.Lock; created on first use


def _try_tts_manager():
    """Return the TTS BackendManager if it is initialised+ready, else None.

    Kept for ASR-only profiles where TTS isn't wired at all. For LAZY_TTS the
    ``_ensure_tts_manager_started`` coroutine should be awaited first so the
    manager is in READY state before this is consulted.
    """
    try:
        from app.core.backend_manager import tts_manager  # local import; PR3 module
        mgr = tts_manager()
    except RuntimeError:
        return None
    return mgr if mgr.is_ready() else None


async def _ensure_tts_manager_started():
    """FIX_3 / FIX_3_completion: drive the TTS BackendManager to READY.

    Return values:
      * ``mgr`` — manager exists and is READY (caller must use ``acquire``).
      * ``None`` — manager was never installed (ASR-only profile, or
        ``init_backend_managers`` wasn't called). Caller may fall back to the
        legacy ``tts_service.synthesize`` path.

    Raises ``HTTPException(503)`` when the manager exists but is *not*
    serviceable — FAILED, DRAINING, RELOADING, or when ``start()`` fails to
    bring an INIT-state manager to READY. This is intentional: a FAILED
    manager indicates a configuration / resource problem and silently
    falling back to legacy ``tts_service`` would bypass the drain contract
    and mask the failure from operators.
    """
    import asyncio as _asyncio
    global _tts_lazy_start_lock
    try:
        from app.core.backend_manager import tts_manager, BackendState
        mgr = tts_manager()
    except RuntimeError:
        # Manager singleton never installed → legacy fallback is OK.
        return None

    if mgr.is_ready():
        return mgr

    # FAILED is non-recoverable here — surface as 503, never fall through to
    # legacy tts_service (which would skip drain / hide the failure).
    if mgr.state == BackendState.FAILED:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_failed", "state": "failed"},
        )

    # DRAINING / RELOADING are transient — surface 503 so the client retries.
    if mgr.state != BackendState.INIT:
        raise HTTPException(
            status_code=503,
            detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
        )

    if _tts_lazy_start_lock is None:
        _tts_lazy_start_lock = _asyncio.Lock()
    async with _tts_lazy_start_lock:
        if mgr.is_ready():
            return mgr
        if mgr.state == BackendState.FAILED:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_failed", "state": "failed"},
            )
        if mgr.state != BackendState.INIT:
            raise HTTPException(
                status_code=503,
                detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
            )
        try:
            await mgr.start()
        except Exception as exc:
            logger.exception("lazy TTS manager.start() failed")
            # start() failure flips state to FAILED. Surface 503 instead of
            # silently falling back to legacy tts_service.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "tts_manager_start_failed",
                    "state": mgr.state.value,
                    "message": str(exc),
                },
            ) from exc
    if mgr.is_ready():
        return mgr
    # Defensive: start() returned without exception but state isn't READY.
    raise HTTPException(
        status_code=503,
        detail={"error": "tts_manager_unavailable", "state": mgr.state.value},
    )


def _try_asr_manager():
    """Return the ASR BackendManager if it is initialised+ready, else None."""
    try:
        from app.core.backend_manager import asr_manager
        mgr = asr_manager()
    except RuntimeError:
        return None
    return mgr if mgr.is_ready() else None


def _get_tts_stream_executor() -> ThreadPoolExecutor:
    global _tts_stream_executor
    if _tts_stream_executor is None:
        # Phase 3b-B-4 part-4 INVESTIGATION RESULT: lifting max_workers above
        # 1 exposes a deeper bug in the C++ stateful Code2WavRunner reset
        # path. Two concurrent /tts/stream requests cause:
        #
        #   CUDA runtime error in cudaMemsetAsync(state.read.rawPointer(), ...)
        #   an illegal memory access was encountered
        #
        # The C++ engine slot pools (Phase 3b-B-1) + worker thread-dispatch
        # (Phase 3b-B-2) + per-slot Code2Wav (Phase 3b-B-4 part-2 commit
        # `5e1323f`) all carry the right per-slot data, but per-slot
        # StatefulCode2WavRunner state buffer initialization isn't actually
        # multi-slot safe yet — that's the next bottleneck to fix. Until
        # that's resolved, keep this serializing at the HTTP layer so the
        # worker never sees two in-flight requests simultaneously. The
        # `OVS_TTS_WORKER_CONCURRENCY` env is wired all the way down (engine
        # pools sized to it, worker dispatcher uses it, _WorkerIO semaphore
        # picks it up) but its only practical effect today is making the
        # cold-start eager-init less of a spike when the cap eventually
        # rises.
        _tts_stream_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-stream"
        )
    return _tts_stream_executor


def _get_asr_executor() -> ThreadPoolExecutor:
    global _asr_executor
    if _asr_executor is None:
        _asr_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="asr-stream"
        )
    return _asr_executor


def _default_vad_backend() -> str:
    return (
        os.environ.get("OVS_VAD_BACKEND")
        or "silero"
    ).strip() or "silero"


def _default_vad_silence_ms() -> int:
    raw = os.environ.get("OVS_VAD_SILENCE_MS") or "400"
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid OVS_VAD_SILENCE_MS=%r; using 400", raw)
        return 400
    return max(0, value)

@app.on_event("startup")
async def startup():
    global _asr_backend

    try:
        from app.core.profile_loader import apply_profile_from_env, current_profile
        apply_profile_from_env()
    except Exception as exc:
        logger.error("Failed to apply OpenVoiceStream profile: %s", exc)
        raise

    # Initialise the execution coordinator from the loaded profile's
    # execution_policy block. Default to concurrent (no lock) when the
    # profile does not declare one — matches the previous behaviour.
    from app.core.coordinator import init_coordinator, get_coordinator
    init_coordinator(current_profile().get("execution_policy", {"mode": "concurrent"}))

    # Rockchip userspace runtime is vendored in the RK image. Validate it
    # before importing rkvoice-stream backends so version/hash mismatches fail
    # with a clear operator action instead of opaque native runtime errors.
    if (current_profile().get("env") or {}).get("LANGUAGE_MODE") == "rk" or os.environ.get("LANGUAGE_MODE") == "rk":
        from app.core.rk_runtime import check_rk_runtime
        check_rk_runtime(current_profile())

    # Log language mode configuration
    language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    logger.info("=" * 60)
    logger.info("LANGUAGE_MODE: %s", language_mode)
    logger.info(
        "VAD default: backend=%s silence_ms=%d",
        _default_vad_backend(),
        _default_vad_silence_ms(),
    )
    if language_mode == "multilanguage":
        logger.info("  → Using Qwen3 TTS + ASR (52 languages, voice cloning)")
    else:
        logger.info("  → Using Sherpa TTS + ASR (zh/en mode)")
    logger.info("=" * 60)

    from app.core import model_downloader
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    model_downloader.ensure_models(language_mode, model_dir)

    # Resolve any TRT engines declared by the active profile. Must run
    # AFTER model_downloader (ONNX inputs may be needed for fallback
    # compile) and BEFORE any backend module is imported by the factories
    # (backends read engine paths from env vars at module import time).
    try:
        from app.core.engine_resolver import resolve_all
        resolved = resolve_all(current_profile())
        if resolved:
            logger.info("engine_resolver: resolved %d engine(s)", len(resolved))
            for env_var, path in resolved.items():
                logger.info("  %s → %s", env_var, path)
    except Exception as exc:
        logger.error("engine_resolver failed: %s", exc)
        raise

    # ASR backend (load before TTS to avoid ORT session conflicts)
    # Note: create_asr_backend() will auto-select based on LANGUAGE_MODE
    try:
        from app.core.asr_backend import create_asr_backend
        _asr_backend = create_asr_backend()  # Let it auto-detect from LANGUAGE_MODE
        logger.info("Pre-loading ASR (%s)...", _asr_backend.name)
        _asr_backend.preload()
        logger.info("ASR backend: %s (capabilities: %s)",
                     _asr_backend.name, [c.value for c in _asr_backend.capabilities])

        # Warm up ASR executor thread so its CUDA per-thread context is
        # initialised before the first streaming request.  Without this the
        # very first accept_waveform pays a cold-context tax on encoder.
        # SKIP_ASR_WARMUP=1 skips this on memory-constrained devices (Nano 8GB):
        # saves ~300-400 MB at startup, costs ~100ms one-time cold-context tax
        # on the very first ASR request.
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("ASR streaming warmup skipped (SKIP_ASR_WARMUP set).")
        else:
            _asyncio = __import__("asyncio")
            _executor = _get_asr_executor()

            def _warm_asr():
                # Some backends (e.g. SherpaASRBackend) don't expose a
                # transcribe_audio convenience method; their warmup is
                # implicit in preload(). Skip silently to avoid log noise.
                if not hasattr(_asr_backend, "transcribe_audio"):
                    logger.info(
                        "ASR warmup skipped: %s has no transcribe_audio (preload already warmed).",
                        type(_asr_backend).__name__,
                    )
                    return
                try:
                    import numpy as _np
                    silence = _np.zeros(16000, dtype=_np.float32)
                    _asr_backend.transcribe_audio(silence)
                    logger.info("ASR streaming executor warmed up (1 thread, CUDA primed).")
                except Exception as exc:
                    logger.warning("ASR warm-up failed: %s", exc)

            await _asyncio.get_event_loop().run_in_executor(_executor, _warm_asr)
    except Exception as e:
        logger.warning("ASR backend failed: %s", e)

    from app.core import tts_service
    if not tts_service.is_configured():
        logger.info("ASR-only mode: profile declares no tts_backend; TTS endpoints will return 503.")
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS preload skipped (LAZY_TTS set); will load on first request.")
    else:
        logger.info("Pre-loading TTS model...")
        tts_service.preload()

    # Warm up the dedicated streaming-TTS executor thread so its CUDA
    # per-thread context is initialized before the first /tts/stream
    # request lands. Without this, the very first streaming request
    # pays a ~30ms cold-context tax on prefill.
    # Skip when LAZY_TTS or ASR-only — TTS not loaded yet, can't warm what isn't there.
    if not tts_service.is_configured():
        pass  # ASR-only mode, no TTS warmup
    elif os.environ.get("LAZY_TTS", "").lower() in ("1", "true", "yes"):
        logger.info("TTS streaming warmup skipped (LAZY_TTS).")
    else:
      try:
        from app.core.tts_backend import TTSCapability
        if tts_service.has_capability(TTSCapability.STREAMING):
            backend = tts_service.get_backend()
            executor = _get_tts_stream_executor()

            def _warm_stream():
                try:
                    # Run one tiny streaming synthesis on the executor
                    # thread to materialize CUDA context state.
                    stream_kwargs = {}
                    profile = os.environ.get("EDGE_LLM_TTS_WARMUP_STREAMING_PROFILE")
                    if profile:
                        stream_kwargs["streaming_profile"] = profile
                    warmup_text = os.environ.get("EDGE_LLM_TTS_WARMUP_TEXT", "你好")
                    for _ in backend.generate_streaming(warmup_text, **stream_kwargs):
                        pass
                except Exception as exc:  # pragma: no cover
                    logger.warning("TTS streaming warm-up failed: %s", exc)

            import asyncio as _asyncio
            await _asyncio.get_event_loop().run_in_executor(executor, _warm_stream)
            logger.info("TTS streaming executor warmed up (1 thread, CUDA primed).")
      except Exception as exc:  # pragma: no cover
        logger.warning("TTS streaming executor warm-up skipped: %s", exc)

    # Register backend getters with the coordinator so 'exclusive' policy can
    # call unload() on the dormant slot. Lambdas resolve lazily so they cope
    # with backends loaded after this point (LAZY_TTS).
    try:
        from app.core import tts_service as _tts_service_mod
        coord = get_coordinator()
        coord.register_backend("asr", lambda: _asr_backend)
        coord.register_backend("tts", lambda: _tts_service_mod._backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("Coordinator backend registration skipped: %s", exc)

    # ── BackendManager wiring (PR4) ─────────────────────────────────────
    # Wrap the already-preloaded ASR/TTS instances in lifecycle managers so
    # /admin/backend/reload + acquire()-based request gating can drain
    # inflight work and hot-swap backends. The factories below return the
    # *current* singleton; on a reload the manager will call them again
    # after unloading the previous one, by which point tts_service /
    # _asr_backend have been re-bound to fresh instances. preloader is a
    # no-op on initial start (already loaded above); on reload the factory
    # invokes the real backend factory which performs its own preload.
    try:
        from app.core import backend_manager as _bm
        from app.core import tts_service as _tts_service_mod
        from app.core.asr_backend import create_asr_backend as _create_asr
        from app.core.tts_backend import create_tts_backend as _create_tts

        # On reload, build a fresh instance and rebind the legacy module
        # globals so downstream code (which still reads tts_service /
        # _asr_backend directly) sees the new backend.
        def _asr_factory():
            global _asr_backend
            if _asr_backend is None:
                _asr_backend = _create_asr()
            return _asr_backend

        def _asr_preloader(b):
            # Initial start: already preloaded above. On reload, the
            # factory returns a freshly constructed (un-preloaded)
            # instance, so we call preload() here.
            if not b.is_ready():
                b.preload()

        def _asr_unloader(b):
            global _asr_backend
            try:
                b.unload()
            finally:
                _asr_backend = None

        def _tts_factory():
            if _tts_service_mod._backend is None:
                _tts_service_mod._backend = _create_tts()
            return _tts_service_mod._backend

        def _tts_preloader(b):
            if not b.is_ready():
                b.preload()

        def _tts_unloader(b):
            try:
                b.unload()
            finally:
                _tts_service_mod._backend = None

        # FIX_4_completion: seed both managers with the profile ref used at
        # startup (OVS_PROFILE_JSON / OVS_PROFILE / OVS_PROFILE_DEFAULT). Same
        # precedence as profile_loader.apply_profile_from_env so rollback
        # re-applies via the identical source.
        _initial_profile_ref = (
            os.environ.get("OVS_PROFILE_JSON")
            or os.environ.get("OVS_PROFILE")
            or os.environ.get("OVS_PROFILE_DEFAULT")
        )
        _bm.init_backend_managers(
            tts_factory=_tts_factory,
            tts_preloader=_tts_preloader,
            tts_unloader=_tts_unloader,
            asr_factory=_asr_factory,
            asr_preloader=_asr_preloader,
            asr_unloader=_asr_unloader,
            initial_profile_ref=_initial_profile_ref,
        )

        # Bring up managers. ASR is always started if a backend exists;
        # TTS respects ASR-only profiles and LAZY_TTS env (matches the
        # legacy preload skip above).
        if _asr_backend is not None:
            await _bm.asr_manager().start()
        if tts_service.is_configured() and tts_service.is_ready():
            await _bm.tts_manager().start()
    except Exception as exc:  # pragma: no cover
        logger.warning("BackendManager wiring skipped: %s", exc)

    logger.info("Speech service ready.")


# ── Health & Capabilities ────────────────────────────────────────

@app.get("/health")
async def health():
    from app.core import tts_service

    result = {
        "tts": tts_service.is_ready(),
        "tts_backend": tts_service.backend_name() if tts_service.is_ready() else None,
        "tts_capabilities": [c.value for c in tts_service.capabilities()] if tts_service.is_ready() else [],
    }

    # Part D disconnect-watcher instrumentation: expose _WorkerIO cancel
    # counter so stress harness can read it. Temporary; remove once stable.
    try:
        from app.backends.jetson.trt_edge_llm_tts import _WorkerIO
        with _WorkerIO._cancel_count_lock:
            result["tts_worker_cancel_count"] = _WorkerIO._cancel_count
    except Exception:
        pass

    # ASR
    try:
        from app.core.asr_backend import create_asr_backend
        asr_be = _get_asr_backend()
        result["asr"] = asr_be.is_ready() if asr_be else False
        result["asr_backend"] = asr_be.name if asr_be and asr_be.is_ready() else None
        result["asr_capabilities"] = [c.value for c in asr_be.capabilities] if asr_be and asr_be.is_ready() else []
        if asr_be and asr_be.is_ready() and hasattr(asr_be, "providers"):
            result["asr_providers"] = asr_be.providers
    except Exception:
        result["asr"] = False
        result["asr_backend"] = None
        result["asr_capabilities"] = []

    return result


@app.get("/asr/capabilities")
async def asr_capabilities():
    """Return ASR backend info and supported capabilities."""
    asr_be = _get_asr_backend()
    if not asr_be or not asr_be.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)
    caps = {
        "backend": asr_be.name,
        "capabilities": [c.value for c in asr_be.capabilities],
        "sample_rate": asr_be.sample_rate,
    }
    if hasattr(asr_be, "providers"):
        caps["providers"] = asr_be.providers
    return caps


@app.get("/tts/capabilities")
async def tts_capabilities():
    """Return TTS backend info and supported capabilities."""
    from app.core import tts_service
    from app.core.tts_speakers import available_speakers
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    backend = tts_service.get_backend()
    return {
        "backend": tts_service.backend_name(),
        "model_id": backend.model_id,
        "capabilities": [c.value for c in tts_service.capabilities()],
        "sample_rate": tts_service.get_sample_rate(),
        "speakers": available_speakers(backend.model_id),
    }


# ── Speaker Management ─────────────────────────────────────────────


class RegisterSpeakerRequest(BaseModel):
    speaker_embedding_b64: str
    label: str | None = None
    speaker_id: int | None = None


@app.get("/tts/speakers")
async def tts_speakers_list():
    """List all speakers registered for the active TTS model."""
    from app.core import tts_service
    from app.core.tts_speakers import available_speakers, default_speaker_id
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    backend = tts_service.get_backend()
    return {
        "model_id": backend.model_id,
        "default_speaker_id": default_speaker_id(backend.model_id),
        "speakers": available_speakers(backend.model_id),
    }


@app.post("/tts/speakers/register")
async def tts_speakers_register(req: RegisterSpeakerRequest):
    """Register a voice-clone embedding as a persistent speaker.

    Accepts a base64-encoded speaker embedding (from /tts/clone/embedding)
    and assigns it a permanent speaker_id for subsequent /tts calls.
    """
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.tts_speakers import register_speaker

    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone"},
            status_code=501,
        )

    try:
        emb = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    backend = tts_service.get_backend()
    try:
        spec = register_speaker(
            model_id=backend.model_id,
            payload=req.speaker_embedding_b64,
            label=req.label or "",
            meta={"dim": len(emb) // 4, "dtype": "float32"},
            speaker_id=req.speaker_id,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {
        "speaker_id": spec.id,
        "type": spec.type,
        "label": spec.label,
        "model_id": backend.model_id,
    }


@app.delete("/tts/speakers/{speaker_id}")
async def tts_speakers_delete(speaker_id: int):
    """Delete a registered embedding speaker. Preset speakers cannot be deleted."""
    from app.core import tts_service
    from app.core.tts_speakers import unregister_speaker
    if not tts_service.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    backend = tts_service.get_backend()
    try:
        ok = unregister_speaker(backend.model_id, speaker_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not ok:
        return JSONResponse({"error": f"Speaker {speaker_id} not found"}, status_code=404)
    return {"deleted": True, "speaker_id": speaker_id}


# ── TTS ──────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(req: TTSRequest):
    from app.core import tts_service
    from app.core.coordinator import get_coordinator

    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        async with mgr.acquire() as backend:
            try:
                voice_kwargs = _request_voice_kwargs(req, backend=backend)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            async with get_coordinator().acquire("tts"):
                # FIX_2: speed/pitch_shift come from voice_kwargs (merged with
                # runtime overrides). Do NOT pass req.speed/req.pitch directly.
                wav_bytes, meta = backend.synthesize(
                    text=req.text,
                    language=req.language,
                    **voice_kwargs,
                )
    else:
        # Manager not initialised (ASR-only or wiring failed at startup) —
        # legacy tts_service path. Kept for ASR-only profiles where the
        # TTS manager is intentionally never started; LAZY_TTS is now handled
        # by _ensure_tts_manager_started above.
        try:
            voice_kwargs = _request_voice_kwargs(req)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        async with get_coordinator().acquire("tts"):
            wav_bytes, meta = tts_service.synthesize(
                text=req.text,
                language=req.language,
                **voice_kwargs,
            )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


@app.options("/tts/stream")
async def tts_stream_options():
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest, request: Request):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    import asyncio
    import struct
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.coordinator import get_coordinator

    # FIX_1+FIX_3: prefer the BackendManager path so /admin/backend/reload's
    # drain logic sees streaming requests in flight. Fall back to the legacy
    # tts_service path only when the manager isn't initialised (ASR-only).
    mgr = await _ensure_tts_manager_started()

    # Capability gate uses tts_service so the response shape stays consistent
    # — both paths read the same underlying backend.
    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    # Sentence-level streaming: split the request text into sentences (via
    # pysbd when the language is supported, regex fallback otherwise) and
    # call the TTS backend per sentence.
    from app.core.v2v import SentenceBuffer
    sbuf = SentenceBuffer(language=req.language)
    sentences = list(sbuf.add(req.text or "")) + list(sbuf.flush())

    if mgr is not None:
        # Acquire OUTSIDE the generator so inflight_http is bumped synchronously
        # at endpoint entry — otherwise it would only increment when the client
        # starts iterating the StreamingResponse, and reload drain could miss it.
        acquire_cm = mgr.acquire()
        backend = await acquire_cm.__aenter__()
        try:
            try:
                voice_kwargs = _request_voice_kwargs(req, backend=backend)
            except ValueError as exc:
                await acquire_cm.__aexit__(None, None, None)
                return JSONResponse({"error": str(exc)}, status_code=400)
            sr = backend.sample_rate

            async def stream():
                # Part D disconnect watcher (spec §3): Starlette cancellation
                # does not reliably close the inner sync generator running in
                # _tts_stream_executor — so poll request.is_disconnected()
                # every 100 ms and explicitly close the generator on
                # disconnect. The for-loop break path in _run calls .close()
                # on the wrapped generator, which raises GeneratorExit into
                # _generate_streaming_single() and triggers
                # _WorkerIO.cancel(req_id) (trt_edge_llm_tts.py:1255-1269).
                import threading as _threading
                cancel_flag = _threading.Event()
                gen_holder: list = [None]
                gen_lock = _threading.Lock()
                watcher_task: asyncio.Task | None = None

                async def _disconnect_watcher():
                    # Directly drain the ASGI receive channel; Starlette's
                    # is_disconnected() uses a tight cancel-scope that often
                    # misses uvicorn's http.disconnect events under
                    # StreamingResponse on Python 3.10. Blocking on raw
                    # request.receive() is reliable: uvicorn pushes
                    # http.disconnect there as soon as the socket closes.
                    logger.info("tts/stream: disconnect watcher started")
                    try:
                        while not cancel_flag.is_set():
                            try:
                                message = await request.receive()
                            except Exception:
                                logger.debug(
                                    "disconnect watcher receive() failed",
                                    exc_info=True,
                                )
                                return
                            if message.get("type") == "http.disconnect":
                                cancel_flag.set()
                                with gen_lock:
                                    g = gen_holder[0]
                                if g is not None:
                                    try:
                                        g.close()
                                    except Exception:
                                        logger.debug(
                                            "disconnect watcher gen.close() raised",
                                            exc_info=True,
                                        )
                                logger.info(
                                    "tts/stream: client disconnected — cancel flag raised"
                                )
                                return
                    except asyncio.CancelledError:
                        pass

                try:
                    async with get_coordinator().acquire("tts"):
                        yield struct.pack("<I", sr)
                        if not sentences:
                            return
                        loop = asyncio.get_event_loop()
                        watcher_task = asyncio.create_task(_disconnect_watcher())
                        for sentence in sentences:
                            if cancel_flag.is_set():
                                break
                            queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                            def _run(text=sentence):
                                gen = None
                                try:
                                    gen = backend.generate_streaming(
                                        text,
                                        language=req.language,
                                        **voice_kwargs,
                                    )
                                    with gen_lock:
                                        gen_holder[0] = gen
                                    for chunk in gen:
                                        if cancel_flag.is_set():
                                            break
                                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                                except Exception:
                                    logger.exception("tts/stream synthesis failed for sentence=%r", text)
                                finally:
                                    # Explicit close → triggers GeneratorExit
                                    # in _generate_streaming_single, which
                                    # calls worker_io.cancel(req_id).
                                    if gen is not None:
                                        try:
                                            gen.close()
                                        except Exception:
                                            logger.debug(
                                                "gen.close() in _run raised",
                                                exc_info=True,
                                            )
                                    with gen_lock:
                                        gen_holder[0] = None
                                    loop.call_soon_threadsafe(queue.put_nowait, None)

                            loop.run_in_executor(_get_tts_stream_executor(), _run)
                            while True:
                                chunk = await queue.get()
                                if chunk is None:
                                    break
                                yield chunk
                finally:
                    if watcher_task is not None:
                        watcher_task.cancel()
                        try:
                            await watcher_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    await acquire_cm.__aexit__(None, None, None)

            return StreamingResponse(stream(), media_type="application/octet-stream")
        except Exception:
            await acquire_cm.__aexit__(None, None, None)
            raise

    # Manager not initialised — legacy direct-backend path.
    backend = tts_service.get_backend()
    sr = tts_service.get_sample_rate()
    try:
        voice_kwargs = _request_voice_kwargs(req, backend=backend)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if not sentences:
        async def empty():
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
        return StreamingResponse(empty(), media_type="application/octet-stream")

    async def stream_legacy():
        # Part D disconnect watcher — mirrors the manager-branch logic above.
        import threading as _threading
        cancel_flag = _threading.Event()
        gen_holder: list = [None]
        gen_lock = _threading.Lock()
        watcher_task: asyncio.Task | None = None

        async def _disconnect_watcher():
            logger.info("tts/stream (legacy): disconnect watcher started")
            try:
                while not cancel_flag.is_set():
                    try:
                        message = await request.receive()
                    except Exception:
                        logger.debug(
                            "legacy disconnect watcher receive() failed",
                            exc_info=True,
                        )
                        return
                    if message.get("type") == "http.disconnect":
                        cancel_flag.set()
                        with gen_lock:
                            g = gen_holder[0]
                        if g is not None:
                            try:
                                g.close()
                            except Exception:
                                logger.debug(
                                    "legacy disconnect watcher gen.close() raised",
                                    exc_info=True,
                                )
                        logger.info(
                            "tts/stream (legacy): client disconnected — cancel flag raised"
                        )
                        return
            except asyncio.CancelledError:
                pass

        try:
            async with get_coordinator().acquire("tts"):
                yield struct.pack("<I", sr)
                loop = asyncio.get_event_loop()
                watcher_task = asyncio.create_task(_disconnect_watcher())
                for sentence in sentences:
                    if cancel_flag.is_set():
                        break
                    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                    def _run(text=sentence):
                        gen = None
                        try:
                            gen = backend.generate_streaming(
                                text,
                                language=req.language,
                                **voice_kwargs,
                            )
                            with gen_lock:
                                gen_holder[0] = gen
                            for chunk in gen:
                                if cancel_flag.is_set():
                                    break
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        except Exception:
                            logger.exception("tts/stream synthesis failed for sentence=%r", text)
                        finally:
                            if gen is not None:
                                try:
                                    gen.close()
                                except Exception:
                                    logger.debug(
                                        "legacy gen.close() in _run raised",
                                        exc_info=True,
                                    )
                            with gen_lock:
                                gen_holder[0] = None
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    loop.run_in_executor(_get_tts_stream_executor(), _run)

                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        yield chunk
        finally:
            if watcher_task is not None:
                watcher_task.cancel()
                try:
                    await watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(stream_legacy(), media_type="application/octet-stream")


# ── Voice Clone ───��──────────────────────────────────────────────

@app.post("/tts/clone")
async def tts_clone(req: CloneRequest):
    """Synthesize with voice cloning. Requires voice_clone capability."""
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability
    from app.core.coordinator import get_coordinator

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    # FIX_1: route through manager.acquire() so reload drain sees this request.
    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        async with mgr.acquire() as backend:
            async with get_coordinator().acquire("tts"):
                wav_bytes, meta = backend.clone_voice(
                    text=req.text,
                    speaker_embedding=speaker_embedding,
                    language=req.language,
                )
    else:
        async with get_coordinator().acquire("tts"):
            wav_bytes, meta = tts_service.clone_voice(
                text=req.text,
                speaker_embedding=speaker_embedding,
                language=req.language,
            )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta.get("duration", meta.get("duration_s", 0))),
            "X-Inference-Time": str(meta.get("inference_time", meta.get("inference_time_s", 0))),
            "X-RTF": str(meta.get("rtf", 0)),
        },
    )


@app.post("/tts/clone/embedding")
async def tts_extract_embedding(file: UploadFile = File(...)):
    """Extract speaker embedding from reference audio WAV.

    Returns base64-encoded speaker embedding that can be reused
    across multiple /tts/clone calls.
    """
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    audio_bytes = await file.read()
    from app.core.coordinator import get_coordinator
    async with get_coordinator().acquire("tts"):
        embedding = tts_service.extract_speaker_embedding(audio_bytes)
    return {
        "speaker_embedding_b64": base64.b64encode(embedding).decode(),
        "embedding_size": len(embedding),
    }


@app.post("/tts/clone/stream")
async def tts_clone_stream(req: CloneStreamRequest):
    """Stream TTS with voice cloning.

    Returns raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks.
    Requires voice_clone capability.
    """
    import asyncio
    import struct
    import base64
    from app.core import tts_service
    from app.core.tts_backend import TTSCapability

    if not tts_service.has_capability(TTSCapability.VOICE_CLONE):
        return JSONResponse(
            {"error": "Voice cloning not supported by current backend",
             "required_capability": "voice_clone",
             "backend": tts_service.backend_name()},
            status_code=501,
        )

    if not tts_service.has_capability(TTSCapability.STREAMING):
        return JSONResponse(
            {"error": "Streaming not supported by current backend",
             "required_capability": "streaming"},
            status_code=501,
        )

    try:
        speaker_embedding = base64.b64decode(req.speaker_embedding_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 speaker_embedding_b64"}, status_code=400)

    from app.core.coordinator import get_coordinator

    stream_kwargs: dict = {
        "speaker_embedding": speaker_embedding,
        "language": req.language,
    }
    if req.first_chunk_frames is not None:
        stream_kwargs["first_chunk_frames"] = req.first_chunk_frames
    if req.chunk_frames is not None:
        stream_kwargs["chunk_frames"] = req.chunk_frames
    if req.streaming_profile is not None:
        stream_kwargs["streaming_profile"] = req.streaming_profile

    # FIX_1: enter manager.acquire() at endpoint scope so reload drain
    # observes the inflight streaming request immediately.
    mgr = await _ensure_tts_manager_started()
    if mgr is not None:
        acquire_cm = mgr.acquire()
        backend = await acquire_cm.__aenter__()
        try:
            sr = backend.sample_rate

            async def stream():
                try:
                    async with get_coordinator().acquire("tts"):
                        yield struct.pack("<I", sr)
                        loop = asyncio.get_event_loop()
                        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

                        def _run():
                            try:
                                for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                            except Exception:
                                logger.exception("tts/clone/stream synthesis failed")
                            finally:
                                loop.call_soon_threadsafe(queue.put_nowait, None)

                        loop.run_in_executor(_get_tts_stream_executor(), _run)
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break
                            yield chunk
                finally:
                    await acquire_cm.__aexit__(None, None, None)

            return StreamingResponse(stream(), media_type="application/octet-stream")
        except Exception:
            await acquire_cm.__aexit__(None, None, None)
            raise

    # Legacy fallback (manager not initialised).
    sr = tts_service.get_sample_rate()
    backend = tts_service.get_backend()

    async def stream_legacy():
        async with get_coordinator().acquire("tts"):
            yield struct.pack("<I", sr)
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            def _run():
                try:
                    for chunk in backend.generate_streaming(req.text, **stream_kwargs):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception:
                    logger.exception("tts/clone/stream synthesis failed")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(_get_tts_stream_executor(), _run)

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

    return StreamingResponse(stream_legacy(), media_type="application/octet-stream")


# ── ASR ──────────────────────────────────────────────────────────

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    audio_bytes = await file.read()

    from app.core.coordinator import get_coordinator
    mgr = _try_asr_manager()
    if mgr is not None:
        async with mgr.acquire() as asr_be:
            async with get_coordinator().acquire("asr"):
                result = asr_be.transcribe(audio_bytes, language=language)
            return {
                "text": result.text,
                "language": result.language,
                "backend": asr_be.name,
                **result.meta,
            }
    asr_be = _get_asr_backend()
    if asr_be and asr_be.is_ready():
        async with get_coordinator().acquire("asr"):
            result = asr_be.transcribe(audio_bytes, language=language)
        return {
            "text": result.text,
            "language": result.language,
            "backend": asr_be.name,
            **result.meta,
        }
    else:
        return JSONResponse(
            status_code=503,
            content={"error": "ASR backend not available"},
        )


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
    vad: Optional[str] = None,           # default from OVS_VAD_BACKEND
    vad_silence_ms: Optional[int] = None,
):
    """Streaming ASR via WebSocket.

    Client sends: raw int16 PCM bytes
    Client sends: empty bytes b"" to signal end
    Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}

    Server-side VAD auto-finalizes on silence — client no longer needs to
    send the empty b"" frame in open-mic dialogue mode. ``vad_silence_ms``
    controls the silence threshold (default 400 ms). Pass ?vad=none for
    legacy forced-EOS-only behavior.

    Requires an ASR backend with STREAMING capability.
    """
    import asyncio
    import numpy as np
    from app.core.asr_backend import ASRCapability

    await ws.accept()

    # Register this WS session with the BackendManager (if available) so a
    # subsequent /admin/backend/reload can force-close it (code 1012) and
    # cancel the handler task instead of waiting forever for drain.
    _asr_mgr = _try_asr_manager()
    _ws_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _asr_mgr is not None:
        _asr_mgr.register_ws(_ws_handle)
    vad_backend = vad if vad is not None else _default_vad_backend()
    vad_silence = _default_vad_silence_ms() if vad_silence_ms is None else max(0, int(vad_silence_ms))

    # Choose backend: prefer ASR backend with STREAMING, fall back to sherpa
    asr_be = _get_asr_backend()
    use_backend_stream = (
        asr_be is not None
        and asr_be.is_ready()
        and asr_be.has_capability(ASRCapability.STREAMING)
    )

    # Lazy-init server-side VAD only if requested. Reuses the shared
    # singleton model from app.core.vad — no extra model load per
    # connection.
    vad_session = None
    if vad_backend and vad_backend not in ("none", "off", "disabled"):
        try:
            from app.core import vad as vad_mod
            vad_session = vad_mod.create_vad(
                vad_backend, sample_rate=sample_rate, silence_ms=vad_silence
            )
        except Exception as e:
            logger.warning("VAD '%s' init failed (%s); falling back to forced-EOS", vad_backend, e)
            vad_session = None

    try:
        if use_backend_stream:
            from app.core.coordinator import get_coordinator
            async with get_coordinator().acquire("asr"):
                await _asr_stream_backend(ws, asr_be, language, sample_rate, vad_session)
        else:
            await ws.send_json({"error": "no streaming ASR available"})
            await ws.close()
    finally:
        if _asr_mgr is not None:
            _asr_mgr.unregister_ws(_ws_handle)


async def _asr_stream_backend(
    ws: WebSocket,
    asr_be,
    language: str,
    sample_rate: int,
    vad_session=None,
):
    """Streaming ASR using ASR backend (accumulate-then-transcribe).

    Supports a ``reset`` control command: the client may send a JSON text
    message ``{"command": "reset"}`` at any time.  This discards the
    current stream and creates a fresh one without closing the WebSocket.
    """
    import asyncio
    import json as _json
    import numpy as np

    stream = asr_be.create_stream(language=language)
    logger.info("ASR stream opened (backend=%s)", asr_be.name)

    try:
        while True:
            msg = await ws.receive()

            # ── Text message: control command ──
            if "text" in msg and msg["text"]:
                try:
                    cmd = _json.loads(msg["text"])
                except (ValueError, TypeError):
                    continue
                if cmd.get("command") == "reset":
                    stream = asr_be.create_stream(language=language)
                    await ws.send_json({
                        "type": "reset",
                        "text": "",
                        "is_final": True,
                        "is_stable": True,
                        "reset": True,
                    })
                    logger.debug("ASR stream reset by client command (backend=%s)", asr_be.name)
                elif cmd.get("command") == "end_utterance" or (cmd.get("type") or "").lower() == "eou":
                    _loop = asyncio.get_event_loop()
                    force_endpoint = getattr(stream, "force_endpoint", None)
                    if force_endpoint is not None:
                        final_text = await _loop.run_in_executor(_get_asr_executor(), force_endpoint)
                    else:
                        await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                        final_text = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                    await ws.send_json({
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "is_stable": True,
                    })
                    logger.debug("ASR utterance endpoint forced (backend=%s)", asr_be.name)
                continue

            # ── Binary message: audio data ──
            data = msg.get("bytes", b"")
            if data is None:
                # WebSocket disconnect frame — no bytes key
                break

            if len(data) == 0:
                # End of audio — pre-encode tail, then decode
                _loop = asyncio.get_event_loop()
                await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                final_text = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                await ws.send_json({
                    "type": "final",
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                })
                break

            # Buffer audio (run in thread to avoid blocking event loop)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            _loop = asyncio.get_event_loop()
            await _loop.run_in_executor(_get_asr_executor(), stream.accept_waveform, sample_rate, samples)

            # Server-side VAD endpoint detection (opt-in via ?vad=)
            if vad_session is not None:
                from app.core.vad import VADSession
                event = vad_session.process(samples)
                if event == VADSession.SPEECH_END:
                    # Emit vad_endpoint BEFORE finalize so the client can split
                    # VAD silence-wait from ASR compute time.
                    await ws.send_json({"type": "vad_endpoint"})
                    await _loop.run_in_executor(_get_asr_executor(), stream.prepare_finalize)
                    final_text = await _loop.run_in_executor(_get_asr_executor(), stream.finalize)
                    try:
                        await ws.send_json({
                            "type": "final",
                            "text": final_text,
                            "is_final": True,
                            "is_stable": True,
                            "endpoint": "vad",
                        })
                    except Exception:
                        # Client may have disconnected during a slow finalize
                        # (e.g. TRT-EdgeLLM on Jetson). Nothing to send to.
                        pass
                    break

            # Check for partial results
            partial_text, is_endpoint = stream.get_partial()
            if partial_text:
                if is_endpoint:
                    await ws.send_json({
                        "type": "final",
                        "text": partial_text,
                        "is_final": True,
                        "is_stable": True,
                    })
                else:
                    await ws.send_json({
                        "type": "partial",
                        "text": partial_text,
                        "is_final": False,
                        "is_stable": False,
                    })

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected (backend=%s)", asr_be.name)
    except Exception as e:
        logger.error("ASR stream error (backend=%s): %s", asr_be.name, e, exc_info=True)
        # Surface backend failures to clients as structured terminal frames.
        # Otherwise benchmark clients only observe a socket close and lose the
        # actual failure reason.
        try:
            await ws.send_json({
                "type": "error",
                "error": f"{type(e).__name__}: {e}",
                "backend": asr_be.name,
                "is_final": True,
                "is_stable": True,
            })
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Unified V2V WebSocket: ASR + TTS + VAD + barge-in
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/v2v/stream")
async def v2v_stream(ws: WebSocket):
    """Unified bi-directional WebSocket: speech in, partials + audio out.

    Client may enable any subset of features via the first ``config``
    JSON frame. See ``docs/api/v2v-stream.md`` for the protocol spec.
    Minimum viable patterns:

      TTS-only (LLM token stream → audio):
        send {"type":"config", "tts_language":"zh"}
        send {"type":"text", "text":"..."} repeatedly
        send {"type":"tts_flush"}; await binary chunks + tts_done

      ASR-only (mic → text, with auto VAD endpoint):
        send {"type":"config", "asr_language":"zh"}
        send PCM binary chunks
        await asr_partial / asr_endpoint / asr_final

      V2V (full duplex):
        config with both asr_language + tts_language
        interleave binary (mic) with text (LLM tokens)
        receive partials, endpoints, audio
        send {"type":"abort"} to barge-in
    """
    import asyncio
    import json as _json
    import struct
    import numpy as np

    from app.core import tts_service, v2v as v2v_proto
    from app.core.asr_backend import ASRCapability
    from app.core.tts_backend import TTSCapability
    from app.core import vad as vad_mod
    from app.core.coordinator import get_coordinator

    coord = get_coordinator()

    await ws.accept()

    # Register this v2v session with whichever BackendManager(s) are
    # available so /admin/backend/reload of either kind can hard-close
    # the WS (code 1012) instead of letting the connection linger.
    _v2v_asr_mgr = _try_asr_manager()
    _v2v_tts_mgr = _try_tts_manager()
    _v2v_handle = _WSHandle(websocket=ws, task=asyncio.current_task())
    if _v2v_asr_mgr is not None:
        _v2v_asr_mgr.register_ws(_v2v_handle)
    if _v2v_tts_mgr is not None:
        _v2v_tts_mgr.register_ws(_v2v_handle)

    # ── Stage 1: receive initial config ─────────────────────────────
    try:
        first_msg = await ws.receive()
    except WebSocketDisconnect:
        return
    cfg_text = first_msg.get("text", "")
    if not cfg_text:
        await ws.close(code=1003); return
    try:
        cfg = _json.loads(cfg_text)
    except (ValueError, TypeError):
        await ws.close(code=1003); return
    if cfg.get("type") != v2v_proto.CLIENT_CONFIG:
        await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                            "error": "first message must be a config frame"})
        await ws.close(code=1003); return

    asr_language    = cfg.get("asr_language")  # e.g. "zh" / "Chinese" / "en" / "auto" / None
    tts_language    = cfg.get("tts_language")  # truthy = enable TTS; "auto" = let backend detect
    # "auto" enables TTS but tells downstream (sentence buffer + backend) to
    # not assume a language — backends with auto-detect (e.g. qwen3) will pick
    # one from the text content; SentenceBuffer falls back to a regex splitter.
    tts_language_norm = None if tts_language == "auto" else tts_language
    # Normalize common client-supplied aliases to the lowercase full names
    # qwen3 TTS expects ("chinese"/"english"/...). Sherpa TTS ignores the
    # value entirely, so this is a no-op there.
    if tts_language_norm:
        _TTS_LANG_ALIAS = {
            "zh": "chinese", "zh-cn": "chinese", "zh-hans": "chinese",
            "en": "english", "en-us": "english", "en-gb": "english",
            "ja": "japanese", "jp": "japanese",
            "ko": "korean", "kr": "korean",
        }
        key = tts_language_norm.strip().lower()
        tts_language_norm = _TTS_LANG_ALIAS.get(key, key)
    tts_voice       = cfg.get("tts_voice")
    tts_speaker_id = cfg.get("tts_speaker_id")
    tts_speed       = cfg.get("tts_speed")
    # Resolve speaker once at config time — avoids mid-session changes
    # (e.g. unregister) affecting later sentences in the same session.
    tts_speaker_kwargs: dict = {}
    if tts_speaker_id is not None and tts_language:
        from app.core.tts_speakers import speaker_kwargs_for_id
        if tts_service.is_ready():
            tts_speaker_kwargs = speaker_kwargs_for_id(
                int(tts_speaker_id), tts_service.get_backend().model_id
            )
    sample_rate     = int(cfg.get("sample_rate", 16000))
    vad_backend     = cfg.get("vad", _default_vad_backend() if asr_language else "none")
    vad_silence_ms  = int(cfg.get("vad_silence_ms", _default_vad_silence_ms()))
    multi_utterance = bool(cfg.get("multi_utterance", False))

    if not asr_language and not tts_language:
        await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                            "error": "config must enable asr_language and/or tts_language"})
        await ws.close(code=1003); return

    # ── Stage 2: bring up the backends ──────────────────────────────
    asr_be = None
    asr_manager = None  # ASRSessionManager — owns per-utterance lifecycle.
    asr_enabled = False
    vad = None
    if asr_language:
        asr_be = _get_asr_backend()
        if asr_be is None or not asr_be.is_ready() or not asr_be.has_capability(ASRCapability.STREAMING):
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "asr_language requested but no streaming ASR backend ready"})
            await ws.close(code=1011); return
        # Defer stream creation until first speech-start (or first audio
        # without VAD) — the manager creates a fresh stream per utterance.
        from app.core.asr_session_manager import ASRSessionManager
        asr_manager = ASRSessionManager(
            backend=asr_be,
            language=asr_language,
            coord=coord,
            executor=_get_asr_executor(),
        )
        asr_enabled = True
        # VAD init runs in executor: silero ONNX first-load takes ~500ms and
        # would otherwise stall the event loop. ValueError (e.g. unsupported
        # sample rate) is a hard config error → reject and close. Other init
        # failures fall back to no-VAD with a warning.
        try:
            _loop_init = asyncio.get_event_loop()
            vad = await _loop_init.run_in_executor(
                None,
                lambda: vad_mod.create_vad(vad_backend, sample_rate=sample_rate, silence_ms=vad_silence_ms),
            )
        except ValueError as e:
            await ws.send_json({"type": v2v_proto.SERVER_ERROR, "error": f"VAD config: {e}"})
            await ws.close(code=1003); return
        except Exception as e:
            logger.warning("v2v VAD init (%s) failed: %s — running without VAD", vad_backend, e)
            vad = None

    tts_be = None
    tts_buffer = None
    if tts_language:
        if not tts_service.is_ready() or not tts_service.has_capability(TTSCapability.STREAMING):
            await ws.send_json({"type": v2v_proto.SERVER_ERROR,
                                "error": "tts_language requested but no streaming TTS backend ready"})
            await ws.close(code=1011); return
        tts_be = tts_service.get_backend()
        low_latency_tts = os.environ.get("OVS_TTS_LOW_LATENCY_CHUNKING", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if low_latency_tts:
            tts_buffer = v2v_proto.LowLatencyTTSBuffer(language=tts_language_norm)
        else:
            tts_buffer = v2v_proto.SentenceBuffer(language=tts_language_norm)

    logger.info("v2v stream opened (asr=%s tts=%s vad=%s spk_id=%s spk_kwargs=%s)",
                asr_language or "off", tts_language or "off", vad_backend if asr_language else "off",
                tts_speaker_id, list(tts_speaker_kwargs.keys()) if tts_speaker_kwargs else None)

    # ── Stage 3: per-connection state + write serialization ─────────
    send_lock = asyncio.Lock()
    state = {
        # Per-utterance ASR endpoint signalling. Replaces the old
        # asr_eos / vad_endpoint / vad_endpoint_pending flags now that
        # ASRSessionManager owns the stream lifecycle.
        "asr_session_closed": False,   # client explicitly ended ASR (asr_eos or ws close)
        "endpoint_pending":   None,    # ("vad" | "client_eos"), set by dispatcher,
                                       # consumed by asr_out_task
        "endpoint_pending_gen": None,  # generation tag for endpoint_pending;
                                       # if it no longer matches asr_active_gen
                                       # by the time asr_out_task observes it,
                                       # the endpoint belongs to a preempted
                                       # utterance and must NOT fire finalize
                                       # against the new one (gen-race fix,
                                       # codex root-cause 2026-05-19).
        "tts_flush":        False,   # set when client tts_flush
        "current_tts_task": None,    # running TTS synth task (cancellable)
        "current_tts_stop": None,    # threading.Event to signal synth thread
                                     # to stop on barge-in (avoids orphan
                                     # synth blocking the TTS executor)
        "tts_started":      False,   # tts_started frame sent for current sentence
        "client_closed":    False,
        "asr_active":       False,   # tracks whether manager.on_speech_start
                                     # has been called for the current utterance
        "asr_active_gen":   0,       # generation tagged onto asr_active so a
                                     # stale finalize doesn't clear a fresh
                                     # utterance's asr_active flag (BUG 2)
        "endpoint_finalize_pending": False,  # set when dispatcher already
                                     # accepted the speech-end chunk and the
                                     # asr_out_task should finalize next tick
                                     # (BUG 3: avoids the flag-set/audio-accept
                                     # race that lost the tail of utterances)
    }
    tts_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    async def send_json(payload):
        async with send_lock:
            try:
                await ws.send_json(payload)
            except Exception:
                state["client_closed"] = True

    async def send_bytes(data):
        async with send_lock:
            try:
                await ws.send_bytes(data)
            except Exception:
                state["client_closed"] = True

    async def send_error(msg):
        await send_json({"type": v2v_proto.SERVER_ERROR, "error": msg})

    # ── Stage 4: tasks ──────────────────────────────────────────────

    async def dispatcher():
        """Receive incoming binary (audio) + text (control) frames."""
        try:
            while not state["client_closed"]:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    state["client_closed"] = True
                    break
                # binary → ASR input
                data = msg.get("bytes")
                if data:
                    if not asr_enabled:
                        continue  # ignored in TTS-only mode
                    # After session close, drop further audio. Spec: client
                    # must open a new WebSocket to start another session.
                    if state["asr_session_closed"]:
                        continue
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    speech_started_now = False
                    speech_ended_now = False
                    if vad is not None:
                        event = vad.process(samples)
                        if event == vad_mod.VADSession.SPEECH_START:
                            # Notify client FIRST so it can stop buffering /
                            # playing TTS audio, then perform the server-side
                            # barge-in (cancel in-flight TTS, open fresh ASR).
                            await send_json({
                                "type": v2v_proto.SERVER_VAD_EVENT,
                                "event": v2v_proto.VAD_EVENT_SPEECH_START,
                            })
                            # Auto barge-in: cancel any in-flight TTS, then
                            # open a fresh ASR utterance (pre-empts any
                            # still-active session per spec).
                            t = state["current_tts_task"]
                            if t is not None and not t.done():
                                t.cancel()
                            stop = state["current_tts_stop"]
                            if stop is not None:
                                stop.set()
                            async with coord.acquire("asr"):
                                new_gen = await asr_manager.on_speech_start()
                            # Clear any stale endpoint from the previous
                            # utterance — a VAD speech-end that was pending
                            # finalize while this new speech-start preempted
                            # it must NOT cause asr_out_task to call
                            # finalize() against the fresh generation
                            # (codex root-cause 2026-05-19: stale endpoint
                            # firing on the wrong generation).
                            state["endpoint_pending"] = None
                            state["endpoint_pending_gen"] = None
                            state["asr_active"] = True
                            state["asr_active_gen"] = new_gen
                            speech_started_now = True
                        elif event == vad_mod.VADSession.SPEECH_END:
                            # Defer setting endpoint_pending until AFTER we
                            # accept this final chunk below — otherwise the
                            # asr_out_task observes the flag and calls
                            # finalize() while the tail audio is still
                            # in-flight, silently dropping it (BUG 3).
                            speech_ended_now = True
                    # No-VAD mode: open the session lazily on first audio.
                    if vad is None and not state["asr_active"]:
                        async with coord.acquire("asr"):
                            new_gen = await asr_manager.on_speech_start()
                        state["endpoint_pending"] = None
                        state["endpoint_pending_gen"] = None
                        state["asr_active"] = True
                        state["asr_active_gen"] = new_gen
                    if state["asr_active"]:
                        async with coord.acquire("asr"):
                            await asr_manager.accept_audio(samples)
                    # Now safe to flag the endpoint — audio chunk that
                    # carried the speech-end has been delivered to the
                    # stream. asr_out_task will pick this up on the next
                    # poll and call finalize().
                    if speech_ended_now:
                        state["endpoint_pending"] = "vad"
                        state["endpoint_pending_gen"] = state["asr_active_gen"]
                        if not multi_utterance:
                            state["asr_session_closed"] = True
                        # Notify client of VAD speech_end so it can update
                        # its state machine (e.g. show "thinking" indicator,
                        # await asr_final). Sent AFTER endpoint_pending is
                        # latched to keep ordering deterministic w.r.t. the
                        # asr_final that follows from asr_out_task.
                        await send_json({
                            "type": v2v_proto.SERVER_VAD_EVENT,
                            "event": v2v_proto.VAD_EVENT_SPEECH_END,
                        })
                    continue
                # text → JSON control
                text = msg.get("text", "")
                if not text:
                    continue
                try:
                    payload = _json.loads(text)
                except (ValueError, TypeError):
                    continue
                typ = payload.get("type")
                if typ == v2v_proto.CLIENT_TEXT and tts_buffer is not None:
                    for sentence in tts_buffer.add(payload.get("text", "")):
                        await tts_q.put(sentence)
                elif typ == v2v_proto.CLIENT_TTS_FLUSH:
                    if tts_buffer is not None:
                        for sentence in tts_buffer.flush():
                            await tts_q.put(sentence)
                    state["tts_flush"] = True
                elif typ == v2v_proto.CLIENT_ASR_EOS:
                    state["endpoint_pending"] = "client_eos"
                    state["endpoint_pending_gen"] = state["asr_active_gen"]
                    if not multi_utterance:
                        state["asr_session_closed"] = True
                elif typ == v2v_proto.CLIENT_ABORT:
                    t = state["current_tts_task"]
                    if t is not None and not t.done():
                        t.cancel()
                    stop = state["current_tts_stop"]
                    if stop is not None:
                        stop.set()
                    # Drain queue so flush doesn't replay queued sentences
                    while not tts_q.empty():
                        try: tts_q.get_nowait()
                        except asyncio.QueueEmpty: break
                    # Cancel any in-flight ASR utterance too — spec: barge-in
                    # discards pending finals and resets to IDLE.
                    if asr_manager is not None and state["asr_active"]:
                        async with coord.acquire("asr"):
                            await asr_manager.cancel("bargein")
                        state["asr_active"] = False
        except WebSocketDisconnect:
            state["client_closed"] = True

    async def asr_out_task():
        """Drive partial polling + per-utterance finalize via the manager.

        Each utterance is its own ``ASRSessionManager`` stream. We poll
        the *active* stream (manager.stream) for partials, then on an
        endpoint trigger (VAD speech-end, client asr_eos, or backend
        is_endpoint) we call ``manager.finalize()`` which destroys the
        stream and returns the final text.
        """
        last_streamed_final = None
        while not state["client_closed"]:
            # Pull a stream snapshot under the manager's lock so we can
            # tag any partial with the generation it came from. If the
            # generation has advanced by emit-time, drop the partial —
            # it belongs to an utterance that's already been replaced
            # (BUG 4: stale-stream partial leak).
            partial, is_endpoint, partial_gen = "", False, 0
            if state["asr_active"]:
                try:
                    async with coord.acquire("asr"):
                        partial_gen, partial, is_endpoint = (
                            await asr_manager.get_partial_for_generation()
                        )
                except Exception:
                    partial, is_endpoint, partial_gen = "", False, 0
                if partial and partial_gen == asr_manager.current_generation \
                        and partial_gen == state["asr_active_gen"]:
                    await send_json({"type": v2v_proto.SERVER_ASR_PARTIAL,
                                     "text": partial, "is_stable": bool(is_endpoint)})

            endpoint_reason = state["endpoint_pending"]
            # Gen-race gate: if endpoint_pending was stamped against a
            # generation that has since been preempted (VAD speech-start
            # of a new utterance, or post-worker-restart on_speech_start),
            # drop it on the floor instead of firing finalize against the
            # *new* active utterance. Without this gate the new utterance
            # gets finalized too early and the manager rejects the result
            # with "finalize result discarded (state=ACTIVE)"
            # (codex root-cause 2026-05-19).
            if (
                endpoint_reason
                and state.get("endpoint_pending_gen") is not None
                and state.get("endpoint_pending_gen") != state["asr_active_gen"]
            ):
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                endpoint_reason = None

            endpoint_fired = (
                bool(endpoint_reason)
                or (is_endpoint and state["asr_active"])
            )

            if endpoint_fired:
                # Drain pending flag now to avoid double-firing.
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                # Emit asr_endpoint only for VAD / backend endpoints,
                # not client-driven eos.
                if endpoint_reason != "client_eos":
                    await send_json({"type": v2v_proto.SERVER_ASR_ENDPOINT})

                if state["asr_active"]:
                    finalize_gen = state["asr_active_gen"]
                    async with coord.acquire("asr"):
                        ran_gen, final_text, finalize_accepted = (
                            await asr_manager.finalize_with_status(
                                endpoint_reason or "backend_endpoint"
                            )
                        )
                    # Only clear asr_active if the generation we finalized
                    # is still the active one. If a new speech_start
                    # bumped the generation while finalize was in flight,
                    # leaving asr_active=True is correct — audio for the
                    # new utterance must continue to flow (BUG 2).
                    if finalize_accepted and state["asr_active_gen"] == finalize_gen:
                        state["asr_active"] = False
                else:
                    final_text = ""
                    ran_gen = state["asr_active_gen"]
                    finalize_accepted = True

                if not finalize_accepted:
                    logger.info(
                        "suppressing discarded asr_final from gen=%s current_gen=%s reason=%s",
                        ran_gen,
                        state["asr_active_gen"],
                        endpoint_reason or "backend_endpoint",
                    )
                    continue

                # Multi-utterance: mid-session finals carry
                # session_complete=False; close-out final on
                # asr_session_closed carries True.
                if multi_utterance:
                    is_closing = state["asr_session_closed"]
                    if is_closing:
                        duplicate = (final_text or "") == (last_streamed_final or "")
                        await send_json({
                            "type": v2v_proto.SERVER_ASR_FINAL,
                            "text": final_text or "",
                            "session_complete": True,
                            "duplicate_of_streamed": duplicate,
                        })
                        return
                    else:
                        await send_json({
                            "type": v2v_proto.SERVER_ASR_FINAL,
                            "text": final_text or "",
                            "session_complete": False,
                        })
                        last_streamed_final = final_text or ""
                        # keep the loop running for the next utterance
                else:
                    await send_json({"type": v2v_proto.SERVER_ASR_FINAL,
                                     "text": final_text or ""})
                    return

            # Exit only when the session is closed and there's nothing
            # left to finalize — single-utterance terminates above on
            # the endpoint; multi-utterance terminates on close-out.
            if state["asr_session_closed"] and not state["asr_active"]:
                return

            await asyncio.sleep(0.05)

    async def tts_out_task():
        """Drain sentence queue → synthesize → emit audio.

        Sends the 4-byte sample-rate header on first successful synth
        (NOT first attempted synth — so a cancelled-mid-flight first
        sentence doesn't leave the client with a header but no audio).
        """
        sr_header_sent = False
        while not state["client_closed"]:
            # Exit when client said flush and the queue is drained.
            if state["tts_flush"] and tts_q.empty():
                # Multi-utterance: per-turn flush ends one turn but not
                # the SESSION. Reset the sticky flag, emit a per-turn
                # tts_done (session_complete=False mirroring ASR's
                # mid-session final at :1763-1779), then loop back to
                # wait for the next turn. Without this the task returns
                # after round 1 → asyncio.gather() unblocks → WS closes,
                # which is the "TTS stuck after round 1" bug.
                if multi_utterance and not state.get("asr_session_closed", False):
                    state["tts_flush"] = False
                    if not state["client_closed"]:
                        await send_json({
                            "type": v2v_proto.SERVER_TTS_DONE,
                            "session_complete": False,
                        })
                    continue
                break
            try:
                sentence = await asyncio.wait_for(tts_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            audio_queue: asyncio.Queue = asyncio.Queue()
            # Likely #1 fix: signal the synth thread to stop mid-iteration
            # on barge-in (single-thread TTS executor would otherwise be
            # blocked by the orphaned generator until it completes).
            import threading as _threading
            stop_event = _threading.Event()
            state["current_tts_stop"] = stop_event

            def _run_synth(s, synth_be):
                try:
                    stream_kwargs = {"language": tts_language_norm}
                    if tts_speaker_kwargs:
                        stream_kwargs.update(tts_speaker_kwargs)
                    elif tts_voice is not None:
                        stream_kwargs["voice"] = tts_voice  # deprecated
                    if tts_speed is not None:    stream_kwargs["speed"] = tts_speed
                    # FIX_C: use the manager-acquired backend so a reload
                    # waiting for drain sees this request as inflight.
                    for chunk in synth_be.generate_streaming(s, **stream_kwargs):
                        if stop_event.is_set():
                            break
                        loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                except Exception as e:
                    logger.exception("v2v tts synthesis failed for sentence=%r", s)
                    loop.call_soon_threadsafe(audio_queue.put_nowait, ("__error__", str(e)))
                finally:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            async def drain():
                nonlocal sr_header_sent
                # PR5 / FIX_C: take BackendManager.acquire() *per utterance*
                # so admin reload's drain logic sees this synth as inflight.
                # Per-utterance (vs per-session) is intentional: v2v sessions
                # can run for minutes; holding acquire across the whole
                # session would block every reload until the user hangs up.
                # _v2v_tts_mgr is captured from the enclosing scope (set
                # earlier from _try_tts_manager()); fall back to the
                # already-bound tts_be when manager wiring is absent (partial
                # config / tests).
                tts_mgr_local = _v2v_tts_mgr
                if tts_mgr_local is not None:
                    acquire_cm = tts_mgr_local.acquire()
                    synth_backend = await acquire_cm.__aenter__()
                else:
                    acquire_cm = None
                    synth_backend = tts_be
                try:
                    # Coord lock per-sentence: cheap on concurrent profiles;
                    # serializes sentences against ASR on serialized profiles.
                    async with coord.acquire("tts"):
                        if not sr_header_sent:
                            sr = tts_service.get_sample_rate() if hasattr(tts_service, "get_sample_rate") else 16000
                            await send_bytes(struct.pack("<I", sr))
                            sr_header_sent = True
                        await send_json({"type": v2v_proto.SERVER_TTS_STARTED, "sentence": sentence})
                        loop.run_in_executor(_get_tts_stream_executor(), _run_synth, sentence, synth_backend)
                        state["tts_started"] = True
                        while True:
                            item = await audio_queue.get()
                            if item is None:
                                break
                            if isinstance(item, tuple) and item[0] == "__error__":
                                await send_error(f"tts: {item[1]}")
                                break
                            await send_bytes(item)
                        await send_json({"type": v2v_proto.SERVER_TTS_SENTENCE_DONE, "sentence": sentence})
                finally:
                    if acquire_cm is not None:
                        try:
                            await acquire_cm.__aexit__(None, None, None)
                        except Exception:
                            logger.exception("v2v tts acquire exit failed")

            task = asyncio.create_task(drain())
            state["current_tts_task"] = task
            try:
                await task
            except asyncio.CancelledError:
                # Barge-in: tell the synth thread to break out of the
                # generator loop, then drain any chunks it produced
                # before noticing the flag.
                stop_event.set()
                try:
                    while True:
                        item = audio_queue.get_nowait()
                        if item is None: break
                except asyncio.QueueEmpty:
                    pass
            finally:
                state["current_tts_task"] = None
                state["current_tts_stop"] = None
        if not state["client_closed"]:
            # Session-final tts_done. In multi-utterance mode tag it as
            # session_complete=True so the client can distinguish it from
            # the per-turn dones emitted above. Single-utterance mode
            # omits the field for backward compatibility.
            payload = {"type": v2v_proto.SERVER_TTS_DONE}
            if multi_utterance:
                payload["session_complete"] = True
            await send_json(payload)

    # ── Stage 5: orchestrate ────────────────────────────────────────
    # Bug #3 fix: dispatcher loops on ws.receive() forever (only exits
    # on disconnect). If we asyncio.gather all three, the server hangs
    # after asr_final / tts_done. Spawn work tasks separately, wait for
    # them, then cancel the dispatcher.
    dispatcher_task = asyncio.create_task(dispatcher())
    work_tasks = []
    if asr_enabled:
        work_tasks.append(asyncio.create_task(asr_out_task()))
    if tts_be is not None:
        work_tasks.append(asyncio.create_task(tts_out_task()))

    try:
        if work_tasks:
            await asyncio.gather(*work_tasks, return_exceptions=False)
        else:
            # No work tasks (shouldn't happen — config rejected earlier),
            # just keep the dispatcher running until the client closes.
            await dispatcher_task
    except Exception as e:
        logger.error("v2v stream error: %s", e, exc_info=True)
        try:
            await send_error(f"{type(e).__name__}: {e}")
        except Exception:
            pass
    finally:
        if not dispatcher_task.done():
            dispatcher_task.cancel()
            try:
                await dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
        for t in work_tasks:
            if not t.done():
                t.cancel()
        # Tell the synth thread to bail (if running) so the TTS executor
        # frees up for the next connection.
        stop = state["current_tts_stop"]
        if stop is not None:
            stop.set()
        # Cancel any in-flight ASR utterance before closing the socket
        # so the worker doesn't leak the session.
        if asr_manager is not None:
            try:
                await asr_manager.cancel("ws_close")
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
        if _v2v_asr_mgr is not None:
            _v2v_asr_mgr.unregister_ws(_v2v_handle)
        if _v2v_tts_mgr is not None:
            _v2v_tts_mgr.unregister_ws(_v2v_handle)
        logger.info("v2v stream closed")


# ── Admin: TTS runtime overrides ────────────────────────────────────────────

class TTSRuntimePatch(BaseModel):
    speaker_id: Optional[int] = None
    speed: Optional[float] = None
    pitch_shift: Optional[float] = None


def _current_tts_model_id() -> Optional[str]:
    from app.core import tts_service
    if not tts_service.is_ready():
        return None
    try:
        return tts_service.get_backend().model_id
    except Exception:
        return None


def _effective_tts_values(model_id: Optional[str]) -> dict:
    from app.core import tts_runtime
    from app.core.tts_speakers import default_speaker_id
    snap = tts_runtime.get_overrides()
    if snap.default_speaker_id is not None:
        eff_speaker = snap.default_speaker_id
    elif model_id is not None:
        try:
            eff_speaker = default_speaker_id(model_id)
        except Exception:
            eff_speaker = None
    else:
        eff_speaker = None
    return {
        "speaker_id": eff_speaker,
        "speed": snap.default_speed,
        "pitch_shift": snap.default_pitch_shift,
    }


def _admin_dep():
    from app.core.admin_auth import require_admin
    return require_admin


@app.get("/admin/tts/runtime")
async def admin_tts_runtime_get(_: None = Depends(_admin_dep())):
    from app.core import tts_runtime
    snap = tts_runtime.get_overrides()
    model_id = _current_tts_model_id()
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.patch("/admin/tts/runtime")
async def admin_tts_runtime_patch(
    req: TTSRuntimePatch,
    _: None = Depends(_admin_dep()),
):
    from app.core import tts_runtime
    fields = req.model_fields_set
    kwargs: dict = {}
    if "speaker_id" in fields:
        kwargs["speaker_id"] = req.speaker_id
    if "speed" in fields:
        kwargs["speed"] = req.speed
    if "pitch_shift" in fields:
        kwargs["pitch_shift"] = req.pitch_shift
    model_id = _current_tts_model_id()
    if model_id is not None:
        kwargs["model_id"] = model_id
    try:
        snap = tts_runtime.update_overrides(**kwargs)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return {
        "model_id": model_id,
        "overrides": {
            "speaker_id": snap.default_speaker_id,
            "speed": snap.default_speed,
            "pitch_shift": snap.default_pitch_shift,
            "updated_at": snap.updated_at,
        },
        "effective": _effective_tts_values(model_id),
    }


@app.post("/admin/tts/speakers/reload")
async def admin_tts_speakers_reload(
    _: None = Depends(_admin_dep()),
):
    from app.core.tts_speakers import reload_speakers, available_speakers
    reload_speakers()
    model_id = _current_tts_model_id()
    count = 0
    if model_id is not None:
        try:
            count = len(available_speakers(model_id))
        except Exception:
            count = 0
    return {"reloaded": True, "model_id": model_id, "count": count}


# ── Admin: Backend hot-reload ───────────────────────────────────────────────

class BackendReloadRequest(BaseModel):
    kind: Literal["tts", "asr"]
    profile: str
    drain_timeout_s: Optional[float] = None


@app.post("/admin/backend/reload")
async def admin_backend_reload(
    payload: BackendReloadRequest,
    _: None = Depends(_admin_dep()),
):
    from app.core.backend_manager import tts_manager, asr_manager

    if payload.kind == "tts":
        mgr = tts_manager()
    else:  # "asr"  (Literal already constrains the values)
        mgr = asr_manager()
    # drain_timeout_s override is plumbed into the request schema for
    # forward compatibility; the manager does not yet expose a setter,
    # so we ignore it for now (TODO: surface a per-call drain timeout
    # on BackendManager.reload).
    return await mgr.reload(payload.profile, reason="admin")


@app.get("/admin/backend/status")
async def admin_backend_status(_: None = Depends(_admin_dep())):
    from app.core.backend_manager import tts_manager, asr_manager
    return {"tts": tts_manager().status(), "asr": asr_manager().status()}
