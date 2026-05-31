"""RK TTS adapter — wraps rkvoice_stream.create_tts() output.

rkvoice-stream's TTSBackend ABC is smaller than ours (no `capabilities`,
no `language` arg, `speaker_id` is int with default 0); the adapter
forwards everything the OpenVoiceStream contract requires and exposes
a conservative default capability set.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from app.core.language import detect_zh_en
from app.core.tts_backend import TTSBackend, TTSCapability
from app.core.tts_speakers import resolve_speaker_kwargs


# rkvoice-stream's TTSBackend doesn't expose a capability set. The shipped
# backends (matcha_rknn, piper_rknn, qwen3_rknn) all do basic + streaming
# TTS, so declare that as the floor. The wire layer feature-detects optional
# things (voice clone, etc.) via has_capability().
_DEFAULT_RK_TTS_CAPS = {
    TTSCapability.BASIC_TTS,
    TTSCapability.STREAMING,
    TTSCapability.MULTI_LANGUAGE,
}


class RKTTSBackend(TTSBackend):
    """Adapter around rkvoice_stream.create_tts(). Backend selection is
    delegated to rkvoice-stream via the ``TTS_BACKEND`` env var (set in
    the rk3576/rk3588 profile)."""

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for RK NPU TTS.

        Spec §1 sample row "RK ASR/TTS": rkvoice-stream owns NPU lifecycle
        (see ``app/backends/rk/tts.py:79`` for the unload-belongs-to-rkvoice
        comment), serializes through one NPU device, and cannot be safely
        multiplexed across slots. Single-session only.
        """
        from app.core.concurrency_capability import ConcurrencyCapability
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=1,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="external_managed",
        )

    def __init__(self):
        from rkvoice_stream import create_tts
        self._inner = create_tts()
        # PR5c FIX_2: cache metadata at construction time so post-unload
        # status queries (manager.status() / health checks) don't crash on
        # ``self._inner is None``.
        try:
            self._cached_name = f"rk:{self._inner.name}"
        except Exception:
            self._cached_name = "rk:unknown"
        try:
            self._cached_sample_rate = int(self._inner.get_sample_rate())
        except Exception:
            self._cached_sample_rate = 0

    @property
    def name(self) -> str:
        if self._inner is None:
            return self._cached_name
        return f"rk:{self._inner.name}"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return set(_DEFAULT_RK_TTS_CAPS)

    @property
    def sample_rate(self) -> int:
        if self._inner is None:
            return self._cached_sample_rate
        return self._inner.get_sample_rate()

    def is_ready(self) -> bool:
        if self._inner is None:
            return False
        return self._inner.is_ready()

    def preload(self) -> None:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        self._inner.preload()

    def unload(self) -> None:
        """Drop the rkvoice-stream inner backend handle. Idempotent.

        PR5: ``supports_hot_reload`` stays False — the NPU is held by the
        rkvoice-stream backend and a deeper teardown contract belongs to
        that repo (PR scope excludes it). Provide a best-effort release here
        so future support can plug in without touching the manager.
        """
        if self._inner is None:
            return
        try:
            self._inner = None
            import gc
            gc.collect()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "RKTTSBackend.unload failed; continuing"
            )

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs)
        sid = voice.get("speaker_id", 0)
        # rkvoice-stream's synthesize() doesn't take `language`; pass it
        # through kwargs only when explicitly set so backends that ignore it
        # are unaffected.
        language = detect_zh_en(text, language)
        kwargs.setdefault("language", language)
        return self._inner.synthesize(
            text=text,
            speaker_id=sid,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )

    def generate_streaming(self, text: str, **kwargs):
        """Bridge our base-class generate_streaming() to rkvoice-stream's
        synthesize_stream().

        rkvoice-stream yields ``(audio, metadata)`` tuples where ``audio`` is
        either float32 [-1,1], int16 PCM, or raw bytes. The wire layer
        (`/tts/stream`) expects int16 PCM bytes per chunk, so coerce here
        — starlette's StreamingResponse calls ``.encode()`` on non-bytes
        items and explodes on tuples (`'tuple' object has no attribute
        'encode'`).
        """
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, **kwargs)
        speaker_id = voice.get("speaker_id", 0)
        kwargs.pop("speaker_id", None)
        speed = kwargs.pop("speed", None)
        pitch_shift = kwargs.pop("pitch_shift", None)
        language = detect_zh_en(text, kwargs.pop("language", None))
        kwargs.setdefault("language", language)
        for item in self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        ):
            audio = item[0] if isinstance(item, tuple) else item
            if audio is None:
                continue
            if isinstance(audio, (bytes, bytearray)):
                if len(audio) == 0:
                    continue
                yield bytes(audio)
                continue
            if isinstance(audio, np.ndarray):
                if audio.size == 0:
                    continue
                if audio.dtype == np.int16:
                    yield audio.tobytes()
                else:
                    a = np.asarray(audio, dtype=np.float32)
                    a = np.clip(a, -1.0, 1.0)
                    yield (a * 32767.0).astype(np.int16).tobytes()
                continue
            # Unknown payload — skip rather than poison the stream.
            continue

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if self._inner is None:
            raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
        language = detect_zh_en(text, language)
        kwargs.setdefault("language", language)
        yield from self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id if speaker_id is not None else 0,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )
