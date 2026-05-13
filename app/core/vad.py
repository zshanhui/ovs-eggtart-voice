"""Server-side VAD with a shared singleton model.

Used by:
- WS /v2v/stream — auto-detect end-of-speech to finalize ASR
- WS /asr/stream?vad=silero — opt-in VAD for the ASR-only endpoint

Design constraint: silero-vad ONNX model is ~1.7 MB but loads in ~500 ms
and locks an ORT session. We load it ONCE at module-init and share it
across every WS connection. Per-connection state (rolling speech
probability + silence timer) is held in a Session object.

The webrtcvad fallback is stateless and cheap; no singleton needed.

Backends:
- 'silero' (recommended) — multilingual, accurate, ~200 lines model;
   requires `pip install silero-vad` (already a transitive dep via
   onnxruntime in our Jetson + RK images)
- 'webrtcvad' — light C extension, less accurate; requires
   `pip install webrtcvad-wheels` (already in jetson image; missing
   on rk-v1.2 but optional)
- 'none' — disable VAD (caller must finalize ASR explicitly)
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# Singleton: load silero ONNX exactly once per process
# ───────────────────────────────────────────────────────────────
_silero_model = None
_silero_load_lock = threading.Lock()


def _get_silero_model():
    """Lazy-load the silero VAD model; subsequent callers reuse it."""
    global _silero_model
    if _silero_model is not None:
        return _silero_model
    with _silero_load_lock:
        if _silero_model is not None:
            return _silero_model
        try:
            from silero_vad import load_silero_vad
            _silero_model = load_silero_vad(onnx=True)
            logger.info("silero-vad loaded (ONNX, shared across all sessions)")
        except ImportError as e:
            raise RuntimeError(
                "silero VAD requested but `silero-vad` is not installed. "
                "Add `pip install silero-vad` to your image, or use "
                "vad=webrtcvad / vad=none."
            ) from e
        return _silero_model


# ───────────────────────────────────────────────────────────────
# Per-connection VAD sessions
# ───────────────────────────────────────────────────────────────

class VADSession:
    """Abstract base. Feed PCM, get speech / endpoint events."""

    SPEECH_START   = "speech_start"
    SPEECH_END     = "speech_end"      # silence threshold crossed
    NONE           = None              # no transition this chunk

    def process(self, samples: np.ndarray) -> Optional[str]:
        """Feed one chunk of int16 or float32 PCM (16 kHz mono assumed).

        Returns one of:
          SPEECH_START  — first speech frame after silence (onset)
          SPEECH_END    — silence sustained past threshold (endpoint)
          None          — no transition this chunk
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset state (e.g., after a forced finalize)."""
        raise NotImplementedError


class SileroVADSession(VADSession):
    """silero-vad streaming wrapper. Holds per-connection running state
    (rolling probability + silence counter) but shares the underlying
    ONNX model with all other sessions in the process."""

    # silero expects exactly 512 samples at 16 kHz (32 ms) or 256 at 8 kHz
    WINDOW_16K = 512

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_ms: int = 500,
        speech_pad_ms: int = 100,
    ):
        if sample_rate != 16000:
            raise ValueError(f"silero VAD expects 16 kHz, got {sample_rate}")
        from silero_vad import VADIterator
        self._iter = VADIterator(
            _get_silero_model(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._leftover: np.ndarray = np.empty(0, dtype=np.float32)
        self._in_speech = False

    def process(self, samples: np.ndarray) -> Optional[str]:
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        # silero VADIterator needs exactly WINDOW_16K samples per step;
        # buffer leftover bytes across calls.
        buf = np.concatenate([self._leftover, samples])
        event: Optional[str] = None
        i = 0
        while i + self.WINDOW_16K <= len(buf):
            window = buf[i : i + self.WINDOW_16K]
            ev = self._iter(window, return_seconds=False)
            if ev:
                if "start" in ev and not self._in_speech:
                    self._in_speech = True
                    event = self.SPEECH_START
                elif "end" in ev and self._in_speech:
                    self._in_speech = False
                    event = self.SPEECH_END
            i += self.WINDOW_16K
        self._leftover = buf[i:]
        return event

    def reset(self) -> None:
        self._iter.reset_states()
        self._leftover = np.empty(0, dtype=np.float32)
        self._in_speech = False


class WebRTCVADSession(VADSession):
    """webrtcvad streaming wrapper. Per-frame is_speech check + a
    silence-frame counter."""

    FRAME_MS = 30   # webrtcvad supports 10/20/30 ms frames
    FRAME_BYTES_16K = 16000 * (FRAME_MS / 1000) * 2  # int16 samples * 2 bytes

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,    # 0..3, higher = more aggressive
        silence_ms: int = 500,
    ):
        import webrtcvad
        self._vad = webrtcvad.Vad(int(aggressiveness))
        self._sr = sample_rate
        self._frame_bytes = int(sample_rate * (self.FRAME_MS / 1000) * 2)
        self._silence_frames_threshold = max(1, silence_ms // self.FRAME_MS)
        self._silence_count = 0
        self._leftover_bytes = b""
        self._in_speech = False

    def process(self, samples: np.ndarray) -> Optional[str]:
        if samples.dtype == np.float32:
            samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        elif samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        buf = self._leftover_bytes + samples.tobytes()
        event: Optional[str] = None
        i = 0
        while i + self._frame_bytes <= len(buf):
            frame = buf[i : i + self._frame_bytes]
            speech = self._vad.is_speech(frame, self._sr)
            if speech:
                if not self._in_speech:
                    self._in_speech = True
                    event = self.SPEECH_START
                self._silence_count = 0
            else:
                if self._in_speech:
                    self._silence_count += 1
                    if self._silence_count >= self._silence_frames_threshold:
                        self._in_speech = False
                        event = self.SPEECH_END
                        self._silence_count = 0
            i += self._frame_bytes
        self._leftover_bytes = buf[i:]
        return event

    def reset(self) -> None:
        self._silence_count = 0
        self._leftover_bytes = b""
        self._in_speech = False


def create_vad(
    backend: str = "silero",
    sample_rate: int = 16000,
    silence_ms: int = 500,
    **kwargs,
) -> Optional[VADSession]:
    """Factory. Returns None for backend='none'."""
    if backend in (None, "", "none", "off", "disabled"):
        return None
    if backend == "silero":
        return SileroVADSession(sample_rate=sample_rate, silence_ms=silence_ms, **kwargs)
    if backend in ("webrtc", "webrtcvad"):
        return WebRTCVADSession(sample_rate=sample_rate, silence_ms=silence_ms, **kwargs)
    raise ValueError(f"unknown VAD backend: {backend!r}; use 'silero' | 'webrtcvad' | 'none'")
