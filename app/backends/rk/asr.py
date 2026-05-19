"""RK ASR adapter — wraps rkvoice_stream.create_asr() output to fit the
OpenVoiceStream ASRBackend interface.

The two ABCs (ours in app.core.asr_backend and theirs in
rkvoice_stream.engine.asr) are intentionally near-identical; this module
bridges the capability enum and forwards every method.

Long-audio guard (added 2026-05): the Qwen3-on-RKLLM pipeline has a 512-token
decoder context cap and a sliding-window decoder that snowballs garbage from
one chunk into the next. On audio >~10s the model often bails to its own
instruction suffix ("转录") or hallucinates only the last segment. We fix this
in the adapter layer (no submodule changes) by:
  (a) energy-RMS splitting long audio into <=4.5s segments at silence
  (b) running each segment as an INDEPENDENT inner.transcribe() (fresh
      StreamSession internally → no cross-segment prefix poisoning)
  (c) discarding placeholder echoes (e.g. just "转录" or "转录：")
  (d) joining with a language-aware separator
"""
from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np

from app.core.asr_backend import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Long-audio segmentation (energy-RMS, no webrtcvad dep — RK image doesn't
# ship webrtcvad). Mirrors qwen3_asr.py:_split_at_silence_energy but copied
# inline so this module is importable from the RK container without pulling
# in the Jetson TRT-Edge-LLM dependency tree.
# ---------------------------------------------------------------------------

_VAD_MAX_SEG_SEC = 4.5
_VAD_MIN_SEG_SEC = 0.5
_VAD_FRAME_MS = 20
_LONG_AUDIO_THRESHOLD_S = 15.0       # below this, trust the inner streaming path (raised from 5.0 — the true streaming path with ROLLING_BUFFER_SEC=15 can handle up to 15s cleanly)

# Outputs that mean "model gave up and echoed its own instruction suffix" —
# drop these from the joined transcript.
_PLACEHOLDER_OUTPUTS = {
    "", "转录", "转录。", "转录：", "转录:",
    "transcription", "transcription.", "transcription:",
}


def _split_at_silence_energy(audio: np.ndarray, sr: int = 16000) -> list[np.ndarray]:
    max_seg = int(_VAD_MAX_SEG_SEC * sr)
    min_seg = int(_VAD_MIN_SEG_SEC * sr)
    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(_VAD_FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    framed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    threshold = float(os.environ.get("ASR_ENERGY_SPLIT_RMS", "0.003"))
    is_silence = rms < threshold
    min_run = max(1, int(os.environ.get("ASR_ENERGY_MIN_SILENCE_MS", "120")) // _VAD_FRAME_MS)

    cut_candidates: list[int] = []
    run_start: Optional[int] = None
    for i, silent in enumerate(is_silence):
        if silent:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                cut_candidates.append(((run_start + i) // 2) * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        cut_candidates.append(((run_start + n_frames) // 2) * frame_len)
    cand = np.array(cut_candidates, dtype=np.int64)

    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        hi = target
        mask = (cand >= lo) & (cand <= hi)
        if mask.any():
            pick = int(cand[mask][np.argmax(cand[mask])])
        else:
            pick = int(target)
        cuts.append(pick)
    cuts.append(len(audio))

    # Merge mid-fragments <1s into the previous segment to avoid model bailout
    min_frag = int(1.0 * sr)
    min_tail = int(1.5 * sr)
    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)
        else:
            i += 1
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]


def _float_to_wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_to_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    ratio = 16000 / sr
    new_len = int(len(audio) * ratio)
    idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _to_str(value) -> str:
    """rkvoice-stream's inner.finalize() returns a structured dict, not a
    plain string. Unwrap the canonical 'text' field recursively (and tolerate
    plain str / None inputs)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _to_str(value.get("text", ""))
    if value is None:
        return ""
    return str(value)


def _clean_segment_text(text) -> str:
    """Drop model-bailout placeholders. Tolerates dict/str/None inputs."""
    s = _to_str(text)
    if not s:
        return ""
    stripped = s.strip()
    if stripped in _PLACEHOLDER_OUTPUTS:
        return ""
    # Some bailouts pad with whitespace/newlines around the placeholder
    if stripped.rstrip("。：:.\n ") in _PLACEHOLDER_OUTPUTS:
        return ""
    return stripped


def _join_segments(texts: list[str], language: str) -> str:
    texts = [t for t in texts if t]
    if not texts:
        return ""
    if len(texts) > 1:
        # Trim trailing CJK/Latin punctuation off all-but-last segments
        trail = "。，、！？；,.!?;"
        texts = [t.rstrip(trail).rstrip() for t in texts[:-1]] + [texts[-1]]
    cjk = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
    sep = "" if (language in cjk or any(language.startswith(p) for p in ("zh", "ja", "ko"))) else " "
    return sep.join(texts).strip()


# ---------------------------------------------------------------------------
# Stream adapter
# ---------------------------------------------------------------------------


class _UnloadRaceError(RuntimeError):
    """Raised by the stream adapter when its backing inner has been unloaded
    mid-stream. Distinguishes "known unload race" from native RuntimeError
    surfaced by rkvoice_stream itself."""


class _RKASRStreamAdapter(ASRStream):
    """Forwards accept_waveform to the inner stream for partial emission, but
    intercepts finalize: if the accumulated audio is longer than the long-audio
    threshold, segment + per-segment transcribe instead of trusting the inner's
    sliding-window decoder (which snowballs garbage past ~10s)."""

    def __init__(self, inner, backend: "RKASRBackend", language: str = "auto"):
        self._inner = inner
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._sample_rate = 16000

    # PR5d FIX_1: every inner/backend._inner deref guards against the unload
    # race window — manager hard-closes the WS on profile swap, but a stream
    # task in flight may still touch the adapter one more time. Raise a
    # dedicated _UnloadRaceError (subclass of RuntimeError) so callers can
    # distinguish "known unload race" from a native RuntimeError surfaced by
    # the rkvoice_stream inner. See finalize() long-path for why the
    # narrower except matters.
    def _live_inner(self):
        inner = self._inner
        if inner is None:
            raise _UnloadRaceError(
                "RK ASR stream adapter inner was unloaded; stream is dead"
            )
        return inner

    def _live_backend_inner(self):
        backend = self._backend
        inner = getattr(backend, "_inner", None) if backend is not None else None
        if inner is None:
            raise _UnloadRaceError(
                "RK ASR backend was unloaded; stream is dead"
            )
        return inner

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        inner = self._live_inner()
        self._sample_rate = sample_rate
        # Buffer for our own finalize path. Cheap copy; the underlying memory
        # is already a numpy array.
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._chunks.append(samples)
        inner.accept_waveform(sample_rate, samples)

    def finalize(self) -> str:
        inner = self._live_inner()
        if not self._chunks:
            return inner.finalize() or ""
        audio = np.concatenate(self._chunks)
        dur_s = len(audio) / max(self._sample_rate, 1)
        if dur_s <= _LONG_AUDIO_THRESHOLD_S:
            text = inner.finalize() or ""
            return _clean_segment_text(text)

        # Long path: segment + per-segment offline transcribe via inner.
        audio = _resample_to_16k(audio, self._sample_rate)
        try:
            segments = _split_at_silence_energy(audio, 16000)
        except Exception as e:
            logger.warning("RK ASR splitter failed (%.1fs audio): %s", dur_s, e)
            segments = [audio]

        texts: list[str] = []
        for seg in segments:
            if len(seg) / 16000 < 0.4:
                continue
            wav_bytes = _float_to_wav_bytes(seg, 16000)
            try:
                backend_inner = self._live_backend_inner()
                result = backend_inner.transcribe(
                    wav_bytes, language=self._language
                )
            except _UnloadRaceError:
                # Backend was unloaded mid-finalize — propagate so the
                # caller distinguishes this from a true transcribe error.
                # Native RuntimeError from transcribe() still falls through
                # to the broad except below and is logged-and-skipped.
                raise
            except Exception as e:
                logger.warning(
                    "RK ASR segment failed (%.1fs): %s", len(seg) / 16000, e
                )
                continue
            seg_text = _clean_segment_text(getattr(result, "text", "") or "")
            if seg_text:
                texts.append(seg_text)

        # Discard the inner's sliding-window result entirely — it's the
        # poisoned snowball. Some inners need their state torn down; trust
        # the GC and a fresh stream next call.
        return _join_segments(texts, self._language)

    def prepare_finalize(self) -> None:
        self._live_inner().prepare_finalize()

    def cancel_and_finalize(self) -> None:
        self._live_inner().cancel_and_finalize()

    def get_partial(self) -> tuple[str, bool]:
        return self._live_inner().get_partial()


# ---------------------------------------------------------------------------
# Capability map + backend
# ---------------------------------------------------------------------------

_CAP_MAP = {
    "offline": ASRCapability.OFFLINE,
    "streaming": ASRCapability.STREAMING,
    "multi_language": ASRCapability.MULTI_LANGUAGE,
}


class RKASRBackend(ASRBackend):
    """Adapter around rkvoice_stream.create_asr().

    Backend selection is delegated to rkvoice-stream itself via the
    ``ASR_BACKEND`` env var (set in the rk3576/rk3588 profile).
    """

    def __init__(self):
        from rkvoice_stream import create_asr
        self._inner = create_asr()
        self._platform = os.environ.get("RK_PLATFORM", "rk3576")
        # PR5c FIX_2: cache metadata at construction time so post-unload
        # status queries don't crash on ``self._inner is None``.
        try:
            self._cached_name = f"rk:{self._inner.name}"
        except Exception:
            self._cached_name = "rk:unknown"
        try:
            self._cached_sample_rate = int(self._inner.sample_rate)
        except Exception:
            self._cached_sample_rate = 0
        try:
            cached_caps: set[ASRCapability] = set()
            for cap in self._inner.capabilities:
                value = cap.value if hasattr(cap, "value") else str(cap)
                mapped = _CAP_MAP.get(value)
                if mapped is not None:
                    cached_caps.add(mapped)
            self._cached_capabilities = cached_caps
        except Exception:
            self._cached_capabilities = set()

    @property
    def name(self) -> str:
        if self._inner is None:
            return self._cached_name
        return f"rk:{self._inner.name}"

    @property
    def capabilities(self) -> set[ASRCapability]:
        if self._inner is None:
            return set(self._cached_capabilities)
        out: set[ASRCapability] = set()
        for cap in self._inner.capabilities:
            value = cap.value if hasattr(cap, "value") else str(cap)
            mapped = _CAP_MAP.get(value)
            if mapped is not None:
                out.add(mapped)
        return out

    @property
    def sample_rate(self) -> int:
        if self._inner is None:
            return self._cached_sample_rate
        return self._inner.sample_rate

    def is_ready(self) -> bool:
        if self._inner is None:
            return False
        return self._inner.is_ready()

    def preload(self) -> None:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        self._inner.preload()

    def unload(self) -> None:
        """Drop the rkvoice-stream inner backend handle. Idempotent.

        PR5: ``supports_hot_reload`` stays False — see RKTTSBackend.unload().
        """
        if self._inner is None:
            return
        try:
            self._inner = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("RKASRBackend.unload failed; continuing")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        # Long-audio guard: if WAV is >5s, split at silence and run each
        # segment through inner.transcribe() independently (fresh session
        # internally), then concatenate. Mirrors the streaming finalize path.
        try:
            audio, sr = _wav_to_float(audio_bytes)
        except Exception:
            audio, sr = np.empty(0, dtype=np.float32), 16000
        dur_s = len(audio) / max(sr, 1)

        if dur_s <= _LONG_AUDIO_THRESHOLD_S:
            result = self._inner.transcribe(audio_bytes, language=language)
            text = _clean_segment_text(getattr(result, "text", "") or "")
            meta = getattr(result, "meta", {}) or {}
            return TranscriptionResult(text=text, language=result.language, **meta)

        audio = _resample_to_16k(audio, sr)
        try:
            segments = _split_at_silence_energy(audio, 16000)
        except Exception as e:
            logger.warning("RK ASR splitter failed offline (%.1fs): %s", dur_s, e)
            segments = [audio]

        texts: list[str] = []
        meta_acc: dict = {}
        last_lang = language
        for seg in segments:
            if len(seg) / 16000 < 0.4:
                continue
            wav_seg = _float_to_wav_bytes(seg, 16000)
            try:
                result = self._inner.transcribe(wav_seg, language=language)
            except Exception as e:
                logger.warning(
                    "RK ASR offline segment failed (%.1fs): %s", len(seg) / 16000, e
                )
                continue
            seg_text = _clean_segment_text(getattr(result, "text", "") or "")
            if seg_text:
                texts.append(seg_text)
            last_lang = getattr(result, "language", last_lang) or last_lang

        return TranscriptionResult(
            text=_join_segments(texts, language),
            language=last_lang,
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        return _RKASRStreamAdapter(
            self._inner.create_stream(language=language), self, language=language
        )
