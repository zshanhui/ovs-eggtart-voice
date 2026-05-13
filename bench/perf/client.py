"""ASR / TTS clients with millisecond-level instrumentation.

Single source of truth for perf timing. Every runner uses these — keeps
timestamp semantics consistent across asr/tts/v2v/concurrent.
"""
from __future__ import annotations
import io, json, time, wave
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import requests
import websocket  # websocket-client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_text(value) -> str:
    """ASR servers vary in `text` field type. Coerce to a single string.
    - str: pass through
    - dict: try common per-language keys, then fall back to first string value
    - None / other: empty string
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("zh", "en", "text", "result", "transcript"):
            v = value.get(k)
            if isinstance(v, str) and v:
                return v
        for v in value.values():
            if isinstance(v, str) and v:
                return v
        return ""
    if value is None:
        return ""
    return str(value)


def wav_duration_s(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()


def wav_to_pcm_chunks(wav_bytes: bytes, chunk_ms: int = 250) -> tuple[list[bytes], int]:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    chunk_n = int(sr * chunk_ms / 1000)
    return [samples[i:i + chunk_n].tobytes() for i in range(0, len(samples), chunk_n)], sr


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class ASRResult:
    text: str
    audio_dur_s: float
    processing_ms: float          # wall-clock from request start to final text
    tfd_ms: float | None = None   # streaming only: first PCM sent -> first partial
    eos_to_final_ms: float | None = None  # streaming only
    rtf: float = 0.0              # wall-clock RTF: processing_ms / audio_dur
                                  # NOTE: in --realtime streaming mode this is ≥ 1.0
                                  # by construction (client sleeps between chunks).
    finalize_rtf: float | None = None  # compute-bound RTF for streaming:
                                       # eos_to_final_ms / audio_dur. This is the
                                       # cross-device-comparable number — independent
                                       # of how the client paces chunks.

    @property
    def as_dict(self) -> dict:
        return {**self.__dict__}


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_dur_s: float            # synthesized audio duration
    tfd_ms: float                 # request -> first audio chunk
    total_ms: float               # request -> last audio chunk
    rtf: float = 0.0              # total_ms / audio_dur

    @property
    def as_dict(self) -> dict:
        return {
            "audio_dur_s": self.audio_dur_s,
            "tfd_ms": self.tfd_ms,
            "total_ms": self.total_ms,
            "rtf": self.rtf,
        }


# ---------------------------------------------------------------------------
# ASR client
# ---------------------------------------------------------------------------

class ASRClient:
    def __init__(self, base_url: str, ws_url: str | None = None,
                 chunk_ms: int = 250, realtime: bool = True,
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.ws_url = (ws_url or base_url).replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self.timeout = timeout

    # ----- offline POST /asr -----
    def transcribe_offline(self, wav_bytes: bytes, language: str = "Chinese") -> ASRResult:
        dur = wav_duration_s(wav_bytes)
        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/asr",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language": language},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        proc_ms = (time.monotonic() - t0) * 1000
        return ASRResult(
            text=text, audio_dur_s=dur, processing_ms=proc_ms,
            rtf=proc_ms / (dur * 1000) if dur else 0.0,
        )

    # ----- streaming WS /asr/stream -----
    def transcribe_streaming(self, wav_bytes: bytes, language: str = "Chinese",
                             eos_mode: str = "forced") -> ASRResult:
        """
        eos_mode:
          - "forced": send b"" after last PCM chunk (immediate finalize)
          - "vad":    don't send b""; let server VAD trigger finalize (more realistic)
        """
        assert eos_mode in ("forced", "vad")
        chunks, sr = wav_to_pcm_chunks(wav_bytes, self.chunk_ms)
        dur = wav_duration_s(wav_bytes)
        chunk_dur = self.chunk_ms / 1000.0

        ws = websocket.create_connection(
            f"{self.ws_url}/asr/stream?language={language}&sample_rate={sr}",
            timeout=self.timeout,
        )
        t_first_send = time.monotonic()
        t_first_partial: float | None = None

        # Pump audio in a thread so we can read partials concurrently
        # Simpler approach: send each chunk, then non-blocking poll for partials
        # via short ws timeout. websocket-client doesn't make this easy with
        # a single connection — we just send everything, then read until final.
        # That's good enough for TFD measurement because partials arrive
        # eagerly anyway; we just sample TFD by checking right after each
        # send for a backlog message.
        ws.settimeout(0.001)
        for c in chunks:
            ws.send_binary(c)
            if t_first_partial is None:
                try:
                    msg = ws.recv()
                    data = json.loads(msg)
                    if data.get("text"):
                        t_first_partial = time.monotonic()
                except websocket.WebSocketTimeoutException:
                    pass
            if self.realtime:
                time.sleep(chunk_dur)
        ws.settimeout(self.timeout)

        t_eos = time.monotonic()
        if eos_mode == "forced":
            ws.send_binary(b"")

        final_text = ""
        while True:
            raw = ws.recv()
            if not raw:
                # Server closed without a frame. Older servers do this on
                # backend errors; newer ones send {"type":"error",...} first.
                raise RuntimeError("server closed WebSocket without a final frame (likely backend error)")
            data = json.loads(raw)
            if data.get("type") == "error":
                raise RuntimeError(f"server error: {data.get('error', '(no detail)')}")
            text_field = data.get("text", final_text)
            text_str = _coerce_text(text_field)
            if t_first_partial is None and text_str:
                t_first_partial = time.monotonic()
            if data.get("type") == "final":
                final_text = text_str.strip()
                break
            final_text = text_str.strip()
        t_final = time.monotonic()
        ws.close()

        eos_to_final_ms = (t_final - t_eos) * 1000
        return ASRResult(
            text=final_text,
            audio_dur_s=dur,
            processing_ms=(t_final - t_first_send) * 1000,
            tfd_ms=((t_first_partial - t_first_send) * 1000) if t_first_partial else None,
            eos_to_final_ms=eos_to_final_ms,
            rtf=((t_final - t_first_send) * 1000) / (dur * 1000) if dur else 0.0,
            finalize_rtf=eos_to_final_ms / (dur * 1000) if dur else None,
        )


# ---------------------------------------------------------------------------
# TTS client
# ---------------------------------------------------------------------------

class TTSClient:
    def __init__(self, base_url: str, timeout: int = 120,
                 stream: bool = True, voice: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stream = stream
        self.voice = voice

    def synthesize(self, text: str, language: str = "zh") -> TTSResult:
        payload: dict = {"text": text}
        if self.voice:
            payload["voice"] = self.voice
        endpoint = "/tts/stream" if self.stream else "/tts"

        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload, stream=self.stream, timeout=self.timeout,
        )
        resp.raise_for_status()

        buf = bytearray()
        t_first: float | None = None
        if self.stream:
            for chunk in resp.iter_content(4096):
                if not chunk:
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                buf.extend(chunk)
        else:
            t_first = time.monotonic()
            buf.extend(resp.content)
        t_end = time.monotonic()

        audio = bytes(buf)
        try:
            dur = wav_duration_s(audio)
        except Exception:
            # Server may return raw PCM; estimate from byte size at 16k/16-bit mono
            dur = len(audio) / (16000 * 2)
        total_ms = (t_end - t0) * 1000
        tfd_ms = ((t_first or t_end) - t0) * 1000
        return TTSResult(
            audio_bytes=audio, audio_dur_s=dur,
            tfd_ms=tfd_ms, total_ms=total_ms,
            rtf=total_ms / (dur * 1000) if dur else 0.0,
        )


# ---------------------------------------------------------------------------
# V2V composite
# ---------------------------------------------------------------------------

@dataclass
class V2VResult:
    audio_dur_s: float
    asr_text: str
    tts_audio_dur_s: float
    eos_to_first_audio_ms: float
    asr_finalize_ms: float
    llm_delay_ms: float
    tts_tfd_ms: float
    tts_total_ms: float

    @property
    def as_dict(self) -> dict:
        return self.__dict__.copy()


def run_v2v(asr: ASRClient, tts: TTSClient, wav_bytes: bytes,
            language_asr: str = "Chinese", language_tts: str = "zh",
            eos_mode: str = "forced", llm_delay_ms: float = 0.0) -> V2VResult:
    """End-to-end voice-to-voice. LLM stage is a sleep placeholder."""
    asr_res = asr.transcribe_streaming(wav_bytes, language_asr, eos_mode)
    t_after_asr = time.monotonic()
    if llm_delay_ms > 0:
        time.sleep(llm_delay_ms / 1000.0)
    tts_res = tts.synthesize(asr_res.text, language_tts)
    return V2VResult(
        audio_dur_s=asr_res.audio_dur_s,
        asr_text=asr_res.text,
        tts_audio_dur_s=tts_res.audio_dur_s,
        # client-side EOS-to-first-audio: ASR finalize + LLM delay + TTS TFD
        eos_to_first_audio_ms=(asr_res.eos_to_final_ms or 0) + llm_delay_ms + tts_res.tfd_ms,
        asr_finalize_ms=asr_res.eos_to_final_ms or 0,
        llm_delay_ms=llm_delay_ms,
        tts_tfd_ms=tts_res.tfd_ms,
        tts_total_ms=tts_res.total_ms,
    )
