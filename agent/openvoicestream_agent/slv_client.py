"""Async client for SLV's /v2v/stream WebSocket.

Single persistent connection per App lifetime (invariant 1). All public
send methods are serialized through an internal lock so frames never
interleave. Reader task decodes JSON vs binary frames and pushes typed
V2VEvent values onto a queue exposed via `events()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import connect as ws_connect

# Re-use SLV's protocol constants (invariant 6: never redeclare).
from app.core.v2v import (  # type: ignore[import-not-found]
    CLIENT_ABORT,
    CLIENT_ASR_EOS,
    CLIENT_CONFIG,
    CLIENT_TEXT,
    CLIENT_TTS_FLUSH,
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_ASR_PARTIAL,
    SERVER_ERROR,
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
)

logger = logging.getLogger(__name__)


# ── Typed events ──────────────────────────────────────────────────────


class V2VEvent:
    """Base class for events emitted by SLVClient."""


@dataclass
class ASRPartial(V2VEvent):
    text: str
    is_stable: bool = False


@dataclass
class ASREndpoint(V2VEvent):
    pass


@dataclass
class ASRFinal(V2VEvent):
    text: str
    session_complete: bool = True
    duplicate_of_streamed: bool = False
    # Per-utterance detected language as reported by the ASR backend
    # (e.g. "Chinese", "English"). None when the backend doesn't perform
    # language ID. Modes (e.g. InterpreterMode) consume this to pick a
    # translator src language at runtime.
    language: "str | None" = None


@dataclass
class TTSStarted(V2VEvent):
    sentence: str


@dataclass
class TTSSentenceDone(V2VEvent):
    sentence: str


@dataclass
class TTSDone(V2VEvent):
    pass


@dataclass
class TTSAudio(V2VEvent):
    pcm: bytes
    sample_rate: int


@dataclass
class SLVError(V2VEvent):
    message: str


# ── Client ───────────────────────────────────────────────────────────


class SLVClient:
    """One persistent WS to /v2v/stream for the entire App lifetime."""

    def __init__(self, url: str, config: dict[str, Any]) -> None:
        self.url = url
        self.config = dict(config)
        # Make sure multi_utterance is on (invariant 1).
        self.config["multi_utterance"] = True

        self._ws: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._queue: asyncio.Queue[V2VEvent] = asyncio.Queue()
        self._tts_sample_rate: int | None = None
        self._closed = False
        # Set when reader exits for any reason; events() uses this to
        # break out of `await queue.get()` instead of hanging forever.
        self._reader_done: asyncio.Event = asyncio.Event()

    # ── lifecycle ───────────────────────────────────────────────────

    # Time we wait after sending CLIENT_CONFIG before declaring a fresh
    # WS healthy. If the server closes within this window the limiter
    # almost certainly rejected us (SLV's per-client WS slot is still
    # busy with the previous session's teardown); back off and retry.
    _RECONNECT_GRACE_S = 0.05
    # Backoff schedule for the limiter race. Empirically the slot
    # releases inside ~40ms, but Jetson under thermal throttle can be
    # slower; cover up to ~1.75s of contention before giving up.
    _RECONNECT_BACKOFFS = (0.25, 0.5, 1.0)

    async def connect(self) -> None:
        if self._ws is not None:
            return  # idempotent
        await self._open_with_retry()

    async def reconnect(self) -> None:
        """Tear down current WS and open a fresh one, replaying config.

        Self-healing against SLV's session-limiter race (server `app/main.py`
        ``try_acquire_ws`` admission): the previous session's WS slot is
        held until a teardown chain (dispatcher cancel → ASR cancel →
        ws.close → manager unregister → token release) completes. A
        reconnect that arrives inside that 3–40 ms window is closed with
        WS code 4429. We detect the immediate close by waiting on the
        reader-done signal for ``_RECONNECT_GRACE_S`` after sending the
        config frame; if reader fires inside the grace, we back off and
        retry.
        """
        if self._closed:
            return
        async with self._send_lock:
            old_reader = self._reader_task
            old_ws = self._ws
            self._ws = None
            self._reader_task = None
            if old_reader is not None and not old_reader.done():
                old_reader.cancel()
                try:
                    await old_reader
                except (asyncio.CancelledError, Exception):
                    pass
            if old_ws is not None:
                try:
                    await old_ws.close()
                except Exception:
                    pass
            self._reader_done.clear()
            self._tts_sample_rate = None
            await self._open_with_retry()

    async def _open_with_retry(self) -> None:
        """Open a WS and verify it survived the limiter grace window.

        Must be called with ``self._send_lock`` already held by the
        caller (or in a single-threaded init path like ``connect``).
        """
        attempts = list(self._RECONNECT_BACKOFFS) + [None]
        for attempt_idx, backoff in enumerate(attempts):
            self._reader_done.clear()
            self._tts_sample_rate = None
            self._ws = await ws_connect(self.url, max_size=None)
            await self._ws.send(json.dumps({"type": CLIENT_CONFIG, **self.config}))
            self._reader_task = asyncio.create_task(
                self._reader_loop(), name="slv-reader"
            )
            try:
                # If the server rejects this connection (limiter race,
                # bad config, etc.) the reader sees ConnectionClosed
                # almost immediately and sets _reader_done. Survive the
                # grace window → connection is healthy, return.
                await asyncio.wait_for(
                    self._reader_done.wait(), timeout=self._RECONNECT_GRACE_S
                )
            except asyncio.TimeoutError:
                # Healthy: reader is still running after grace window.
                return
            # Reader fired → connection died inside grace window.
            # Tear down what we just built and back off.
            dead_reader = self._reader_task
            dead_ws = self._ws
            self._ws = None
            self._reader_task = None
            if dead_reader is not None and not dead_reader.done():
                dead_reader.cancel()
                try:
                    await dead_reader
                except (asyncio.CancelledError, Exception):
                    pass
            if dead_ws is not None:
                try:
                    await dead_ws.close()
                except Exception:
                    pass
            # Drain any SLVError the reader may have queued so subsequent
            # events() consumers don't see this rejection.
            try:
                while not self._queue.empty():
                    _ = self._queue.get_nowait()
            except Exception:
                pass
            if backoff is None:
                raise ConnectionError(
                    f"SLV reconnect: server closed within {self._RECONNECT_GRACE_S}s "
                    f"on all {len(attempts)} attempts (limiter race?)"
                )
            logger.warning(
                "SLV reconnect attempt %d rejected within %.0fms; retrying in %.2fs",
                attempt_idx + 1, self._RECONNECT_GRACE_S * 1000, backoff,
            )
            await asyncio.sleep(backoff)

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._ws = None

    # ── send helpers ────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                await self.connect()
            try:
                await self._ws.send(json.dumps(payload))
            except websockets.ConnectionClosed:
                logger.info("send_json: WS closed mid-send, dropping %s", payload.get("type"))
                # Null the dead handle so the next caller triggers
                # connect() instead of replaying onto the closed WS.
                self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                await self.connect()
            try:
                await self._ws.send(pcm)
            except websockets.ConnectionClosed:
                # Reader will notice and signal _reader_done; dispatch can
                # decide to reconnect. Audio chunks during reconnect are
                # naturally dropped — first ASR utterance after reconnect
                # picks up from the new chunks.
                self._ws = None

    async def send_text(self, text: str) -> None:
        if text:
            logger.info("SLV send text chunk len=%d", len(text))
        await self._send_json({"type": CLIENT_TEXT, "text": text})

    async def flush_tts(self) -> None:
        logger.info("SLV send tts_flush")
        await self._send_json({"type": CLIENT_TTS_FLUSH})

    async def abort(self) -> None:
        await self._send_json({"type": CLIENT_ABORT})

    async def asr_eos(self) -> None:
        await self._send_json({"type": CLIENT_ASR_EOS})

    # ── reader ──────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[V2VEvent]:
        while True:
            # Drain anything already queued first so SLVError emitted on
            # reader exit is still surfaced to the consumer.
            if not self._queue.empty():
                yield self._queue.get_nowait()
                continue
            if self._reader_done.is_set() or self._closed:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    # A manual reconnect can cancel an old reader and start a
                    # new one while this iterator is still alive. The old
                    # reader's finally may set the shared event after the new
                    # reader is already running; don't make dispatch exit and
                    # reconnect a second time in that case.
                    self._reader_done.clear()
                    continue
                return
            get_task = asyncio.create_task(self._queue.get())
            done_task = asyncio.create_task(self._reader_done.wait())
            try:
                done, pending = await asyncio.wait(
                    {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                get_task.cancel()
                done_task.cancel()
                raise
            for t in pending:
                t.cancel()
            if get_task in done:
                yield get_task.result()
            else:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    self._reader_done.clear()
                    continue
                # Reader finished; flush any final items it pushed.
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._handle_binary(bytes(msg))
                else:
                    await self._handle_json(msg)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as e:
            logger.info("SLV WS closed: %s", e)
            await self._queue.put(SLVError(f"connection closed: {e}"))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("SLV reader crashed")
            await self._queue.put(SLVError(str(e)))
        finally:
            # Wake any consumer blocked on events().
            self._reader_done.set()

    async def _handle_binary(self, data: bytes) -> None:
        if self._tts_sample_rate is None:
            if len(data) < 4:
                await self._queue.put(SLVError("first binary frame < 4 bytes"))
                return
            (sr,) = struct.unpack("<I", data[:4])
            self._tts_sample_rate = sr
            pcm = data[4:]
            logger.info("SLV tts sample_rate=%d first_pcm=%d", sr, len(pcm))
            if pcm:
                await self._queue.put(TTSAudio(pcm=pcm, sample_rate=sr))
            return
        logger.info("SLV tts audio bytes=%d", len(data))
        await self._queue.put(TTSAudio(pcm=data, sample_rate=self._tts_sample_rate))

    async def _handle_json(self, raw: str) -> None:
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError as e:
            await self._queue.put(SLVError(f"bad json: {e}"))
            return
        t = evt.get("type")
        if t == SERVER_ASR_PARTIAL:
            await self._queue.put(
                ASRPartial(text=evt.get("text", ""), is_stable=bool(evt.get("is_stable", False)))
            )
        elif t == SERVER_ASR_ENDPOINT:
            await self._queue.put(ASREndpoint())
        elif t == SERVER_ASR_FINAL:
            await self._queue.put(
                ASRFinal(
                    text=evt.get("text", ""),
                    session_complete=bool(evt.get("session_complete", True)),
                    duplicate_of_streamed=bool(evt.get("duplicate_of_streamed", False)),
                    language=evt.get("language"),
                )
            )
        elif t == SERVER_TTS_STARTED:
            logger.info("SLV tts_started sentence=%r", evt.get("sentence", "")[:80])
            await self._queue.put(TTSStarted(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_SENTENCE_DONE:
            await self._queue.put(TTSSentenceDone(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_DONE:
            logger.info("SLV tts_done")
            await self._queue.put(TTSDone())
        elif t == SERVER_ERROR:
            await self._queue.put(SLVError(evt.get("error", "unknown")))
        else:
            logger.debug("Unknown SLV message type: %r", t)
