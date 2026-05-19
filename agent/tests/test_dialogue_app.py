"""Default dialogue turn: tokens stream directly to SLV; no client-side batching.

Historically this was DialogueApp.on_user_utterance; the same logic now
lives in ModeContext.run_default_dialogue_turn (invoked by ChatMode).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_mode import ModeContext, ModeManager
from openvoicestream_agent.apps_dialogue_shim import DialogueApp  # back-compat alias
from openvoicestream_agent.llm.base import LLMBackend
from openvoicestream_agent.llm import LLMStreamError
from openvoicestream_agent.modes import ChatMode


class FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0
        self.aborted: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self) -> None:
        self.aborted += 1


class FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.last_messages: list[dict[str, str]] | None = None
        self.last_session: Any = None

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_session = kw.get("session")
        for t in self.tokens:
            yield t


class FakeAudio:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop_playback(self) -> None:
        self.stopped += 1


async def _noop_broadcast(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_default_dialogue_turn_streams_tokens_directly_to_slv():
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    llm = FakeLLM(["你", "好", "，", "世界。"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Every LLM token forwarded individually (no batching/joining).
    assert slv.text_frames == ["你", "好", "，", "世界。"]
    # flush_tts called exactly once after stream ends.
    assert slv.flushed == 1
    # History has user + assistant entries.
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好，世界。"},
    ]
    # LLM saw full messages including the configured system prompt.
    assert llm.last_messages[0] == {"role": "system", "content": "SYS"}
    assert llm.last_messages[-1] == {"role": "user", "content": "hi"}
    # session was passed through to LLM (for prefix-cache control).
    assert llm.last_session is session


@pytest.mark.asyncio
async def test_cancelled_dialogue_turn_closes_llm_stream_without_tts_flush():
    """Barge-in cancels the dialogue task. That must close the upstream LLM
    stream (edge-llm maps client disconnect to channel.cancel()), but must
    not flush partial old tokens into TTS."""
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()
    first_token_sent = asyncio.Event()
    stream_closed = asyncio.Event()

    class CancellableLLM(LLMBackend):
        async def stream(self, messages, **kw):  # type: ignore[override]
            try:
                yield "old"
                first_token_sent.set()
                await asyncio.sleep(60)
                yield "tail"  # pragma: no cover
            finally:
                stream_closed.set()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=CancellableLLM(), session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )

    task = asyncio.create_task(ctx.run_default_dialogue_turn("hi"))
    await asyncio.wait_for(first_token_sent.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream_closed.is_set()
    assert slv.text_frames == ["old"]
    assert slv.flushed == 0
    assert session.history == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_midstream_llm_error_aborts_partial_tts_without_history_pollution():
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    audio = FakeAudio()
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    class PartialThenErrorLLM(LLMBackend):
        async def stream(self, messages, **kw):  # type: ignore[override]
            yield "你"
            yield "可能"
            raise LLMStreamError("finish_reason=error")

    ctx = ModeContext(
        config=cfg, slv=slv, llm=PartialThenErrorLLM(), session=session,
        audio=audio, events=events, broadcast=_noop_broadcast,
    )

    with pytest.raises(LLMStreamError):
        await ctx.run_default_dialogue_turn("hi")

    assert slv.text_frames == ["你", "可能"]
    assert slv.flushed == 0
    assert slv.aborted == 1
    assert audio.stopped == 1
    assert session.history == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_multi_mode_app_class_is_back_compat_dialogue_shim():
    """The legacy `DialogueApp` import path now resolves to MultiModeApp."""
    from apps.multi_mode.app import MultiModeApp

    assert DialogueApp is MultiModeApp
