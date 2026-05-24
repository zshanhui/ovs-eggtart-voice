"""Dashboard /api/translator/runtime endpoints (Phase 2a).

Verifies:
  - GET returns backend / src_lang / tgt_lang / supported_targets
  - PATCH with a legal tgt_lang updates ``config.translator_tgt_lang``
  - PATCH with malformed / unsupported / missing body returns 400
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from openvoicestream_agent.app_mode import AppMode, ModeManager
from openvoicestream_agent.config import Config
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin


class _Chat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test chat"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, config: Config):
    app = MagicMock()
    app.config = config
    app.events = EventBus()
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(
        _ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock()
    )
    app.session = SimpleNamespace(history=[])
    app.broadcast = AsyncMock()
    mgr = ModeManager(lambda: None)
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    return app


def _cfg(port: int, **overrides) -> Config:
    base = dict(
        metadata={"dashboard_port": port},
        translator_backend="ctranslate2",
        translator_url="http://localhost:9001",
        translator_src_lang="zho_Hans",
        translator_tgt_lang="eng_Latn",
        translator_timeout_s=5.0,
    )
    base.update(overrides)
    return Config(**base)


@pytest.mark.asyncio
async def test_translator_runtime_get_returns_full_payload(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/translator/runtime")
            assert r.status == 200
            data = await r.json()
        assert data["backend"] == "ctranslate2"
        assert data["src_lang"] == "zho_Hans"
        assert data["tgt_lang"] == "eng_Latn"
        assert data["url"] == "http://localhost:9001"
        assert data["timeout_s"] == 5.0
        # supported_targets covers at least the 11 NLLB targets we ship.
        targets = data["supported_targets"]
        assert isinstance(targets, list)
        assert len(targets) >= 11
        codes = {t["code"] for t in targets}
        assert {"eng_Latn", "zho_Hans", "jpn_Jpan"}.issubset(codes)
        for t in targets:
            assert set(t.keys()) == {"code", "name"}
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_updates_target_language(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "jpn_Jpan"},
            )
            assert r.status == 200
            data = await r.json()
        assert data == {"ok": True, "tgt_lang": "jpn_Jpan"}
        # Live config mutated.
        assert cfg.translator_tgt_lang == "jpn_Jpan"
        # Broadcast fired with the runtime payload.
        app.broadcast.assert_awaited()
        evt_name, evt_payload = app.broadcast.await_args.args
        assert evt_name == "on_translator_runtime_change"
        assert evt_payload["tgt_lang"] == "jpn_Jpan"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_malformed_code(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "english"},  # wrong format
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "NLLB format" in data["error"]
        # Config NOT mutated.
        assert cfg.translator_tgt_lang == "eng_Latn"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_unsupported_code(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # Well-formed FLORES code but not in our supported_targets.
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "vie_Latn"},
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "supported" in data["error"]
        assert cfg.translator_tgt_lang == "eng_Latn"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_missing_field(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={},
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "tgt_lang" in data["error"]
    finally:
        await plugin.stop()
