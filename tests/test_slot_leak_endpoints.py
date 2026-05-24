"""Regression tests for codex MUST-FIX 1 / 2.

These tests verify that session-limiter slots are NOT leaked when
exceptions happen between ``try_acquire()`` and the streaming generator's
finally block (e.g. lazy ``_ensure_tts_manager_started()`` raising, or
``metrics.inc_sessions_active()`` raising).
"""

from __future__ import annotations

import pytest

from app.core import session_limiter, metrics


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    session_limiter._reset_for_tests()
    metrics._reset_for_tests()


# ---------------------------------------------------------------------------
# MUST-FIX 2: metrics failure must not desync limiter state
# ---------------------------------------------------------------------------

def test_metrics_failure_does_not_leak_slot(monkeypatch):
    """Even if metrics raises, try_acquire/_release keep ``active`` honest."""
    sl = session_limiter.SessionLimiter(2)

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated metrics outage")

    monkeypatch.setattr(metrics, "inc_sessions_active", _boom)
    monkeypatch.setattr(metrics, "dec_sessions_active", _boom)

    t = sl.try_acquire()
    assert t is not None, "try_acquire must succeed even when metrics is broken"
    assert sl.active == 1, "limiter must count the slot regardless of metrics"

    t.release()
    assert sl.active == 0, "release must decrement even when metrics is broken"

    # Subsequent acquires still work.
    t2 = sl.try_acquire()
    assert t2 is not None
    assert sl.active == 1


def test_rejection_metric_failure_still_rejects(monkeypatch):
    """If inc_sessions_rejected raises, the limit decision still holds."""
    import asyncio
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    sl = session_limiter.init_limiter({})
    held = sl.try_acquire()
    assert held is not None

    monkeypatch.setattr(
        metrics, "inc_sessions_rejected",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("metrics down")),
    )

    async def _try_http():
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            async with session_limiter.acquire_http("/x"):
                pass
        assert ei.value.status_code == 429

    asyncio.run(_try_http())
    held.release()


# ---------------------------------------------------------------------------
# Double-release safety
# ---------------------------------------------------------------------------

def test_double_release_under_lock_is_idempotent():
    sl = session_limiter.SessionLimiter(2)
    t = sl.try_acquire()
    assert sl.active == 1
    t.release()
    t.release()
    t.release()
    assert sl.active == 0


# ---------------------------------------------------------------------------
# MUST-FIX 1: /tts/stream lazy-start exception releases the slot
# ---------------------------------------------------------------------------

def test_tts_stream_lazy_start_failure_releases_slot(monkeypatch, tmp_path):
    """If ``_ensure_tts_manager_started()`` raises mid-handler, the slot
    must not leak. We patch the lazy-start coroutine to raise, then drive
    the endpoint through TestClient and assert the limiter is empty.
    """
    # Required env BEFORE importing app.main (startup reads them).
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LAZY_TTS", "1")
    monkeypatch.setenv("LANGUAGE_MODE", "disabled")
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "2")

    from fastapi.testclient import TestClient

    session_limiter.init_limiter({})

    # Defer-import the app so the fresh limiter init is picked up.
    from app import main as appmod

    async def _boom():
        raise RuntimeError("simulated FAILED tts manager")

    monkeypatch.setattr(appmod, "_ensure_tts_manager_started", _boom)

    sl = session_limiter.get_limiter()
    assert sl is not None and sl.active == 0

    with TestClient(appmod.app, raise_server_exceptions=False) as c:
        r = c.post("/tts/stream", json={"text": "hello", "language": "en"})
        # The handler should propagate the synthetic RuntimeError → 500.
        # We only require that the slot is returned, not a specific status.
        assert r.status_code >= 400

    # Allow Starlette to finish wrapping up the request.
    assert sl.active == 0, (
        f"slot leaked: limiter.active={sl.active} after failing "
        f"/tts/stream lazy-start"
    )


# ---------------------------------------------------------------------------
# MUST-FIX 1 round 2: CancelledError mid-setup must release the slot
# ---------------------------------------------------------------------------

def test_tts_stream_cancelled_during_backend_acquire_releases_slot(
    monkeypatch, tmp_path,
):
    """If `backend_cm.__aenter__()` raises CancelledError (e.g. client
    disconnect or shutdown), the slot must not leak. Reproduces the round-1
    miss where `except Exception` didn't cover BaseException."""
    import asyncio

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LAZY_TTS", "1")
    monkeypatch.setenv("LANGUAGE_MODE", "disabled")
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "2")

    from fastapi.testclient import TestClient

    session_limiter.init_limiter({})

    from app import main as appmod

    # Build a fake "manager" whose acquire() context-manager raises
    # CancelledError in __aenter__. The slot is acquired BEFORE this
    # await, so this is exactly the leak path codex flagged.
    class _CMRaisesCancelled:
        async def __aenter__(self):
            raise asyncio.CancelledError("simulated cancel")

        async def __aexit__(self, *a):
            return False

    class _FakeMgr:
        def acquire(self):
            return _CMRaisesCancelled()

    async def _ensure_ok():
        return _FakeMgr()

    monkeypatch.setattr(appmod, "_ensure_tts_manager_started", _ensure_ok)

    # Capability gate must pass so we reach the acquire path.
    from app.core import tts_service as _svc
    monkeypatch.setattr(_svc, "has_capability", lambda _c: True)

    sl = session_limiter.get_limiter()
    assert sl is not None and sl.active == 0

    with TestClient(appmod.app, raise_server_exceptions=False) as c:
        r = c.post("/tts/stream", json={"text": "x", "language": "en"})
        assert r.status_code >= 400

    assert sl.active == 0, (
        f"slot leaked on CancelledError during backend_cm.__aenter__: "
        f"limiter.active={sl.active}"
    )


def test_tts_clone_stream_cancelled_during_backend_acquire_releases_slot(
    monkeypatch, tmp_path,
):
    """Same shape as test_tts_stream_cancelled_during_backend_acquire — but
    against /tts/clone/stream, which had an identical leak path."""
    import asyncio

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LAZY_TTS", "1")
    monkeypatch.setenv("LANGUAGE_MODE", "disabled")
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "2")
    # Voice clone capability must be advertised so the early reject path
    # doesn't short-circuit before we hit the BackendManager acquire.
    monkeypatch.setenv("OVS_ENABLE_VOICE_CLONE", "1")

    from fastapi.testclient import TestClient

    session_limiter.init_limiter({})

    from app import main as appmod

    class _CMRaisesCancelled:
        async def __aenter__(self):
            raise asyncio.CancelledError("simulated cancel")

        async def __aexit__(self, *a):
            return False

    class _FakeMgr:
        def acquire(self):
            return _CMRaisesCancelled()

    async def _ensure_ok():
        return _FakeMgr()

    monkeypatch.setattr(appmod, "_ensure_tts_manager_started", _ensure_ok)

    from app.core import tts_service as _svc
    monkeypatch.setattr(_svc, "has_capability", lambda _c: True)

    sl = session_limiter.get_limiter()
    assert sl is not None and sl.active == 0

    with TestClient(appmod.app, raise_server_exceptions=False) as c:
        # The endpoint may 404 if voice-clone isn't wired in this build;
        # in that case the slot was never acquired so the test is moot.
        try:
            r = c.post(
                "/tts/clone/stream",
                json={"text": "x", "reference_audio": "ZHVtbXk=",
                      "language": "en"},
            )
        except Exception:
            r = None
        # Whether 404 / 400 / 500 — what matters is no slot leak.
        _ = r

    assert sl.active == 0, (
        f"slot leaked on CancelledError during /tts/clone/stream backend "
        f"acquire: limiter.active={sl.active}"
    )
