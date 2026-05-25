"""WorkerIO migration regression coverage for TRTEdgeLLMASRBackend.

Verifies that the post-migration paths (`_transcribe_worker`,
`_worker_request`, streaming begin/chunk/finalize, cancel_and_finalize)
still preserve the legacy ASR worker protocol contracts:

  * offline transcribe sends one input line and receives one terminal
    ``done`` event keyed by ``id``;
  * streaming events (begin_ack / partial / final / segment_rotation /
    end_ack) are read as single-event responses despite the WorkerIO
    request iterator's "loop until done" default — `_worker_request`
    breaks after the first event so each ``session_id``-keyed
    inflight queue is reset between protocol lines;
  * worker subprocess death surfaces as ``WorkerExitError``;
  * the same ``session_id`` is reused across begin → chunk → end
    without leaking inflight queues.

Mocks the subprocess with ``_FakeProc`` / ``_FakeStdoutQueue`` (same
fixtures as ``test_worker_io.py``) so we exercise the real
``WorkerIO`` instance.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from app.core.worker_io import WorkerIO
from app.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    WorkerExitError,
)


# ---------------------------------------------------------------------------
# Subprocess mocks (mirror of test_worker_io.py fixtures)
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self.writes.append(s)
        return len(s)

    def flush(self) -> None:
        pass


class _FakeStdoutQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[str | None]" = queue.Queue()

    def feed(self, line: str) -> None:
        self._q.put(line if line.endswith("\n") else line + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdoutQueue()


def _make_backend_with_wio() -> tuple[TRTEdgeLLMASRBackend, _FakeProc, WorkerIO]:
    """Build a backend wired to a fake subprocess + WorkerIO.

    Bypasses ``_ensure_worker`` since that would try to spawn a real
    binary. Sets ``_worker`` + ``_wio`` directly so the request paths
    under test can run end-to-end against the mock stdin/stdout.
    """
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    backend = TRTEdgeLLMASRBackend()
    backend._worker = proc  # type: ignore[assignment]
    backend._wio = wio
    backend._ensure_worker = lambda: None  # already wired
    return backend, proc, wio


# ---------------------------------------------------------------------------
# offline _transcribe_worker — single done event terminates
# ---------------------------------------------------------------------------


def test_transcribe_worker_routes_through_wio_and_returns_text():
    backend, proc, _wio = _make_backend_with_wio()

    def _feeder():
        # Wait for the request to land on stdin, then echo back a done
        # event with matching id.
        for _ in range(50):
            if proc.stdin.writes:
                break
            time.sleep(0.01)
        assert proc.stdin.writes, "expected the transcribe request on stdin"
        first = json.loads(proc.stdin.writes[0])
        rid = first["id"]
        proc.stdout.feed(json.dumps({
            "id": rid,
            "event": "done",
            "ok": True,
            "responses": [{"output_text": "hello world"}],
        }))

    threading.Thread(target=_feeder, daemon=True).start()
    res = backend._transcribe_worker("/tmp/fake.safetensors", elapsed_mel_s=0.05)
    assert res.text == "hello world"


def test_transcribe_worker_worker_exit_raises_runtime_error():
    """``_transcribe_worker`` translates WorkerExitError → RuntimeError.

    Preserves the legacy error contract that downstream callers (offline
    transcribe path) rely on — a worker death surfaces as a generic
    RuntimeError with the stderr tail appended, not the WorkerIO sentinel.
    """
    backend, proc, _wio = _make_backend_with_wio()

    def _kill():
        time.sleep(0.02)
        proc.stdout.eof()

    threading.Thread(target=_kill, daemon=True).start()
    with pytest.raises(RuntimeError):
        backend._transcribe_worker("/tmp/fake.safetensors", elapsed_mel_s=0.05)


# ---------------------------------------------------------------------------
# streaming _worker_request — single-event-per-line semantics
# ---------------------------------------------------------------------------


def test_worker_request_returns_first_event_only():
    """Streaming events (begin_ack, partial, ...) are one event per line.

    Even though ``wio.request()`` would normally loop until ``done``,
    ``_worker_request`` must break after the first event so the same
    ``id`` can be reused for the next begin/chunk/end line.
    """
    backend, proc, _wio = _make_backend_with_wio()

    def _feeder():
        for _ in range(50):
            if proc.stdin.writes:
                break
            time.sleep(0.01)
        proc.stdout.feed(json.dumps({"id": "sess-1", "event": "begin_ack"}))

    threading.Thread(target=_feeder, daemon=True).start()
    resp = backend._worker_request({"event": "begin", "id": "sess-1"})
    assert resp == {"id": "sess-1", "event": "begin_ack"}


def test_worker_request_session_id_persists_across_lines():
    """begin → chunk → end can all use the same session_id without
    leaking inflight queues in the underlying WorkerIO.

    After each call, the inflight queue for ``sess-2`` should be empty
    so the next ``_worker_request`` can re-register the same id cleanly.
    """
    backend, proc, _wio = _make_backend_with_wio()

    pending = ["begin_ack", "partial", "end_ack"]
    pending_lock = threading.Lock()

    def _feeder():
        idx = 0
        while idx < len(pending):
            # Wait for the next stdin line to land before responding.
            target_writes = idx + 1
            for _ in range(100):
                if len(proc.stdin.writes) >= target_writes:
                    break
                time.sleep(0.01)
            with pending_lock:
                event = pending[idx]
            payload = {"id": "sess-2", "event": event}
            if event == "partial":
                payload["text"] = "hi"
            proc.stdout.feed(json.dumps(payload))
            idx += 1

    threading.Thread(target=_feeder, daemon=True).start()
    r1 = backend._worker_request({"event": "begin", "id": "sess-2"})
    assert r1["event"] == "begin_ack"
    r2 = backend._worker_request({"event": "chunk", "id": "sess-2"})
    assert r2["event"] == "partial"
    assert r2["text"] == "hi"
    r3 = backend._worker_request({"event": "end", "id": "sess-2"})
    assert r3["event"] == "end_ack"
    # Inflight map fully drained — no leak across same-id reuse.
    assert _wio._inflight == {}, _wio._inflight


def test_worker_request_worker_exit_raises_worker_exit_error():
    backend, proc, _wio = _make_backend_with_wio()

    def _kill():
        time.sleep(0.02)
        proc.stdout.eof()

    threading.Thread(target=_kill, daemon=True).start()
    with pytest.raises(WorkerExitError):
        backend._worker_request({"event": "begin", "id": "sess-3"})
    # _worker cleared so next call rebuilds.
    assert backend._worker is None


def test_worker_request_classifies_no_active_session():
    """Typed error classification still fires through the WorkerIO path."""
    from app.backends.jetson.trt_edge_llm_asr import NoActiveSessionError

    backend, proc, _wio = _make_backend_with_wio()

    def _feeder():
        for _ in range(50):
            if proc.stdin.writes:
                break
            time.sleep(0.01)
        proc.stdout.feed(json.dumps({
            "id": "sess-4",
            "event": "error",
            "ok": False,
            "error": "no active session",
        }))

    threading.Thread(target=_feeder, daemon=True).start()
    with pytest.raises(NoActiveSessionError):
        backend._worker_request({"event": "chunk", "id": "sess-4"})
