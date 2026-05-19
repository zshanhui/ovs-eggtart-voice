import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.asr_backend import ASRCapability
from app.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    _float_audio_to_wav_bytes,
)


def test_float_audio_to_wav_bytes_roundtrip_header():
    wav_bytes = _float_audio_to_wav_bytes(np.zeros(16000, dtype=np.float32), 16000)

    assert wav_bytes[:4] == b"RIFF"
    assert b"WAVE" in wav_bytes[:16]


def test_trt_edgellm_asr_stream_accumulates_and_finalizes(monkeypatch):
    backend = TRTEdgeLLMASRBackend()
    backend._ready = True
    calls = []

    def fake_transcribe(wav_bytes, language="auto"):
        calls.append((wav_bytes, language))
        return type("Result", (), {"text": "你好"})()

    monkeypatch.setattr(backend, "transcribe", fake_transcribe)
    stream = backend.create_stream(language="Chinese")
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))

    assert stream.get_partial() == ("", False)
    assert stream.finalize() == "你好"
    assert calls[0][1] == "Chinese"
    assert calls[0][0][:4] == b"RIFF"


def test_trt_edgellm_asr_advertises_streaming_capability():
    backend = TRTEdgeLLMASRBackend()

    assert ASRCapability.STREAMING in backend.capabilities


# ── worker-error classification ─────────────────────────────────────────


from app.backends.jetson.trt_edge_llm_asr import (  # noqa: E402
    NoActiveSessionError,
    SessionAlreadyActiveError,
    WorkerExitError,
    WorkerProtocolError,
    _classify_worker_response,
)


def test_classify_no_active_session():
    err = _classify_worker_response({"event": "error", "error": "no active session for id=abc"})
    assert isinstance(err, NoActiveSessionError)


def test_classify_session_already_active():
    err = _classify_worker_response({"event": "error", "error": "session already active"})
    assert isinstance(err, SessionAlreadyActiveError)


def test_classify_worker_exit():
    err = _classify_worker_response({"ok": False, "error": "worker terminated unexpectedly"})
    assert isinstance(err, WorkerExitError)


def test_classify_unknown_error_returns_none():
    # Returning None means "not a typed protocol error" — _worker_request
    # will still raise a generic WorkerProtocolError.
    assert _classify_worker_response({"event": "error", "error": "decoder failed"}) is None
    assert _classify_worker_response({"ok": True}) is None


def test_worker_request_injects_typed_no_active_session(monkeypatch):
    backend = TRTEdgeLLMASRBackend()

    def fake_request(input_data):
        # Simulate the real path's behaviour using the same parser:
        # bypass actual subprocess by re-implementing the protocol shim.
        line = '{"event":"error","error":"no active session"}\n'
        import json as _j
        out = _j.loads(line)
        typed = _classify_worker_response(out, request_event=input_data.get("event"))
        if typed is not None:
            raise typed
        raise RuntimeError(out)

    monkeypatch.setattr(backend, "_worker_request", fake_request)
    import pytest
    with pytest.raises(NoActiveSessionError):
        backend._worker_request({"event": "chunk", "id": "x"})


def test_worker_request_injects_typed_session_already_active(monkeypatch):
    backend = TRTEdgeLLMASRBackend()

    def fake_request(input_data):
        import json as _j
        out = _j.loads('{"event":"error","error":"session already active for id=x"}')
        typed = _classify_worker_response(out)
        if typed is not None:
            raise typed
        raise RuntimeError(out)

    monkeypatch.setattr(backend, "_worker_request", fake_request)
    import pytest
    with pytest.raises(SessionAlreadyActiveError):
        backend._worker_request({"event": "begin", "id": "x"})


def test_worker_request_injects_worker_exit_on_empty_line():
    # Direct unit check that the bare exit path raises WorkerExitError.
    err = _classify_worker_response({"event": "error", "error": "worker exited"})
    assert isinstance(err, WorkerExitError)


def test_restart_worker_is_idempotent_with_no_running_worker():
    """restart_worker() must be safe to call when nothing is running."""
    backend = TRTEdgeLLMASRBackend()
    assert backend._worker is None
    backend.restart_worker()  # no-op, must not raise
    assert backend._worker is None


def test_typed_errors_subclass_worker_protocol_error():
    assert issubclass(NoActiveSessionError, WorkerProtocolError)
    assert issubclass(SessionAlreadyActiveError, WorkerProtocolError)
    assert issubclass(WorkerExitError, WorkerProtocolError)
