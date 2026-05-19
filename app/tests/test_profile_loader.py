"""Tests for app.core.profile_loader hot-reload semantics (PR1)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.core import profile_loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    """Reset profile_loader module-level state between tests."""
    # Start each test with empty operator set + empty applied set + no profile.
    monkeypatch.setattr(profile_loader, "_OPERATOR_KEYS", frozenset())
    monkeypatch.setattr(profile_loader, "_APPLIED_KEYS", set())
    monkeypatch.setattr(profile_loader, "_CURRENT_PROFILE", {})
    yield


def _write_profile(tmp_path: Path, name: str, body: dict) -> Path:
    body = {"name": name, **body}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_apply_writes_env_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_BAR", raising=False)
    p = _write_profile(tmp_path, "pA", {"env": {"FOO_BAR": "x"}})

    profile = profile_loader.apply_profile(str(p))

    assert profile["name"] == "pA"
    import os
    assert os.environ["FOO_BAR"] == "x"
    assert "FOO_BAR" in profile_loader.get_applied_keys()
    assert "OVS_PROFILE_NAME" in profile_loader.get_applied_keys()


def test_second_apply_overwrites_previous_profile_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"env": {"A_KEY": "1"}})
    profile_loader.apply_profile(str(a))
    assert os.environ["A_KEY"] == "1"

    b = _write_profile(tmp_path, "B", {"env": {"A_KEY": "2"}})
    profile_loader.apply_profile(str(b))
    assert os.environ["A_KEY"] == "2"  # bug #1: previously stuck at "1"


def test_second_apply_clears_keys_only_in_old_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("OVS_X", raising=False)
    monkeypatch.delenv("OVS_Y", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"env": {"OVS_X": "1"}})
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_X"] == "1"

    b = _write_profile(tmp_path, "B", {"env": {"OVS_Y": "2"}})
    profile_loader.apply_profile(str(b))
    assert "OVS_X" not in os.environ  # bug #5: stale key cleared
    assert os.environ["OVS_Y"] == "2"


def test_operator_env_never_overwritten(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("OVS_OPERATOR_TEST", "operator-value")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"OVS_OPERATOR_TEST"})
    )

    p = _write_profile(
        tmp_path, "P", {"env": {"OVS_OPERATOR_TEST": "profile-value"}}
    )
    profile_loader.apply_profile(str(p))

    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"
    assert "OVS_OPERATOR_TEST" not in profile_loader.get_applied_keys()


def test_snapshot_operator_keys_excludes_empty_values(monkeypatch):
    """docker-compose passes declared-but-unset vars as empty strings,
    not unset; these must not be treated as operator-owned (otherwise
    profile defaults silently fail to apply — orin-nx regression
    2026-05-20 with QWEN3_ARTIFACT_MANIFEST="")."""
    monkeypatch.setenv("QWEN3_ARTIFACT_MANIFEST", "")
    monkeypatch.setenv("QWEN3_ARTIFACT_SET", "")
    monkeypatch.setenv("QWEN3_ARTIFACT_ROOT", "/opt/models/qwen3-edgellm")

    snapshot = profile_loader._snapshot_operator_keys()

    assert "QWEN3_ARTIFACT_MANIFEST" not in snapshot
    assert "QWEN3_ARTIFACT_SET" not in snapshot
    assert "QWEN3_ARTIFACT_ROOT" in snapshot


def test_operator_env_not_cleared_on_reapply(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("OVS_OPERATOR_TEST", "operator-value")
    monkeypatch.setattr(
        profile_loader, "_OPERATOR_KEYS", frozenset({"OVS_OPERATOR_TEST"})
    )

    a = _write_profile(
        tmp_path, "A", {"env": {"OVS_OPERATOR_TEST": "p1", "OTHER": "o1"}}
    )
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"

    b = _write_profile(tmp_path, "B", {"env": {"OTHER": "o2"}})
    profile_loader.apply_profile(str(b))
    assert os.environ["OVS_OPERATOR_TEST"] == "operator-value"


def test_tts_model_id_recomputed_on_reload(tmp_path, monkeypatch):
    monkeypatch.delenv("OVS_TTS_MODEL_ID", raising=False)
    import os

    a = _write_profile(tmp_path, "A", {"tts_model_id": "kokoro-en", "env": {}})
    profile_loader.apply_profile(str(a))
    assert os.environ["OVS_TTS_MODEL_ID"] == "kokoro-en"

    b = _write_profile(tmp_path, "B", {"tts_model_id": "matcha-zh", "env": {}})
    profile_loader.apply_profile(str(b))
    assert os.environ["OVS_TTS_MODEL_ID"] == "matcha-zh"  # bug #3


def test_apply_profile_with_explicit_ref_param(tmp_path, monkeypatch):
    """Explicit profile_ref bypasses env resolution (bug #4 fix)."""
    monkeypatch.delenv("OVS_PROFILE", raising=False)
    monkeypatch.delenv("OVS_PROFILE_JSON", raising=False)
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)

    p = _write_profile(tmp_path, "explicit", {"env": {"K": "v"}})

    profile = profile_loader.apply_profile(str(p))
    assert profile["name"] == "explicit"
    import os
    assert os.environ["K"] == "v"


def test_concurrent_apply_thread_safe(tmp_path, monkeypatch):
    import os

    profiles = []
    for i in range(4):
        p = _write_profile(
            tmp_path, f"P{i}", {"env": {f"KEY_{i}": f"v{i}"}}
        )
        profiles.append(str(p))

    errors: list[BaseException] = []

    def worker(path: str) -> None:
        try:
            for _ in range(20):
                profile_loader.apply_profile(path)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in profiles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []

    # Final state must be self-consistent: the current profile's name must
    # match one of the inputs, and _APPLIED_KEYS must reflect that profile
    # (i.e. exactly one KEY_i is present in env among the four).
    current = profile_loader.current_profile()
    assert current.get("name", "").startswith("P")
    final_idx = int(current["name"][1:])

    applied = profile_loader.get_applied_keys()
    assert f"KEY_{final_idx}" in applied
    assert os.environ.get(f"KEY_{final_idx}") == f"v{final_idx}"
    for i in range(4):
        if i == final_idx:
            continue
        assert f"KEY_{i}" not in os.environ, (
            f"stale KEY_{i} leaked; final profile was P{final_idx}"
        )


def test_apply_profile_from_env_still_works(tmp_path, monkeypatch):
    """apply_profile_from_env() honors OVS_PROFILE (compat path)."""
    import os

    monkeypatch.delenv("OVS_PROFILE_JSON", raising=False)
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("COMPAT_KEY", raising=False)

    p = _write_profile(tmp_path, "compat", {"env": {"COMPAT_KEY": "ok"}})
    # OVS_PROFILE resolves via _profile_path which expects either a filename
    # under configs/profiles or an absolute path. Use absolute path here.
    monkeypatch.setenv("OVS_PROFILE", str(p))

    profile = profile_loader.apply_profile_from_env()
    assert profile["name"] == "compat"
    assert os.environ["COMPAT_KEY"] == "ok"


def test_apply_profile_returns_empty_when_no_ref(monkeypatch):
    """No env hints + no explicit ref → returns {} without touching state."""
    for k in ("OVS_PROFILE_JSON", "OVS_PROFILE", "OVS_PROFILE_DEFAULT",
              "LANGUAGE_MODE", "OVS_PRESET"):
        monkeypatch.delenv(k, raising=False)

    result = profile_loader.apply_profile()
    assert result == {}
    assert profile_loader.current_profile() == {}
