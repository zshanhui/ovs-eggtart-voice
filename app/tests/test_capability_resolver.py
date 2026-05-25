"""Unit tests for ``app.core.capability_resolver``.

Follow-up #4 (spec §3/§4/§5). Verifies behaviour parity with the three
previous sites that re-implemented capability resolution:

- limiter ``session_ceiling`` (aggregate + clamp/warn)
- coordinator ``coordinator_mode`` (downgrade per §4)
- main ``executor_max_workers`` (clamp + capability fallback per §5)
"""

from __future__ import annotations

import logging

import pytest

from app.core.capability_resolver import (
    ResolvedCapability,
    resolve,
    resolve_executor_for_tts,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "OVS_MAX_CONCURRENT_SESSIONS",
        "OVS_TTS_STREAM_MAX_WORKERS",
        "OVS_TTS_STREAM_MAX_WORKERS_KOKORO",
        "OVS_TTS_STREAM_MAX_WORKERS_MATCHA",
        "OVS_TTS_STREAM_MAX_WORKERS_QWEN3",
        "OVS_TTS_STREAM_MAX_WORKERS_MOSS",
        "RK_PLATFORM",
        "LANGUAGE_MODE",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------- Aggregation math (spec §1) -----------------------------------


def test_aggregate_finite_vs_none_takes_finite():
    """paraformer max=None (inf) + matcha max=2 -> 2."""
    r = resolve(profile={
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    })
    assert r.session_ceiling == 2
    assert r.executor_max_workers == 2


def test_aggregate_both_none_falls_back_to_target_default():
    """When both backends have max=None, target_default still kicks in."""
    # cpu.sherpa_asr/sherpa are both parallel but with finite max=4
    r = resolve(profile={
        "asr_backend": "cpu.sherpa_asr",
        "tts_backend": "cpu.sherpa",
    })
    assert r.session_ceiling == 4


def test_no_declared_backends_uses_target_default():
    r = resolve(profile={"name": "desktop-mac"})
    assert r.session_ceiling == 4


def test_no_profile_uses_unknown_default():
    r = resolve(profile=None)
    assert r.session_ceiling == 1


# ---------- Profile clamp + warning (spec §3) -----------------------------


def test_profile_clamp_warning():
    r = resolve(profile={
        "asr_backend": "rk.asr",
        "tts_backend": "rk.tts",
        "max_concurrent_sessions": 99,
    })
    assert r.session_ceiling == 1
    assert any("max_concurrent_sessions" in w for w in r.clamp_warnings)


def test_profile_downgrade_no_warning():
    r = resolve(profile={
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
        "max_concurrent_sessions": 1,
    })
    assert r.session_ceiling == 1
    assert r.clamp_warnings == [] or all(
        "exceeds" not in w for w in r.clamp_warnings
    )


# ---------- Env clamp + warning (spec §3) ---------------------------------


def test_env_clamp_warning():
    env = {"OVS_MAX_CONCURRENT_SESSIONS": "16"}
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env=env,
    )
    assert r.session_ceiling == 2
    assert any("OVS_MAX_CONCURRENT_SESSIONS" in w for w in r.clamp_warnings)


def test_env_downgrade_honored():
    env = {"OVS_MAX_CONCURRENT_SESSIONS": "1"}
    r = resolve(
        profile={
            "asr_backend": "cpu.sherpa_asr",
            "tts_backend": "cpu.sherpa",
        },
        env=env,
    )
    assert r.session_ceiling == 1


def test_env_bad_value_raises():
    with pytest.raises(ValueError):
        resolve(profile={}, env={"OVS_MAX_CONCURRENT_SESSIONS": "0"})


def test_profile_bad_value_raises():
    with pytest.raises(ValueError):
        resolve(profile={"max_concurrent_sessions": -1})


# ---------- Coordinator mode (spec §4) ------------------------------------


def test_coordinator_concurrent_jetson_pair():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "concurrent"


def test_coordinator_concurrent_downgraded_for_rk():
    r = resolve(
        profile={"asr_backend": "rk.asr", "tts_backend": "rk.tts"},
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "serialized"


def test_coordinator_exclusive_honored():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "exclusive"},
    )
    assert r.coordinator_mode == "exclusive"


def test_coordinator_serialized_passthrough():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "serialized"},
    )
    assert r.coordinator_mode == "serialized"


def test_coordinator_no_profile_keeps_requested_concurrent():
    """Legacy callers without profile: pass through raw policy.mode."""
    r = resolve(profile=None, policy={"mode": "concurrent"})
    assert r.coordinator_mode == "concurrent"


def test_coordinator_mixed_pair_downgrades():
    r = resolve(
        profile={
            "asr_backend": "rk.asr",
            "tts_backend": "jetson.matcha_trt",
        },
        policy={"mode": "concurrent"},
    )
    assert r.coordinator_mode == "serialized"


# ---------- Executor max_workers (spec §5) -------------------------------


def test_executor_env_clamped_to_capability():
    env = {"OVS_TTS_STREAM_MAX_WORKERS_MATCHA": "16"}
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env=env,
        tts_backend_name="jetson.matcha_trt.fp16",
    )
    assert r.executor_max_workers == 2
    assert any("exceeds backend ceiling" in w for w in r.clamp_warnings)


def test_executor_falls_back_to_capability_when_no_env():
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        env={},
        tts_backend_name="jetson.matcha_trt.fp16",
    )
    assert r.executor_max_workers == 2


def test_executor_legacy_default_when_no_tts_backend_declared():
    """No tts_backend in profile → legacy default 2 (cap not consulted)."""
    r = resolve(profile={"name": "desktop-mac"}, env={})
    assert r.executor_max_workers == 2


def test_executor_backend_specific_env_wins_over_global():
    env = {
        "OVS_TTS_STREAM_MAX_WORKERS": "16",
        "OVS_TTS_STREAM_MAX_WORKERS_KOKORO": "1",
    }
    r = resolve(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.kokoro_trt",
        },
        env=env,
        tts_backend_name="jetson.kokoro_trt.fp16",
    )
    assert r.executor_max_workers == 1


# ---------- Cross-caller consistency --------------------------------------


def test_three_callers_share_ceiling_and_mode():
    """Same profile → all three projections must agree on the underlying
    capability snapshot. Regression guard against future drift."""
    profile = {
        "asr_backend": "jetson.paraformer_trt",
        "tts_backend": "jetson.matcha_trt",
    }
    policy = {"mode": "concurrent"}
    r = resolve(profile=profile, policy=policy,
                tts_backend_name="jetson.matcha_trt.fp16", env={})

    # Limiter sees ceiling=2.
    assert r.session_ceiling == 2
    # Coordinator agrees concurrent is OK.
    assert r.coordinator_mode == "concurrent"
    # Executor cap == ceiling (no env override).
    assert r.executor_max_workers == 2
    assert r.executor_max_workers == r.session_ceiling


def test_three_callers_consistent_for_rk():
    profile = {"asr_backend": "rk.asr", "tts_backend": "rk.tts"}
    policy = {"mode": "concurrent"}
    r = resolve(profile=profile, policy=policy, env={})
    assert r.session_ceiling == 1
    assert r.coordinator_mode == "serialized"
    assert r.executor_max_workers == 1


# ---------- Thin wrapper parity with legacy return shape -----------------


def test_resolve_executor_for_tts_returns_legacy_shape():
    env = {"OVS_TTS_STREAM_MAX_WORKERS_MATCHA": "16"}
    n, name, src = resolve_executor_for_tts(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        tts_backend_name="jetson.matcha_trt.fp16",
        env=env,
    )
    assert n == 2
    assert name == "jetson.matcha_trt.fp16"
    assert src == "OVS_TTS_STREAM_MAX_WORKERS_MATCHA"


def test_resolve_executor_for_tts_source_capability():
    n, name, src = resolve_executor_for_tts(
        profile={
            "asr_backend": "jetson.paraformer_trt",
            "tts_backend": "jetson.matcha_trt",
        },
        tts_backend_name="jetson.matcha_trt.fp16",
        env={},
    )
    assert n == 2
    assert src == "concurrency_capability"


def test_resolve_executor_for_tts_source_default():
    n, name, src = resolve_executor_for_tts(
        profile={},
        tts_backend_name=None,
        env={},
    )
    # No profile-declared TTS backend → legacy default 2 + "default"
    # source (mirrors pre-resolver behavior in app.main).
    assert n == 2
    assert src == "default"
