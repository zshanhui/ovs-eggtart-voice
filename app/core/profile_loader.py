"""Runtime profile loader for deploy-time backend selection.

Profiles set environment defaults before backend modules are imported.
Operator-supplied env (present at process start) is preserved across reloads;
profile-applied keys are tracked so a subsequent ``apply_profile()`` cleans
stale keys before writing the new profile's defaults.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_OPERATOR_KEY_PREFIXES: tuple[str, ...] = (
    "OVS_", "LANGUAGE_MODE", "MODEL_DIR", "ASR_", "TTS_", "EDGE_LLM_",
    "QWEN3_", "KOKORO_", "MATCHA_", "RK_", "RKLLM_", "SHERPA_",
    "STREAMING_", "VOCOS_", "HF_", "CUDA_", "TRT_", "NVIDIA_",
)


def _snapshot_operator_keys() -> frozenset[str]:
    """Snapshot every env key matching an operator prefix at import time.

    These keys are considered owned by the operator (or the surrounding
    container/CI environment) and will NEVER be overwritten or cleared by
    ``apply_profile``.
    """
    return frozenset(
        k for k in os.environ.keys()
        if k.startswith(_OPERATOR_KEY_PREFIXES)
    )


# Snapshot taken exactly once at module import.
_OPERATOR_KEYS: frozenset[str] = _snapshot_operator_keys()

# Keys written by the most recent ``apply_profile`` call. Used to clear stale
# values when reloading a different profile.
_APPLIED_KEYS: set[str] = set()

# Most recently loaded profile dict (empty when none has been applied).
_CURRENT_PROFILE: dict = {}

# Reentrant lock guarding the three pieces of state above.
_PROFILE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _project_root() -> Path:
    # __file__ = <repo>/app/core/profile_loader.py → parents[2] = <repo>
    return Path(__file__).resolve().parents[2]


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return _project_root() / "configs" / "profiles" / candidate.name


def _select_profile_ref() -> str | None:
    """Resolve a profile reference from env (legacy ``apply_profile_from_env`` path)."""
    profile_ref = _env("OVS_PROFILE_JSON") or _env("OVS_PROFILE") or _env("OVS_PROFILE_DEFAULT")
    if not profile_ref:
        language_mode = os.environ.get("LANGUAGE_MODE", "").strip()
        if language_mode == "zh_en":
            profile_ref = "jetson-zh-en"
        elif language_mode == "multilanguage":
            profile_ref = "jetson-multilang-highperf"
    if not profile_ref:
        preset = _env("OVS_PRESET")
        if preset:
            from app.core.profile_selector import resolve_profile_name, UnsupportedPreset
            try:
                profile_ref = resolve_profile_name(preset)
            except UnsupportedPreset as exc:
                logger.error("preset %r not supported on this device: %s", preset, exc)
                raise
            logger.info("preset %r → profile %r", preset, profile_ref)
    return profile_ref


def _derive_tts_model_id(profile: dict) -> str | None:
    """Compute the OVS_TTS_MODEL_ID value implied by ``profile``.

    Returns ``None`` when the profile has no usable hint; callers decide
    whether to write it into env (and whether operator ownership overrides).
    """
    model_id = profile.get("tts_model_id")
    if model_id:
        return str(model_id)

    engines = profile.get("required_engines") or []
    tts_keys = ("kokoro", "matcha", "qwen3", "piper")
    for engine in engines:
        mid = engine.get("model_id", "")
        if any(k in str(mid).lower() for k in tts_keys):
            return str(mid)

    logger.warning(
        "Profile %r has no tts_model_id; speaker tables may mis-scope. "
        "Add tts_model_id to your profile JSON.",
        profile.get("name"),
    )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_profile() -> dict:
    """Return the most recently loaded profile dict (or ``{}``)."""
    with _PROFILE_LOCK:
        return _CURRENT_PROFILE


def get_applied_keys() -> frozenset[str]:
    """Return a snapshot of env keys written by the most recent ``apply_profile``."""
    with _PROFILE_LOCK:
        return frozenset(_APPLIED_KEYS)


def apply_profile(
    profile_ref: str | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
    resolve_engines: bool = False,
) -> dict:
    """Load a profile and reconcile process env against it.

    Args:
        profile_ref: Profile name or absolute path. ``None`` falls back to env
            resolution (compatible with ``apply_profile_from_env``).
        overrides: Extra env entries layered on top of the profile (respecting
            operator ownership).
        resolve_engines: Reserved for PR4 — when True, future versions will
            call ``engine_resolver`` to materialize artifact paths. Currently
            a no-op.

    Returns:
        The parsed profile dict (or ``{}`` when no profile could be resolved).
    """
    global _CURRENT_PROFILE

    with _PROFILE_LOCK:
        ref = profile_ref if profile_ref else _select_profile_ref()
        if not ref:
            return {}

        path = _profile_path(ref)
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Compute the desired env from profile JSON (with $VAR expansion).
        new_env: dict[str, str] = {}
        for key, value in (profile.get("env") or {}).items():
            new_env[key] = os.path.expandvars(str(value))

        derived: dict[str, str] = {
            "OVS_PROFILE_NAME": str(profile.get("name", path.stem)),
        }
        tts_model_id = _derive_tts_model_id(profile)
        if tts_model_id is not None:
            derived["OVS_TTS_MODEL_ID"] = tts_model_id

        merged: dict[str, str] = {**new_env, **derived}
        if overrides:
            for k, v in overrides.items():
                merged[k] = str(v)

        new_keys = set(merged.keys())

        # 1. Clear stale keys (in previous profile but not in this one),
        #    skipping operator-owned keys.
        stale = _APPLIED_KEYS - new_keys
        for k in stale:
            if k not in _OPERATOR_KEYS:
                os.environ.pop(k, None)

        # 2. Write new values, unconditionally overwriting unless operator-owned.
        for k, v in merged.items():
            if k in _OPERATOR_KEYS:
                continue
            os.environ[k] = v

        # 3. Update bookkeeping.
        _APPLIED_KEYS.clear()
        _APPLIED_KEYS.update(k for k in new_keys if k not in _OPERATOR_KEYS)
        _CURRENT_PROFILE = profile

        if resolve_engines:
            # TODO(PR4): wire app.core.engine_resolver here.
            pass

        logger.info(
            "Applied profile %s from %s (%d env keys; %d stale cleared)",
            derived["OVS_PROFILE_NAME"],
            path,
            len(_APPLIED_KEYS),
            len(stale),
        )
        return profile


def apply_profile_from_env() -> dict:
    """Compatibility shim: resolve profile from env and apply it.

    Resolution order:
      1. ``OVS_PROFILE_JSON`` — explicit path.
      2. ``OVS_PROFILE`` / ``OVS_PROFILE_DEFAULT`` — profile name.
      3. ``LANGUAGE_MODE`` heuristic (``zh_en`` / ``multilanguage``).
      4. ``OVS_PRESET`` — resolved via ``profile_selector``.
    """
    return apply_profile(None)
