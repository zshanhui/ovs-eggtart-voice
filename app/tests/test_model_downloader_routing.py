"""Tests for app.core.model_downloader profile-driven routing.

Regression: orin-nano 2026-05-25 silent Qwen3-skip bug. When OVS_PROFILE
selected a Qwen3 ASR profile but the environment had LANGUAGE_MODE=zh_en
pre-set, ensure_models() routed by language_mode and skipped
_ensure_qwen3_artifacts(). After fix, routing is profile-driven first
(asr_backend/tts_backend) and language_mode-driven second; both UNION.
"""

from __future__ import annotations

from unittest.mock import patch

from app.core import model_downloader


def _no_op_download(url, dest_dir):  # pragma: no cover - safety net
    raise AssertionError(
        f"unexpected real download attempt: {url} -> {dest_dir}"
    )


def test_profile_with_trt_edge_llm_triggers_qwen3_even_when_lang_mode_zh_en(
    tmp_path, monkeypatch,
):
    """Profile asr_backend=jetson.trt_edge_llm must call _ensure_qwen3_artifacts
    even when language_mode='zh_en' (the legacy zh_en path would otherwise
    never call it). This is the core orin-nano regression fix."""
    from app.core import profile_loader
    monkeypatch.setattr(
        profile_loader,
        "current_profile",
        lambda: {
            "asr_backend": "jetson.trt_edge_llm",
            "tts_backend": "jetson.matcha_trt",
        },
    )
    # Patch the symbol the function actually calls (module-level lookup).
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend zh_en assets (matcha + paraformer) are already present
        # so no actual download fires. language_mode='zh_en' unions in
        # the legacy zh_en requirements alongside profile-driven matcha.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_called_once()


def test_no_profile_zh_en_legacy_path_does_not_call_qwen3(tmp_path, monkeypatch):
    """Backward-compat: no profile + LANGUAGE_MODE=zh_en must NOT trigger
    Qwen3 (legacy behaviour for users who never opted into profiles)."""
    from app.core import profile_loader
    monkeypatch.setattr(profile_loader, "current_profile", lambda: {})
    with patch.object(
        model_downloader, "_ensure_qwen3_artifacts"
    ) as mock_qwen3, patch.object(
        model_downloader, "_download_and_extract", side_effect=_no_op_download,
    ):
        # Pretend both zh_en models exist so we don't trip the downloader.
        for sub, files in (
            ("matcha-icefall-zh-en", ("model-steps-3.onnx", "tokens.txt", "lexicon.txt")),
            ("paraformer-streaming", ("encoder.onnx", "tokens.txt")),
        ):
            d = tmp_path / sub
            d.mkdir()
            for f in files:
                (d / f).write_text("x")

        model_downloader.ensure_models(
            language_mode="zh_en", model_dir=str(tmp_path),
        )

    mock_qwen3.assert_not_called()
