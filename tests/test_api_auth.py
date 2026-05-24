"""Unit tests for app.core.api_auth."""

import os
import pytest

from app.core import api_auth, metrics


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    metrics._reset_for_tests()


def test_unset_env_disables_auth():
    assert api_auth.is_enabled() is False
    assert api_auth._configured_keys() == []


def test_empty_env_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "")
    assert api_auth.is_enabled() is False


def test_whitespace_only_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "   ")
    assert api_auth.is_enabled() is False


def test_comma_only_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", ",,,")
    assert api_auth.is_enabled() is False


def test_single_key_enables(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "abc")
    assert api_auth.is_enabled() is True
    assert api_auth._configured_keys() == ["abc"]


def test_multiple_keys_with_whitespace(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", " abc , def ,,ghi ")
    assert api_auth._configured_keys() == ["abc", "def", "ghi"]


def test_extract_bearer():
    assert api_auth._extract_bearer(None) is None
    assert api_auth._extract_bearer("") is None
    assert api_auth._extract_bearer("Basic abc") is None
    # "Bearer " with trailing whitespace has no second token after split.
    assert api_auth._extract_bearer("Bearer ") is None
    assert api_auth._extract_bearer("Bearer abc") == "abc"
    assert api_auth._extract_bearer("bearer ABC") == "ABC"
    assert api_auth._extract_bearer("Bearer  abc  ") == "abc"


def test_mask_key():
    assert api_auth.mask_key(None) == "<missing>"
    assert api_auth.mask_key("") == "<missing>"
    assert api_auth.mask_key("short") == "short..."
    # Never returns the raw long key (>8 chars truncated).
    assert api_auth.mask_key("supersecretkey1234567") == "supersec..."
    assert "1234567" not in api_auth.mask_key("supersecretkey1234567")


def test_matches_constant_time():
    keys = ["abc", "def"]
    assert api_auth._matches("abc", keys) is True
    assert api_auth._matches("def", keys) is True
    assert api_auth._matches("ghi", keys) is False
    assert api_auth._matches("", keys) is False


def test_hot_update_changes_check(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "k1")
    assert api_auth._matches("k1", api_auth._configured_keys())
    monkeypatch.setenv("OVS_API_KEYS", "k2")
    assert not api_auth._matches("k1", api_auth._configured_keys())
    assert api_auth._matches("k2", api_auth._configured_keys())


# ── HTTP integration via TestClient ────────────────────────────────────

def _build_app():
    from fastapi import FastAPI, Depends, Request
    from app.core.api_auth import check_http
    app = FastAPI()

    @app.get("/protected")
    def protected(request: Request, _: None = Depends(check_http)):
        return {"ok": True}

    return app


def test_http_disabled_lets_through(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected")
        assert r.status_code == 200


def test_http_enabled_missing_key_401(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected")
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"
        body = r.json()
        # FastAPI nests HTTPException.detail under "detail"
        assert body["detail"]["error"] == "unauthorized"


def test_http_enabled_wrong_key_401(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401


def test_http_enabled_valid_key_passes(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Bearer mysecret"})
        assert r.status_code == 200


def test_http_non_bearer_scheme_fails(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Basic mysecret"})
        assert r.status_code == 401
