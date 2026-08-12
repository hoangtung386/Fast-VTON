"""Tests for the Hub prefetch helper."""

import pytest

from src.vton import hub


def test_asset_patterns_are_narrow_and_well_formed() -> None:
    """Every entry names a repo and restricts what is pulled from it."""
    assert hub.HUB_ASSETS
    seen = set()
    for repo_id, patterns in hub.HUB_ASSETS:
        assert "/" in repo_id, f"{repo_id!r} is not a Hub repo id"
        assert repo_id not in seen, f"{repo_id} listed twice"
        seen.add(repo_id)
        assert patterns, f"{repo_id} would download the entire repository"
        # `.bin` duplicates of every `.safetensors` weight are pure waste.
        assert not any(p.endswith(".bin") for p in patterns)


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def flaky(repo_id: str, **_: object) -> str:
        calls.append(repo_id)
        if len(calls) < 3:
            raise ConnectionError("connection broken")
        return "/cache"

    monkeypatch.setattr(hub, "snapshot_download", flaky)
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)

    hub.prefetch_hub_assets([("org/repo", ["*.json"])], attempts=5, initial_backoff=0.0)

    assert calls == ["org/repo"] * 3


def test_reraises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fails(repo_id: str, **_: object) -> str:
        raise ConnectionError("connection broken")

    monkeypatch.setattr(hub, "snapshot_download", always_fails)
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)

    with pytest.raises(ConnectionError):
        hub.prefetch_hub_assets([("org/repo", ["*.json"])], attempts=2, initial_backoff=0.0)


def test_stops_calling_after_first_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda repo_id, **_: calls.append(repo_id))

    hub.prefetch_hub_assets([("a/b", ["*"]), ("c/d", ["*"])], attempts=3)

    assert calls == ["a/b", "c/d"]


def test_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="attempts"):
        hub.prefetch_hub_assets([("a/b", ["*"])], attempts=0)
