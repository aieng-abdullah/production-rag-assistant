"""BM25 TTL cache tests (PLAN.md PR-3): build-once, invalidate, empty-tenant."""

from src.services import bm25_cache as cache


def _fake_pipeline(monkeypatch, chunks):
    """Count builds; return a stable sentinel per build."""
    calls = {"load": 0, "build": 0}

    def fake_load(tenant_id):
        calls["load"] += 1
        return chunks

    def fake_build(chunk_list):
        calls["build"] += 1
        return f"index-{calls['build']}"

    monkeypatch.setattr(cache, "load_all_chunks", fake_load)
    monkeypatch.setattr(cache, "build_bm25_index", fake_build)
    cache.clear_all()
    return calls


def test_get_bm25_builds_once_until_invalidated(monkeypatch):
    calls = _fake_pipeline(monkeypatch, [{"page_content": "hello"}])

    first = cache.get_bm25("tenant-a")
    second = cache.get_bm25("tenant-a")

    assert first == second == "index-1"
    assert calls["load"] == 1

    cache.invalidate("tenant-a")
    third = cache.get_bm25("tenant-a")

    assert third == "index-2"
    assert calls["load"] == 2


def test_get_bm25_caches_empty_tenant(monkeypatch):
    """No chunks → None cached; repeated queries don't re-hit Chroma."""
    calls = _fake_pipeline(monkeypatch, [])

    first = cache.get_bm25("tenant-empty")
    second = cache.get_bm25("tenant-empty")

    assert first is None and second is None
    assert calls["load"] == 1
    assert calls["build"] == 0


def test_tenants_do_not_share_index(monkeypatch):
    calls = _fake_pipeline(monkeypatch, [{"page_content": "x"}])

    a = cache.get_bm25("tenant-a")
    b = cache.get_bm25("tenant-b")

    assert a != b
    assert calls["load"] == 2


def test_invalidate_unknown_tenant_is_noop():
    cache.clear_all()

    cache.invalidate("never-seen")

    assert "never-seen" not in cache._cache
