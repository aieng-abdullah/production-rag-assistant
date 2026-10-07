"""Per-IP sliding-window rate limiting (security/rate-limit).

Clock is injected by monkeypatching `rate_limit._now`, so window expiry is
tested deterministically instead of sleeping.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import rate_limit
from src.api.app import create_app
from src.config import Config
from src.db.database import Base, get_engine, reset_engine

ANON_PAYLOAD = {"device_id": "device-0001-abc"}


@pytest.fixture()
def clock(monkeypatch):
    state = {"now": 1_000_000.0}
    monkeypatch.setattr(rate_limit, "_now", lambda: state["now"])
    return state


@pytest.fixture()
def api():
    return create_app()


@pytest.fixture()
def client(api, clock):
    return TestClient(api)


@pytest.fixture()
def auth_api(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'rl.db'}")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield create_app()
    reset_engine()


@pytest.fixture()
def auth_client(auth_api):
    return TestClient(auth_api)


def test_burst_trips_429_after_global_limit(client):
    statuses = [client.get("/probe").status_code for _ in range(61)]

    assert statuses[:60] == [404] * 60
    assert statuses[60] == 429


def test_429_body_and_retry_after_contract(client):
    for _ in range(60):
        client.get("/probe")

    response = client.get("/probe")

    assert response.headers["Retry-After"] == "60"
    assert response.json() == {"detail": "Too many requests. Try again in 60s."}


def test_window_refills_after_time_advances(client, clock):
    for _ in range(60):
        client.get("/probe")
    assert client.get("/probe").status_code == 429

    clock["now"] += rate_limit.GLOBAL_WINDOW_SECONDS + 1

    assert client.get("/probe").status_code == 404


def test_partial_window_drains_progressively(client, clock):
    first_hit = clock["now"]
    for _ in range(60):
        client.get("/probe")

    clock["now"] = first_hit + rate_limit.GLOBAL_WINDOW_SECONDS / 2
    blocked = client.get("/probe")
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "30"

    clock["now"] = first_hit + rate_limit.GLOBAL_WINDOW_SECONDS + 1
    assert client.get("/probe").status_code == 404


def test_ip_isolation(api, clock):
    blocked = TestClient(api, client=("203.0.113.7", 50000))
    other = TestClient(api, client=("198.51.100.9", 50000))

    for _ in range(60):
        assert blocked.get("/probe").status_code == 404
    assert blocked.get("/probe").status_code == 429

    assert other.get("/probe").status_code == 404


def test_health_always_exempt(client):
    statuses = [client.get("/health").status_code for _ in range(200)]

    assert statuses == [200] * 200


def test_options_preflight_exempt(client):
    statuses = [client.options("/chat").status_code for _ in range(200)]

    assert 429 not in statuses


def test_success_path_unaffected_under_limit(client):
    first = client.get("/health")
    second = client.get("/probe")

    assert first.status_code == 200
    assert second.status_code == 404
    assert "Retry-After" not in first.headers


def test_anonymous_capped_at_three_per_hour(auth_client):
    for _ in range(3):
        response = auth_client.post("/auth/anonymous", json=ANON_PAYLOAD)
        assert response.status_code == 200

    blocked = auth_client.post("/auth/anonymous", json=ANON_PAYLOAD)

    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "3600"
    assert blocked.json() == {"detail": "Too many requests. Try again in 3600s."}


def test_anonymous_window_refills_after_hour(auth_client, clock):
    for _ in range(3):
        auth_client.post("/auth/anonymous", json=ANON_PAYLOAD)
    assert auth_client.post("/auth/anonymous", json=ANON_PAYLOAD).status_code == 429

    clock["now"] += rate_limit.ANON_WINDOW_SECONDS + 1

    assert auth_client.post("/auth/anonymous", json=ANON_PAYLOAD).status_code == 200


def test_other_auth_routes_separate_bucket(auth_client):
    for _ in range(3):
        assert auth_client.post("/auth/anonymous", json=ANON_PAYLOAD).status_code == 200

    for _ in range(5):
        assert (
            auth_client.get("/auth/google", follow_redirects=False).status_code
            in (302, 501)
        )

    assert auth_client.get("/auth/google", follow_redirects=False).status_code == 429


def test_auth_429_does_not_block_global_bucket(client):
    for _ in range(5):
        assert client.get("/auth/google", follow_redirects=False).status_code in (
            302,
            501,
        )
    assert client.get("/auth/google", follow_redirects=False).status_code == 429

    assert client.get("/probe").status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/health", None),
        ("HEAD", "/health", None),
        ("OPTIONS", "/chat", None),
        ("POST", "/auth/anonymous", rate_limit.ANON),
        ("POST", "/auth/anonymous/", rate_limit.ANON),
        ("GET", "/auth/anonymous", rate_limit.AUTH),
        ("GET", "/auth/google", rate_limit.AUTH),
        ("POST", "/auth/google/callback", rate_limit.AUTH),
        ("POST", "/chat", rate_limit.GLOBAL),
        ("GET", "/no-such-route", rate_limit.GLOBAL),
        ("GET", "/authoring", rate_limit.GLOBAL),
    ],
)
def test_route_class_buckets(method, path, expected):
    assert rate_limit.route_class(method, path) == expected


def test_tracking_is_bounded(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_KEYS", 4)
    window = rate_limit.SlidingWindow()

    for i in range(50):
        assert window.hit((rate_limit.GLOBAL, f"10.0.0.{i}"), 60, 60.0) is None

    assert window.key_count() <= 4


def test_eviction_drops_expired_keys_before_fresh_ones(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED_KEYS", 4)
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(rate_limit, "_now", lambda: now["t"])
    window = rate_limit.SlidingWindow()
    for i in range(4):
        window.hit((rate_limit.GLOBAL, f"10.0.0.{i}"), 60, 60.0)

    now["t"] += 61

    assert window.hit((rate_limit.GLOBAL, "10.0.0.4"), 60, 60.0) is None
    assert window.key_count() == 1


def test_rejected_hits_do_not_extend_window(client, clock):
    for _ in range(60):
        client.get("/probe")
    blocked_at = clock["now"]
    for _ in range(10):
        assert client.get("/probe").status_code == 429
    clock["now"] = blocked_at + rate_limit.GLOBAL_WINDOW_SECONDS

    assert client.get("/probe").status_code == 404
