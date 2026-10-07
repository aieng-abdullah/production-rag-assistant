"""Billing endpoint tests (PLAN.md PR-5): flag gating, checkout, portal,
Stripe webhook (signature check, event handling, idempotent replay)."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import Subscription, User

# Fake secrets — tests never call real Stripe (all SDK calls monkeypatched).
FAKE_STRIPE_KEY = "sk_test_fake_0000000000000000"
FAKE_WEBHOOK_SECRET = "whsec_test_fake_000000000000"
FAKE_PRICE = "price_pro_monthly_fake"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'billing.db'}")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield
    reset_engine()


@pytest.fixture()
def client_off(db, monkeypatch):
    """Flag off: STRIPE_SECRET_KEY absent — billing routers never register."""
    monkeypatch.setattr(Config, "STRIPE_SECRET_KEY", "")
    monkeypatch.setattr(Config, "STRIPE_WEBHOOK_SECRET", "")
    monkeypatch.setattr(Config, "STRIPE_PRO_PRICE", "")
    return TestClient(create_app())


@pytest.fixture()
def client(db, monkeypatch):
    """Flag on: full Stripe config present."""
    monkeypatch.setattr(Config, "STRIPE_SECRET_KEY", FAKE_STRIPE_KEY)
    monkeypatch.setattr(Config, "STRIPE_WEBHOOK_SECRET", FAKE_WEBHOOK_SECRET)
    monkeypatch.setattr(Config, "STRIPE_PRO_PRICE", FAKE_PRICE)
    return TestClient(create_app())


@pytest.fixture()
def headers():
    return {"Authorization": f"Bearer {create_token(1)}"}


def _seed_user(user_id: int = 1, email: str = "user@example.com") -> int:
    with session_scope() as session:
        user = User(id=user_id, email=email)
        session.add(user)
        session.flush()
        return user.id


def _post_webhook(client, event: dict):
    """Send an already-built event; the signature is faked per test."""
    return client.post(
        "/webhooks/stripe",
        content=json.dumps(event).encode(),
        headers={"stripe-signature": "t=1,v1=deadbeef"},
    )


def _subscription_event(
    event_type: str,
    *,
    sub_id: str = "sub_123",
    customer: str = "cus_123",
    status: str = "active",
    price_id: str = FAKE_PRICE,
    user_id: int | None = 1,
    period_end: int | None = 1_900_000_000,
) -> dict:
    metadata = {"user_id": str(user_id)} if user_id is not None else {}
    return {
        "type": event_type,
        "data": {
            "object": {
                "id": sub_id,
                "customer": customer,
                "status": status,
                "metadata": metadata,
                "items": {
                    "data": [
                        {
                            "price": {"id": price_id},
                            "current_period_end": period_end,
                        }
                    ]
                },
            }
        },
    }


# --- Flag gating (flag-off deployments must not expose billing) ---


def test_billing_routes_absent_when_flag_off(client_off):
    assert client_off.post("/billing/checkout").status_code == 404
    assert client_off.post("/billing/portal").status_code == 404
    assert client_off.post("/webhooks/stripe").status_code == 404


# --- Auth & config guards ---


def test_checkout_requires_auth(client):
    assert client.post("/billing/checkout").status_code == 401


def test_portal_requires_auth(client):
    assert client.post("/billing/portal").status_code == 401


def test_checkout_partial_config_500(client, headers, monkeypatch):
    monkeypatch.setattr(Config, "STRIPE_PRO_PRICE", "")
    response = client.post("/billing/checkout", headers=headers)

    assert response.status_code == 500
    assert "not configured" in response.json()["detail"]


def test_portal_without_customer_404(client, headers):
    _seed_user()
    assert client.post("/billing/portal", headers=headers).status_code == 404


# --- Checkout ---


def test_checkout_creates_customer_and_returns_url(client, headers, monkeypatch):
    _seed_user()
    created = {}

    def fake_customer_create(**kwargs):
        created["email"] = kwargs.get("email")
        return SimpleNamespace(id="cus_new_1")

    def fake_checkout_create(**kwargs):
        created["price"] = kwargs["line_items"][0]["price"]
        return SimpleNamespace(url="https://checkout.stripe.com/pay/cs_1")

    monkeypatch.setattr("stripe.Customer.create", fake_customer_create)
    monkeypatch.setattr(
        "stripe.checkout.Session.create", fake_checkout_create
    )

    response = client.post("/billing/checkout", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"url": "https://checkout.stripe.com/pay/cs_1"}
    assert created["email"] == "user@example.com"
    assert created["price"] == FAKE_PRICE
    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.stripe_customer_id == "cus_new_1"


def test_checkout_reuses_stored_customer(client, headers, monkeypatch):
    _seed_user()
    with session_scope() as session:
        session.add(Subscription(user_id=1, stripe_customer_id="cus_saved"))
    calls = {"customer": 0}

    def boom_customer(**kwargs):
        calls["customer"] += 1
        raise AssertionError("must not create a second customer")

    monkeypatch.setattr("stripe.Customer.create", boom_customer)
    monkeypatch.setattr(
        "stripe.checkout.Session.create",
        lambda **kwargs: SimpleNamespace(url="https://checkout.stripe.com/pay/cs_2"),
    )

    response = client.post("/billing/checkout", headers=headers)

    assert response.status_code == 200
    assert calls["customer"] == 0
    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.stripe_customer_id == "cus_saved"


# --- Portal ---


def test_portal_returns_url(client, headers, monkeypatch):
    _seed_user()
    with session_scope() as session:
        session.add(Subscription(user_id=1, stripe_customer_id="cus_1"))
    monkeypatch.setattr(
        "stripe.billing_portal.Session.create",
        lambda **kwargs: SimpleNamespace(url="https://billing.stripe.com/p/session"),
    )

    response = client.post("/billing/portal", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"url": "https://billing.stripe.com/p/session"}


# --- Webhook: signature ---


def test_webhook_rejects_bad_signature(client):
    """Real construct_event verifies the header — a fake one fails."""
    response = client.post(
        "/webhooks/stripe",
        content=b'{"type": "checkout.session.completed"}',
        headers={"stripe-signature": "t=1,v1=notavalidsignature"},
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_webhook_partial_config_500(client, monkeypatch):
    monkeypatch.setattr(Config, "STRIPE_WEBHOOK_SECRET", "")
    response = client.post(
        "/webhooks/stripe", content=b"{}", headers={"stripe-signature": "x"}
    )

    assert response.status_code == 500
    assert "not configured" in response.json()["detail"]


# --- Webhook: event handling ---


def test_checkout_completed_stores_customer(client, monkeypatch):
    _seed_user()
    event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {"client_reference_id": "1", "customer": "cus_checkout"}
        },
    }
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    response = _post_webhook(client, event)

    assert response.status_code == 200
    assert response.json() == {"received": True}
    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.stripe_customer_id == "cus_checkout"


def test_subscription_updated_sets_pro_tier(client, monkeypatch):
    _seed_user()
    event = _subscription_event("customer.subscription.updated")
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    response = _post_webhook(client, event)

    assert response.status_code == 200
    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.tier == "pro"
        assert sub.status == "active"
        assert sub.stripe_subscription_id == "sub_123"
        assert sub.stripe_customer_id == "cus_123"
        assert sub.current_period_end is not None


def test_subscription_wrong_price_stays_free(client, monkeypatch):
    _seed_user()
    event = _subscription_event(
        "customer.subscription.updated", price_id="price_other_plan"
    )
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    _post_webhook(client, event)

    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.tier == "free"
        assert sub.status == "active"


def test_subscription_incomplete_status_stays_free(client, monkeypatch):
    _seed_user()
    event = _subscription_event(
        "customer.subscription.updated", status="incomplete"
    )
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    _post_webhook(client, event)

    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.tier == "free"
        assert sub.status == "incomplete"


def test_subscription_deleted_marks_canceled(client, monkeypatch):
    _seed_user()
    with session_scope() as session:
        session.add(
            Subscription(
                user_id=1,
                tier="pro",
                status="active",
                stripe_subscription_id="sub_123",
                stripe_customer_id="cus_123",
            )
        )
    event = {"type": "customer.subscription.deleted", "data": {"object": {"id": "sub_123"}}}
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    response = _post_webhook(client, event)

    assert response.status_code == 200
    with session_scope() as session:
        sub = session.query(Subscription).filter_by(user_id=1).one()
        assert sub.tier == "free"
        assert sub.status == "canceled"
        assert sub.current_period_end is None


def test_subscription_unknown_customer_ignored(client, monkeypatch):
    event = _subscription_event("customer.subscription.updated", user_id=None)
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    response = _post_webhook(client, event)

    assert response.status_code == 200
    with session_scope() as session:
        assert session.query(Subscription).count() == 0


# --- Webhook: idempotency (Stripe event replays converge) ---


def test_webhook_replay_creates_single_row(client, monkeypatch):
    _seed_user()
    event = _subscription_event("customer.subscription.updated")
    monkeypatch.setattr(
        "stripe.Webhook.construct_event",
        lambda payload, sig, secret: event,
    )

    first = _post_webhook(client, event)
    second = _post_webhook(client, event)

    assert first.status_code == 200
    assert second.status_code == 200
    with session_scope() as session:
        rows = session.query(Subscription).filter_by(user_id=1).all()
        assert len(rows) == 1
        assert rows[0].tier == "pro"
        assert rows[0].stripe_subscription_id == "sub_123"
