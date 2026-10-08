"""Tests for quota enforcement (Phase 2)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import Config
from src.db.database import Base
from src.db.models import Document, Subscription, User
from src.services import quotas as quotas_module
from src.services.quotas import (
    ANON_DOCUMENT_LIMIT,
    ANON_QUERY_LIMIT,
    QuotaExceeded,
    check_query_quota,
    check_tier_document_quota,
    clear_session_factory_override,
    document_limit_for,
    query_limit_for,
    reserve_query_slot,
    set_session_factory_override,
    tier_for,
    TIER_MULTIPLIERS,
)
from src.services.usage import record_usage, get_daily_usage, get_document_count


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    # Override session factory for quota checks
    set_session_factory_override(lambda: factory)
    yield s
    clear_session_factory_override()
    s.close()
    engine.dispose()


def _create_user_with_sub(session, tier="free"):
    user = User(email=f"test_{tier}@example.com", name="Test", google_sub=f"sub_{tier}")
    session.add(user)
    session.flush()
    sub = Subscription(user_id=user.id, tier=tier, status="active")
    session.add(sub)
    session.commit()
    return user


class TestQuotas:
    def test_tier_multipliers(self):
        assert TIER_MULTIPLIERS["free"] == 1
        assert TIER_MULTIPLIERS["pro"] == 10

    def test_check_query_quota_free(self, session):
        user = _create_user_with_sub(session, tier="free")
        result = check_query_quota(user.id, "free")
        assert result.allowed is True
        assert result.limit == 20  # DAILY_QUERY_LIMIT * 1
        assert result.used == 0
        assert result.remaining == 20

    def test_check_query_quota_pro(self, session):
        user = _create_user_with_sub(session, tier="pro")
        result = check_query_quota(user.id, "pro")
        assert result.allowed is True
        assert result.limit == 200  # DAILY_QUERY_LIMIT * 10
        assert result.used == 0
        assert result.remaining == 200

    def test_check_query_quota_exceeded(self, session):
        user = _create_user_with_sub(session, tier="free")
        # Add 20 queries worth of usage (10 events * 2 units = 20)
        for _ in range(10):
            record_usage(session, user.id, kind="query", units=2)
        session.commit()

        result = check_query_quota(user.id, "free")
        assert result.allowed is False
        assert result.used == 20
        assert result.limit == 20
        assert result.remaining == 0

    def test_check_document_quota_free(self, session):
        user = _create_user_with_sub(session, tier="free")
        result = check_tier_document_quota(user.id, "free")
        assert result.allowed is True
        assert result.limit == 5  # DOCUMENT_LIMIT * 1
        assert result.used == 0
        assert result.remaining == 5

    def test_check_document_quota_pro(self, session):
        user = _create_user_with_sub(session, tier="pro")
        result = check_tier_document_quota(user.id, "pro")
        assert result.allowed is True
        assert result.limit == 50  # DOCUMENT_LIMIT * 10
        assert result.used == 0
        assert result.remaining == 50


class TestSessionTierLimits:
    """Session surface (FastAPI) must read `Subscription.tier` — the admin
    tier endpoint and Stripe webhook only write that row, so limits that
    ignore it make pro grants a no-op."""

    def test_tier_for_reads_subscription(self, session):
        member = User(email="member@example.com", name="M", google_sub="sub-m")
        guest = User(email="anon-device-0001@local", name="Guest")
        pro = _create_user_with_sub(session, tier="pro")
        session.add_all([member, guest])
        session.commit()

        assert tier_for(session, pro.id) == "pro"
        assert tier_for(session, member.id) == "free"
        assert tier_for(session, guest.id) == "anonymous"
        # Stale/malformed token: unknown user falls back to free, never pro.
        assert tier_for(session, 999_999) == "free"

    def test_limits_apply_pro_multiplier(self, session):
        pro = _create_user_with_sub(session, tier="pro")
        free = _create_user_with_sub(session, tier="free")

        assert query_limit_for(session, pro.id) == Config.DAILY_QUERY_LIMIT * 10
        assert query_limit_for(session, free.id) == Config.DAILY_QUERY_LIMIT
        assert document_limit_for(session, pro.id) == Config.DOCUMENT_LIMIT * 10
        assert document_limit_for(session, free.id) == Config.DOCUMENT_LIMIT

    def test_guest_limits_ignore_subscription(self, session):
        guest = User(email="anon-device-0002@local", name="Guest")
        session.add(guest)
        session.commit()

        assert query_limit_for(session, guest.id) == ANON_QUERY_LIMIT
        assert document_limit_for(session, guest.id) == ANON_DOCUMENT_LIMIT

    def test_reserve_query_slot_uses_pro_multiplier(self, session, monkeypatch):
        monkeypatch.setattr(quotas_module, "DAILY_QUERY_LIMIT", 1)
        pro = _create_user_with_sub(session, tier="pro")

        for _ in range(10):  # 1 unit/day * pro multiplier 10
            reserve_query_slot(session, pro.id)

        with pytest.raises(QuotaExceeded):
            reserve_query_slot(session, pro.id)

    def test_reserve_query_slot_free_multiplier_unchanged(self, session, monkeypatch):
        monkeypatch.setattr(quotas_module, "DAILY_QUERY_LIMIT", 1)
        free = _create_user_with_sub(session, tier="free")

        reserve_query_slot(session, free.id)

        with pytest.raises(QuotaExceeded):
            reserve_query_slot(session, free.id)


class TestUsageTracking:
    def test_record_usage(self, session):
        user = _create_user_with_sub(session)
        event = record_usage(session, user.id, kind="query", units=2)
        session.commit()

        assert event.id is not None
        assert event.user_id == user.id
        assert event.kind == "query"
        assert event.units == 2

    def test_get_daily_usage(self, session):
        user = _create_user_with_sub(session)
        record_usage(session, user.id, kind="query", units=2)
        record_usage(session, user.id, kind="query", units=2)
        record_usage(session, user.id, kind="ingest", units=1)
        session.commit()

        # Only query kind
        query_usage = get_daily_usage(session, user.id, kind="query")
        assert query_usage == 4

        # Only ingest kind
        ingest_usage = get_daily_usage(session, user.id, kind="ingest")
        assert ingest_usage == 1

    def test_get_document_count(self, session):
        user = _create_user_with_sub(session)
        assert get_document_count(session, user.id) == 0

        # Add documents
        doc1 = Document(user_id=user.id, filename="doc1.pdf")
        doc2 = Document(user_id=user.id, filename="doc2.pdf")
        session.add_all([doc1, doc2])
        session.commit()

        assert get_document_count(session, user.id) == 2