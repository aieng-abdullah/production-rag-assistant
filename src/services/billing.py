"""Subscription persistence for billing (PLAN PR-5).

Framework-free: session in, state out. The webhook handler calls the
`apply_*` helpers; upserts keyed by the unique `user_id` make Stripe
event replays idempotent (same event twice → same row, one subscription).
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Subscription, User

__all__ = [
    "get_subscription",
    "store_customer",
    "apply_subscription",
    "mark_canceled",
    "user_by_stripe_customer",
]


def get_subscription(session: Session, user_id: int) -> Subscription | None:
    stmt = select(Subscription).where(Subscription.user_id == user_id)
    return session.scalars(stmt).first()


def store_customer(session: Session, user_id: int, stripe_customer_id: str) -> None:
    """Attach (or migrate) the Stripe customer id for this user."""
    sub = get_subscription(session, user_id)
    if sub is None:
        session.add(
            Subscription(user_id=user_id, stripe_customer_id=stripe_customer_id)
        )
    else:
        sub.stripe_customer_id = stripe_customer_id


def user_by_stripe_customer(session: Session, stripe_customer_id: str) -> int | None:
    stmt = select(Subscription.user_id).where(
        Subscription.stripe_customer_id == stripe_customer_id
    )
    return session.scalars(stmt).first()


def apply_subscription(
    session: Session,
    *,
    user_id: int,
    stripe_subscription_id: str,
    stripe_customer_id: str | None,
    tier: str,
    status: str,
    current_period_end: datetime | None,
) -> None:
    """Upsert subscription state from a Stripe event (idempotent)."""
    sub = get_subscription(session, user_id)
    if sub is None:
        sub = Subscription(user_id=user_id)
        session.add(sub)
    sub.stripe_subscription_id = stripe_subscription_id
    if stripe_customer_id:
        sub.stripe_customer_id = stripe_customer_id
    sub.tier = tier
    sub.status = status
    sub.current_period_end = current_period_end


def mark_canceled(session: Session, stripe_subscription_id: str) -> None:
    """Subscription deleted → free tier, keep the row for audit."""
    stmt = select(Subscription).where(
        Subscription.stripe_subscription_id == stripe_subscription_id
    )
    sub = session.scalars(stmt).first()
    if sub is not None:
        sub.tier = "free"
        sub.status = "canceled"
        sub.current_period_end = None
    else:
        # Late/never-seen deletion with no local row: nothing to cancel.
        return


def email_for_user(session: Session, user_id: int) -> str | None:
    user = session.get(User, user_id)
    return user.email if user is not None else None
