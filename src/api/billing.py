"""Billing endpoints (PLAN PR-5): checkout, portal, Stripe webhook.

Flag discipline: `src.api.app.create_app` registers these routers ONLY
when `STRIPE_SECRET_KEY` is set — flag-off deployments have no /billing
and no /webhooks routes at all. The webhook authenticates via Stripe's
signature header (400 on bad signature, never trusts the body).

Idempotency: every state change is an upsert keyed by the unique
`user_id`/`stripe_subscription_id`, so Stripe event replays converge to
the same row — no event table needed.
"""

from datetime import datetime, timezone

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from src.api.deps import require_user
from src.config import Config
from src.db.database import session_scope
from src.services import billing

__all__ = ["router", "webhook_router"]

router = APIRouter(prefix="/billing", tags=["billing"])
webhook_router = APIRouter(prefix="/webhooks", tags=["billing"])


def _require_stripe_configured(*keys: str) -> None:
    """Fail loud on partial config (flag on, price/secret missing)."""
    missing = [key for key in keys if not getattr(Config, key)]
    if missing:
        logger.error(f"Billing misconfigured, missing: {', '.join(missing)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Billing not configured",
        )


def _period_end(subscription_obj: dict) -> datetime | None:
    """Stripe moved period end onto items — accept both shapes."""
    epoch = None
    items = (subscription_obj.get("items") or {}).get("data") or []
    if items:
        epoch = items[0].get("current_period_end")
    epoch = epoch or subscription_obj.get("current_period_end")
    return datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch else None


@router.post("/checkout")
def create_checkout(user_id: int = Depends(require_user)) -> dict:
    """Hosted Stripe Checkout for the pro subscription → returns its URL."""
    _require_stripe_configured("STRIPE_SECRET_KEY", "STRIPE_PRO_PRICE")
    stripe.api_key = Config.STRIPE_SECRET_KEY

    with session_scope() as session:
        email = billing.email_for_user(session, user_id)
        subscription = billing.get_subscription(session, user_id)
        customer_id = (
            subscription.stripe_customer_id if subscription is not None else None
        )

    try:
        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                metadata={"user_id": str(user_id)},
            )
            customer_id = customer.id
            with session_scope() as session:
                billing.store_customer(session, user_id, customer_id)
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": Config.STRIPE_PRO_PRICE, "quantity": 1}],
            client_reference_id=str(user_id),
            subscription_data={"metadata": {"user_id": str(user_id)}},
            success_url=f"{Config.APP_BASE_URL}/?billing=success",
            cancel_url=f"{Config.APP_BASE_URL}/?billing=cancel",
        )
    except stripe.StripeError as exc:
        logger.error(f"Stripe checkout failed user={user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment provider unavailable",
        ) from exc

    return {"url": checkout_session.url}


@router.post("/portal")
def create_portal(user_id: int = Depends(require_user)) -> dict:
    """Stripe billing portal (cancel/update) → returns its URL."""
    _require_stripe_configured("STRIPE_SECRET_KEY")
    stripe.api_key = Config.STRIPE_SECRET_KEY

    with session_scope() as session:
        subscription = billing.get_subscription(session, user_id)
        customer_id = (
            subscription.stripe_customer_id if subscription is not None else None
        )
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No billing customer yet",
        )

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=Config.APP_BASE_URL,
        )
    except stripe.StripeError as exc:
        logger.error(f"Stripe portal failed user={user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment provider unavailable",
        ) from exc

    return {"url": portal_session.url}


def _resolve_user(session, subscription_obj: dict) -> int | None:
    """User for a subscription event: checkout metadata first, then the
    stored customer id (events can arrive before/without metadata)."""
    metadata = subscription_obj.get("metadata") or {}
    raw = metadata.get("user_id")
    if raw and str(raw).isdigit():
        return int(raw)
    customer_id = subscription_obj.get("customer")
    return billing.user_by_stripe_customer(session, customer_id) if customer_id else None


def _handle_event(event: dict) -> None:
    event_type = event.get("type", "")
    obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        user_id = obj.get("client_reference_id")
        customer_id = obj.get("customer")
        if user_id and customer_id and str(user_id).isdigit():
            with session_scope() as session:
                billing.store_customer(session, int(user_id), customer_id)
        return

    if event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
    ):
        with session_scope() as session:
            user_id = _resolve_user(session, obj)
            if user_id is None:
                logger.warning(
                    f"Subscription event for unknown customer sub={obj.get('id')}"
                )
                return
            price_id = None
            items = (obj.get("items") or {}).get("data") or []
            if items:
                price_id = (items[0].get("price") or {}).get("id")
            active = obj.get("status") in ("active", "trialing")
            tier = "pro" if active and price_id == Config.STRIPE_PRO_PRICE else "free"
            if active and Config.STRIPE_PRO_PRICE and price_id != Config.STRIPE_PRO_PRICE:
                logger.warning(
                    f"Active sub with unexpected price={price_id} user={user_id} — staying free"
                )
            billing.apply_subscription(
                session,
                user_id=user_id,
                stripe_subscription_id=obj.get("id", ""),
                stripe_customer_id=obj.get("customer"),
                tier=tier,
                status=obj.get("status", "unknown"),
                current_period_end=_period_end(obj),
            )
        return

    if event_type == "customer.subscription.deleted":
        with session_scope() as session:
            billing.mark_canceled(session, obj.get("id", ""))
        return

    logger.debug(f"Ignoring Stripe event type={event_type}")


@webhook_router.post("/stripe")
async def stripe_webhook(request: Request) -> dict:
    """Stripe receiver: signature-verified, replay-safe, never leaks internals."""
    _require_stripe_configured("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET")
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload, signature, Config.STRIPE_WEBHOOK_SECRET
        )
    except (stripe.SignatureVerificationError, ValueError, TypeError) as exc:
        logger.warning(f"Webhook rejected: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature"
        ) from exc

    _handle_event(dict(event))
    return {"received": True}
