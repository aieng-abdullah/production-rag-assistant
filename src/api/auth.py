"""Google OAuth → JWT routes (PLAN.md PR-2b).

Env-gated: missing GOOGLE_CLIENT_ID/SECRET → 501 with setup hint
(open-source rule: never require private credentials for a local run).
The `state` round-trip is enforced via a short-lived httpOnly cookie (CSRF).
"""

import hmac
from typing import Any

import httpx
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.httpx_client import OAuth2Client
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from src.api.security import create_token
from src.config import Config
from src.db.database import session_scope
from src.db.models import User

__all__ = ["router", "demo_router"]

router = APIRouter(prefix="/auth", tags=["auth"])
# Registered by create_app only while ENABLE_DEMO_LOGIN resolves on (PR-6).
demo_router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
STATE_COOKIE = "oauth_state"

_SETUP_HINT = (
    "Google sign-in is not configured. Set GOOGLE_CLIENT_ID and "
    "GOOGLE_CLIENT_SECRET in .env — see README 'Google OAuth setup'."
)


def _credentials_missing() -> bool:
    return not (Config.GOOGLE_CLIENT_ID and Config.GOOGLE_CLIENT_SECRET)


def _oauth_client() -> OAuth2Client:
    return OAuth2Client(
        Config.GOOGLE_CLIENT_ID,
        Config.GOOGLE_CLIENT_SECRET,
        redirect_uri=Config.GOOGLE_REDIRECT_URI,
        scope="openid email profile",
    )


def _exchange_code(code: str) -> dict[str, Any]:
    """Code → Google token → userinfo. Tests monkeypatch this seam."""
    client = _oauth_client()
    client.fetch_token(GOOGLE_TOKEN_URL, code=code, grant_type="authorization_code")
    response = client.get(GOOGLE_USERINFO_URL)
    response.raise_for_status()
    return response.json()


def _upsert_user(profile: dict[str, Any]) -> int:
    """Find by `sub` (fallback email), update profile fields, or create. Returns user id.

    One retry: concurrent callbacks can race the unique index — second attempt
    finds the row the winner committed.
    """
    sub = profile.get("sub")
    email = profile.get("email")
    if not sub or not email:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google profile missing sub/email claims",
        )
    for attempt in (1, 2):
        try:
            with session_scope() as session:
                user = session.query(User).filter(User.google_sub == sub).first()
                if user is None:
                    user = session.query(User).filter(User.email == email).first()
                if user is None:
                    user = User(email=email, google_sub=sub)
                    session.add(user)
                user.google_sub = sub
                user.name = profile.get("name") or user.name
                user.picture_url = profile.get("picture") or user.picture_url
                session.flush()
                return int(user.id)
        except IntegrityError:
            if attempt == 2:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="User upsert failed twice under concurrent sign-in",
                ) from None
            logger.warning("User upsert race, retrying sub={sub}", sub=sub)


@router.get("/google")
def start_google_oauth() -> RedirectResponse:
    """Redirect to Google's consent screen (302), state stored in httpOnly cookie."""
    if _credentials_missing():
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_SETUP_HINT)
    client = _oauth_client()
    url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
    response = RedirectResponse(url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=Config.GOOGLE_REDIRECT_URI.startswith("https://"),
    )
    return response


@router.get("/google/callback")
def google_callback(
    request: Request, code: str | None = None, state: str | None = None
) -> RedirectResponse:
    """Exchange code, upsert user, redirect to frontend with JWT in `token`."""
    if _credentials_missing():
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_SETUP_HINT)
    expected = request.cookies.get(STATE_COOKIE)
    if (
        not code
        or not expected
        or not state
        or not hmac.compare_digest(expected, state)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state"
        )
    try:
        profile = _exchange_code(code)
    except (httpx.HTTPError, OAuthError) as exc:
        # Expired/replayed code (invalid_grant) surfaces as authlib OAuthError.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google token exchange failed",
        ) from exc
    user_id = _upsert_user(profile)
    logger.info("Google sign-in ok user_id={user_id}", user_id=user_id)
    token = create_token(user_id)
    frontend = Config.FRONTEND_URL.rstrip("/")
    # Fragment, not query: tokens in query strings land in history/Referer logs.
    response = RedirectResponse(f"{frontend}/#token={token}", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(STATE_COOKIE)
    return response


@demo_router.post("/demo")
def demo_login() -> dict:
    """Local demo account: upsert `demo@local`, issue a real HS256 JWT.

    Lets the Streamlit UI authenticate like any other user (PR-6) without
    Google creds. Registered only while ENABLE_DEMO_LOGIN resolves on —
    404 otherwise. Moves the legacy `default`-tenant corpus to this user
    on first sign-in so pre-API uploads stay visible. The token is never
    logged (security gate 3)."""
    if Config.ENABLE_DEMO_LOGIN != "on":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Demo login disabled"
        )
    if not Config.JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not set",
        )

    from src.db.chroma_client import (  # lazy: keeps API boot off torch
        DEFAULT_TENANT,
        has_chunks,
        reassign_tenant,
    )

    with session_scope() as session:
        user = session.query(User).filter(User.email == "demo@local").first()
        if user is None:
            user = User(email="demo@local", name="Demo User")
            session.add(user)
        session.flush()
        user_id = int(user.id)

    if not has_chunks(str(user_id)):
        moved = reassign_tenant(DEFAULT_TENANT, str(user_id))
        if moved:
            logger.info(f"Demo sign-in adopted {moved} legacy chunks")

    token = create_token(user_id)
    logger.info("Demo sign-in user_id={user_id}", user_id=user_id)
    return {"token": token, "user_id": user_id, "email": "demo@local"}


class AnonymousLogin(BaseModel):
    """Guest device id — uuid4 hex from the browser, used as the account key."""

    device_id: str = Field(pattern=r"^[a-z0-9-]{8,64}$")


@router.post("/anonymous")
def anonymous_login(body: AnonymousLogin) -> dict:
    """Guest session (PLAN PR-6): upsert a per-device account, issue JWT.

    Guests get their own tenant (uploads stay theirs) plus the smaller
    guest quotas enforced by `src.services.quotas` tier limits. One row
    per device id — no verification, no email, expires nothing; replace
    with real sign-in at any time via the login wall.
    """
    if not Config.JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not set",
        )

    email = f"anon-{body.device_id}@local"
    with session_scope() as session:
        user = session.query(User).filter(User.email == email).first()
        if user is None:
            user = User(email=email, name="Guest")
            session.add(user)
        session.flush()
        user_id = int(user.id)

    token = create_token(user_id)
    logger.info("Anonymous sign-in user_id={user_id}", user_id=user_id)
    return {"token": token, "user_id": user_id, "email": "Guest", "tier": "anonymous"}
