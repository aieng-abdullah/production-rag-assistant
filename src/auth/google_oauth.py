"""Google OAuth authentication flow."""

import urllib.parse
from typing import Optional

import httpx
import streamlit as st
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import Config
from src.db.database import get_session_factory
from src.db.models import Subscription, User
from src.db.chroma_client import reassign_tenant


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

OAUTH_SCOPES = ["openid", "email", "profile"]


def _get_client_id() -> str:
    cid = Config.GOOGLE_CLIENT_ID
    if not cid:
        raise EnvironmentError("GOOGLE_CLIENT_ID not set in environment")
    return cid


def _get_client_secret() -> str:
    secret = Config.GOOGLE_CLIENT_SECRET
    if not secret:
        raise EnvironmentError("GOOGLE_CLIENT_SECRET not set in environment")
    return secret


def _get_redirect_uri() -> str:
    """OAuth redirect target = app root.

    Streamlit serves app.py at the app URL root; /auth/* paths 404 on
    Streamlit Cloud. Callback code arrives as ?code= on the root page.
    APP_BASE_URL must match the deployed host (localhost or *.streamlit.app).
    """
    return Config.APP_BASE_URL.rstrip("/") + "/"


def get_google_auth_url() -> str:
    """Build the Google OAuth authorization URL."""
    params = {
        "client_id": _get_client_id(),
        "redirect_uri": _get_redirect_uri(),
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access/refresh tokens."""
    data = {
        "code": code,
        "client_id": _get_client_id(),
        "client_secret": _get_client_secret(),
        "redirect_uri": _get_redirect_uri(),
        "grant_type": "authorization_code",
    }
    resp = httpx.post(GOOGLE_TOKEN_URL, data=data, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _fetch_userinfo(access_token: str) -> dict:
    """Fetch user profile from Google."""
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = httpx.get(GOOGLE_USERINFO_URL, headers=headers, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _upsert_user(session: Session, google_sub: str, email: str, name: Optional[str], picture: Optional[str]) -> User:
    """Insert or update user by google_sub."""
    stmt = select(User).where(User.google_sub == google_sub)
    user = session.scalars(stmt).first()

    if user:
        user.email = email
        user.name = name
        user.picture_url = picture
    else:
        user = User(
            email=email,
            name=name,
            picture_url=picture,
            google_sub=google_sub,
        )
        session.add(user)
        session.flush()

    return user


def _ensure_subscription(session: Session, user_id: int) -> Subscription:
    """Get or create subscription for user (defaults to free)."""
    from src.db.models import Subscription
    stmt = select(Subscription).where(Subscription.user_id == user_id)
    sub = session.scalars(stmt).first()
    if not sub:
        sub = Subscription(user_id=user_id, tier="free", status="inactive")
        session.add(sub)
        session.flush()
    return sub


def _migrate_guest_data(old_tenant: str, new_user_id: int) -> None:
    """Migrate guest tenant data to real user."""
    if old_tenant and old_tenant.startswith("guest_"):
        try:
            reassign_tenant(old_tenant, str(new_user_id))
            logger.info(f"Migrated guest tenant {old_tenant} -> user {new_user_id}")
        except Exception as e:
            logger.warning(f"Guest migration failed: {e}")


def handle_callback(code: str) -> tuple[int, str, str]:
    """
    Handle Google OAuth callback.

    Returns:
        (user_id, email, tier)
    """
    logger.info("Processing Google OAuth callback")

    # Exchange code for tokens
    tokens = _exchange_code_for_tokens(code)
    access_token = tokens.get("access_token")
    if not access_token:
        raise ValueError("No access token in OAuth response")

    # Fetch user info
    userinfo = _fetch_userinfo(access_token)
    google_sub = userinfo.get("id")
    email = userinfo.get("email")
    name = userinfo.get("name")
    picture = userinfo.get("picture")

    if not google_sub or not email:
        raise ValueError("Missing required user info from Google")

    # Upsert user in database
    session_factory = get_session_factory()
    with session_factory() as session:
        user = _upsert_user(session, google_sub, email, name, picture)
        _ensure_subscription(session, user.id)
        session.commit()
        user_id = user.id

    # Migrate guest data if coming from guest session
    old_tenant = st.session_state.get("user_id", "")
    _migrate_guest_data(old_tenant, user_id)

    # Get tier from subscription
    with session_factory() as session:
        sub = _ensure_subscription(session, user_id)
        tier = sub.tier or "free"

    # Update session state (identity only — no JWT; session-local auth)
    st.session_state.user_id = str(user_id)
    st.session_state.user_email = email
    st.session_state.user_tier = tier
    st.session_state.guest_queries = 0
    st.session_state.guest_docs = 0

    logger.info(f"Google OAuth success: user_id={user_id} email={email} tier={tier}")
    return user_id, email, tier


def create_guest_session() -> str:
    """Create a new guest session."""
    import uuid
    guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    st.session_state.user_id = guest_id
    st.session_state.user_email = "guest@local"
    st.session_state.user_tier = "free"
    st.session_state.guest_queries = 0
    st.session_state.guest_docs = 0
    logger.info(f"Created guest session: {guest_id}")
    return guest_id