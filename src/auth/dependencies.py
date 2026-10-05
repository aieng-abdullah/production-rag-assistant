"""Authentication dependencies for Streamlit pages."""

import streamlit as st
from typing import Optional
from loguru import logger

from src.auth.jwt_handler import verify_token
from src.config import Config


LOGIN_PAGE = "app.py"


def require_user() -> int:
    """
    Require authenticated user.

    Returns user_id if authenticated, otherwise redirects to login page.
    Works with both JWT tokens (real users) and guest sessions.
    """
    # Check if auth is enabled
    if not getattr(Config, "APP_AUTH_ENABLED", False):
        # Auth disabled - allow guest access
        return _ensure_guest_session()

    jwt_token = st.session_state.get("jwt")
    if not jwt_token:
        logger.debug("No JWT in session, redirecting to login")
        st.switch_page(LOGIN_PAGE)

    # Guest token - no verification needed
    if jwt_token == "guest-token":
        return _ensure_guest_session()

    # Verify JWT
    try:
        claims = verify_token(jwt_token)
        user_id = claims.get("user_id")
        if not user_id:
            logger.warning("JWT missing user_id claim")
            st.switch_page(LOGIN_PAGE)

        # Sync session state with token claims
        st.session_state.user_id = str(user_id)
        st.session_state.user_email = claims.get("email")
        st.session_state.user_tier = claims.get("tier", "free")

        return int(user_id)

    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        # Clear invalid token
        st.session_state.jwt = None
        st.switch_page(LOGIN_PAGE)


def _ensure_guest_session() -> int:
    """Ensure guest session exists, create if needed."""
    user_id = st.session_state.get("user_id")
    if not user_id or not user_id.startswith("guest_"):
        from src.auth.google_oauth import create_guest_session
        create_guest_session()
        user_id = st.session_state.user_id

    # For guest, return a numeric-ish ID for tenant isolation
    # Use hash of guest_id for Chroma tenant_id
    import hashlib
    guest_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
    return guest_hash


def get_current_user_id() -> Optional[int]:
    """Get current user_id without redirecting. Returns None if not authenticated."""
    jwt_token = st.session_state.get("jwt")
    if not jwt_token:
        return None

    if jwt_token == "guest-token":
        return _ensure_guest_session()

    try:
        claims = verify_token(jwt_token)
        return int(claims.get("user_id", 0))
    except Exception:
        return None


def get_current_tier() -> str:
    """Get current user's tier (free/pro)."""
    return st.session_state.get("user_tier", "free")


def is_guest() -> bool:
    """Check if current session is a guest."""
    user_id = st.session_state.get("user_id", "")
    return user_id.startswith("guest_")