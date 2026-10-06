"""Authentication dependencies for Streamlit pages."""

from typing import Optional

import streamlit as st
from loguru import logger

from src.config import Config


LOGIN_PAGE = "app.py"


def require_user() -> int:
    """
    Require a signed-in session (real user or guest).

    Returns user_id if a session exists, otherwise redirects to login page.
    Identity lives in st.session_state (no JWT — PR-6.1 single-process auth).
    """
    if not getattr(Config, "APP_AUTH_ENABLED", False):
        return _ensure_guest_session()

    user_id = st.session_state.get("user_id")
    if not user_id:
        logger.debug("No session user_id, redirecting to login")
        st.switch_page(LOGIN_PAGE)

    if user_id.startswith("guest_"):
        return _ensure_guest_session()

    return int(user_id)


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
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None

    if user_id.startswith("guest_"):
        return _ensure_guest_session()

    return int(user_id)


def get_current_tier() -> str:
    """Get current user's tier (free/pro)."""
    return st.session_state.get("user_tier", "free")


def is_guest() -> bool:
    """Check if current session is a guest."""
    user_id = st.session_state.get("user_id", "")
    return user_id.startswith("guest_")
