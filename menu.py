"""Sidebar navigation — pattern adapted from antoineross/streamlit-saas-starter.

Single-process app (PLAN PR-6.1): every visitor is a local user
(session state). APP_AUTH=on turns app.py into a login gate before the
content pages — still session-local, no backend involved.
"""

import streamlit as st

from pricing_modal import maybe_show_pricing_modal
from src.auth.google_oauth import create_guest_session
from src.config import Config
from ui_core import brand_html

AUTH_ENABLED = Config.APP_AUTH_ENABLED

_LOGIN = "app.py"
_LANDING = "pages/1_Landing.py"
_CHAT = "pages/2_Chat.py"
_DOCUMENTS = "pages/3_Documents.py"
_DASHBOARD = "pages/4_Dashboard.py"
_SETTINGS = "pages/5_Settings.py"
_BILLING = "pages/6_Billing.py"
_ADMIN = "pages/7_Admin.py"

ADMIN_EMAILS = ["admin@example.com"]  # Must match pages/7_Admin.py


def _is_admin() -> bool:
    email = st.session_state.get("user_email", "")
    return email in ADMIN_EMAILS


def unauthenticated_menu() -> None:
    """Navigation for visitors without a session."""
    st.sidebar.page_link(_LANDING, label="Landing", icon=":material/home:")
    st.sidebar.page_link(_LOGIN, label="Sign in", icon=":material/login:")
    st.sidebar.divider()
    if st.sidebar.button(
        "Continue as guest",
        type="secondary",
        use_container_width=True,
        key="sidebar_guest",
    ):
        create_guest_session()
        st.rerun()


def authenticated_menu() -> None:
    """Navigation for signed-in users: core flow + account line + logout."""
    st.sidebar.page_link(_CHAT, label="Chat", icon=":material/chat:")
    st.sidebar.page_link(
        _DOCUMENTS, label="Documents", icon=":material/upload_file:"
    )
    st.sidebar.page_link(
        _DASHBOARD, label="Dashboard", icon=":material/dashboard:"
    )
    st.sidebar.page_link(_BILLING, label="Billing", icon=":material/credit_card:")
    st.sidebar.page_link(_SETTINGS, label="Settings", icon=":material/settings:")

    # Admin link (only for admin users)
    if _is_admin():
        st.sidebar.page_link(_ADMIN, label="Admin", icon=":material/admin_panel_settings:")

    st.sidebar.divider()

    # User info
    email = st.session_state.get("user_email") or "guest"
    tier = st.session_state.get("user_tier", "free")
    avatar, name = st.sidebar.columns([1, 4])
    avatar.markdown(
        '<div class="avatar">'
        f"{(email[:1] or 'G').upper()}</div>",
        unsafe_allow_html=True,
    )
    tier_badge = "🟢 Pro" if tier == "pro" else "⚪ Free"
    name.markdown(f"<small>{email} · {tier_badge}</small>", unsafe_allow_html=True)

    if st.sidebar.button(
        "Log out",
        key="logout_btn",
        icon=":material/logout:",
        use_container_width=True,
    ):
        logout()


def logout() -> None:
    """Clear every session state and return to Login (session-local only)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page(_LOGIN)


def menu() -> None:
    """Render the sidebar menu for the current auth state."""
    st.sidebar.markdown(brand_html(26), unsafe_allow_html=True)

    # Guest users get authenticated menu but with guest limits
    if st.session_state.get("user_id"):
        authenticated_menu()
        return

    # No session - show unauthenticated menu
    unauthenticated_menu()


def menu_with_redirect() -> None:
    """Render menu; bounce to Login when auth is on and no session."""
    if AUTH_ENABLED and not st.session_state.get("user_id"):
        st.switch_page(_LOGIN)
    menu()
    maybe_show_pricing_modal()


@st.dialog("Sign in to continue")
def show_login_wall() -> None:
    """Guest quota exhausted: offer sign in or reset guest session."""
    limit = Config.GUEST_QUERY_LIMIT
    st.markdown(f"### You've used all {limit} free queries")
    st.caption(
        "Sign in with Google to continue — your documents and history "
        "stay associated with your account."
    )

    st.divider()

    # Option 1: Sign in with Google
    from src.auth.google_oauth import get_google_auth_url

    google_url = get_google_auth_url()
    st.link_button(
        "Sign in with Google",
        google_url,
        type="primary",
        use_container_width=True,
        icon=":material/login:",
    )

    st.caption("Or reset your guest session:")

    # Option 2: Reset guest session
    if st.button(
        "Continue as guest (reset limits)",
        type="secondary",
        use_container_width=True,
        key="wall_guest_reset",
    ):
        create_guest_session()
        st.rerun()