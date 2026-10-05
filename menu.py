"""Sidebar navigation — pattern adapted from antoineross/streamlit-saas-starter.

Single-process demo (PLAN PR-6.1): every visitor is a local demo user
(session state). APP_AUTH=on turns app.py into a login gate before the
content pages — still session-local, no backend involved.
"""

import os

import streamlit as st

from pricing_modal import maybe_show_pricing_modal
from src.config import Config
from ui_core import brand_html

AUTH_ENABLED = os.getenv("APP_AUTH", "off") == "on"

_LOGIN = "app.py"
_LANDING = "pages/1_Landing.py"
_CHAT = "pages/2_Chat.py"
_DOCUMENTS = "pages/3_Documents.py"
_DASHBOARD = "pages/4_Dashboard.py"
_SETTINGS = "pages/5_Settings.py"
_BILLING = "pages/6_Billing.py"


def unauthenticated_menu() -> None:
    """Navigation for visitors without a session (spec §2)."""
    st.sidebar.page_link(_LANDING, label="Landing", icon=":material/home:")
    st.sidebar.page_link(_LOGIN, label="Login", icon=":material/login:")


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
    st.sidebar.divider()
    email = st.session_state.get("user_email") or "demo user"
    avatar, name = st.sidebar.columns([1, 4])
    avatar.markdown(
        '<div class="avatar">'
        f"{(email[:1] or 'D').upper()}</div>",
        unsafe_allow_html=True,
    )
    name.markdown(f"<small>{email}</small>", unsafe_allow_html=True)
    if st.sidebar.button(
        "Log out",
        key="logout_btn",
        icon=":material/logout:",
        use_container_width=True,
    ):
        logout()


def logout() -> None:
    """Clear every session key and return to Login (spec §3 — token never persisted)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page(_LOGIN)


def menu() -> None:
    """Render the sidebar menu for the current auth state."""
    st.sidebar.markdown(brand_html(26), unsafe_allow_html=True)
    if AUTH_ENABLED and not st.session_state.get("jwt"):
        unauthenticated_menu()
        return
    if st.session_state.get("jwt"):
        authenticated_menu()
        return
    # APP_AUTH=off and not yet signed in (Login page): public links only.
    unauthenticated_menu()


def menu_with_redirect() -> None:
    """Render menu; bounce to Login when auth is on and jwt missing;
    open the plan chooser once after each login."""
    if AUTH_ENABLED and not st.session_state.get("jwt"):
        st.switch_page(_LOGIN)
    menu()
    maybe_show_pricing_modal()


@st.dialog("Sign in to keep going")
def show_login_wall() -> None:
    """Anonymous quota exhausted: offer the local demo session to continue."""
    limit = Config.ANON_QUERY_LIMIT
    st.markdown(f"### You've used all {limit} free queries")
    st.caption(
        "Sign in to continue — your documents and history stay associated "
        "with your account."
    )

    st.caption("Local demo — no account, stays in this browser session.")
    if st.button(
        "Continue as demo",
        type="primary",
        use_container_width=True,
        key="wall_demo",
    ):
        st.session_state.jwt = "demo-token"
        st.session_state.user_email = "demo@local"
        st.session_state.user_id = "demo"
        st.session_state.show_pricing_modal = True
        st.rerun()
