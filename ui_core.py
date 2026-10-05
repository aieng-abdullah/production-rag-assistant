"""Shared Streamlit wiring: session state, CSS, workspace accents.

PR-6: pages reach the backend through `api_client` (httpx + JWT); this
module owns session/OAuth plumbing and chrome only.
"""

import re
from pathlib import Path

import streamlit as st
from loguru import logger

from src.api.security import TokenError, decode_token
from src.config import Config
from src.db.database import session_scope
from src.db.models import User

STYLES_PATH = Path(__file__).parent / "styles" / "main.css"

# Spec §3: the allowed session keys (PR-6: no provider keys — server-side).
_SESSION_DEFAULTS = {
    "jwt": None,
    "user_email": None,
    "user_id": None,
    # Guest tier (PLAN PR-6): stable per-browser device id + tier flag.
    "device_id": None,
    "anon_tier": False,
    "workspace": "academic",
    "messages": [],
    "quota": 0,
    # Progressive auth wall: free anonymous queries, plan choice after login.
    "anon_queries": 0,
    "user_tier": "free",
    "show_pricing_modal": False,
    # One-shot success banner (Documents page flash after ready/deleted).
    "success_flash": None,
}

# Spec §4.2: workspace accents override the base --ws-accent token.
_WS_ACCENTS = {
    "legal": "#1E3A5F",
    "academic": "#0D9488",
}


def init_session_state() -> None:
    """Initialize session state variables with spec defaults."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def google_login_url() -> str | None:
    """OAuth entrypoint on the API host; None while Google creds are absent."""
    if not (Config.GOOGLE_CLIENT_ID and Config.GOOGLE_CLIENT_SECRET):
        return None
    return f"{Config.API_BASE_URL}/auth/google"


def _email_for_user(user_id: int) -> str | None:
    """Best-effort email for the sidebar; sign-in survives a DB hiccup (logged)."""
    try:
        with session_scope() as session:
            user = session.get(User, user_id)
            return user.email if user is not None else None
    except Exception as exc:
        logger.error(
            "Could not load email for user {uid}: {err}", uid=user_id, err=exc
        )
        return None


def capture_oauth_token() -> bool:
    """Consume `?token=` from the OAuth redirect (fragment bridge in app.py).

    Stores JWT + identity, flags the pricing modal, then removes the param
    so the token stops appearing in the URL/history. Returns True on a new
    login. Bad/expired tokens are logged and dropped, never raised.
    """
    token = st.query_params.get("token")
    if not token:
        return False
    del st.query_params["token"]
    try:
        user_id = decode_token(token)
    except (TokenError, EnvironmentError) as exc:
        logger.warning("OAuth token rejected: {err}", err=str(exc))
        return False
    st.session_state.jwt = token
    st.session_state.user_id = str(user_id)
    st.session_state.user_email = _email_for_user(user_id)
    st.session_state.anon_tier = False
    st.session_state.show_pricing_modal = True
    logger.info("OAuth sign-in captured user_id={uid}", uid=user_id)
    return True


def load_css() -> None:
    """Inject semantic CSS tokens (docs/UI_DESIGN.md §4.2)."""
    if STYLES_PATH.exists():
        st.markdown(
            f"<style>{STYLES_PATH.read_text()}</style>",
            unsafe_allow_html=True,
        )


def apply_workspace_accent() -> None:
    """Hot-swap the workspace accent token for the active workspace.

    Streamlit's theme cannot change per session state — CSS override can
    (spec §4.2). Applies to sidebar active states, send/upload buttons and
    workspace badges until the accent is swapped again.
    """
    accent = _WS_ACCENTS.get(st.session_state.get("workspace", "legal"), "#4F46E5")
    st.markdown(
        f"<style>:root {{ --ws-accent: {accent}; }}</style>",
        unsafe_allow_html=True,
    )


# --- Branding (docs/UI_DESIGN.md §8 — product name TBD, domain = groundedai) ---
_LOGO_SVG = """<svg class="logo-mark" viewBox="0 0 32 32" width="{size}" height="{size}" aria-label="GroundedAI logo">
  <rect x="1" y="1" width="30" height="30" rx="8" fill="url(#logo-grad)"/>
  <path d="M9 17.5l4.5 4.5L23 12.5" stroke="#ffffff" stroke-width="3.2" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
  <defs>
    <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#4F46E5"/>
      <stop offset="1" stop-color="#7C3AED"/>
    </linearGradient>
  </defs>
</svg>"""


def logo_mark(size: int = 28) -> str:
    """Return the brand mark SVG at the requested pixel size."""
    return _LOGO_SVG.format(size=size)


def brand_html(size: int = 28) -> str:
    """Logo + wordmark row for markdown containers."""
    return (
        f'<div class="brand">{logo_mark(size)}'
        '<span class="brand-name">Grounded<span class="brand-ai">AI</span></span>'
        "</div>"
    )


def page_config(title: str) -> None:
    """Common page config — Material Symbols icon, no emoji (spec §1.6)."""
    st.set_page_config(
        page_title=title,
        page_icon=":material/library_books:",
        layout="wide",
    )


def lottie(name: str, height: int = 180, *, loop: bool = True) -> None:
    """Render a bundled Lottie animation (`static/lottie/<name>.json`).

    Streamlit serves the sibling `static/` directory at `/app/static/...`,
    and the lottie-player web component is vendored there too — the page
    never calls a CDN at runtime. `name` must match `[a-z0-9_-]+` (it is
    interpolated into HTML).
    """
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError(f"Invalid lottie animation name: {name!r}")
    st.html(
        f"""
        <script src="/app/static/lottie-player.js"></script>
        <lottie-player
            src="/app/static/lottie/{name}.json"
            background="transparent"
            autoplay
            loop="{str(loop).lower()}"
            style="width:100%;height:{height}px;display:block;margin:0 auto;">
        </lottie-player>
        """,
        unsafe_allow_javascript=True,
    )
