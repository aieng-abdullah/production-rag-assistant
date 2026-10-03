"""Shared Streamlit wiring: session state, CSS, workspace accents.

Interim state (pre-PR-6): pages still call the service layer directly —
the httpx/JWT swap to the API happens in PLAN PR-6 once PR-3 lands.
"""

from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).parent / "styles" / "main.css"

# Spec §3: the only allowed session keys (plus legacy provider keys until PR-6).
_SESSION_DEFAULTS = {
    "jwt": None,
    "user_email": None,
    "user_id": None,
    "workspace": "legal",
    "messages": [],
    "doc_statuses": {},
    "quota": 0,
    "bm25_index": None,
    "ingested_docs": [],
    "anthropic_key": "",
    "anthropic_model": "claude-sonnet-4-20250514",
    "openai_key": "",
    "openai_model": "gpt-4o",
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


def page_config(title: str) -> None:
    """Common page config — Material Symbols icon, no emoji (spec §1.6)."""
    st.set_page_config(
        page_title=title,
        page_icon=":material/library_books:",
        layout="wide",
    )
