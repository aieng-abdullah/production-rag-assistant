"""Shared Streamlit wiring: session state, CSS, workspace accents.

Single-process mode (PLAN PR-6.1): pages call `src.services` directly;
this module owns session/chrome only — no backend, no JWT plumbing.
"""

import re
from pathlib import Path
from threading import Lock, Thread

import streamlit as st
from loguru import logger

from src.config import Config

STYLES_PATH = Path(__file__).parent / "styles" / "main.css"

# Background RAG warm-up: one per process (file-watcher reloads re-import
# this module, which is the only case it restarts — harmless).
_warmup_lock = Lock()
_warmup_started = False

# Spec §3: the only allowed session keys (session-local auth, PR-6.1).
_SESSION_DEFAULTS = {
    "user_email": None,
    "user_id": None,
    "workspace": "academic",
    "messages": [],
    "doc_statuses": {},
    "quota": 0,
    "ingested_docs": [],
    # Guest tier: free try-out before the login wall.
    "guest_queries": 0,
    "guest_docs": 0,
    "user_tier": "free",
    "show_pricing_modal": False,
    # One-shot success banner (Documents page flash after ingest/delete).
    "success_flash": None,
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


def start_rag_warmup() -> None:
    """Validate config + ping Voyage in a background thread at startup.

    There is no local model left to warm (PLAN-render-react Phase 1): this
    checks keys and reaches the Voyage API so the first chat does not pay
    for a dead connection. Log-only — warm-up failures never raise into UI.
    """
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _warm() -> None:
        try:
            Config.validate()

            # Voyage reachability ping: any HTTP response means the route is
            # live; only transport errors count as unreachable.
            try:
                if not Config.VOYAGE_API_KEY:
                    logger.warning(
                        "VOYAGE_API_KEY missing — embeddings/rerank will fail on first use"
                    )
                else:
                    import httpx

                    resp = httpx.get(
                        Config.VOYAGE_BASE_URL,
                        timeout=5.0,
                        follow_redirects=True,
                    )
                    logger.info(f"Voyage API reachable (HTTP {resp.status_code})")
            except Exception as ping_exc:
                logger.warning(f"Voyage API unreachable: {ping_exc}")

            logger.info("RAG warm-up complete")
        except Exception as exc:
            logger.error("RAG warm-up failed (first query pays): {err}", err=exc)

    Thread(target=_warm, daemon=True, name="rag-warmup").start()


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
    """Common page config — Material Symbols icon, no emoji (spec §1.6).

    menu_items clears the default About/Get Help entries; footer chrome is
    hidden via styles/main.css (Streamlit Cloud "Hosted with Streamlit").
    """
    st.set_page_config(
        page_title=title,
        page_icon=":material/library_books:",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )


def lottie(name: str, height: int = 180, *, loop: bool = True) -> None:
    """Render a bundled Lottie animation (`static/lottie/<name>.json`).

    Streamlit serves the sibling `static/` directory at `/app/static/...`,
    and the lottie-player web component is vendored there too — the page
    never calls a CDN at runtime. `name` must match `[a-z0-9_-]+` (it is
    interpolated into HTML).

    The `<lottie-player>` element is created from inside a `<script>`:
    Streamlit sanitizes `st.html` bodies with DOMPurify, which drops the
    unknown custom tag but keeps `<script>` (and re-executes it).
    """
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError(f"Invalid lottie animation name: {name!r}")
    loop_js = "true" if loop else "false"
    st.html(
        f"""
        <script src="/app/static/lottie-player.js"></script>
        <div class="lottie-host" style="width:100%;height:{height}px;"></div>
        <script>
          (() => {{
            const host = document.currentScript.previousElementSibling;
            if (!host || host.dataset.lottieReady) return;
            host.dataset.lottieReady = "1";
            const player = document.createElement("lottie-player");
            player.setAttribute("src", "/app/static/lottie/{name}.json");
            player.setAttribute("background", "transparent");
            player.setAttribute("autoplay", "");
            player.setAttribute("loop", "{loop_js}");
            player.style.cssText =
              "width:100%;height:{height}px;display:block;margin:0 auto;";
            host.appendChild(player);
          }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
