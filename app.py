"""Login gate — entrypoint (docs/UI_DESIGN.md §3).

Streamlit Cloud + Docker both run `streamlit run app.py` — the whole app
is one process (PLAN PR-6.1); there is no backend.
APP_AUTH=on (default): force this gate before content pages.
APP_AUTH=off: allow guest access without login gate.
No `src.*` imports here (AGENTS.md guardrail).
"""

import streamlit as st

from menu import menu
from ui_core import (
    brand_html,
    init_session_state,
    load_css,
    page_config,
    start_rag_warmup,
)

# Handle OAuth callback before rendering UI
if "code" in st.query_params:
    from src.auth.google_oauth import handle_callback

    code = st.query_params["code"]
    st.query_params.clear()
    try:
        handle_callback(code)
        st.switch_page("pages/2_Chat.py")
    except Exception as e:
        st.error(f"Authentication failed: {e}")

# Guest entry point: /?guest=1 — the auth card's "Continue as guest"
# link (HTML anchor; no Streamlit widget involved).
if st.query_params.get("guest") == "1":
    st.query_params.clear()
    from src.auth.google_oauth import create_guest_session

    create_guest_session()
    st.switch_page("pages/2_Chat.py")

page_config("Sign in — GroundedAI")
init_session_state()
load_css()
start_rag_warmup()  # config + Voyage ping in background while user reads login

menu()

from src.auth.google_oauth import get_google_auth_url  # noqa: E402

_google_url = get_google_auth_url().replace("&", "&amp;")

st.html(
    f"""
<div class="auth-wrap">
  <div class="auth-brand">{brand_html(34)}</div>

  <div class="auth-hero">
    <span class="auth-hero-badge">Legal &amp; academic · citation-enforced RAG</span>
    <h1>Citation-verified answers from your documents</h1>
    <p>Ask your statutes, contracts, and papers questions — every sentence
       cites its page-level source, validated by code before you see it.
       The assistant is warm when it can help, and honest when it can't.</p>
  </div>

  <div class="auth-card">
    <h2>Sign in to GroundedAI</h2>
    <a class="auth-btn auth-btn-google" href="{_google_url}">
      <svg viewBox="0 0 48 48" width="18" height="18" aria-hidden="true">
        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
      </svg>
      Sign in with Google
    </a>

    <div class="auth-divider"><span>or</span></div>

    <a class="auth-btn auth-btn-guest" href="/?guest=1">Continue as guest</a>

    <p class="auth-note">No account needed for guest access — 3 questions
       free to try it out. Your documents stay private and isolated to
       your session.</p>
  </div>

  <div class="auth-trust">
    <div class="trust-card">
      <strong>1.00</strong>
      <span>Faithfulness on golden set (Ragas)</span>
    </div>
    <div class="trust-card">
      <strong>Page-level</strong>
      <span>Every sentence cites its source page</span>
    </div>
    <div class="trust-card">
      <strong>Abstains</strong>
      <span>Says "not in your documents" instead of guessing</span>
    </div>
  </div>

  <p class="auth-foot">
    Open source stack: Groq · LangChain · Chroma · Streamlit
    &nbsp;·&nbsp; <a href="/Landing">About the product</a>
  </p>
</div>
""",
)
