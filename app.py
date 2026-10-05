"""Login gate — entrypoint (docs/UI_DESIGN.md §3).

Streamlit Cloud + Docker both run `streamlit run app.py`.
APP_AUTH=off (default): local demo session, no backend.
APP_AUTH=on: Google→JWT via API /auth/google; a same-origin bridge moves
the `#token=` fragment (security gate: never a query string in the
redirect) into `?token=` once so the server can consume it.
No `src.*` imports here (AGENTS.md guardrail).
"""

import streamlit as st

from api_client import APIError, complete_demo_login
from menu import AUTH_ENABLED, menu
from ui_core import (
    capture_oauth_token,
    google_login_url,
    init_session_state,
    load_css,
    logo_mark,
    page_config,
)

# Fragment → query bridge: OAuth lands on /#token=..., Streamlit cannot read
# fragments. Runs in the page document, so it reads our own location directly;
# capture_oauth_token() deletes the param before any widget renders. Content
# is a constant string (no user input). Fails silent — login stays off rather
# than leaking a token anywhere unexpected (fail safe, not fail open).
_OAUTH_BRIDGE_JS = """
try {
  const m = window.location.hash.match(/[#&]token=([^&]+)/);
  if (m) {
    const u = new URL(window.location.href);
    u.searchParams.set("token", m[1]);
    u.hash = "";
    window.location.replace(u.toString());
  }
} catch (e) {}
"""

page_config("Sign in — RAG Research Assistant")
init_session_state()
load_css()
st.html(_OAUTH_BRIDGE_JS, unsafe_allow_javascript=True)
if capture_oauth_token():
    st.switch_page("pages/2_Chat.py")

menu()

st.markdown(
    f'<div class="brand-center">{logo_mark(56)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero hero-center">
  <h1>Citation-Verified RAG</h1>
  <p>ChatGPT guesses. We verify — every answer grounded in your documents,
     cited to the page.</p>
</div>
""",
    unsafe_allow_html=True,
)

login_box, _ = st.columns([2, 1])
with login_box:
    with st.container(border=True):
        st.subheader("Sign in")
        google_url = google_login_url()
        if google_url:
            st.link_button(
                "Continue with Google",
                google_url,
                type="primary",
                use_container_width=True,
                key="google_signin",
            )
        elif AUTH_ENABLED:
            st.button(
                "Continue with Google",
                type="primary",
                disabled=True,
                use_container_width=True,
                key="google_signin",
            )
            st.error(
                "Google sign-in needs GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET "
                "in .env (see README 'Google OAuth setup'). "
                "Set APP_AUTH=off for local demo access.",
                icon=":material/info:",
            )
        if not AUTH_ENABLED:
            st.caption("Demo mode — no account needed. Your session stays local.")
            if st.button(
                "Continue as demo",
                type="secondary" if google_url else "primary",
                use_container_width=True,
                key="demo_continue",
            ):
                try:
                    complete_demo_login()
                except APIError as exc:
                    st.error(exc.detail)
                else:
                    st.switch_page("pages/2_Chat.py")

        st.divider()
        st.page_link(
            "pages/1_Landing.py",
            label="Back to landing",
            icon=":material/home:",
        )

f1, f2 = st.columns(2)
with f1:
    with st.container(border=True):
        st.markdown("**How is this different from ChatGPT?**")
        st.markdown(
            "Retrieves from YOUR documents only, cites every claim with page "
            "numbers, and refuses to answer when the corpus does not support it."
        )
with f2:
    with st.container(border=True):
        st.markdown("**What does 'citation-verified' mean?**")
        st.markdown(
            "Generated sentences are checked against retrieved chunks before "
            "display. Unsupported claims are rejected."
        )

st.caption("Open source stack: Groq · LangChain · Chroma · Streamlit")
