"""Login gate — entrypoint (docs/UI_DESIGN.md §3).

Streamlit Cloud + Docker both run `streamlit run app.py`.
APP_AUTH=off (default): local demo session, no backend.
APP_AUTH=on: Google→JWT arrives with PLAN PR-2b — button disabled with
an inline notice until then (fail loud, never a fake token).
No `src.*` imports here (AGENTS.md guardrail).
"""

import streamlit as st

from menu import AUTH_ENABLED, menu
from ui_core import init_session_state, load_css, logo_mark, page_config

page_config("Sign in — RAG Research Assistant")
init_session_state()
load_css()
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
        if AUTH_ENABLED:
            st.button(
                "Continue with Google",
                type="primary",
                disabled=True,
                use_container_width=True,
                key="google_signin",
            )
            st.error(
                "Google sign-in connects with PLAN PR-2b (auth service). "
                "Set APP_AUTH=off for local demo access.",
                icon=":material/info:",
            )
        else:
            st.caption("Demo mode — no account needed. Your session stays local.")
            if st.button(
                "Continue as demo",
                type="primary",
                use_container_width=True,
                key="demo_continue",
            ):
                st.session_state.jwt = "demo-token"
                st.session_state.user_email = "demo@local"
                st.session_state.user_id = "demo"
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
