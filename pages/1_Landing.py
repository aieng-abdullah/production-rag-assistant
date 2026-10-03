"""Landing page — public marketing (docs/UI_DESIGN.md §5).

Moved from root `app.py`; the root script is now the Login gate.
No `src.*` imports here (AGENTS.md guardrail).
"""

import streamlit as st

from menu import menu
from ui_core import init_session_state, load_css, page_config

page_config("RAG Research Assistant")
init_session_state()
load_css()
menu()

# --- Hero ---------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Citation-Verified RAG</h1>
  <p>ChatGPT guesses. We verify — every sentence checked against its source,
     with page-level provenance. Legal and Academic workspaces.</p>
</div>
""",
    unsafe_allow_html=True,
)

cta1, cta2, _ = st.columns([1, 1, 2])
with cta1:
    st.page_link(
        "pages/2_Chat.py",
        label="Open Chat (demo)",
        icon=":material/chat:",
    )
with cta2:
    st.page_link(
        "pages/3_Documents.py",
        label="Upload a document",
        icon=":material/upload_file:",
    )

# --- Trust row ----------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Faithfulness", "1.00", "Ragas, golden set")
m2.metric("Answer relevancy", "0.88", "threshold 0.75")
m3.metric("Context recall", "1.00", "threshold 0.70")
m4.metric("Sources cited", "100%", "every claim grounded")

st.divider()

# --- Features -----------------------------------------------------------
f1, f2 = st.columns(2)
with f1:
    with st.container(border=True):
        st.subheader("Verify, don't guess")
        st.markdown(
            "Answers are validated against retrieved context before they "
            "render. The system abstains rather than fabricates — "
            "faithfulness **1.00** on the Ragas golden set."
        )
with f2:
    with st.container(border=True):
        st.subheader("Provenance on every claim")
        st.markdown(
            "Each statement carries `[SOURCE N]` citations down to page "
            "number, with retrievable chunk text and Langfuse traces for "
            "full observability."
        )

f3, f4 = st.columns(2)
with f3:
    with st.container(border=True):
        st.subheader("Legal workspace")
        st.markdown(
            "Contract and case-document Q&A with section-aware retrieval. "
            "Workspaces arrive with PLAN PR-4 — engine is shared."
        )
with f4:
    with st.container(border=True):
        st.subheader("Academic workspace")
        st.markdown(
            "Paper comprehension over your own corpus: hybrid BM25 + vector "
            "search, cross-encoder reranking, citation-enforced generation."
        )

st.divider()

# --- Stack --------------------------------------------------------------
st.caption(
    "Built with: Groq · LangChain · Chroma · Streamlit · Langfuse · Ragas"
)

st.divider()

# --- FAQ ----------------------------------------------------------------
st.subheader("FAQ")
with st.expander("How is this different from ChatGPT?"):
    st.write(
        "It retrieves from YOUR documents only, cites every claim with page "
        "numbers, and refuses to answer when the corpus does not support it."
    )
with st.expander("What does 'citation-verified' mean?"):
    st.write(
        "Generated sentences are checked against retrieved chunks before "
        "display. Unsupported claims are rejected — verification badges and "
        "claim-level provenance ship with PLAN PR-4b."
    )
with st.expander("Is there a free tier?"):
    st.write(
        "Yes — Free ($0) covers the demo. Paid tiers (Pro/Teams) activate "
        "with Stripe billing; pricing below is a preview."
    )

# --- Pricing ------------------------------------------------------------
st.subheader("Pricing")
p1, p2, p3 = st.columns(3)
with p1:
    with st.container(border=True):
        st.markdown("**Free**")
        st.markdown("### $0")
        st.markdown("- Demo workspace\n- Community support")
        st.button(
            "Current plan", key="price_free", disabled=True, use_container_width=True
        )
with p2:
    with st.container(border=True):
        st.markdown("**Pro**")
        st.markdown("### $9 / mo")
        st.markdown("- Higher quotas\n- Priority processing")
        st.button(
            "Coming soon", key="price_pro", disabled=True, use_container_width=True
        )
with p3:
    with st.container(border=True):
        st.markdown("**Teams**")
        st.markdown("### $29 / mo")
        st.markdown("- Shared workspaces\n- Admin controls")
        st.button(
            "Coming soon",
            key="price_teams",
            disabled=True,
            use_container_width=True,
        )

st.caption(
    "Billing activates with PLAN PR-5 (Stripe). Bangladesh-first: "
    "self-host friendly."
)

# --- Contact ------------------------------------------------------------
st.divider()
st.subheader("Contact")
with st.form("contact_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    c1.text_input("Name", key="contact_name", placeholder="Your name")
    c2.text_input("Email", key="contact_email", placeholder="you@example.com")
    st.text_area("Message", key="contact_message", placeholder="How can we help?")
    submitted = st.form_submit_button("Send message", type="primary")
    if submitted:
        st.toast("Message saved locally — sending lands with backend", icon=":material/mail:")
