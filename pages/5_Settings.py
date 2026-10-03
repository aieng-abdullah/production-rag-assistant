"""Settings — account, workspace default, provider keys, danger zone (spec §5).

Provider keys stay session-local until PR-6 moves calls behind the API.
Danger-zone actions need the PR-2b auth service — disabled, never faked.
"""

import streamlit as st

from menu import logout, menu_with_redirect
from ui_core import init_session_state, load_css, page_config

page_config("Settings — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()

# --- Account ------------------------------------------------------------
st.subheader("Account")
with st.container(border=True):
    a1, a2 = st.columns([3, 1])
    a1.markdown(f"**{st.session_state.user_email or 'demo@local'}**")
    a1.caption("Signed in locally — Google accounts arrive with PLAN PR-2b.")
    a2.button("Log out", icon=":material/logout:", on_click=logout, use_container_width=True)

# --- Workspace default --------------------------------------------------
st.subheader("Workspace")
with st.container(border=True):
    st.radio(
        "Default workspace",
        options=["legal", "academic"],
        format_func=lambda w: w.capitalize(),
        key="workspace",
        horizontal=True,
    )
    st.caption("Legal = navy accent, Academic = teal. Chat can switch per session.")

# --- Provider keys ------------------------------------------------------
st.subheader("LLM providers")
st.caption("Groq is free and built in. Add your own key for other providers:")
with st.container(border=True):
    k1, k2 = st.columns(2)
    k1.text_input(
        "Anthropic API key",
        type="password",
        key="anthropic_key",
        placeholder="sk-ant-…",
    )
    k1.selectbox(
        "Anthropic model",
        ["claude-sonnet-4-20250514"],
        key="anthropic_model",
        disabled=True,
    )
    k2.text_input(
        "OpenAI API key",
        type="password",
        key="openai_key",
        placeholder="sk-…",
    )
    k2.selectbox(
        "OpenAI model",
        ["gpt-4o"],
        key="openai_model",
        disabled=True,
    )
    st.caption("Keys live in this browser session only — never persisted.")

# --- Danger zone --------------------------------------------------------
st.subheader("Danger zone")
with st.container(border=True):
    d1, d2 = st.columns([3, 1])
    d1.markdown("**Delete account**")
    d1.caption("Removes your workspaces, documents and history permanently.")
    d2.button(
        "Delete",
        key="delete_account",
        disabled=True,
        use_container_width=True,
    )
    st.caption("Account lifecycle arrives with PLAN PR-2b (auth service).")
