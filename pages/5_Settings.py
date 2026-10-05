"""Settings — account, workspace default, provider keys, danger zone, usage (spec §5).

Provider keys stay session-local (single-process mode, PLAN PR-6.1).
"""

import streamlit as st
from datetime import date
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from menu import logout, menu_with_redirect
from src.auth.dependencies import is_guest, get_current_user_id
from src.config import Config
from src.db.database import get_session_factory
from src.db.models import User, UsageEvent, Subscription
from ui_core import init_session_state, load_css, page_config

page_config("Settings — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()


def _persist_workspace(user_id: int, workspace: str) -> None:
    """Persist default workspace to database."""
    session_factory = get_session_factory()
    with session_factory() as session:
        user = session.get(User, user_id)
        if user:
            user.default_workspace = workspace
            session.commit()


def _get_user_tier(user_id: int) -> str:
    """Get user's tier from subscription."""
    from src.db.models import Subscription
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = select(Subscription.tier).where(Subscription.user_id == user_id)
        tier = session.scalars(stmt).first()
        return tier or "free"


# --- Account ------------------------------------------------------------
st.subheader("Account")
with st.container(border=True):
    a1, a2 = st.columns([3, 1])

    user_email = st.session_state.get("user_email") or "guest@local"
    user_tier = st.session_state.get("user_tier", "free")
    tier_badge = "🟢 Pro" if user_tier == "pro" else "⚪ Free"

    a1.markdown(f"**{user_email}**")
    a1.caption(f"{tier_badge} · Signed in with Google")
    a2.button("Log out", icon=":material/logout:", on_click=logout, use_container_width=True)

# --- Workspace default --------------------------------------------------
st.subheader("Workspace")
with st.container(border=True):
    # Load default from database for real users
    default_ws = "academic"
    user_id = get_current_user_id()
    if user_id and not is_guest():
        session_factory = get_session_factory()
        with session_factory() as session:
            user = session.get(User, user_id)
            if user and user.default_workspace:
                default_ws = user.default_workspace

    new_ws = st.radio(
        "Default workspace",
        options=["legal", "academic"],
        format_func=lambda w: w.capitalize(),
        index=0 if default_ws == "legal" else 1,
        horizontal=True,
    )
    if new_ws != st.session_state.workspace:
        st.session_state.workspace = new_ws
        if user_id and not is_guest():
            _persist_workspace(user_id, new_ws)
            st.toast(f"Default workspace saved: {new_ws}")
    st.caption("Legal = navy accent, Academic = teal. Chat can switch per session.")

# --- Usage stats --------------------------------------------------------
if not is_guest():
    st.subheader("Usage")
    user_id = get_current_user_id()
    tier = st.session_state.get("user_tier", "free")
    multiplier = 10 if tier == "pro" else 1

    session_factory = get_session_factory()
    with session_factory() as session:
        # Today's query usage
        today = date.today()
        query_used = session.execute(
            select(func.coalesce(func.sum(UsageEvent.units), 0))
            .where(
                UsageEvent.user_id == user_id,
                UsageEvent.kind == "query",
                func.date(UsageEvent.created_at) == today,
            )
        ).scalar() or 0

        query_limit = Config.DAILY_QUERY_LIMIT * multiplier
        query_remaining = max(query_limit - query_used, 0)

        # Document count
        doc_count = session.execute(
            select(func.count(Subscription.id))
            .where(Subscription.user_id == user_id)
        ).scalar() or 0

        # Actually count documents from Chroma
        from src.db.chroma_client import count_chunks
        tenant_id = str(user_id)
        total_chunks = count_chunks(tenant_id)

    u1, u2, u3 = st.columns(3)
    with u1:
        st.metric("Queries today", f"{query_used}/{query_limit}", f"{query_remaining} left")
    with u2:
        st.metric("Documents", f"{doc_count}/{Config.DOCUMENT_LIMIT * multiplier}")
    with u3:
        st.metric("Chunks indexed", total_chunks)

    st.caption(f"Tier: {'🟢 Pro' if tier == 'pro' else '⚪ Free'} · Limits reset daily at midnight UTC")

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
    if is_guest():
        st.info("Sign in with Google to access account deletion.")
        st.button(
            "Delete account",
            key="delete_account",
            disabled=True,
            use_container_width=True,
        )
    else:
        d1, d2 = st.columns([3, 1])
        d1.markdown("**Delete account**")
        d1.caption("Removes your workspaces, documents and history permanently.")
        if d2.button(
            "Delete",
            key="delete_account",
            type="primary",
            use_container_width=True,
        ):
            # Two-step confirmation
            if st.session_state.get("confirm_delete"):
                # Actually delete
                from src.services import RAGService
                rag = RAGService()
                tenant_id = str(user_id)
                rag.delete_tenant_data(tenant_id)

                # Delete user from database
                session_factory = get_session_factory()
                with session_factory() as session:
                    user = session.get(User, user_id)
                    if user:
                        session.delete(user)
                        session.commit()

                logout()
                st.success("Account deleted.")
                st.rerun()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm deletion.")
                st.rerun()