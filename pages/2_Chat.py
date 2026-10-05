"""Chat page (docs/UI_DESIGN.md §5 — 2_Chat).

PR-6: every query goes through `api_client` → `POST /chat` with the
session JWT. No `src.*` service imports — retrieval, quotas, and
generation live behind the API. Guests (no session) auto-provision a
per-device guest account and hit the login wall after the free tier.
"""

import streamlit as st
from loguru import logger

import api_client
from api_client import APIError
from menu import menu_with_redirect, show_login_wall
from src.config import Config
from ui_core import (
    apply_workspace_accent,
    brand_html,
    init_session_state,
    load_css,
    page_config,
)

page_config("Chat — RAG Research Assistant")
init_session_state()
load_css()
apply_workspace_accent()
menu_with_redirect()


def display_answer(answer: str, sources: list[dict]) -> None:
    """Display answer text, then sources as expanders."""
    st.markdown(answer)

    if sources:
        st.markdown("---")
        st.markdown("**Sources:**")
        for i, source in enumerate(sources, 1):
            with st.expander(
                f"[{i}] {source['doc_id']} - Page {source['page_num']}"
            ):
                st.markdown(f"**Document:** `{source['doc_id']}`")
                st.markdown(f"**Page:** {source['page_num']}")
                st.markdown("**Text:**")
                text = source["text"]
                st.text(
                    text[:500] + "..." if len(text) > 500 else text
                )


def _assistant_error(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})


def _handle_api_error(exc: APIError) -> None:
    """Map transport/quota/auth failures onto UI actions."""
    logger.error(f"Chat API error status={exc.status}: {exc.detail}")
    if exc.status == 401:
        st.warning("Your session expired — please sign in again.")
        show_login_wall()
        _assistant_error("Please sign in to keep chatting.")
    elif exc.status == 429 and st.session_state.get("anon_tier"):
        show_login_wall()
        _assistant_error("Free limit reached — sign in to keep chatting.")
    elif exc.status == 429:
        st.error(exc.detail)
        _assistant_error(exc.detail)
    elif exc.status == 0:
        st.error(exc.detail)
        _assistant_error("Backend unreachable. Please retry once it's up.")
    else:
        st.error(f"Error generating answer: {exc.detail}")
        _assistant_error("I encountered an error while generating the answer. Please try again.")


def handle_query(query: str) -> None:
    """Handle user query through the API (PLAN PR-6)."""
    # Guest free tier: proactive wall on the client counter; the API
    # enforces the authoritative per-device unit budget server-side.
    if not st.session_state.get("jwt"):
        if st.session_state.anon_queries >= Config.ANON_QUERY_LIMIT:
            show_login_wall()
            return
        st.session_state.anon_queries += 1
        try:
            api_client.ensure_guest_session()
            # Sidebar rendered before this first guest session existed.
            st.session_state.doc_count = len(api_client.list_documents())
        except APIError as exc:
            _handle_api_error(exc)
            return

    workspace = st.session_state.workspace
    if not st.session_state.get("doc_count"):
        st.warning("Please upload a PDF in the Documents page first.")
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = api_client.ask(query, workspace)
            except APIError as exc:
                _handle_api_error(exc)
                return

            answer = result["answer"]
            sources = result.get("sources") or []
            display_answer(answer, sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [
                        {
                            "doc_id": s["doc_id"],
                            "page_num": s["page_num"],
                            "text": s["text"],
                        }
                        for s in sources
                    ],
                }
            )


def render_sidebar() -> None:
    """Sidebar: brand, quota meter, workspace switcher, document list."""
    st.sidebar.markdown(brand_html(24), unsafe_allow_html=True)
    st.sidebar.divider()

    _render_quota()

    # Workspace switcher (spec §5 — accent + toast on change)
    def _on_workspace_change() -> None:
        st.toast(
            f"Workspace: {st.session_state.workspace}",
            icon=":material/scale:",
        )

    st.sidebar.selectbox(
        "Workspace",
        options=list(Config.WORKSPACES),
        key="workspace",
        on_change=_on_workspace_change,
        help="Legal and Academic profiles share one engine (PLAN PR-4)",
    )
    st.sidebar.divider()

    st.sidebar.markdown("### Documents")
    docs = _load_documents()
    st.session_state.doc_count = len(docs) if docs is not None else None
    if docs:
        for doc in docs:
            status = doc.get("status", "ready")
            suffix = "" if status == "ready" else f" · {status}"
            st.sidebar.markdown(f"- `{doc['filename']}`{suffix}")
        st.sidebar.page_link(
            "pages/3_Documents.py",
            label="Manage documents",
            icon=":material/upload_file:",
        )
    else:
        # Covers "no session yet" and "session with no uploads" alike —
        # the Documents page provisions the guest session on demand.
        st.sidebar.info("No documents yet")
        st.sidebar.page_link(
            "pages/3_Documents.py",
            label="Upload a document",
            icon=":material/upload_file:",
        )


def _render_quota() -> None:
    """Quota meter: guest/member from /usage, free counter pre-sign-in."""
    if not st.session_state.get("jwt"):
        left = max(
            Config.ANON_QUERY_LIMIT - st.session_state.anon_queries, 0
        )
        st.sidebar.caption(f"Free questions left: {left}")
        return

    try:
        meter = api_client.usage()
    except APIError as exc:
        logger.warning(f"Usage meter unavailable: {exc.detail}")
        return

    queries = meter["queries"]
    if meter.get("tier") == "anonymous":
        # Guests think in questions; the API budgets in query+verify units.
        used_questions = (queries["used"] + 1) // 2
        left = max(Config.ANON_QUERY_LIMIT - used_questions, 0)
        st.sidebar.caption(f"Free questions left: {left}")
    else:
        st.sidebar.caption(
            f"Queries today: {queries['used']}/{queries['limit']}"
        )


def _load_documents() -> list[dict] | None:
    """Document list for the sidebar; None while no session is established."""
    if not st.session_state.get("jwt"):
        return None
    try:
        return api_client.list_documents()
    except APIError as exc:
        logger.warning(f"Document list unavailable: {exc.detail}")
        return None


def render_chat() -> None:
    """Render main chat area."""
    st.markdown("### Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                st.markdown("---")
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    with st.expander(
                        f"[{i}] {source['doc_id']} - Page {source['page_num']}"
                    ):
                        st.markdown(f"**Document:** `{source['doc_id']}`")
                        st.markdown(f"**Page:** {source['page_num']}")
                        st.markdown("**Text:**")
                        text = source["text"]
                        st.text(
                            text[:500] + "..." if len(text) > 500 else text
                        )

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_query(prompt)


def main() -> None:
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
