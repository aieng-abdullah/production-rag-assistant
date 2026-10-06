"""Chat page (docs/UI_DESIGN.md §5 — 2_Chat).

Single-process mode (PLAN PR-6.1): calls `src.services.RAGService`
directly — no backend, no api_client.
"""

from pathlib import Path

import streamlit as st
from loguru import logger

from menu import menu_with_redirect, show_login_wall
from src.auth.dependencies import is_guest, get_current_tier
from src.config import Config
from src.db.chroma_client import count_chunks, has_chunks
from src.generation.providers import ProviderOverrides
from src.services import (
    RAGService,
    TIER_MULTIPLIERS,
    check_query_quota,
    record_query_usage,
)
from src.services.bm25_cache import get_bm25
from ui_core import (
    apply_workspace_accent,
    brand_html,
    init_session_state,
    load_css,
    lottie,
    page_config,
)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

rag_service = RAGService()

page_config("Chat — RAG Research Assistant")
init_session_state()
load_css()
apply_workspace_accent()
menu_with_redirect()


def _ui_provider_overrides() -> ProviderOverrides:
    """Read provider keys/models from sidebar UI into a per-request object."""
    return ProviderOverrides(
        anthropic_api_key=st.session_state.get("anthropic_key", "").strip(),
        anthropic_model=st.session_state.get("anthropic_model", "").strip(),
        openai_api_key=st.session_state.get("openai_key", "").strip(),
        openai_model=st.session_state.get("openai_model", "").strip(),
    )


_WARM_ABSTAIN = (
    "I couldn't find that in your documents — I'd rather tell you than guess. "
    "Try rephrasing your question, or upload a document that covers it."
)


def _display_text(cited_answer) -> str:
    """Warm human copy for abstentions; the raw exact reason stays in
    verification/trace for audit."""
    if (cited_answer.verification or {}).get("status") == "abstained":
        return _WARM_ABSTAIN
    return cited_answer.answer


def display_cited_answer(cited_answer) -> None:
    """Display answer text, then sources as expanders."""
    st.markdown(_display_text(cited_answer))

    if cited_answer.sources:
        st.markdown("---")
        st.markdown("**Sources:**")
        for i, source in enumerate(cited_answer.sources, 1):
            with st.expander(
                f"[{i}] {source.doc_id} - Page {source.page_num}"
            ):
                st.markdown(f"**Document:** `{source.doc_id}`")
                st.markdown(f"**Page:** {source.page_num}")
                st.markdown("**Text:**")
                text = source.text
                st.text(
                    text[:500] + "..." if len(text) > 500 else text
                )


def _get_tenant_id() -> str:
    """Get tenant_id from current session."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return "default"
    return str(user_id)


def handle_query(query: str) -> None:
    """Handle user query and generate response."""
    tenant_id = _get_tenant_id()
    workspace = st.session_state.workspace
    tier = get_current_tier()

    # Check if user has any chunks for this workspace
    if not has_chunks(tenant_id, workspace):
        st.warning("Please upload a PDF in the Documents page first.")
        return

    bm25 = get_bm25(tenant_id, workspace)
    if bm25 is None:
        st.warning(f"No documents indexed for the '{workspace}' workspace yet.")
        return

    # Quota check
    if is_guest():
        if st.session_state.guest_queries >= Config.GUEST_QUERY_LIMIT:
            show_login_wall()
            return
        st.session_state.guest_queries += 1
    else:
        # Authenticated user - check persisted quota
        quota = check_query_quota(int(tenant_id), tier)
        if not quota.allowed:
            st.error(
                f"Daily query limit reached ({quota.used}/{quota.limit}). "
                f"Upgrade to Pro for {TIER_MULTIPLIERS['pro']}x quota."
            )
            return
        record_query_usage(int(tenant_id))

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        _t_l, _t_anim, _t_r = st.columns([1, 2, 1])
        with _t_anim:
            lottie("thinking", height=150)
        st.caption("Thinking…")
        try:
            cited_answer = rag_service.generate_answer(
                tenant_id=tenant_id,
                query=query,
                bm25_index=bm25,
                provider_overrides=_ui_provider_overrides(),
                workspace=workspace,
            )

            display_cited_answer(cited_answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _display_text(cited_answer),
                    "sources": [
                        {
                            "doc_id": s.doc_id,
                            "page_num": s.page_num,
                            "text": s.text,
                        }
                        for s in cited_answer.sources
                    ],
                }
            )

        except Exception as e:
            st.error(f"Error generating answer: {e}")
            logger.error(f"Generation failed: {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I encountered an error while generating the "
                        "answer. Please try again."
                    ),
                }
            )


def render_sidebar() -> None:
    """Sidebar: brand, workspace switcher, documents, stats, provider keys."""
    st.sidebar.markdown(brand_html(24), unsafe_allow_html=True)
    st.sidebar.divider()

    # Show quota remaining
    tier = get_current_tier()
    tenant_id = _get_tenant_id()

    if is_guest():
        left = max(Config.GUEST_QUERY_LIMIT - st.session_state.guest_queries, 0)
        st.sidebar.caption(f"Guest queries left: {left}")
    else:
        quota = check_query_quota(int(tenant_id), tier)
        st.sidebar.caption(f"Queries today: {quota.used}/{quota.limit} ({tier})")
        if quota.remaining < 3:
            st.sidebar.warning(f"Only {quota.remaining} queries left today")

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
    doc_ids = rag_service.list_documents(tenant_id)
    if doc_ids:
        for doc_id in doc_ids:
            st.sidebar.markdown(f"- `{doc_id}`")
        st.sidebar.page_link(
            "pages/3_Documents.py",
            label="Manage documents",
            icon=":material/upload_file:",
        )
    else:
        st.sidebar.info("No documents yet")
        st.sidebar.page_link(
            "pages/3_Documents.py",
            label="Upload a document",
            icon=":material/upload_file:",
        )

    if has_chunks(tenant_id):
        st.sidebar.divider()
        st.sidebar.markdown("### Stats")
        st.sidebar.caption(f"Total chunks: {count_chunks(tenant_id)}")

    st.sidebar.divider()
    st.sidebar.markdown("### LLM Providers")
    st.sidebar.caption("Groq is free. Add your own key for other providers:")

    with st.sidebar.expander("Anthropic (optional)"):
        st.text_input(
            "API Key",
            type="password",
            key="anthropic_key",
            help="Get your key from console.anthropic.com",
        )
        st.text_input(
            "Model",
            key="anthropic_model",
            help="e.g. claude-sonnet-4-20250514, claude-3-5-haiku-20241022",
        )

    with st.sidebar.expander("OpenAI (optional)"):
        st.text_input(
            "API Key",
            type="password",
            key="openai_key",
            help="Get your key from platform.openai.com",
        )
        st.text_input("Model", key="openai_model", help="e.g. gpt-4o, gpt-4o-mini")


def render_chat() -> None:
    """Render main chat area."""
    st.markdown("### Chat")

    if not st.session_state.messages:
        _e_left, _e_anim, _e_right = st.columns([1, 2, 1])
        with _e_anim:
            lottie("chat", height=170)
        st.caption("Ask anything about your documents — every answer cites its page.")

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
    try:
        Config.validate()
    except EnvironmentError:
        st.sidebar.warning("No LLM API keys found in .env. Add one via the sidebar.")

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()