"""Dashboard — quota meters + answer history (spec §5).

PR-6: quota cards come from `GET /usage` (tier-aware); answer history is
this browser session's messages. No sample charts shown as fact — a real
per-day trend needs a usage-history endpoint (future ticket).
"""

import pandas as pd
import streamlit as st
from loguru import logger

import api_client
from api_client import APIError
from menu import menu_with_redirect
from ui_core import init_session_state, load_css, page_config

page_config("Dashboard — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()

# --- Metric cards -------------------------------------------------------
turns = [m for m in st.session_state.messages if m.get("role") == "user"]

if not st.session_state.get("jwt"):
    st.info("Sign in to see your quota meters.")
else:
    try:
        meter = api_client.usage()
    except APIError as exc:
        logger.warning(f"Usage meter unavailable: {exc.detail}")
        st.warning(f"Usage unavailable: {exc.detail}")
        meter = None

    if meter:
        plan = "Guest" if meter.get("tier") == "anonymous" else "Free"
        queries = meter["queries"]
        documents = meter["documents"]
        storage = meter["storage"]

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("#### :material/help_circle: Queries today")
                st.markdown(f"## {queries['used']} / {queries['limit']}")
                st.caption("Verified answers cost 2 units (query + verify).")
        with c2:
            with st.container(border=True):
                st.markdown("#### :material/folder: Documents")
                st.markdown(f"## {documents['used']} / {documents['limit']}")
                st.caption("Uploaded to your account.")
        with c3:
            with st.container(border=True):
                st.markdown("#### :material/workspace_premium: Plan")
                st.markdown(f"## {plan}")
                st.caption("Upgrade in Billing")

        used_mb = storage["used_bytes"] / (1024 * 1024)
        limit_mb = storage["limit_bytes"] / (1024 * 1024)
        st.subheader("Storage")
        st.progress(
            min(storage["used_bytes"] / max(storage["limit_bytes"], 1), 1.0)
        )
        st.caption(f"{used_mb:.1f} MB of {limit_mb:.0f} MB used")

# --- Answer history -----------------------------------------------------
st.subheader("Answer history")
if not turns:
    st.info("No queries yet — ask something in Chat to build your history.")
else:
    rows = [
        {"#": i + 1, "Question": m["content"][:120]}
        for i, m in enumerate(turns)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("This browser session only — full history arrives with the usage API.")
