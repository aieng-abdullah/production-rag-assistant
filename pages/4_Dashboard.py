"""Dashboard — quota metrics, usage trend, answer history (spec §5).

Interim data sources are session counters; PR-4b swaps in real usage
and provenance traces. No new session keys (spec §3).
"""

import pandas as pd
import streamlit as st

from menu import menu_with_redirect
from ui_core import init_session_state, load_css, lottie, page_config

page_config("Dashboard — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()

# --- Metric cards -------------------------------------------------------
turns = [m for m in st.session_state.messages if m.get("role") == "user"]
plan_name = "Free" if st.session_state.get("jwt") else "—"

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("#### :material/help_circle: Queries today")
        st.markdown(f"## {st.session_state.quota}")
        st.caption("Free plan demo cap: 500")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/folder: Documents")
        st.markdown(f"## {len(st.session_state.ingested_docs)}")
        st.caption("Uploaded this session")
with c3:
    with st.container(border=True):
        st.markdown("#### :material/workspace_premium: Plan")
        st.markdown(f"## {plan_name}")
        st.caption("Upgrade in Billing")

# --- Usage trend --------------------------------------------------------
st.subheader("Usage — last 7 days")
usage = pd.DataFrame(
    {
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "queries": [12, 18, 9, 24, 15, 4, max(st.session_state.quota, 1)],
    }
)
st.bar_chart(usage.set_index("day"), height=240)
st.caption(
    "Sample week except today — live per-day counts arrive with PLAN PR-4b."
)

# --- Answer history -----------------------------------------------------
st.subheader("Answer history")
if not turns:
    _d_l, _d_anim, _d_r = st.columns([1, 2, 1])
    with _d_anim:
        lottie("waiting", height=150)
    st.info("No queries yet — ask something in Chat to build your history.")
else:
    rows = [
        {"#": i + 1, "Question": m["content"][:120]}
        for i, m in enumerate(turns)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Expandable provenance traces arrive with PLAN PR-4b.")
