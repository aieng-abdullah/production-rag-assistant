"""Billing — current plan, quota, comparison, invoices (spec addendum).

Upgrade buttons are toast-only until PLAN PR-5 flips the Stripe
env-flag; no dead links (AGENTS PR contract).
"""

import streamlit as st

from menu import menu_with_redirect
from ui_core import init_session_state, load_css, page_config

page_config("Billing — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()

QUOTA_CAP = 500  # Free-plan demo cap until PR-5 defines tiers

# --- Current plan -------------------------------------------------------
st.subheader("Current plan")
with st.container(border=True):
    p1, p2 = st.columns([3, 1])
    p1.markdown("### Free")
    p1.caption("Demo workspaces · community support · 500 queries/day")
    p2.button(
        "Upgrade",
        key="billing_upgrade",
        type="primary",
        use_container_width=True,
    )
    if st.session_state.get("billing_upgrade"):
        st.toast("Stripe activates with PLAN PR-5", icon=":material/credit_card:")

# --- Quota usage --------------------------------------------------------
st.subheader("Usage this session")
used = min(st.session_state.quota, QUOTA_CAP)
st.progress(used / QUOTA_CAP, text=f"{used} / {QUOTA_CAP} queries")
if used / QUOTA_CAP >= 0.8:
    st.warning("Quota nearly reached — upgrade opens with PR-5.", icon=":material/warning:")

# --- Plan comparison ----------------------------------------------------
st.subheader("Compare plans")
st.table(
    {
        "": ["Price", "Daily queries", "Workspaces", "Documents", "Support"],
        "Free": ["$0", "500", "1", "5", "Community"],
        "Pro": ["$9 / mo", "5,000", "3", "100", "Priority"],
        "Teams": ["$29 / mo", "25,000", "Unlimited", "Unlimited", "Admin + SSO"],
    }
)

# --- Invoices -----------------------------------------------------------
st.subheader("Invoices")
with st.container(border=True):
    st.info(
        "No invoices yet — billing activates with PLAN PR-5 (Stripe). "
        "Bangladesh-first: local payment rails planned alongside Stripe.",
        icon=":material/receipt:",
    )
