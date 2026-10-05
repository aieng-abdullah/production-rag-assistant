"""Post-login plan chooser: Free vs Pro (paid = coming soon).

One-shot dialog driven by the `show_pricing_modal` session flag — set on
every login (demo or Google), cleared when the user picks a plan.
"""

import streamlit as st

from src.config import Config

__all__ = ["maybe_show_pricing_modal"]


@st.dialog("Choose your plan")
def _pricing_dialog() -> None:
    st.markdown("### Pick your plan")
    free, pro = st.columns(2)

    with free:
        st.markdown("#### Free")
        st.caption(
            f"{Config.DAILY_QUERY_LIMIT} queries/day · "
            f"{Config.DOCUMENT_LIMIT} documents · community support"
        )
        if st.button(
            "Continue with Free",
            type="primary",
            use_container_width=True,
            key="plan_free",
        ):
            st.session_state.user_tier = "free"
            st.session_state.show_pricing_modal = False
            st.rerun()

    with pro:
        st.markdown("#### Pro — $9 / mo")
        st.caption("5,000 queries/day · 100 documents · priority support")
        if st.button(
            "Buy Pro",
            use_container_width=True,
            key="plan_pro",
        ):
            st.toast(
                "Paid plans are coming soon — stay tuned.",
                icon=":material/credit_card:",
            )


def maybe_show_pricing_modal() -> None:
    """Open the plan chooser once per login while the flag is set."""
    if st.session_state.get("show_pricing_modal"):
        _pricing_dialog()
