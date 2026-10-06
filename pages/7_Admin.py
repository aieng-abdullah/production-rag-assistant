"""Admin page — user tier management (Phase 3: polish).

Only accessible to admin users (hardcoded email for simplicity).
"""

import streamlit as st
from sqlalchemy import select, func

from menu import menu_with_redirect
from src.config import Config
from src.db.database import get_session_factory
from src.db.models import User, Subscription, UsageEvent
from ui_core import init_session_state, load_css, page_config

# --- Admin config ---
ADMIN_EMAILS = ["admin@example.com"]  # Replace with your admin email(s)


def _is_admin() -> bool:
    email = st.session_state.get("user_email", "")
    return email in ADMIN_EMAILS


page_config("Admin — RAG Research Assistant")
init_session_state()
load_css()
menu_with_redirect()

if not _is_admin():
    st.error("Access denied. Admin only.")
    st.stop()

st.title("🔧 Admin: User Management")

# --- User list with tier management ---
st.subheader("Users")

session_factory = get_session_factory()
with session_factory() as session:
    users = session.execute(select(User).order_by(User.created_at.desc())).scalars().all()

    for user in users:
        # Get subscription
        sub = session.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        ).scalars().first()

        tier = sub.tier if sub else "free"
        status = sub.status if sub else "inactive"

        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])

        col1.markdown(f"**{user.email}**")
        col1.caption(f"ID: {user.id} · {user.default_workspace}")

        # Tier badge
        tier_badge = "🟢 Pro" if tier == "pro" else "⚪ Free"
        col2.markdown(tier_badge)

        # Usage today
        today = func.date(UsageEvent.created_at) == func.date(func.now())
        query_used = session.execute(
            select(func.coalesce(func.sum(UsageEvent.units), 0))
            .where(
                UsageEvent.user_id == user.id,
                UsageEvent.kind == "query",
                today,
            )
        ).scalar() or 0

        multiplier = 10 if tier == "pro" else 1
        limit = Config.DAILY_QUERY_LIMIT * multiplier
        col3.metric("Queries", f"{query_used}/{limit}")

        # Tier change buttons
        with col4:
            if tier == "free":
                if st.button("Upgrade → Pro", key=f"upgrade_{user.id}", type="primary"):
                    if sub:
                        sub.tier = "pro"
                        sub.status = "active"
                    else:
                        session.add(Subscription(user_id=user.id, tier="pro", status="active"))
                    session.commit()
                    st.toast(f"Upgraded {user.email} to Pro")
                    st.rerun()
            else:
                if st.button("Downgrade → Free", key=f"downgrade_{user.id}"):
                    if sub:
                        sub.tier = "free"
                        sub.status = "inactive"
                    session.commit()
                    st.toast(f"Downgraded {user.email} to Free")
                    st.rerun()

        # Delete user button
        with col5:
            if st.button("Delete User", key=f"delete_{user.id}", type="secondary"):
                if st.session_state.get(f"confirm_delete_{user.id}"):
                    # Delete data
                    from src.services import RAGService
                    rag = RAGService()
                    rag.delete_tenant_data(str(user.id))

                    # Delete user
                    session.delete(user)
                    session.commit()
                    st.toast(f"Deleted {user.email}")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_{user.id}"] = True
                    st.warning("Click again to confirm")
                    st.rerun()

        st.divider()

# --- Stats ---
st.subheader("System Stats")
with session_factory() as session:
    total_users = session.execute(select(func.count(User.id))).scalar()
    pro_users = session.execute(
        select(func.count(Subscription.id)).where(Subscription.tier == "pro")
    ).scalar()
    total_docs = session.execute(select(func.count(UsageEvent.id)).where(UsageEvent.kind == "ingest")).scalar()

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Users", total_users)
    s2.metric("Pro Users", pro_users or 0)
    s3.metric("Total Documents", total_docs or 0)