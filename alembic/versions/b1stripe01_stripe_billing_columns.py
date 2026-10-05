"""stripe billing columns on subscriptions

Revision ID: b1stripe01
Revises: 4c23c4b04a65
Create Date: 2026-10-04 21:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1stripe01'
down_revision: Union[str, Sequence[str], None] = '4c23c4b04a65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add Stripe linkage + lifecycle columns (PLAN PR-5)."""
    with op.batch_alter_table('subscriptions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('stripe_customer_id', sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column('stripe_subscription_id', sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column('status', sa.String(length=20), nullable=False, server_default='inactive'))
        batch_op.add_column(sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_unique_constraint('uq_subscriptions_stripe_customer_id', ['stripe_customer_id'])
        batch_op.create_unique_constraint('uq_subscriptions_stripe_subscription_id', ['stripe_subscription_id'])


def downgrade() -> None:
    """Drop Stripe columns."""
    with op.batch_alter_table('subscriptions', schema=None) as batch_op:
        batch_op.drop_constraint('uq_subscriptions_stripe_subscription_id', type_='unique')
        batch_op.drop_constraint('uq_subscriptions_stripe_customer_id', type_='unique')
        batch_op.drop_column('current_period_end')
        batch_op.drop_column('status')
        batch_op.drop_column('stripe_subscription_id')
        batch_op.drop_column('stripe_customer_id')
