"""add chart_spec to chat_messages

Revision ID: c8f2a1b3d4e5
Revises: 91a7b2c4d5e6
Create Date: 2026-05-15 16:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c8f2a1b3d4e5"
down_revision: Union[str, Sequence[str], None] = "91a7b2c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("chart_spec", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "chart_spec")
