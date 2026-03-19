"""add user_id to datasets

Revision ID: 4b7c6a1d2c3e
Revises: 0774812a95a5
Create Date: 2026-03-19 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "4b7c6a1d2c3e"
down_revision: Union[str, Sequence[str], None] = "0774812a95a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("datasets", sa.Column("user_id", sa.String(), nullable=True))
    op.create_index(op.f("ix_datasets_user_id"), "datasets", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_datasets_user_id"), table_name="datasets")
    op.drop_column("datasets", "user_id")

