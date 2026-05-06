"""extend cleaning profile and log

Revision ID: 7d8a3e9c1f02
Revises: 52e516ec0852
Create Date: 2026-05-06 13:00:00.000000

Adds new columns to support the smarter cleaning algorithm:

- ``cleaning_profiles``: ``duplicate_keep``, ``dedup_normalize_text``,
  ``numeric_coerce_min_rate``, ``outlier_max_rows``, ``column_rules``.
- ``cleaning_logs``: ``backup_path`` so we can record the path of the
  pre-cleaning backup created when the source file is overwritten in place.

The migration is additive and backwards compatible. Existing rows pick up
sensible defaults via the ``server_default`` clauses below.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7d8a3e9c1f02"
down_revision: Union[str, Sequence[str], None] = "52e516ec0852"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cleaning_profiles",
        sa.Column(
            "duplicate_keep",
            sa.String(),
            nullable=True,
            server_default="best",
        ),
    )
    op.add_column(
        "cleaning_profiles",
        sa.Column(
            "dedup_normalize_text",
            sa.Boolean(),
            nullable=True,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        "cleaning_profiles",
        sa.Column(
            "numeric_coerce_min_rate",
            sa.Float(),
            nullable=True,
            server_default="0.8",
        ),
    )
    op.add_column(
        "cleaning_profiles",
        sa.Column(
            "outlier_max_rows",
            sa.Integer(),
            nullable=True,
            server_default="500000",
        ),
    )
    op.add_column(
        "cleaning_profiles",
        sa.Column("column_rules", sa.JSON(), nullable=True),
    )

    op.add_column(
        "cleaning_logs",
        sa.Column("backup_path", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cleaning_logs", "backup_path")
    op.drop_column("cleaning_profiles", "column_rules")
    op.drop_column("cleaning_profiles", "outlier_max_rows")
    op.drop_column("cleaning_profiles", "numeric_coerce_min_rate")
    op.drop_column("cleaning_profiles", "dedup_normalize_text")
    op.drop_column("cleaning_profiles", "duplicate_keep")
