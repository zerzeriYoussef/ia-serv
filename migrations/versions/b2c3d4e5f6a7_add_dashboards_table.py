"""add_dashboards_table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-01 20:00:00.000000

Creates the `dashboards` table that persists RAG-insight generation runs.
Each row links to a dataset (required) and optionally to a column_analysis
and a user_id string from the auth service.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dashboards",
        sa.Column("id",          sa.Integer(),    nullable=False),
        sa.Column("user_id",     sa.String(),     nullable=True),
        sa.Column("dataset_id",  sa.Integer(),    nullable=False),
        sa.Column("analysis_id", sa.Integer(),    nullable=True),

        # Persisted LLM output
        sa.Column("executive_summary_kpis", sa.JSON(), nullable=True),
        sa.Column("dashboard_charts",       sa.JSON(), nullable=True),

        # Optional RAG debug metadata
        sa.Column("rag_meta", sa.JSON(), nullable=True),

        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),

        # Constraints
        sa.ForeignKeyConstraint(["dataset_id"],  ["datasets.id"],       name="fk_dashboards_dataset_id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["column_analysis.id"], name="fk_dashboards_analysis_id"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes
    op.create_index("ix_dashboards_id",          "dashboards", ["id"],          unique=False)
    op.create_index("ix_dashboards_dataset_id",  "dashboards", ["dataset_id"],  unique=False)
    op.create_index("ix_dashboards_user_id",     "dashboards", ["user_id"],     unique=False)


def downgrade() -> None:
    op.drop_index("ix_dashboards_user_id",    table_name="dashboards")
    op.drop_index("ix_dashboards_dataset_id", table_name="dashboards")
    op.drop_index("ix_dashboards_id",         table_name="dashboards")
    op.drop_table("dashboards")
