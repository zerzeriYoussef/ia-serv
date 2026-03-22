"""add_multi_dataset_and_statistical_fields

Revision ID: a1b2c3d4e5f6
Revises: 0774812a95a5
Create Date: 2026-03-22 21:20:00.000000

Changes:
- Add source_dataset_ids (JSON) for multi-dataset tracking
- Add merge_report (JSON) with merge operation details
- Add target_column (String) from statistical analysis
- Add statistical_validation (String) e.g. 'bonferroni_corrected'
- Drop the UNIQUE constraint on dataset_id (multi-dataset analyses
  share the primary dataset_id)
- Re-create dataset_id as a plain index
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '4b7c6a1d2c3e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new columns for multi-dataset support
    op.add_column(
        'column_analysis',
        sa.Column('source_dataset_ids', sa.JSON(), nullable=True),
    )
    op.add_column(
        'column_analysis',
        sa.Column('merge_report', sa.JSON(), nullable=True),
    )

    # Add statistical validation columns
    op.add_column(
        'column_analysis',
        sa.Column('target_column', sa.String(), nullable=True),
    )
    op.add_column(
        'column_analysis',
        sa.Column('statistical_validation', sa.String(), nullable=True),
    )

    # Drop the unique constraint on dataset_id so multiple multi-dataset
    # analyses can reference the same primary dataset.
    # PostgreSQL names unique constraints automatically; the name used by
    # the original migration was 'column_analysis_dataset_id_key'.
    try:
        op.drop_constraint(
            'column_analysis_dataset_id_key',
            'column_analysis',
            type_='unique',
        )
    except Exception:
        # If the constraint was already dropped or has a different name,
        # continue gracefully.
        pass

    # Ensure there is a plain index on dataset_id for query performance.
    # (Alembic will skip creation silently if it already exists.)
    try:
        op.create_index(
            'ix_column_analysis_dataset_id',
            'column_analysis',
            ['dataset_id'],
            unique=False,
        )
    except Exception:
        pass


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('column_analysis', 'statistical_validation')
    op.drop_column('column_analysis', 'target_column')
    op.drop_column('column_analysis', 'merge_report')
    op.drop_column('column_analysis', 'source_dataset_ids')

    # Restore the unique constraint
    op.create_unique_constraint(
        'column_analysis_dataset_id_key',
        'column_analysis',
        ['dataset_id'],
    )

    try:
        op.drop_index('ix_column_analysis_dataset_id', table_name='column_analysis')
    except Exception:
        pass
