"""add outlier review profile fields

Revision ID: 91a7b2c4d5e6
Revises: 7d8a3e9c1f02
Create Date: 2026-05-06 16:10:00.000000
"""

from typing import Sequence, Union



from alembic import op

import sqlalchemy as sa





revision: str = "91a7b2c4d5e6"

down_revision: Union[str, Sequence[str], None] = "7d8a3e9c1f02"

branch_labels: Union[str, Sequence[str], None] = None

depends_on: Union[str, Sequence[str], None] = None





def upgrade() -> None:

    op.add_column(

        "cleaning_profiles",

        sa.Column("add_flag_columns", sa.Boolean(), nullable=True, server_default=sa.false()),

    )

    op.add_column(

        "cleaning_profiles",

        sa.Column("save_outliers_metadata", sa.Boolean(), nullable=True, server_default=sa.true()),

    )

    op.add_column(

        "cleaning_profiles",

        sa.Column("target_columns", sa.JSON(), nullable=True),

    )





def downgrade() -> None:

    op.drop_column("cleaning_profiles", "target_columns")

    op.drop_column("cleaning_profiles", "save_outliers_metadata")

    op.drop_column("cleaning_profiles", "add_flag_columns")

