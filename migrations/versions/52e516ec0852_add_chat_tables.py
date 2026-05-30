"""add_chat_tables

Revision ID: 52e516ec0852
Revises: b2c3d4e5f6a7
Create Date: 2026-04-11 22:14:08.014293

"""

from typing import Sequence, Union



from alembic import op

import sqlalchemy as sa





revision: str = '52e516ec0852'

down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'

branch_labels: Union[str, Sequence[str], None] = None

depends_on: Union[str, Sequence[str], None] = None





def upgrade() -> None:

    """Add conversations and chat_messages tables."""

    op.create_table(

        'conversations',

        sa.Column('id', sa.Integer(), nullable=False),

        sa.Column('dataset_id', sa.Integer(), nullable=False),

        sa.Column('user_id', sa.String(), nullable=True),

        sa.Column('title', sa.String(), nullable=True),

        sa.Column('analysis_version', sa.String(), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),

        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),

        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),

        sa.PrimaryKeyConstraint('id'),

    )

    op.create_index(op.f('ix_conversations_dataset_id'), 'conversations', ['dataset_id'], unique=False)

    op.create_index(op.f('ix_conversations_id'), 'conversations', ['id'], unique=False)

    op.create_index(op.f('ix_conversations_user_id'), 'conversations', ['user_id'], unique=False)



    op.create_table(

        'chat_messages',

        sa.Column('id', sa.Integer(), nullable=False),

        sa.Column('conversation_id', sa.Integer(), nullable=False),

        sa.Column('role', sa.Enum('user', 'assistant', name='messagerole'), nullable=False),

        sa.Column('content', sa.Text(), nullable=False),

        sa.Column('intent', sa.String(), nullable=True),

        sa.Column('tool_calls', sa.JSON(), nullable=True),

        sa.Column('chunk_ids', sa.JSON(), nullable=True),

        sa.Column('latency_ms', sa.Integer(), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),

        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),

        sa.PrimaryKeyConstraint('id'),

    )

    op.create_index(op.f('ix_chat_messages_conversation_id'), 'chat_messages', ['conversation_id'], unique=False)

    op.create_index(op.f('ix_chat_messages_id'), 'chat_messages', ['id'], unique=False)





def downgrade() -> None:

    """Remove conversations and chat_messages tables."""

    op.drop_index(op.f('ix_chat_messages_id'), table_name='chat_messages')

    op.drop_index(op.f('ix_chat_messages_conversation_id'), table_name='chat_messages')

    op.drop_table('chat_messages')

    op.drop_index(op.f('ix_conversations_user_id'), table_name='conversations')

    op.drop_index(op.f('ix_conversations_id'), table_name='conversations')

    op.drop_index(op.f('ix_conversations_dataset_id'), table_name='conversations')

    op.drop_table('conversations')

