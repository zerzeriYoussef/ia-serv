"""
ConversationRepository — DB access layer for conversations and chat messages.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation, ChatMessage, MessageRole


class ConversationRepository:

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        dataset_id: int,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        analysis_version: Optional[str] = None,
    ) -> Conversation:
        conv = Conversation(
            dataset_id=dataset_id,
            user_id=user_id,
            title=title or "New conversation",
            analysis_version=analysis_version,
        )
        db.add(conv)
        await db.flush()
        await db.refresh(conv)
        return conv

    @staticmethod
    async def get_by_id(db: AsyncSession, conversation_id: int) -> Optional[Conversation]:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_dataset(
        db: AsyncSession,
        dataset_id: int,
        *,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Conversation]:
        q = select(Conversation).where(Conversation.dataset_id == dataset_id)
        if user_id:
            q = q.where(Conversation.user_id == user_id)
        q = q.order_by(desc(Conversation.created_at)).offset(skip).limit(limit)
        result = await db.execute(q)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    @staticmethod
    async def add_message(
        db: AsyncSession,
        *,
        conversation_id: int,
        role: MessageRole,
        content: str,
        intent: Optional[str] = None,
        tool_calls: Optional[list] = None,
        chunk_ids: Optional[list] = None,
        latency_ms: Optional[int] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            intent=intent,
            tool_calls=tool_calls,
            chunk_ids=chunk_ids,
            latency_ms=latency_ms,
        )
        db.add(msg)
        await db.flush()
        await db.refresh(msg)
        return msg

    @staticmethod
    async def get_messages(
        db: AsyncSession,
        conversation_id: int,
        *,
        limit: int = 20,
    ) -> List[ChatMessage]:
        """Return most-recent messages first (caller reverses for chronological order)."""
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_analysis_version(analysis_id: int, updated_at) -> str:
        """Stable short hash used to detect stale conversations."""
        raw = f"{analysis_id}:{updated_at}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]
