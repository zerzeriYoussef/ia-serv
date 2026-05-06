from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from app.models.dataset import Dataset, DatasetStatus

logger = logging.getLogger(__name__)


class DatasetRepository:
    
    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> Dataset:
        """Create new dataset"""
        dataset = Dataset(**kwargs)
        db.add(dataset)
        await db.flush()
        await db.refresh(dataset)
        logger.info(f"Created dataset: {dataset.id}")
        return dataset
    
    @staticmethod
    async def get_by_id(db: AsyncSession, dataset_id: int) -> Optional[Dataset]:
        """Get dataset by ID"""
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dataset]:
        """Get all datasets with pagination"""
        result = await db.execute(
            select(Dataset)
            .order_by(Dataset.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def count(db: AsyncSession) -> int:
        """Count total datasets"""
        result = await db.execute(select(func.count(Dataset.id)))
        return result.scalar()
    
    @staticmethod
    async def update_status(
        db: AsyncSession,
        dataset_id: int,
        status: DatasetStatus
    ) -> Optional[Dataset]:
        """Update dataset status"""
        await db.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(status=status)
        )
        await db.commit()
        return await DatasetRepository.get_by_id(db, dataset_id)
    
    @staticmethod
    async def update_metadata(
        db: AsyncSession,
        dataset_id: int,
        metadata: dict
    ) -> Optional[Dataset]:
        """Update dataset metadata"""
        await db.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(**metadata)
        )
        await db.commit()
        return await DatasetRepository.get_by_id(db, dataset_id)
    
    @staticmethod
    async def get_by_user_id(
        db: AsyncSession,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dataset]:
        """Get datasets belonging to a specific user, newest first."""
        result = await db.execute(
            select(Dataset)
            .where(Dataset.user_id == user_id)
            .order_by(Dataset.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def count_by_user_id(db: AsyncSession, user_id: str) -> int:
        """Count datasets belonging to a specific user."""
        result = await db.execute(
            select(func.count(Dataset.id)).where(Dataset.user_id == user_id)
        )
        return result.scalar()

    @staticmethod
    async def delete(db: AsyncSession, dataset_id: int) -> bool:
        """Delete dataset"""
        result = await db.execute(
            delete(Dataset).where(Dataset.id == dataset_id)
        )
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def delete_by_user_id(db: AsyncSession, user_id: str) -> int:
        """Bulk-delete every dataset belonging to a user and all related tables.

        Returns the number of dataset rows deleted. Callers are responsible for
        deleting the physical files *before* calling this method.
        """
        # Get all dataset IDs for this user
        result = await db.execute(select(Dataset.id).where(Dataset.user_id == user_id))
        dataset_ids = result.scalars().all()

        if not dataset_ids:
            return 0

        # Import all related models to handle cascades manually
        from app.models.column_analysis import ColumnAnalysis
        from app.models.dashboard import Dashboard
        from app.models.conversation import Conversation, ChatMessage
        from app.models.cleaning_profile import CleaningLog

        # 1. Delete ChatMessages linked to the user's conversations
        conv_result = await db.execute(select(Conversation.id).where(Conversation.dataset_id.in_(dataset_ids)))
        conv_ids = conv_result.scalars().all()
        if conv_ids:
            await db.execute(delete(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids)))

        # 2. Delete the rest of the dependent tables
        await db.execute(delete(Conversation).where(Conversation.dataset_id.in_(dataset_ids)))
        await db.execute(delete(Dashboard).where(Dashboard.dataset_id.in_(dataset_ids)))
        await db.execute(delete(CleaningLog).where(CleaningLog.dataset_id.in_(dataset_ids)))
        await db.execute(delete(ColumnAnalysis).where(ColumnAnalysis.dataset_id.in_(dataset_ids)))

        # 3. Finally delete the datasets themselves
        final_result = await db.execute(delete(Dataset).where(Dataset.id.in_(dataset_ids)))
        await db.execute(delete(Dataset).where(Dataset.user_id == user_id)) # just to be totally safe
        await db.commit()
        
        logger.info("Deleted %d datasets for user_id=%s", final_result.rowcount, user_id)
        return final_result.rowcount