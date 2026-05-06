from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from app.models.cleaning_profile import CleaningProfile, CleaningLog

logger = logging.getLogger(__name__)


class CleaningProfileRepository:
    
    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> CleaningProfile:
        """Create new cleaning profile"""
        profile = CleaningProfile(**kwargs)
        db.add(profile)
        await db.flush()
        await db.refresh(profile)
        return profile
    
    @staticmethod
    async def get_by_id(db: AsyncSession, profile_id: int) -> Optional[CleaningProfile]:
        """Get profile by ID"""
        result = await db.execute(
            select(CleaningProfile).where(CleaningProfile.id == profile_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_default(db: AsyncSession) -> Optional[CleaningProfile]:
        """Get default cleaning profile"""
        result = await db.execute(
            select(CleaningProfile).where(CleaningProfile.is_default == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(db: AsyncSession) -> List[CleaningProfile]:
        """Get all cleaning profiles"""
        result = await db.execute(select(CleaningProfile))
        return result.scalars().all()
    
    @staticmethod
    async def update(
        db: AsyncSession,
        profile_id: int,
        **kwargs
    ) -> Optional[CleaningProfile]:
        """Update cleaning profile"""
        await db.execute(
            update(CleaningProfile)
            .where(CleaningProfile.id == profile_id)
            .values(**kwargs)
        )
        await db.commit()
        return await CleaningProfileRepository.get_by_id(db, profile_id)
    
    @staticmethod
    async def delete(db: AsyncSession, profile_id: int) -> bool:
        """Delete cleaning profile"""
        result = await db.execute(
            delete(CleaningProfile).where(CleaningProfile.id == profile_id)
        )
        await db.commit()
        return result.rowcount > 0


class CleaningLogRepository:
    
    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> CleaningLog:
        """Create cleaning log"""
        log = CleaningLog(**kwargs)
        db.add(log)
        await db.flush()
        await db.refresh(log)
        return log
    
    @staticmethod
    async def get_by_dataset(
        db: AsyncSession,
        dataset_id: int
    ) -> List[CleaningLog]:
        """Get all cleaning logs for a dataset"""
        result = await db.execute(
            select(CleaningLog)
            .where(CleaningLog.dataset_id == dataset_id)
            .order_by(CleaningLog.created_at.desc())
        )
        return result.scalars().all()

    @staticmethod
    async def get_by_id(db: AsyncSession, log_id: int) -> Optional[CleaningLog]:
        """Get cleaning log by ID"""
        result = await db.execute(
            select(CleaningLog).where(CleaningLog.id == log_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_by_dataset(db: AsyncSession, dataset_id: int) -> Optional[CleaningLog]:
        """Get latest cleaning log for dataset"""
        result = await db.execute(
            select(CleaningLog)
            .where(CleaningLog.dataset_id == dataset_id)
            .order_by(CleaningLog.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()