"""
Repository for column analysis database operations
"""

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
import logging

from app.models.column_analysis import ColumnAnalysis

logger = logging.getLogger(__name__)


class AnalysisRepository:
    """CRUD operations for column analysis"""
    
    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> ColumnAnalysis:
        """Create new analysis record, replacing any existing one for the same dataset."""
        
        dataset_id = kwargs.get("dataset_id")
        if dataset_id is not None:
            await db.execute(
                delete(ColumnAnalysis).where(ColumnAnalysis.dataset_id == dataset_id)
            )
        
        analysis = ColumnAnalysis(**kwargs)
        db.add(analysis)
        await db.flush()
        await db.refresh(analysis)
        
        logger.info(f"Created analysis {analysis.id} for dataset {analysis.dataset_id}")
        return analysis
    
    @staticmethod
    async def get_by_dataset(
        db: AsyncSession,
        dataset_id: int
    ) -> Optional[ColumnAnalysis]:
        """Get latest analysis for a dataset"""
        
        result = await db.execute(
            select(ColumnAnalysis)
            .where(ColumnAnalysis.dataset_id == dataset_id)
            .order_by(ColumnAnalysis.created_at.desc())
        )
        return result.scalars().first()
    
    @staticmethod
    async def get_all(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[ColumnAnalysis]:
        """Get all analyses with pagination"""
        
        result = await db.execute(
            select(ColumnAnalysis)
            .offset(skip)
            .limit(limit)
            .order_by(ColumnAnalysis.created_at.desc())
        )
        return result.scalars().all()
    
    @staticmethod
    async def update(
        db: AsyncSession,
        analysis_id: int,
        **kwargs
    ) -> Optional[ColumnAnalysis]:
        """Update existing analysis"""
        
        analysis = await db.get(ColumnAnalysis, analysis_id)
        
        if analysis:
            for key, value in kwargs.items():
                setattr(analysis, key, value)
            await db.flush()
            await db.refresh(analysis)
            logger.info(f"Updated analysis {analysis_id}")
        
        return analysis
    
    @staticmethod
    async def delete(db: AsyncSession, dataset_id: int) -> bool:
        """Delete analysis for a dataset"""
        
        result = await db.execute(
            delete(ColumnAnalysis).where(ColumnAnalysis.dataset_id == dataset_id)
        )
        await db.commit()
        
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted analysis for dataset {dataset_id}")
        
        return deleted