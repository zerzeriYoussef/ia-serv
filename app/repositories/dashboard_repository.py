"""
Repository for Dashboard database operations.

Public routes use: create, get_by_id, get_by_dataset.
System/Agent use : update, delete (no public API routes exposed).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dashboard import Dashboard

logger = logging.getLogger(__name__)


class DashboardRepository:

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> Dashboard:
        """Persist a new dashboard record and return it (flushed, refreshed)."""
        dashboard = Dashboard(**kwargs)
        db.add(dashboard)
        await db.flush()
        await db.refresh(dashboard)
        logger.info("Created Dashboard %s for dataset %s", dashboard.id, dashboard.dataset_id)
        return dashboard

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    async def get_by_id(db: AsyncSession, dashboard_id: int) -> Optional[Dashboard]:
        """Fetch a single dashboard by primary key."""
        result = await db.execute(
            select(Dashboard).where(Dashboard.id == dashboard_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_dataset(
        db: AsyncSession,
        dataset_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Dashboard]:
        """Return all dashboards for a dataset, newest first."""
        result = await db.execute(
            select(Dashboard)
            .where(Dashboard.dataset_id == dataset_id)
            .order_by(Dashboard.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # System / Agent only — no public API routes
    # ------------------------------------------------------------------

    @staticmethod
    async def update(db: AsyncSession, dashboard_id: int, **kwargs) -> Optional[Dashboard]:
        """Update fields on an existing dashboard. Returns None if not found."""
        dashboard = await db.get(Dashboard, dashboard_id)
        if not dashboard:
            return None
        for key, value in kwargs.items():
            setattr(dashboard, key, value)
        await db.flush()
        await db.refresh(dashboard)
        logger.info("Updated Dashboard %s", dashboard_id)
        return dashboard

    @staticmethod
    async def delete(db: AsyncSession, dashboard_id: int) -> bool:
        """Hard-delete a dashboard. Returns True if a row was removed."""
        result = await db.execute(
            delete(Dashboard).where(Dashboard.id == dashboard_id)
        )
        await db.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("Deleted Dashboard %s", dashboard_id)
        return deleted
