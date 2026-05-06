from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging

from app.core.database import get_db
from app.api.dependencies.auth import CurrentUser, require_auth
from app.models.dataset import Dataset
from app.models.dashboard import Dashboard

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/users/me/stats")
async def get_user_stats(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    user_id = current_user.user_id
    
    # Number of analyses (datasets uploaded)
    ds_query = select(func.count(Dataset.id)).where(Dataset.user_id == user_id)
    analyses_count = (await db.execute(ds_query)).scalar() or 0
    
    # Number of dashboards
    dash_query = (
        select(func.count(Dashboard.id))
        .join(Dataset, Dashboard.dataset_id == Dataset.id)
        .where(Dataset.user_id == user_id)
    )
    dashboards_count = (await db.execute(dash_query)).scalar() or 0
    
    # Number of KPIs tracked (sum of KPIs across all dashboards)
    kpis_query = (
        select(Dashboard.executive_summary_kpis)
        .join(Dataset, Dashboard.dataset_id == Dataset.id)
        .where(Dataset.user_id == user_id)
    )
    dashboards = (await db.execute(kpis_query)).scalars().all()
    
    kpis_count = 0
    for kpis in dashboards:
        if isinstance(kpis, list):
            kpis_count += len(kpis)
            
    return {
        "analyses": analyses_count,
        "dashboards": dashboards_count,
        "kpis": kpis_count
    }
