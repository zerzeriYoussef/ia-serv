import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.dashboard import Dashboard

async def run():
    async with AsyncSessionLocal() as db:
        r = await db.execute(select(Dashboard.id, Dashboard.user_id, Dashboard.dataset_id, Dashboard.executive_summary_kpis))
        print("Dashboards:", r.all())

if __name__ == "__main__":
    asyncio.run(run())
