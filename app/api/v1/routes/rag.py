"""
RAG-backed insights (Gemini) — executive KPIs + dashboard chart specs.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.api.v1.schemas.rag_schema import (
    DashboardChartItemSchema,
    ExecutiveSummaryKpiItemSchema,
    RagInsightsResponseSchema,
    RagMetaSchema,
)
from app.services.rag.rag_insights_service import run_rag_insights

router = APIRouter()


@router.post(
    "/datasets/{dataset_id}/rag-insights",
    response_model=RagInsightsResponseSchema,
    summary="RAG + Gemini: executive KPIs and dashboard charts",
)
async def dataset_rag_insights(
    dataset_id: int,
    debug: bool = Query(
        default=False,
        description="If true, include rag_meta (retrieved chunk ids, query preview).",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Uses stored analysis + a single file load for KPI hints inside the prompt.
    Response is shaped for UI: ``executive_summary_kpis`` and ``dashboard_charts`` only
    (no raw KPI block or full relationship dump).
    """
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured",
        )

    try:
        llm_raw, meta = await run_rag_insights(db, dataset_id)
    except ValueError as e:
        code = str(e)
        if code == "no_analysis":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis for dataset {dataset_id}. Run POST /datasets/{dataset_id}/analyze first.",
            )
        if code == "no_dataset":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG/Gemini failed: {e!s}",
        ) from e

    try:
        kpis = [
            ExecutiveSummaryKpiItemSchema.model_validate(x)
            for x in (llm_raw.get("executive_summary_kpis") or [])
        ]
        charts = [
            DashboardChartItemSchema.model_validate(x)
            for x in (llm_raw.get("dashboard_charts") or [])
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM returned invalid shape: {e!s}",
        ) from e

    rag_meta = RagMetaSchema.model_validate(meta["rag"]) if debug else None

    return RagInsightsResponseSchema(
        dataset_id=dataset_id,
        executive_summary_kpis=kpis,
        dashboard_charts=charts,
        rag_meta=rag_meta,
    )
