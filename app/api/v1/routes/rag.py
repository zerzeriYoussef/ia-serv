"""
RAG-backed insights (Gemini) — executive KPIs + dashboard chart specs.

Routes
------
POST /datasets/{id}/rag-insights              → generate (no save)
POST /datasets/{id}/rag-insights?save=true    → generate + persist Dashboard row
GET  /datasets/{id}/dashboards                → list saved dashboards for a dataset
POST /dashboards/{id}/execute                 → execute pandas KPI/chart logic on real CSV
"""

from datetime import datetime, timezone
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.api.v1.schemas.rag_schema import (
    ChartDataResponseSchema,
    ChartMetadataSchema,
    DashboardChartItemSchema,
    DashboardExecuteResponseSchema,
    DashboardSchema,
    ExecutiveSummaryKpiItemSchema,
    RagInsightsResponseSchema,
    RagMetaSchema,
)
from app.repositories.dashboard_repository import DashboardRepository
from app.repositories.dataset_repository import DatasetRepository
from app.services.analysis.kpi_executor import KPIExecutor
from app.services.data.parser_service import ParserService
from app.services.rag.rag_insights_service import (
    run_rag_insights,
    run_rag_insights_and_save,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY n'est pas configuré",
        )


def _handle_rag_value_error(e: ValueError, dataset_id: int) -> None:
    code = str(e)
    if code == "no_analysis":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucune analyse pour le jeu de données {dataset_id}. Exécutez d'abord POST /datasets/{dataset_id}/analyze.",
        )
    if code == "no_dataset":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jeu de données {dataset_id} introuvable.",
        )
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


def _raise_rag_upstream_error(e: Exception) -> None:
    """Expose Gemini/API failures with the correct HTTP class for the UI."""
    if isinstance(e, httpx.HTTPStatusError):
        upstream_status = e.response.status_code
        if upstream_status == status.HTTP_429_TOO_MANY_REQUESTS:
            detail = "Quota Gemini dépassé ou limité en débit. Veuillez réessayer plus tard ou utiliser une autre clé API."
            try:
                body = e.response.json()
                detail = body.get("error", {}).get("message") or detail
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
            ) from e

        if upstream_status == status.HTTP_503_SERVICE_UNAVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini est temporairement indisponible ou surchargé. Veuillez réessayer sous peu.",
            ) from e

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Échec RAG/Gemini : {e!s}",
    ) from e


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/rag-insights
# Supports ?save=true to persist the result as a Dashboard row
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/rag-insights",
    summary="RAG + Gemini: executive KPIs and dashboard charts",
    responses={
        200: {"description": "Generated insights (not saved)"},
        201: {"description": "Generated insights saved as a new Dashboard"},
    },
)
async def dataset_rag_insights(
    dataset_id: int,
    save: bool = Query(
        default=False,
        description="If true, persist the result as a Dashboard record and return it with its DB id.",
    ),
    debug: bool = Query(
        default=False,
        description="If true, include rag_meta (retrieved chunk ids, query preview).",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Calls Gemini via RAG to generate executive KPIs and chart specifications.

    - **save=false** (default): returns `RagInsightsResponseSchema` — nothing stored.
    - **save=true**: persists the result and returns `DashboardSchema` (includes DB `id`).
    """
    _require_gemini()

    # ── save=true branch ─────────────────────────────────────────────────
    if save:
        try:
            dashboard = await run_rag_insights_and_save(
                db, dataset_id, user_id=None, debug=debug
            )
        except ValueError as e:
            _handle_rag_value_error(e, dataset_id)
        except Exception as e:
            _raise_rag_upstream_error(e)

        try:
            return DashboardSchema(
                id=dashboard.id,
                dataset_id=dashboard.dataset_id,
                analysis_id=dashboard.analysis_id,
                executive_summary_kpis=[
                    ExecutiveSummaryKpiItemSchema.model_validate(x)
                    for x in (dashboard.executive_summary_kpis or [])
                ],
                dashboard_charts=[
                    DashboardChartItemSchema.model_validate(x)
                    for x in (dashboard.dashboard_charts or [])
                ],
                rag_meta=(
                    RagMetaSchema.model_validate(dashboard.rag_meta)
                    if dashboard.rag_meta
                    else None
                ),
                created_at=dashboard.created_at,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Le LLM a retourné un format invalide : {e!s}",
            ) from e

    # ── save=false branch (original behaviour) ───────────────────────────
    try:
        llm_raw, meta = await run_rag_insights(db, dataset_id)
    except ValueError as e:
        _handle_rag_value_error(e, dataset_id)
    except Exception as e:
        _raise_rag_upstream_error(e)

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
            detail=f"Le LLM a retourné un format invalide : {e!s}",
        ) from e

    rag_meta = RagMetaSchema.model_validate(meta["rag"]) if debug else None

    return RagInsightsResponseSchema(
        dataset_id=dataset_id,
        executive_summary_kpis=kpis,
        dashboard_charts=charts,
        rag_meta=rag_meta,
    )


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/dashboards
# ---------------------------------------------------------------------------

@router.get(
    "/datasets/{dataset_id}/dashboards",
    response_model=List[DashboardSchema],
    summary="List saved dashboards for a dataset",
)
async def list_dashboards(
    dataset_id: int,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all saved dashboard versions for a dataset, newest first.
    Each entry was created by `POST /datasets/{id}/rag-insights?save=true`.
    """
    # Verify dataset exists
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jeu de données {dataset_id} introuvable.",
        )

    dashboards = await DashboardRepository.get_by_dataset(db, dataset_id, skip=skip, limit=limit)

    return [
        DashboardSchema(
            id=d.id,
            dataset_id=d.dataset_id,
            analysis_id=d.analysis_id,
            executive_summary_kpis=[
                ExecutiveSummaryKpiItemSchema.model_validate(x)
                for x in (d.executive_summary_kpis or [])
            ],
            dashboard_charts=[
                DashboardChartItemSchema.model_validate(x)
                for x in (d.dashboard_charts or [])
            ],
            rag_meta=(
                RagMetaSchema.model_validate(d.rag_meta) if d.rag_meta else None
            ),
            created_at=d.created_at,
        )
        for d in dashboards
    ]


# ---------------------------------------------------------------------------
# POST /dashboards/{dashboard_id}/execute
# ---------------------------------------------------------------------------

@router.post(
    "/dashboards/{dashboard_id}/execute",
    response_model=DashboardExecuteResponseSchema,
    summary="Execute KPI calculations for a saved dashboard",
)
async def execute_dashboard(
    dashboard_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Loads the saved dashboard's KPI + chart specs and executes them against
    the actual dataset CSV using **KPIExecutor** (no eval/exec — safe).

    Returns real computed values (not LLM estimates).
    """
    # Load dashboard
    dashboard = await DashboardRepository.get_by_id(db, dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tableau de bord {dashboard_id} introuvable.",
        )

    # Load dataset
    dataset = await DatasetRepository.get_by_id(db, dashboard.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jeu de données {dashboard.dataset_id} introuvable.",
        )

    # Parse CSV/XLSX into DataFrame
    try:
        df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Impossible d'analyser le fichier du jeu de données : {e!s}",
        ) from e

    # Execute KPIs (still returned in full)
    executor = KPIExecutor(df)
    kpi_results = executor.execute_all_kpis(dashboard.executive_summary_kpis or [])

    # Execute charts → metadata only (no data payload)
    chart_meta = executor.execute_all_charts_metadata(dashboard.dashboard_charts or [])

    # Attach data_endpoint URL to each chart
    for cm in chart_meta:
        cm["data_endpoint"] = f"/api/v1/dashboards/{dashboard_id}/charts/{cm['chart_index']}/data"

    return DashboardExecuteResponseSchema(
        dashboard_id=dashboard_id,
        dataset_id=dashboard.dataset_id,
        kpi_results=kpi_results,
        chart_results=[
            ChartMetadataSchema.model_validate(cm) for cm in chart_meta
        ],
        executed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /dashboards/{dashboard_id}/charts/{chart_index}/data
# ---------------------------------------------------------------------------

@router.get(
    "/dashboards/{dashboard_id}/charts/{chart_index}/data",
    response_model=ChartDataResponseSchema,
    summary="Fetch chart data arrays for a single chart",
)
async def get_chart_data(
    dashboard_id: int,
    chart_index: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Lazy-load the actual data for one chart of an executed dashboard.

    Returns compact columnar arrays instead of verbose per-point objects:
    - **bar/line/grouped**: `{"axis_x": [...], "axis_y": [...]}`
    - **scatter**: `{"axis_x": [...], "axis_y": [...]}`
    - **crosstab/heatmap**: `{"labels_x": [...], "labels_y": [...], "matrix": [[...]]}`
    """
    # Load dashboard
    dashboard = await DashboardRepository.get_by_id(db, dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tableau de bord {dashboard_id} introuvable.",
        )

    charts = dashboard.dashboard_charts or []
    if chart_index < 0 or chart_index >= len(charts):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"chart_index {chart_index} introuvable. Ce tableau de bord contient {len(charts)} graphiques (0..{len(charts) - 1}).",
        )

    # Load dataset + parse
    dataset = await DatasetRepository.get_by_id(db, dashboard.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jeu de données {dashboard.dataset_id} introuvable.",
        )

    try:
        df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Impossible d'analyser le fichier du jeu de données : {e!s}",
        ) from e

    executor = KPIExecutor(df)

    try:
        result = executor.execute_single_chart(charts, chart_index)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Échec de l'exécution du graphique : {e!s}",
        ) from e

    return ChartDataResponseSchema(
        chart_index=chart_index,
        chart_title=result.get("chart_title", ""),
        x_axis=result.get("x_axis"),
        y_axis=result.get("y_axis"),
        aggregation=result.get("aggregation"),
        point_count=result.get("point_count", 0),
        data=result.get("data", {}),
        execution_success=result.get("execution_success", False),
        error=result.get("error"),
    )
