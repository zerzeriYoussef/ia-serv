"""
Report API routes.

Routes:
  GET  /datasets/{dataset_id}/reports/stream  → SSE stream of report generation
  POST /datasets/{dataset_id}/reports         → non-streaming (testing / debug)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import CurrentUser, require_auth
from app.api.v1.schemas.report_schema import (
    GenerateReportRequest,
    ReportDocument,
    ReportDoneEvent,
    ReportSseEventType,
)
from app.core.config import settings
from app.core.database import get_db
from app.repositories.dataset_repository import DatasetRepository
from app.services.report.report_service import generate_report_stream

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured.",
        )


async def _get_dataset_or_404(db: AsyncSession, dataset_id: int):
    ds = await DatasetRepository.get_by_id(db, dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )
    return ds


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/reports/stream  — SSE streaming
# ---------------------------------------------------------------------------

@router.get(
    "/datasets/{dataset_id}/reports/stream",
    summary="Generate an AI report via SSE streaming",
    tags=["Reports"],
    response_class=StreamingResponse,
)
async def stream_report(
    dataset_id: int,
    include_web_context: bool = Query(
        default=True,
        description="Fetch external industry context via Serper (requires SERPER_API_KEY)",
    ),
    filters: Optional[str] = Query(
        default=None,
        description="Optional JSON string of dashboard filters, e.g. {\"region\":\"EMEA\"}",
    ),
    conversation_id: Optional[int] = Query(
        default=None,
        description="Optional ID of conversation history to use for context",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    SSE streaming endpoint for AI report generation.

    Connect with `Accept: text/event-stream`.

    Events emitted in order:
    - `event: report_start`   — report ID + dataset name
    - `event: report_step`    — pipeline stage progress
    - `event: report_token`   — narrative text delta (section name in data)
    - `event: report_section` — completed structured section (JSON)
    - `event: report_done`    — full ReportDocument JSON
    - `event: report_error`   — on failure
    """
    _require_gemini()
    await _get_dataset_or_404(db, dataset_id)

    parsed_filters: dict[str, Any] = {}
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="filters must be valid JSON.",
            )

    async def event_stream():
        try:
            async for frame in generate_report_stream(
                db=db,
                dataset_id=dataset_id,
                filters=parsed_filters,
                include_web_context=include_web_context,
                conversation_id=conversation_id,
            ):
                yield frame
        except Exception as exc:
            logger.error("stream_report: unhandled error: %s", exc)
            error_frame = (
                f"event: {ReportSseEventType.report_error.value}\n"
                f"data: {{\"error\": \"{exc!s}\"}}\n\n"
            )
            yield error_frame

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/reports  — non-streaming (collect full report)
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/reports",
    response_model=ReportDocument,
    summary="Generate an AI report (non-streaming, for testing)",
    tags=["Reports"],
)
async def generate_report(
    dataset_id: int,
    body: GenerateReportRequest = GenerateReportRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Non-streaming version — drains the SSE generator internally and returns
    a single complete ReportDocument JSON. Useful for API clients and testing.
    """
    _require_gemini()
    await _get_dataset_or_404(db, dataset_id)

    final_report: Optional[ReportDocument] = None

    async for frame in generate_report_stream(
        db=db,
        dataset_id=dataset_id,
        filters=body.filters,
        include_web_context=body.include_web_context,
        conversation_id=body.conversation_id,
    ):
        # Parse the SSE frame to extract the report_done payload
        if frame.startswith(f"event: {ReportSseEventType.report_done.value}"):
            data_line = [line for line in frame.splitlines() if line.startswith("data:")]
            if data_line:
                try:
                    payload = json.loads(data_line[0][len("data:"):].strip())
                    done_event = ReportDoneEvent(**payload)
                    final_report = done_event.report
                except Exception as exc:
                    logger.error("generate_report: failed to parse report_done: %s", exc)

    if not final_report:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed — no report_done event received.",
        )

    return final_report
