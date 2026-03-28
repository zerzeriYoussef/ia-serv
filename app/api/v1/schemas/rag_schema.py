"""
Response schemas for RAG + Gemini — executive dashboard shape (not raw KPI dumps).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RagMetaSchema(BaseModel):
    """Optional debug info from retrieval."""

    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieval_query_preview: str = ""


class ExecutiveSummaryKpiItemSchema(BaseModel):
    kpi_name: str
    target_column: str
    pandas_function: str = Field(
        ...,
        description="e.g. .sum(), .mean(), or groupby pattern name",
    )
    logic_hint: str = Field(..., description="Copy-pastable one-liner, e.g. df['col'].mean()")
    description: str = Field(..., description="Why this matters for the business (non-technical)")
    icon: str = Field(default="analytics", description="Material-style icon name for UI")


class DashboardChartItemSchema(BaseModel):
    rank: int = Field(..., ge=1)
    title: str
    relationship: str = Field(..., description="e.g. Column A ↔ Column B")
    strength_score: float = Field(..., ge=0, le=1)
    x_axis_column: Optional[str] = None
    y_axis_column: Optional[str] = None
    pandas_grouping: str
    chart_type: str = Field(
        ...,
        description="e.g. Grouped Bar, Line, Scatter, Heatmap, Box Plot",
    )
    business_insight: str


class RagInsightsResponseSchema(BaseModel):
    """Gemini + RAG output formatted for dashboards (no raw analysis/KPI blob)."""

    dataset_id: int
    executive_summary_kpis: List[ExecutiveSummaryKpiItemSchema] = Field(default_factory=list)
    dashboard_charts: List[DashboardChartItemSchema] = Field(default_factory=list)
    rag_meta: Optional[RagMetaSchema] = Field(
        default=None,
        description="Present only when ?debug=true",
    )
