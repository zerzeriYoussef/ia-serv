



from __future__ import annotations



from datetime import datetime

from typing import Any, Dict, List, Optional



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













class DashboardSchema(BaseModel):

    """A dashboard record as stored in the database."""



    id: int

    dataset_id: int

    analysis_id: Optional[int] = None

    executive_summary_kpis: List[ExecutiveSummaryKpiItemSchema] = Field(default_factory=list)

    dashboard_charts: List[DashboardChartItemSchema] = Field(default_factory=list)

    rag_meta: Optional[RagMetaSchema] = None

    created_at: datetime



    model_config = {"from_attributes": True}













class KpiExecutionResultSchema(BaseModel):

    """Result for a single KPI execution."""

    kpi_name: str

    target_column: Optional[str] = None

    value: Optional[float] = None

    formatted_value: Optional[str] = None

    function_used: Optional[str] = None

    error: Optional[str] = None

    execution_success: bool





class ChartMetadataSchema(BaseModel):

    """Lightweight chart metadata — no data payload. Returned in execute response."""

    chart_index: int

    chart_title: str

    chart_type: Optional[str] = None

    x_axis: Optional[str] = None

    y_axis: Optional[str] = None

    aggregation: Optional[str] = None

    point_count: int = 0

    insight: str = ""

    data_endpoint: str = Field(

        ...,

        description="GET this URL to fetch the actual chart data arrays",

    )

    execution_success: bool

    error: Optional[str] = None





class ChartDataResponseSchema(BaseModel):

    """Full chart data returned by the lazy-loading data endpoint.

    Data uses compact columnar arrays:
      - bar/line/grouped: {axis_x: [...], axis_y: [...]}
      - scatter:          {axis_x: [...], axis_y: [...]}
      - crosstab:         {labels_x: [...], labels_y: [...], matrix: [[...]]}
    """

    chart_index: int

    chart_title: str

    x_axis: Optional[str] = None

    y_axis: Optional[str] = None

    aggregation: Optional[str] = None

    point_count: int = 0

    data: Dict[str, Any] = Field(default_factory=dict)

    execution_success: bool

    error: Optional[str] = None





class DashboardExecuteResponseSchema(BaseModel):

    """Response from POST /dashboards/{id}/execute.

    Charts contain metadata only — call each chart's data_endpoint to fetch arrays.
    """

    dashboard_id: int

    dataset_id: int

    kpi_results: List[KpiExecutionResultSchema] = Field(default_factory=list)

    chart_results: List[ChartMetadataSchema] = Field(default_factory=list)

    executed_at: datetime

