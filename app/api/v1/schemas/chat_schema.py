"""
Locked Pydantic schemas for the chat Q&A system.

Covers:
- Tool arg / result contracts (Analysis Agent boundary)
- Intent / plan (Orchestrator output)
- Chart spec (Visualization Agent output)
- SSE event shapes (streamed to client)
- API surface (request / response models)
"""



from __future__ import annotations



from datetime import datetime

from enum import Enum

from typing import Any, Dict, List, Literal, Optional, Union



from pydantic import BaseModel, Field















class AllowedOp(str, Enum):

    groupby_agg  = "groupby_agg"

    value_counts = "value_counts"

    describe     = "describe"

    correlation  = "correlation"

    filter_agg   = "filter_agg"



ALLOWED_AGGS = frozenset(

    {"sum", "mean", "median", "count", "nunique", "min", "max", "std"}

)





class ToolArgs(BaseModel):

    """Free-form Pandas expression the LLM produces.

    `code_expr` is a single Python expression evaluated against the DataFrame
    bound as `df`. The result must be a scalar, Series, or DataFrame — the
    AnalysisAgent normalises it into a ToolResult automatically.

    Example:
        df[df['continent']=='Europe'].groupby('drink_preference')['monthly_spend'].mean()
    """

    code_expr: str

    label: Optional[str] = None





class ToolResult(BaseModel):

    """Structured output of one Analysis Agent tool call — never prose."""

    result_type: Literal["scalar", "series", "frame_preview"]

    payload: Any

    warnings: List[str] = Field(default_factory=list)















class ChartType(str, Enum):

    bar       = "bar"

    line      = "line"

    scatter   = "scatter"

    heatmap   = "heatmap"

    pie       = "pie"

    histogram = "histogram"





class ChartSpec(BaseModel):

    """Visualization Agent output — maps directly to a frontend render call.

    Data is sourced exclusively from ToolResult.payload — never invented.
    Validated by VizAgent before emitting.
    """

    chart_type: ChartType

    x_col: str

    y_col: Optional[str] = None

    agg_func: Optional[str] = None

    title: str

    caption: str

    data: Dict[str, Any] = Field(default_factory=dict)













class Intent(str, Enum):

    retrieve_only   = "retrieve_only"

    analyze         = "analyze"

    visualize       = "visualize"

    clarify         = "clarify"

    generate_report = "generate_report"

    refuse_unsafe   = "refuse_unsafe"





class OrchestratorPlan(BaseModel):

    """Structured plan produced by the Orchestrator LLM call (JSON mode).

    The Orchestrator must emit this exact shape — validated before execution.
    """

    intent: Intent

    tool_calls: List[ToolArgs] = Field(default_factory=list)

    needs_chart: bool = False

    clarification_question: Optional[str] = None

    refuse_reason: Optional[str] = None





class FinalAnswer(BaseModel):

    """Assembled after tool execution — streamed as narrative + metadata."""

    answer: str

    cited_chunk_ids: List[str] = Field(default_factory=list)

    chart_spec: Optional[ChartSpec] = None

    caveats: List[str] = Field(default_factory=list)















class SseEventType(str, Enum):

    intent     = "intent"

    retrieval  = "retrieval"

    tool_start = "tool_start"

    tool_result = "tool_result"

    chart      = "chart"

    token      = "token"

    done       = "done"

    error      = "error"





class IntentEvent(BaseModel):

    intent: str

    plan_summary: str





class RetrievalEvent(BaseModel):

    chunk_ids: List[str]





class ToolStartEvent(BaseModel):

    tool: str

    args_summary: str





class ToolResultEvent(BaseModel):

    preview: str





class ChartEvent(BaseModel):

    chart_spec: Dict[str, Any]





class TokenEvent(BaseModel):

    delta: str





class DoneEvent(BaseModel):

    conversation_id: int

    message_id: int

    latency_ms: int

    chart_spec: Optional[Dict[str, Any]] = None





class ErrorEvent(BaseModel):

    error: str













class CreateConversationRequest(BaseModel):

    title: Optional[str] = Field(default=None, max_length=200)





class CreateConversationResponse(BaseModel):

    conversation_id: int

    dataset_id: int

    title: str

    created_at: datetime



    model_config = {"from_attributes": True}





class ConversationSchema(BaseModel):

    id: int

    dataset_id: int

    user_id: Optional[str] = None

    title: Optional[str] = None

    analysis_version: Optional[str] = None

    created_at: datetime

    updated_at: Optional[datetime] = None



    model_config = {"from_attributes": True}





class MessageSchema(BaseModel):

    id: int

    conversation_id: int

    role: str

    content: str

    intent: Optional[str] = None

    chunk_ids: Optional[List[str]] = None

    chart_spec: Optional[Dict[str, Any]] = None

    latency_ms: Optional[int] = None

    created_at: datetime



    model_config = {"from_attributes": True}





class AskRequest(BaseModel):

    message: str = Field(..., min_length=1, max_length=4096)





class AskResponse(BaseModel):

    message_id: int

    conversation_id: int

    answer: str

    intent: Optional[str] = None

    chunk_ids: List[str] = Field(default_factory=list)

    chart_spec: Optional[Dict[str, Any]] = None

    caveats: List[str] = Field(default_factory=list)

    latency_ms: int

