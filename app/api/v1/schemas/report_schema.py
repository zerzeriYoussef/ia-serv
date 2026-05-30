"""
Report generation schemas.

Covers every SSE event emitted during streaming and the final
assembled ReportDocument that is sent in the `report_done` event.
"""



from __future__ import annotations



from datetime import datetime

from enum import Enum

from typing import Any, Dict, List, Optional

from uuid import uuid4



from pydantic import BaseModel, Field













class ReportSseEventType(str, Enum):

    report_start    = "report_start"

    report_step     = "report_step"

    report_token    = "report_token"

    report_section  = "report_section"

    report_done     = "report_done"

    report_error    = "report_error"





class ReportStep(str, Enum):

    loading_context  = "loading_context"

    rag_retrieval    = "rag_retrieval"

    web_search       = "web_search"

    merging_context  = "merging_context"

    generating       = "generating"

    finalizing       = "finalizing"













class WhatHappened(BaseModel):

    narrative: str = ""





class ReportSource(BaseModel):

    type: str

    title: str

    url: Optional[str] = None





class WhyItHappened(BaseModel):

    narrative: str = ""

    sources: List[ReportSource] = Field(default_factory=list)





class ActionableRecommendation(BaseModel):

    priority: int = 1

    action: str = ""

    expected_outcome: str = ""













class ReportDocument(BaseModel):

    report_id: str = Field(default_factory=lambda: str(uuid4()))

    generated_at: datetime = Field(default_factory=datetime.utcnow)

    dataset_name: str = ""

    filters_applied: Dict[str, Any] = Field(default_factory=dict)



    what_happened: WhatHappened = Field(default_factory=WhatHappened)

    why_it_happened: WhyItHappened = Field(default_factory=WhyItHappened)

    what_to_do: List[ActionableRecommendation] = Field(default_factory=list)

    what_to_avoid: List[str] = Field(default_factory=list)













class ReportStartEvent(BaseModel):

    report_id: str

    dataset_name: str





class ReportStepEvent(BaseModel):

    step: ReportStep

    message: str





class ReportTokenEvent(BaseModel):

    section: str

    delta: str





class ReportSectionEvent(BaseModel):

    """Emitted when a full structured section is ready (non-streamed sections)."""

    section: str

    data: Any





class ReportDoneEvent(BaseModel):

    report: ReportDocument

    latency_ms: int





class ReportErrorEvent(BaseModel):

    error: str













class GenerateReportRequest(BaseModel):

    filters: Dict[str, Any] = Field(default_factory=dict)

    include_web_context: bool = True

    language: str = "en"

    conversation_id: Optional[int] = None



