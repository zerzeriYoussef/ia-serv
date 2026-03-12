"""
Pydantic schemas for column analysis API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class RelationshipSchema(BaseModel):
    """Single relationship between columns"""
    
    columns: List[str] = Field(..., description="Column names involved")
    type: str = Field(..., description="Type: predictive, correlation, hierarchy")
    strength: float = Field(..., ge=0, le=1, description="Strength score 0-1")
    direction: Optional[str] = Field(None, description="Direction: x → y or positive/negative")
    method: str = Field(..., description="Detection method: custom_pps, pearson, etc")
    insight: Optional[str] = Field(None, description="Human-readable explanation")
    chart_suggestion: Optional[str] = Field(None, description="Suggested chart type")


class ColumnCategoriesSchema(BaseModel):
    """Column categorization"""
    
    metrics: List[str] = Field(default_factory=list, description="Numeric metrics (KPIs)")
    dimensions: List[str] = Field(default_factory=list, description="Categorical dimensions")
    identifiers: List[str] = Field(default_factory=list, description="ID columns")
    temporal: List[str] = Field(default_factory=list, description="Date/time columns")
    geographic: List[str] = Field(default_factory=list, description="Geographic columns")
    other: List[str] = Field(default_factory=list, description="Other columns")


class AnalysisResultSchema(BaseModel):
    """Complete analysis result"""
    
    dataset_id: int
    relationships: List[RelationshipSchema]
    dashboard_columns: List[str]
    column_categories: ColumnCategoriesSchema
    primary_metric: Optional[str]
    confidence_score: float = Field(..., ge=0, le=1)
    total_relationships: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class SimpleRelationshipResponse(BaseModel):
    """Simplified format for chart generation"""
    
    relationships: List[List[str]] = Field(
        ...,
        description="List of column pairs: [['ads', 'sales'], ['region', 'sales']]",
        example=[["ads", "sales"], ["region", "sales"]]
    )
    dashboard_columns: List[str] = Field(
        ...,
        description="Recommended dashboard columns",
        example=["sales", "product", "region"]
    )


class AnalysisRequest(BaseModel):
    """Request to analyze dataset"""
    
    force_reanalyze: bool = Field(
        default=False,
        description="Force re-analysis even if recent analysis exists"
    )