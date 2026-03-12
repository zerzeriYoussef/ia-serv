from sqlalchemy import Column, Integer, String, JSON, Float, ForeignKey, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class ColumnAnalysis(Base):
    """
    Store column analysis results for datasets
    
    Captures:
    - Column categorization (metrics, dimensions, identifiers, etc.)
    - Detected relationships between columns
    - Dashboard recommendations
    - Analysis metadata
    """
    __tablename__ = "column_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, unique=True)
    
    # Column categorization
    metrics = Column(JSON, default=list)           # ["sales", "revenue", "quantity"]
    dimensions = Column(JSON, default=list)        # ["region", "product", "category"]
    identifiers = Column(JSON, default=list)       # ["id", "customer_id"]
    temporal = Column(JSON, default=list)          # ["date", "created_at"]
    geographic = Column(JSON, default=list)        # ["city", "region", "country"]
    other = Column(JSON, default=list)             # Uncategorized columns
    
    # Detected relationships
    relationships = Column(JSON, default=list)     # List of relationship objects
    
    # Dashboard recommendations
    dashboard_columns = Column(JSON, default=list) # Top columns for dashboards
    primary_metric = Column(String)                # Main KPI column
    
    # Analysis metadata
    analysis_method = Column(String, default="custom_pps")
    confidence_score = Column(Float)               # Overall confidence (0-1)
    total_relationships = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ColumnAnalysis {self.id} for Dataset {self.dataset_id}>"