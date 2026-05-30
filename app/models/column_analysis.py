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





    metrics = Column(JSON, default=list)

    dimensions = Column(JSON, default=list)

    identifiers = Column(JSON, default=list)

    temporal = Column(JSON, default=list)

    geographic = Column(JSON, default=list)

    other = Column(JSON, default=list)





    relationships = Column(JSON, default=list)





    dashboard_columns = Column(JSON, default=list)

    primary_metric = Column(String)





    analysis_method = Column(String, default="custom_pps")

    confidence_score = Column(Float)

    total_relationships = Column(Integer, default=0)





    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), onupdate=func.now())



    def __repr__(self):

        return f"<ColumnAnalysis {self.id} for Dataset {self.dataset_id}>"
