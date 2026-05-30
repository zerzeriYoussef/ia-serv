"""
Dashboard model — persisted output of a RAG-insights generation run.

One dataset can have multiple dashboard versions (history); the agent
can update/delete them later via DashboardRepository (no public API routes).
"""



from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey

from sqlalchemy.sql import func



from app.core.database import Base





class Dashboard(Base):

    __tablename__ = "dashboards"



    id = Column(Integer, primary_key=True, index=True)





    user_id     = Column(String, nullable=True, index=True)

    dataset_id  = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)

    analysis_id = Column(Integer, ForeignKey("column_analysis.id"), nullable=True)





    executive_summary_kpis = Column(JSON, default=list)

    dashboard_charts       = Column(JSON, default=list)





    rag_meta = Column(JSON, nullable=True)





    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), onupdate=func.now())



    def __repr__(self) -> str:

        return f"<Dashboard {self.id} dataset={self.dataset_id}>"

