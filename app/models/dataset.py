from sqlalchemy import Column, Integer, String, DateTime, JSON, BigInteger, Enum
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class DatasetStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(BigInteger, nullable=False)  # bytes
    file_type = Column(String, nullable=False)  # csv, xlsx, json, etc.
    
    # Status
    status = Column(Enum(DatasetStatus), default=DatasetStatus.UPLOADED)
    
    # Metadata
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns = Column(JSON, nullable=True)  # List of column names
    column_types = Column(JSON, nullable=True)  # Dict of column: type
    summary_stats = Column(JSON, nullable=True)  # Basic statistics
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # User reference (add later when auth is implemented)
    # user_id = Column(Integer, ForeignKey("users.id"))
    
    def __repr__(self):
        return f"<Dataset {self.id}: {self.original_filename}>"