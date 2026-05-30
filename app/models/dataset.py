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

    file_size = Column(BigInteger, nullable=False)

    file_type = Column(String, nullable=False)





    status = Column(Enum(DatasetStatus), default=DatasetStatus.UPLOADED)





    row_count = Column(Integer, nullable=True)

    column_count = Column(Integer, nullable=True)

    columns = Column(JSON, nullable=True)

    column_types = Column(JSON, nullable=True)

    summary_stats = Column(JSON, nullable=True)





    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), onupdate=func.now())









    user_id = Column(String, nullable=True, index=True)



    def __repr__(self):

        return f"<Dataset {self.id}: {self.original_filename}>"
