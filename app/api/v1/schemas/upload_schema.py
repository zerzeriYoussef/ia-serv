from pydantic import BaseModel, Field

from typing import Optional, List, Dict

from datetime import datetime

from enum import Enum





class DatasetStatus(str, Enum):

    UPLOADED = "uploaded"

    PROCESSING = "processing"

    PROCESSED = "processed"

    FAILED = "failed"





class DatasetUploadResponse(BaseModel):

    id: int

    filename: str

    original_filename: str

    file_size: int

    file_type: str

    status: DatasetStatus

    user_id: Optional[str] = None

    created_at: datetime



    class Config:

        from_attributes = True





class DatasetDetailResponse(BaseModel):

    id: int

    filename: str

    original_filename: str

    file_path: str

    file_size: int

    file_type: str

    status: DatasetStatus

    user_id: Optional[str] = None

    row_count: Optional[int] = None

    column_count: Optional[int] = None

    columns: Optional[List[str]] = None

    column_types: Optional[Dict[str, str]] = None

    summary_stats: Optional[Dict] = None

    created_at: datetime

    updated_at: Optional[datetime] = None



    class Config:

        from_attributes = True





class DatasetListResponse(BaseModel):

    datasets: List[DatasetUploadResponse]

    total: int

    page: int

    page_size: int
