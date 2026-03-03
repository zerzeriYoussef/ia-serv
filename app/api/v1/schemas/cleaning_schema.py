from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class MissingValueStrategy(str, Enum):
    DROP = "drop"
    FILL = "fill"
    INTERPOLATE = "interpolate"


class FillStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


class OutlierAction(str, Enum):
    FLAG = "flag"
    REMOVE = "remove"
    CAP = "cap"


class CleaningProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    
    # Missing values
    handle_missing: MissingValueStrategy = MissingValueStrategy.DROP
    missing_fill_strategy: FillStrategy = FillStrategy.MEAN
    missing_fill_value: Optional[str] = None
    
    # Duplicates
    remove_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    
    # Data types
    fix_data_types: bool = True
    
    # Outliers
    detect_outliers: bool = True
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = 1.5
    outlier_action: OutlierAction = OutlierAction.FLAG
    
    # Text
    strip_whitespace: bool = True
    standardize_text: bool = False
    
    # Dates
    standardize_dates: bool = True
    date_format: Optional[str] = None
    
    # Normalization
    normalize_numeric: bool = False
    normalization_method: Optional[str] = None


class CleaningProfileResponse(CleaningProfileCreate):
    id: int
    is_default: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class CleanDatasetRequest(BaseModel):
    dataset_id: int
    profile_id: Optional[int] = None  # Use default if not provided
    save_as_new: bool = False  # Create new dataset or overwrite


class CleaningReportResponse(BaseModel):
    dataset_id: int
    profile_id: Optional[int]
    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_values_handled: int
    outliers_detected: int
    data_types_fixed: int
    operations_performed: List[str]
    cleaning_report: Dict
    created_at: datetime


class DataQualityResponse(BaseModel):
    dataset_id: int
    quality_score: float  # 0-100
    total_rows: int
    total_columns: int
    missing_values: Dict
    duplicates: Dict
    data_types: Dict
    numeric_summary: Dict
    categorical_summary: Dict
    validation_errors: List[str]
    validation_warnings: List[str]