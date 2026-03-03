from sqlalchemy import Column, Integer, String, JSON, Boolean, ForeignKey, DateTime, Float
from sqlalchemy.sql import func
from app.core.database import Base


class CleaningProfile(Base):
    """Store cleaning configurations that can be reused"""
    __tablename__ = "cleaning_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Cleaning rules
    handle_missing = Column(String, default="drop")  # drop, fill, interpolate
    missing_fill_strategy = Column(String, default="mean")  # mean, median, mode, constant
    missing_fill_value = Column(String)  # for constant strategy
    
    remove_duplicates = Column(Boolean, default=True)
    duplicate_subset = Column(JSON)  # columns to consider for duplicates
    
    fix_data_types = Column(Boolean, default=True)
    
    detect_outliers = Column(Boolean, default=True)
    outlier_method = Column(String, default="iqr")  # iqr, zscore, isolation_forest
    outlier_threshold = Column(Float, default=1.5)  # for IQR
    outlier_action = Column(String, default="flag")  # flag, remove, cap
    
    # Text cleaning
    strip_whitespace = Column(Boolean, default=True)
    standardize_text = Column(Boolean, default=True)  # lowercase, remove special chars
    
    # Date handling
    standardize_dates = Column(Boolean, default=True)
    date_format = Column(String)  # target date format
    
    # Numeric handling
    normalize_numeric = Column(Boolean, default=False)
    normalization_method = Column(String)  # minmax, standard, robust
    
    # Metadata
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<CleaningProfile {self.id}: {self.name}>"


class CleaningLog(Base):
    """Log of cleaning operations performed on datasets"""
    __tablename__ = "cleaning_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    profile_id = Column(Integer, ForeignKey("cleaning_profiles.id"))
    
    # What was done
    operations_performed = Column(JSON)  # List of operations
    
    # Statistics
    rows_before = Column(Integer)
    rows_after = Column(Integer)
    duplicates_removed = Column(Integer, default=0)
    missing_values_handled = Column(Integer, default=0)
    outliers_detected = Column(Integer, default=0)
    data_types_fixed = Column(Integer, default=0)
    
    # Results
    cleaning_report = Column(JSON)  # Detailed report
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<CleaningLog {self.id} for Dataset {self.dataset_id}>"