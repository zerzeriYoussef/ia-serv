from sqlalchemy import Column, Integer, String, JSON, Boolean, ForeignKey, DateTime, Float

from sqlalchemy.sql import func

from app.core.database import Base





class CleaningProfile(Base):

    """Store cleaning configurations that can be reused"""

    __tablename__ = "cleaning_profiles"



    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    description = Column(String)





    handle_missing = Column(String, default="fill")

    missing_fill_strategy = Column(String, default="auto")

    missing_fill_value = Column(String)



    remove_duplicates = Column(Boolean, default=True)

    duplicate_subset = Column(JSON)

    duplicate_keep = Column(String, default="best")

    dedup_normalize_text = Column(Boolean, default=True)



    fix_data_types = Column(Boolean, default=True)

    numeric_coerce_min_rate = Column(Float, default=0.8)



    detect_outliers = Column(Boolean, default=True)

    outlier_method = Column(String, default="iqr")

    outlier_threshold = Column(Float, default=1.5)

    outlier_action = Column(String, default="flag")

    outlier_max_rows = Column(Integer, default=500000)

    add_flag_columns = Column(Boolean, default=False)

    save_outliers_metadata = Column(Boolean, default=True)

    target_columns = Column(JSON)







    column_rules = Column(JSON)





    strip_whitespace = Column(Boolean, default=True)

    standardize_text = Column(Boolean, default=True)





    standardize_dates = Column(Boolean, default=True)

    date_format = Column(String)





    normalize_numeric = Column(Boolean, default=False)

    normalization_method = Column(String)





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





    operations_performed = Column(JSON)





    rows_before = Column(Integer)

    rows_after = Column(Integer)

    duplicates_removed = Column(Integer, default=0)

    missing_values_handled = Column(Integer, default=0)

    outliers_detected = Column(Integer, default=0)

    data_types_fixed = Column(Integer, default=0)





    cleaning_report = Column(JSON)





    backup_path = Column(String)



    created_at = Column(DateTime(timezone=True), server_default=func.now())



    def __repr__(self):

        return f"<CleaningLog {self.id} for Dataset {self.dataset_id}>"
