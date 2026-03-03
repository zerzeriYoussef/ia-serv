from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import pandas as pd
import logging
import os

from app.core.database import get_db
from app.services.data.cleaning_service import CleaningService
from app.services.data.validation_service import ValidationService
from app.services.data.parser_service import ParserService
from app.repositories.cleaning_repository import (
    CleaningProfileRepository,
    CleaningLogRepository
)
from app.repositories.dataset_repository import DatasetRepository
from app.api.v1.schemas.cleaning_schema import (
    CleaningProfileCreate,
    CleaningProfileResponse,
    CleanDatasetRequest,
    CleaningReportResponse,
    DataQualityResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== CLEANING PROFILES ====================

@router.post("/cleaning-profiles", response_model=CleaningProfileResponse)
async def create_cleaning_profile(
    profile: CleaningProfileCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new cleaning profile"""
    
    new_profile = await CleaningProfileRepository.create(
        db,
        **profile.dict()
    )
    
    logger.info(f"Created cleaning profile: {new_profile.name}")
    
    return new_profile


@router.get("/cleaning-profiles", response_model=List[CleaningProfileResponse])
async def list_cleaning_profiles(db: AsyncSession = Depends(get_db)):
    """List all cleaning profiles"""
    profiles = await CleaningProfileRepository.get_all(db)
    return profiles


@router.get("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)
async def get_cleaning_profile(
    profile_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get cleaning profile by ID"""
    profile = await CleaningProfileRepository.get_by_id(db, profile_id)
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cleaning profile {profile_id} not found"
        )
    
    return profile


@router.delete("/cleaning-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cleaning_profile(
    profile_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete cleaning profile"""
    deleted = await CleaningProfileRepository.delete(db, profile_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cleaning profile {profile_id} not found"
        )


# ==================== DATA CLEANING ====================

@router.post("/datasets/{dataset_id}/clean", response_model=CleaningReportResponse)
async def clean_dataset(
    dataset_id: int,
    profile_id: Optional[int] = None,
    save_as_new: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Clean a dataset using a cleaning profile
    
    If no profile_id is provided, uses the default profile.
    If save_as_new is True, creates a new dataset with cleaned data.
    """
    
    # Get dataset
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Get cleaning profile
    if profile_id:
        profile = await CleaningProfileRepository.get_by_id(db, profile_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cleaning profile {profile_id} not found"
            )
    else:
        # Use default profile
        profile = await CleaningProfileRepository.get_default(db)
        if not profile:
            # Create a default profile if none exists
            profile = await CleaningProfileRepository.create(
                db,
                name="Default Cleaning Profile",
                description="Auto-created default profile",
                is_default=True
            )
    
    # Load data
    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    
    # Clean data
    profile_dict = {
        "handle_missing": profile.handle_missing,
        "missing_fill_strategy": profile.missing_fill_strategy,
        "missing_fill_value": profile.missing_fill_value,
        "remove_duplicates": profile.remove_duplicates,
        "duplicate_subset": profile.duplicate_subset,
        "fix_data_types": profile.fix_data_types,
        "detect_outliers": profile.detect_outliers,
        "outlier_method": profile.outlier_method,
        "outlier_threshold": profile.outlier_threshold,
        "outlier_action": profile.outlier_action,
        "strip_whitespace": profile.strip_whitespace,
        "standardize_text": profile.standardize_text,
        "standardize_dates": profile.standardize_dates,
        "date_format": profile.date_format,
    }
    
    df_clean, cleaning_report = CleaningService.clean_dataframe(df, profile_dict)
    
    # Save cleaned data
    if save_as_new:
        # Create new dataset
        new_filename = f"cleaned_{dataset.filename}"
        new_path = dataset.file_path.replace(dataset.filename, new_filename)
        
        # Save to file
        if dataset.file_type == 'csv':
            df_clean.to_csv(new_path, index=False)
        elif dataset.file_type in ['xlsx', 'xls']:
            df_clean.to_excel(new_path, index=False)
        
        # Create dataset record
        new_dataset = await DatasetRepository.create(
            db,
            filename=new_filename,
            original_filename=f"cleaned_{dataset.original_filename}",
            file_path=new_path,
            file_size=os.path.getsize(new_path),
            file_type=dataset.file_type,
            row_count=len(df_clean),
            column_count=len(df_clean.columns),
            columns=df_clean.columns.tolist(),
            status="processed"
        )
        
        target_dataset_id = new_dataset.id
    else:
        # Overwrite existing file
        if dataset.file_type == 'csv':
            df_clean.to_csv(dataset.file_path, index=False)
        elif dataset.file_type in ['xlsx', 'xls']:
            df_clean.to_excel(dataset.file_path, index=False)
        
        # Update dataset metadata
        await DatasetRepository.update_metadata(
            db,
            dataset_id,
            {
                "row_count": len(df_clean),
                "column_count": len(df_clean.columns)
            }
        )
        
        target_dataset_id = dataset_id
    
    # Create cleaning log
    log = await CleaningLogRepository.create(
        db,
        dataset_id=target_dataset_id,
        profile_id=profile.id,
        rows_before=cleaning_report["rows_before"],
        rows_after=cleaning_report["rows_after"],
        duplicates_removed=cleaning_report["changes"].get("duplicates", {}).get("duplicates_removed", 0),
        missing_values_handled=cleaning_report["changes"].get("missing_values", {}).get("total_filled", 0),
        outliers_detected=cleaning_report["changes"].get("outliers", {}).get("total_outliers", 0),
        data_types_fixed=cleaning_report["changes"].get("data_types", {}).get("types_changed", 0),
        operations_performed=cleaning_report["operations"],
        cleaning_report=cleaning_report
    )
    
    logger.info(f"Cleaned dataset {dataset_id}, created log {log.id}")
    
    return log


@router.get("/datasets/{dataset_id}/quality", response_model=DataQualityResponse)
async def get_data_quality(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get data quality report for a dataset"""
    
    # Get dataset
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Load data
    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    
    # Generate quality report
    quality_report = CleaningService.get_data_quality_report(df)
    
    # Run validation
    validation_report = ValidationService.validate_dataframe(df)
    
    # Calculate quality score (0-100)
    quality_score = 100.0
    
    # Deduct points for issues
    if quality_report["missing_values"]["percentage"] > 0:
        quality_score -= min(quality_report["missing_values"]["percentage"], 30)
    
    if quality_report["duplicates"]["percentage"] > 0:
        quality_score -= min(quality_report["duplicates"]["percentage"] * 2, 20)
    
    if not validation_report["valid"]:
        quality_score -= 20
    
    quality_score = max(quality_score, 0)
    
    return {
        "dataset_id": dataset_id,
        "quality_score": round(quality_score, 2),
        "total_rows": quality_report["total_rows"],
        "total_columns": quality_report["total_columns"],
        "missing_values": quality_report["missing_values"],
        "duplicates": quality_report["duplicates"],
        "data_types": quality_report["data_types"],
        "numeric_summary": quality_report["numeric_summary"],
        "categorical_summary": quality_report["categorical_summary"],
        "validation_errors": validation_report["errors"],
        "validation_warnings": validation_report["warnings"]
    }


@router.get("/datasets/{dataset_id}/cleaning-history", response_model=List[CleaningReportResponse])
async def get_cleaning_history(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get cleaning history for a dataset"""
    
    logs = await CleaningLogRepository.get_by_dataset(db, dataset_id)
    return logs


# ==================== VALIDATION ====================

@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Validate dataset against default rules"""
    
    # Get dataset
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Load data
    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    
    # Validate
    validation_report = ValidationService.validate_dataframe(df)
    
    return {
        "dataset_id": dataset_id,
        "valid": validation_report["valid"],
        "errors": validation_report["errors"],
        "warnings": validation_report["warnings"],
        "checks_performed": validation_report["checks_performed"],
        "total_checks": validation_report["total_checks"]
    }