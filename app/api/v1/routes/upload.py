from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import logging

from app.core.database import get_db
from app.services.data.ingestion_service import IngestionService
from app.services.data.parser_service import ParserService
from app.repositories.dataset_repository import DatasetRepository
from app.models.dataset import DatasetStatus
from app.api.v1.schemas.upload_schema import (
    DatasetUploadResponse,
    DatasetDetailResponse,
    DatasetListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()
ingestion_service = IngestionService()


async def process_file_background(dataset_id: int, file_path: str, file_type: str):
    """Background task to process uploaded file"""
    from app.core.database import AsyncSessionLocal
    from app.services.analysis.column_analyzer import ColumnAnalyzer
    from app.repositories.analysis_repository import AnalysisRepository
    
    async with AsyncSessionLocal() as db:
        try:
            # Update status to processing
            await DatasetRepository.update_status(db, dataset_id, DatasetStatus.PROCESSING)
            
            # Parse file
            df, metadata = await ParserService.parse_file(file_path, file_type)
            
            # Update dataset with metadata
            await DatasetRepository.update_metadata(db, dataset_id, {
                "row_count": metadata["row_count"],
                "column_count": metadata["column_count"],
                "columns": metadata["columns"],
                "column_types": metadata["column_types"],
                "summary_stats": metadata["summary_stats"],
                "status": DatasetStatus.PROCESSED
            })
            
            await db.commit()  # Commit dataset metadata first
            
            logger.info(f"Dataset {dataset_id} processed successfully")
            
            # AUTO-ANALYZE (separate try-catch so it doesn't fail upload)
            try:
                logger.info(f"Starting auto-analysis for dataset {dataset_id}...")
                
                analyzer = ColumnAnalyzer(df)
                analysis_results = analyzer.analyze()
                
                # Save analysis
                await AnalysisRepository.create(
                    db,
                    dataset_id=dataset_id,
                    metrics=analysis_results["column_categories"]["metrics"],
                    dimensions=analysis_results["column_categories"]["dimensions"],
                    identifiers=analysis_results["column_categories"]["identifiers"],
                    temporal=analysis_results["column_categories"]["temporal"],
                    geographic=analysis_results["column_categories"]["geographic"],
                    other=analysis_results["column_categories"]["other"],
                    relationships=analysis_results["relationships"],
                    dashboard_columns=analysis_results["dashboard_columns"],
                    primary_metric=analysis_results["primary_metric"],
                    confidence_score=analysis_results["confidence_score"],
                    total_relationships=len(analysis_results["relationships"])
                )
                
                await db.commit()
                
                logger.info(
                    f"Auto-analysis complete for dataset {dataset_id}: "
                    f"{len(analysis_results['relationships'])} relationships found"
                )
                
            except Exception as e:
                logger.error(f"Auto-analysis failed for dataset {dataset_id}: {e}")
                # Don't fail the whole upload - analysis can be run manually later
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_id}: {e}")
            await DatasetRepository.update_status(db, dataset_id, DatasetStatus.FAILED)


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a data file (CSV, Excel, JSON, Parquet)
    
    The file will be saved and processed in the background.
    Returns immediately with upload confirmation.
    """
    
    # Save file
    unique_filename, file_path, file_size = await ingestion_service.save_upload_file(file)
    
    # Get file type
    file_type = file.filename.split('.')[-1].lower()
    
    # Create dataset record
    dataset = await DatasetRepository.create(
        db,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type,
        status=DatasetStatus.UPLOADED
    )
    
    # Process file in background
    background_tasks.add_task(
        process_file_background,
        dataset.id,
        file_path,
        file_type
    )
    
    logger.info(f"File uploaded: {file.filename} -> Dataset ID: {dataset.id}")
    
    return dataset


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get dataset details by ID"""
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    return dataset


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    page: int = 1,
    page_size: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """List all datasets with pagination"""
    
    skip = (page - 1) * page_size
    
    datasets = await DatasetRepository.get_all(db, skip=skip, limit=page_size)
    total = await DatasetRepository.count(db)
    
    return {
        "datasets": datasets,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete dataset"""
    
    # Get dataset
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Delete file
    await ingestion_service.delete_file(dataset.file_path)
    
    # Delete from database
    await DatasetRepository.delete(db, dataset_id)
    
    logger.info(f"Dataset {dataset_id} deleted")