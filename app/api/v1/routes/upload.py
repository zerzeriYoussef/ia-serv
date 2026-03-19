from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
)
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.services.data.ingestion_service import IngestionService
from app.services.data.parser_service import ParserService
from app.repositories.dataset_repository import DatasetRepository
from app.models.dataset import DatasetStatus
from app.api.dependencies.auth import CurrentUser, require_auth, require_admin
from app.api.v1.schemas.upload_schema import (
    DatasetUploadResponse,
    DatasetDetailResponse,
    DatasetListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()
ingestion_service = IngestionService()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def process_file_background(dataset_id: int, file_path: str, file_type: str):
    """Background task to process an uploaded file and auto-analyse its columns."""
    from app.core.database import AsyncSessionLocal
    from app.services.analysis.column_analyzer import ColumnAnalyzer
    from app.repositories.analysis_repository import AnalysisRepository

    async with AsyncSessionLocal() as db:
        try:
            await DatasetRepository.update_status(db, dataset_id, DatasetStatus.PROCESSING)

            df, metadata = await ParserService.parse_file(file_path, file_type)

            await DatasetRepository.update_metadata(db, dataset_id, {
                "row_count": metadata["row_count"],
                "column_count": metadata["column_count"],
                "columns": metadata["columns"],
                "column_types": metadata["column_types"],
                "summary_stats": metadata["summary_stats"],
                "status": DatasetStatus.PROCESSED,
            })

            await db.commit()
            logger.info(f"Dataset {dataset_id} processed successfully")

            # Auto-analysis (isolated so it never fails the upload)
            try:
                logger.info(f"Starting auto-analysis for dataset {dataset_id}...")
                analyzer = ColumnAnalyzer(df)
                analysis_results = analyzer.analyze()

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
                    total_relationships=len(analysis_results["relationships"]),
                )

                await db.commit()
                logger.info(
                    f"Auto-analysis complete for dataset {dataset_id}: "
                    f"{len(analysis_results['relationships'])} relationships found"
                )

            except Exception as exc:
                logger.error(f"Auto-analysis failed for dataset {dataset_id}: {exc}")

        except Exception as exc:
            logger.error(f"Error processing dataset {dataset_id}: {exc}")
            await DatasetRepository.update_status(db, dataset_id, DatasetStatus.FAILED)


# ---------------------------------------------------------------------------
# Upload – requires authentication
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Upload a data file (CSV, Excel, JSON, Parquet).

    **Requires** a valid ``Authorization: Bearer <token>`` header.
    The authenticated user's ID is stored with the dataset.
    The file is processed asynchronously; the endpoint returns immediately.
    """
    unique_filename, file_path, file_size = await ingestion_service.save_upload_file(file)
    file_type = file.filename.split(".")[-1].lower()

    dataset = await DatasetRepository.create(
        db,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type,
        user_id=current_user.user_id,
        status=DatasetStatus.UPLOADED,
    )

    background_tasks.add_task(process_file_background, dataset.id, file_path, file_type)

    logger.info(
        f"File uploaded: {file.filename} -> Dataset ID: {dataset.id} "
        f"(user_id={current_user.user_id})"
    )
    return dataset


# ---------------------------------------------------------------------------
# Get a single dataset
# ---------------------------------------------------------------------------

@router.get("/datasets", response_model=DatasetListResponse)
async def list_all_datasets(
    page: int = 1,
    page_size: int = 10,
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """
    Return a paginated list of **all** datasets across every user.

    **Requires** admin role. Non-admins receive a 403 Forbidden response.
    """
    skip = (page - 1) * page_size

    datasets = await DatasetRepository.get_all(db, skip=skip, limit=page_size)
    total = await DatasetRepository.count(db)

    return {
        "datasets": datasets,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/datasets/my-datasets", response_model=DatasetListResponse)
async def list_my_datasets(
    page: int = 1,
    page_size: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Return a paginated list of datasets that belong to the authenticated user.

    **Requires** a valid ``Authorization: Bearer <token>`` header.
    """
    skip = (page - 1) * page_size

    datasets = await DatasetRepository.get_by_user_id(
        db, current_user.user_id, skip=skip, limit=page_size
    )
    total = await DatasetRepository.count_by_user_id(db, current_user.user_id)

    return {
        "datasets": datasets,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Retrieve full details for a single dataset.

    * Regular users can only access their own datasets.
    * Admins can access any dataset.
    """
    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )

    if not current_user.is_admin and dataset.user_id != current_user.user_id:
        logger.warning(
            "Forbidden: user_id=%s tried to access dataset %s owned by user_id=%s",
            current_user.user_id,
            dataset_id,
            dataset.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this dataset",
        )

    return dataset


# Deleted old positions of list_all_datasets and list_my_datasets to move them up


# ---------------------------------------------------------------------------
# Delete a dataset
# ---------------------------------------------------------------------------

@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Delete a dataset and its associated file.

    * Regular users can only delete their own datasets.
    * Admins can delete any dataset.
    """
    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )

    if not current_user.is_admin and dataset.user_id != current_user.user_id:
        logger.warning(
            "Forbidden: user_id=%s tried to delete dataset %s owned by user_id=%s",
            current_user.user_id,
            dataset_id,
            dataset.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to delete this dataset",
        )

    await ingestion_service.delete_file(dataset.file_path)
    await DatasetRepository.delete(db, dataset_id)
    logger.info(f"Dataset {dataset_id} deleted by user_id={current_user.user_id}")