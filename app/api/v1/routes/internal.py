from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.repositories.dataset_repository import DatasetRepository
from app.services.data.ingestion_service import IngestionService

# We can use a simple secret header or just require admin. Let's stick to require_admin or a shared secret.
# The backend needs to communicate. Since backend creates a token, it can just pass an admin token,
# but it's easier to use a shared internal secret if configured, or just trust the call if it's in the private network.
# Let's use a simple dependency for internal auth. For now, we'll accept any call with a valid user_id and maybe a shared secret,
# or we just rely on network isolation. Let's use a shared internal secret.

from fastapi import Header, HTTPException
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
ingestion_service = IngestionService()

def verify_internal_secret(x_internal_secret: str = Header(...)):
    if settings.INTERNAL_SECRET and x_internal_secret != settings.INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Secret interne invalide")

@router.delete("/internal/users/{user_id}/data", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_internal_secret)])
async def purge_user_data(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Internal endpoint: delete all datasets + files for a given user.
    Called by the auth-service when a user account is deleted.
    """
    datasets = await DatasetRepository.get_by_user_id(db, user_id, limit=10000)

    from app.services.rag.dataset_indexer import delete_dataset_index

    for dataset in datasets:
        try:
            await ingestion_service.delete_file(dataset.file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {dataset.file_path}: {e}")
        delete_dataset_index(dataset.id)

    deleted_count = await DatasetRepository.delete_by_user_id(db, user_id)
    logger.info(f"Purged {deleted_count} datasets for deleted user_id={user_id}")
