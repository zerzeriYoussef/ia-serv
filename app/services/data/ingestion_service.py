import os
import uuid
from pathlib import Path
from typing import BinaryIO, Tuple
import aiofiles
import logging

from app.core.config import settings
from fastapi import UploadFile, HTTPException, status

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = settings.MAX_UPLOAD_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    def _validate_file_extension(self, filename: str) -> str:
        """Validate file extension"""
        extension = filename.split('.')[-1].lower()
        if extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '.{extension}' not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            )
        return extension
    
    def _validate_file_size(self, file_size: int):
        """Validate file size"""
        if file_size > self.max_size:
            max_mb = self.max_size / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_mb:.0f}MB"
            )
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename"""
        extension = original_filename.split('.')[-1]
        unique_name = f"{uuid.uuid4()}.{extension}"
        return unique_name
    
    async def save_upload_file(self, upload_file: UploadFile) -> Tuple[str, str, int]:
        """
        Save uploaded file to disk
        
        Returns:
            Tuple of (unique_filename, file_path, file_size)
        """
        try:
            # Validate extension
            file_extension = self._validate_file_extension(upload_file.filename)
            
            # Read file content
            content = await upload_file.read()
            file_size = len(content)
            
            # Validate size
            self._validate_file_size(file_size)
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(upload_file.filename)
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"File saved: {unique_filename} ({file_size} bytes)")
            
            return unique_filename, str(file_path), file_size
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving file: {str(e)}"
            )
    
    async def delete_file(self, file_path: str):
        """Delete file from disk"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file: {e}")