"""Test environment setup"""
import os
import sys
import asyncio
import logging

# Ensure project root is on sys.path so `app` package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import engine
from app.core.cache import redis_client, init_redis
from app.core.config import settings

logger = logging.getLogger(__name__)


async def test_database():
    """Test database connection"""
    try:
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            logger.info("✓ Database connection successful")
            return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


async def test_redis():
    """Test Redis connection"""
    try:
        await init_redis()
        await redis_client.ping()
        logger.info("✓ Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return False


async def run_tests():
    logger.info(f"Testing {settings.APP_NAME} setup...")
    logger.info("-" * 50)
    
    db_ok = await test_database()
    redis_ok = await test_redis()
    
    logger.info("-" * 50)
    if db_ok and redis_ok:
        logger.info("✓ All tests passed!")
    else:
        logger.error("✗ Some tests failed")


if __name__ == "__main__":
    asyncio.run(run_tests())