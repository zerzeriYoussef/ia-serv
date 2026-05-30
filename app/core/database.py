from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from sqlalchemy.orm import declarative_base

from app.core.config import settings

import logging



logger = logging.getLogger(__name__)





engine = create_async_engine(

    settings.DATABASE_URL,

    pool_size=settings.DATABASE_POOL_SIZE,

    max_overflow=settings.DATABASE_MAX_OVERFLOW,

    echo=settings.DEBUG,

    future=True,

)





AsyncSessionLocal = async_sessionmaker(

    engine,

    class_=AsyncSession,

    expire_on_commit=False,

    autoflush=False,

    autocommit=False,

)





Base = declarative_base()





async def get_db() -> AsyncSession:

    """Dependency for getting database session"""

    async with AsyncSessionLocal() as session:

        try:

            yield session

            await session.commit()

        except Exception as e:

            await session.rollback()

            logger.error(f"Database error: {e}")

            raise

        finally:

            await session.close()





async def init_db():

    """Initialize database tables"""

    async with engine.begin() as conn:

        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized")





async def close_db():

    """Close database connection"""

    await engine.dispose()

    logger.info("Database connection closed")
