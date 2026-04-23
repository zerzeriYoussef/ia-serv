from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.api.v1.routes import cleaning

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.cache import init_redis, close_redis
from app.core.logging import logger
from app.api.v1.routes import upload
from app.api.v1.routes import analysis
from app.api.v1.routes import rag
from app.api.v1.routes import chat
from app.api.v1.routes import report

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize connections
    await init_db()
    await init_redis()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    await close_redis()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(cleaning.router, prefix="/api/v1", tags=["Cleaning"])
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(report.router, prefix="/api/v1", tags=["Reports"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )