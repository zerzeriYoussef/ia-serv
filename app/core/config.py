from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Data Analytics Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/ai_service"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600
    
    # File Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx", "xls", "json", "parquet"]
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"

    # Gemini (RAG / insights). Optional — endpoints return 503 if missing when calling LLM.
    GEMINI_API_KEY: Optional[str] = None
    # Override via env if your project only exposes 1.5 models, e.g. gemini-1.5-flash
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Chroma vector store — relative to the project root or absolute
    CHROMA_DIR: str = "./chroma_data"
    # Embedding model used for dataset chunk indexing + query embedding
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # Max chunks retrieved per chat query
    CHAT_RETRIEVAL_TOP_K: int = 8
    # Number of turns before rolling summary kicks in
    CHAT_ROLLING_SUMMARY_EVERY: int = 20
    # Max recent turns loaded from DB and sent to the narrator (×2 = message rows)
    CHAT_MAX_RECENT_TURNS: int = 10
    # Last N chat rows (user + assistant) included in the orchestrator JSON-plan prompt
    # so follow-ups like "yes, the revenue column" stay tied to the thread.
    CHAT_PLANNER_HISTORY_MESSAGES: int = 12

    # ── Report generation ──────────────────────────────────────────────────
    # Serper Google Search API key — used by the external context agent
    SERPER_API_KEY: Optional[str] = None
    # Top-K chunks retrieved per topic during report RAG (5 topics × this value)
    REPORT_RETRIEVAL_TOP_K: int = 5
    # Redis TTL for cached reports in seconds (default 1 hour)
    REPORT_CACHE_TTL: int = 3600

    class Config:  #i will later change setting above for better security
        env_file = ".env"
        case_sensitive = True
        # Allow extra environment variables like PGCLIENTENCODING / PGSSLMODE
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()