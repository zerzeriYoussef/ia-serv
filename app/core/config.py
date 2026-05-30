from typing import List, Optional

from pydantic_settings import BaseSettings

from functools import lru_cache





class Settings(BaseSettings):



    APP_NAME: str = "AI Data Analytics Service"

    APP_VERSION: str = "1.0.0"

    ENVIRONMENT: str = "development"

    DEBUG: bool = True

    HOST: str = "0.0.0.0"

    PORT: int = 8001





    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/ai_service"

    DATABASE_POOL_SIZE: int = 20

    DATABASE_MAX_OVERFLOW: int = 10





    REDIS_URL: str = "redis://localhost:6379"

    REDIS_CACHE_TTL: int = 3600





    UPLOAD_DIR: str = "./uploads"

    MAX_UPLOAD_SIZE: int = 104857600

    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx", "xls", "json", "parquet"]





    SECRET_KEY: str

    ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30





    INTERNAL_SECRET: str = "changeme_in_production"





    CORS_ORIGINS: List[str] = [

        "http://localhost:4200",

        "http://localhost:5173",

        "http://localhost:3000",

        "http://localhost",

        "http://127.0.0.1:4200",

    ]





    LOG_LEVEL: str = "INFO"





    GEMINI_API_KEY: Optional[str] = None



    GEMINI_MODEL: str = "gemini-2.5-flash"





    CHROMA_DIR: str = "./chroma_data"



    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"



    CHAT_RETRIEVAL_TOP_K: int = 8



    CHAT_ROLLING_SUMMARY_EVERY: int = 20



    CHAT_MAX_RECENT_TURNS: int = 10





    CHAT_PLANNER_HISTORY_MESSAGES: int = 12







    SERPER_API_KEY: Optional[str] = None



    REPORT_RETRIEVAL_TOP_K: int = 5



    REPORT_CACHE_TTL: int = 3600



    class Config:

        env_file = ".env"

        case_sensitive = True



        extra = "ignore"





@lru_cache()

def get_settings() -> Settings:

    """Get cached settings instance"""

    return Settings()





settings = get_settings()
