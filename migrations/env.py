from logging.config import fileConfig
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool

# Import your app's configuration and models
from app.core.config import settings
from app.core.database import Base
from app.models.dataset import Dataset  # Import your models here
from app.models.cleaning_profile import CleaningProfile, CleaningLog
from app.models.column_analysis import ColumnAnalysis
from app.models.dashboard import Dashboard
# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set your metadata
target_metadata = Base.metadata

# Get database URL (convert asyncpg to regular postgresql for sync migrations)
def get_url():
    url = settings.DATABASE_URL
    # Replace asyncpg with psycopg2 for sync migrations
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()