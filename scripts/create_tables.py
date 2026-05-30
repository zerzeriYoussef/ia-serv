import asyncio

import os

import sys

from pathlib import Path





def add_project_root_to_path() -> None:

    """Ensure the project root is on sys.path so `app` imports work when running this script directly."""

    project_root = Path(__file__).resolve().parents[1]

    if str(project_root) not in sys.path:

        sys.path.insert(0, str(project_root))





add_project_root_to_path()



from app.core.database import init_db

from app.core.config import settings





async def main() -> None:

    print("Using database URL:")

    print(f"  {settings.DATABASE_URL}")

    print("\nCreating all tables defined on Base metadata...")

    await init_db()

    print("Tables created (or already existed).")





if __name__ == "__main__":

    asyncio.run(main())



