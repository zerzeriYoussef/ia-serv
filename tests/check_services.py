"""
Improved test setup script with detailed diagnostics
"""
import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing AI Service Setup")
print("=" * 60)

# Test 1: Import and Config
print("\n[1] Testing imports and configuration...")
try:
    from app.core.config import settings
    print(f"✓ Imports successful")
    print(f"  - DATABASE_URL: {settings.DATABASE_URL[:50]}...")
    print(f"  - REDIS_URL: {settings.REDIS_URL}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: PostgreSQL Connection
print("\n[2] Testing PostgreSQL connection...")
try:
    # Test with regular psycopg2 first (simpler)
    import psycopg2
    from urllib.parse import urlparse
    
    # Parse connection string
    parsed = urlparse(settings.DATABASE_URL.replace("+asyncpg", ""))
    
    print(f"  Connecting to: {parsed.hostname}:{parsed.port or 5432}/{parsed.path[1:]}...")
    
    conn = psycopg2.connect(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        database=parsed.path[1:] if parsed.path else 'ai_service',
        user=parsed.username or 'postgres',
        password=parsed.password or ''
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✓ PostgreSQL connected successfully")
    print(f"  Version: {version[0][:80]}...")
    
    cursor.close()
    conn.close()
    
except ImportError:
    print("  Installing psycopg2-binary for testing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
    print("  ✓ Installed. Please run this script again")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ PostgreSQL connection failed: {e}")
    print("\n  Troubleshooting:")
    print("  1. Is PostgreSQL running?")
    print("     - Windows: Check Services app for 'postgresql' service")
    print("     - Mac: Run 'brew services list'")
    print("     - Linux: Run 'sudo systemctl status postgresql'")
    print("  2. Is the database 'ai_service' created?")
    print("  3. Check username/password in .env file")
    print("  4. Try manually: psql -h localhost -U postgres -d ai_service")

# Test 3: Redis Connection
print("\n[3] Testing Redis connection...")
try:
    import redis
    from urllib.parse import urlparse
    
    # Parse Redis URL
    parsed = urlparse(settings.REDIS_URL)
    
    print(f"  Connecting to Redis at {parsed.hostname or 'localhost'}:{parsed.port or 6379}...")
    
    r = redis.Redis(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 6379,
        db=int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0,
        decode_responses=True
    )
    
    # Test connection
    response = r.ping()
    print(f"✓ Redis connected successfully")
    print(f"  Response: {response}")
    
    # Test set/get
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    print(f"  Read/Write test: {value}")
    r.delete('test_key')
    
except ImportError:
    print("  Installing redis for testing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "-q"])
    print("  ✓ Installed. Please run this script again")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Redis connection failed: {e}")
    print("\n  Troubleshooting:")
    print("  1. Is Redis running?")
    print("     - Windows: Check Services app for 'Memurai' or 'Redis' service")
    print("     - Mac: Run 'brew services list' or 'redis-cli ping'")
    print("     - Linux: Run 'sudo systemctl status redis-server'")
    print("  2. Try manually: redis-cli ping (should return PONG)")
    print("  3. Windows users: Install Memurai from https://www.memurai.com/")

# Test 4: Async SQLAlchemy (what the app actually uses)
print("\n[4] Testing async SQLAlchemy connection...")
try:
    from sqlalchemy.ext.asyncio import create_async_engine
    
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        future=True,
    )
    
    async def test_async_db():
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            return result.scalar()
    
    result = asyncio.run(test_async_db())
    print(f"✓ Async SQLAlchemy connection successful")
    print(f"  Test query result: {result}")
    
    asyncio.run(engine.dispose())
    
except Exception as e:
    print(f"✗ Async SQLAlchemy failed: {e}")
    print("\n  This might be OK if sync connection worked above")

# Test 5: Async Redis (what the app actually uses)
print("\n[5] Testing async Redis connection...")
try:
    import redis.asyncio as aioredis
    
    async def test_async_redis():
        client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        response = await client.ping()
        await client.close()
        return response
    
    response = asyncio.run(test_async_redis())
    print(f"✓ Async Redis connection successful")
    print(f"  Response: {response}")
    
except Exception as e:
    print(f"✗ Async Redis failed: {e}")
    print("\n  This might be OK if sync connection worked above")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nIf PostgreSQL and Redis (tests 2 & 3) passed, you're good to go!")
print("The async tests (4 & 5) use the actual libraries your app will use.")
print("\nNext steps:")
print("1. Create database tables: python scripts/create_tables.py")
print("2. Run the app: python -m app.main")
print("3. Open: http://localhost:8000/docs")
print("=" * 60)