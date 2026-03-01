import os
import sys
from urllib.parse import urlparse

# Print Python version and system info
print(f"Python version: {sys.version}")
print(f"System encoding: {sys.getdefaultencoding()}")
print(f"Filesystem encoding: {sys.getfilesystemencoding()}")
print(f"Platform: {sys.platform}")

# Your database URL from .env
db_url = "postgresql+asyncpg://postgres:admin@localhost:5432/ai_service"
print(f"\nDatabase URL: {db_url}")

# Parse it
parsed = urlparse(db_url.replace("+asyncpg", ""))
print(f"Parsed connection:")
print(f"  Host: {parsed.hostname}")
print(f"  Port: {parsed.port or 5432}")
print(f"  Database: {parsed.path[1:] if parsed.path else 'ai_service'}")
print(f"  User: {parsed.username}")
print(f"  Password: {'*' * len(parsed.password) if parsed.password else 'None'}")

# Try to decode the password to see if it has special chars
print(f"\nPassword bytes: {parsed.password.encode('ascii', errors='replace')}")

# Try connection with different encodings
import psycopg2

encodings_to_try = ['UTF8', 'LATIN1', 'WIN1252', 'SQL_ASCII']

for encoding in encodings_to_try:
    print(f"\n--- Trying with client_encoding={encoding} ---")
    try:
        conn = psycopg2.connect(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            database=parsed.path[1:] if parsed.path else 'ai_service',
            user=parsed.username or 'postgres',
            password=parsed.password or '',
            options=f'-c client_encoding={encoding}'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ SUCCESS with {encoding}!")
        print(f"Version: {version[0][:50]}...")
        cursor.close()
        conn.close()
        break
    except Exception as e:
        print(f"❌ Failed with {encoding}: {e}")
        
        # Check if it's the same error
        if "0xe9" in str(e):
            print("   ⚠️  Same encoding error (0xe9 detected)")