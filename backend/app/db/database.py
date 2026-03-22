"""
Vita AI – Database Configuration

SQLAlchemy setup for persisting user profiles and health scan results.
Uses SQLite for local development; swap DATABASE_URL for PostgreSQL in production.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# database.py lives at backend/app/db/database.py → parents[2] = backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_URL = f"sqlite:///{_BACKEND_DIR / 'data' / 'vita.db'}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and auto-closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
