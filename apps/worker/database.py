"""Synchronous SQLAlchemy session for Celery workers."""
from contextlib import contextmanager
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Worker uses sync driver — swap asyncpg → psycopg2
import os

_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://regintel:regintel@localhost:5432/regintel",
).replace("postgresql+asyncpg://", "postgresql+psycopg2://")

engine = create_engine(
    _db_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
