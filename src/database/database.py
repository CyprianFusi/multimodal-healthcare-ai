"""
Database connection and initialization
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Generator
import logging

from config.settings import settings
from src.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and operations"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize database engine"""
        try:
            self.engine = create_engine(
                settings.get_database_url(),
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=settings.debug,
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def create_tables(self):
        """Create all database tables"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                conn.commit()
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        if not settings.is_development():
            raise Exception("Cannot drop tables in non-development environment")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_db(self) -> Generator[Session, None, None]:
        """Dependency for FastAPI"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    return db_manager.get_db()


def init_database():
    """Initialize database (create tables, extensions, etc.)"""
    logger.info("Initializing database...")
    db_manager.create_tables()
    logger.info("Database initialization complete")


def reset_database():
    """Reset database (development only)"""
    if not settings.is_development():
        raise Exception("Database reset only allowed in development")
    logger.warning("Resetting database...")
    db_manager.drop_tables()
    db_manager.create_tables()
    logger.info("Database reset complete")
