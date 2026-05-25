"""SQLModel engine + session factory."""
from __future__ import annotations

import os
from typing import Generator

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./treasury.db")

# check_same_thread=False is required for SQLite + FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def init_db() -> None:
    """Create all tables defined on imported SQLModel subclasses."""
    # Import models so SQLModel.metadata sees them before create_all.
    from db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a SQLModel Session per request."""
    with Session(engine) as session:
        yield session
