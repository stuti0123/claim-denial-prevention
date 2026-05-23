"""
src/core/database.py
--------------------
Lightweight SQLite database for demo/production.

- No separate database server needed (SQLite = single .db file).
- Stores: users (with roles) and prediction history.
- In a full cloud deployment, swap DATABASE_URL to PostgreSQL — zero other changes.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# --------------------------------------------------------------------------- #
#  Engine                                                                       #
# --------------------------------------------------------------------------- #

# Defaults to local SQLite file; override with DATABASE_URL env var for
# PostgreSQL in production (e.g. "postgresql://user:pass@host/db")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./claimops.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --------------------------------------------------------------------------- #
#  Base & Models                                                                #
# --------------------------------------------------------------------------- #

class Base(DeclarativeBase):
    pass


class User(Base):
    """Demo users table. Passwords are bcrypt-hashed."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False, default="billing_clerk")
    # roles: billing_clerk | billing_admin


class PredictionHistory(Base):
    """Stores every claim evaluation so billing staff can review history."""
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    claim_id = Column(String(50), nullable=False)
    patient_id = Column(String(50))
    provider_id = Column(String(50))
    billed_amount = Column(Float)
    diagnosis_code = Column(String(20))
    procedure_code = Column(String(20))
    denial_probability = Column(Float)
    risk_level = Column(String(20))
    is_denied = Column(Integer)  # 0 or 1
    flags = Column(Text)         # pipe-separated flags e.g. "WARN_HIGH_BILLING|ERR_INCOMPLETE"
    submitted_at = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and seed demo users on first run."""
    Base.metadata.create_all(bind=engine)
    _seed_demo_users()


def _seed_demo_users() -> None:
    """Insert demo users (admin + clerk) if they don't already exist."""
    # Import here to avoid circular imports at module load time
    from passlib.context import CryptContext  # type: ignore

    pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

    demo_users = [
        {"username": "admin",       "password": "admin123",  "role": "billing_admin"},
        {"username": "clerk",       "password": "clerk123",  "role": "billing_clerk"},
    ]

    db = SessionLocal()
    try:
        for u in demo_users:
            exists = db.query(User).filter(User.username == u["username"]).first()
            if not exists:
                db.add(User(
                    username=u["username"],
                    hashed_password=pwd_ctx.hash(u["password"]),
                    role=u["role"],
                ))
        db.commit()
    finally:
        db.close()
