"""
src/api/auth.py
---------------
JWT Authentication & Role-Based Access Control (RBAC).

Demo credentials:
  username: admin    password: admin123   role: billing_admin
  username: clerk    password: clerk123   role: billing_clerk
"""

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt  # type: ignore
from passlib.context import CryptContext  # type: ignore
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.core.database import User, get_db

# --------------------------------------------------------------------------- #
#  Config                                                                       #
# --------------------------------------------------------------------------- #

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "claimops-demo-secret-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")


# --------------------------------------------------------------------------- #
#  Schemas                                                                      #
# --------------------------------------------------------------------------- #

class TokenData(BaseModel):
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str


# --------------------------------------------------------------------------- #
#  Core Auth Functions                                                          #
# --------------------------------------------------------------------------- #

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """Return User if credentials are valid, else None."""
    user: User | None = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# --------------------------------------------------------------------------- #
#  FastAPI Dependency — get logged-in user from JWT                             #
# --------------------------------------------------------------------------- #

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user: User | None = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Restrict endpoint to billing_admin role only."""
    if current_user.role != "billing_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
