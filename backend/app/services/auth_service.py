"""
Vita AI – Authentication Service (JWT + bcrypt)

Code contributed by Manogna — adapted from api.py to fit Vita AI's
service layer. Replaces the previous SHA-256 password hashing with
bcrypt and adds JWT token-based authentication.

Usage in endpoints:
    from backend.app.services.auth_service import (
        hash_password, verify_password, validate_password,
        create_access_token, get_current_user,
    )
"""

import os
import re
from datetime import datetime, timedelta
from typing import Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Code contributed by Manogna ──────────────────────────────────────────

# bcrypt password hashing — more secure than SHA-256
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a bcrypt *hashed_password*."""
    return pwd_context.verify(plain_password, hashed_password)


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength.

    Returns ``(True, "")`` on success, or ``(False, reason)`` on failure.
    """
    allowed = re.compile(r'^[A-Za-z0-9@#$%^&+=!_\-\.]*$')
    if not allowed.match(password):
        return False, "Password contains unsupported characters. Use letters, numbers, or _ - . @ # $ % ^ & + = !"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if len(password) > 20:
        return False, "Password too long (max 20 characters)"
    return True, ""


# ── JWT configuration ────────────────────────────────────────────────────

# In production, set VITA_JWT_SECRET in the environment
SECRET_KEY = os.getenv("VITA_JWT_SECRET", "vita-ai-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 2


def create_access_token(data: dict) -> str:
    """Create a signed JWT with an expiry claim."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ── FastAPI dependency for protected routes ──────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Decode a JWT and return the user email (``sub`` claim).

    Raise HTTP 401 if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
