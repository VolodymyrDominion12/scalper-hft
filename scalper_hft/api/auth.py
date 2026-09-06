"""Автентифікація для FastAPI: створення і перевірка JWT."""

import time
from typing import Any

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from scalper_hft.config import get_settings

security = HTTPBearer()

ALGORITHM = "HS256"


def create_access_token(data: dict[str, Any], expires_delta_hours: int = 24) -> str:
    """Створює JWT токен для API."""
    settings = get_settings()
    to_encode = data.copy()
    expire = time.time() + (expires_delta_hours * 3600)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.api_secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict[str, Any]:
    """Залежність FastAPI для перевірки Bearer токена."""
    settings = get_settings()
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.api_secret_key, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
