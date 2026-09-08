"""Автентифікація для FastAPI: створення і перевірка JWT."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from scalper_hft.config import get_settings

security = HTTPBearer()

ALGORITHM = "HS256"

# Ефемерний ключ для localhost-dev, коли API_SECRET_KEY не заданий: випадковий
# per-process (secrets), НЕ захардкоджений у репозиторії. Токени інвалідуються
# при рестарті API — прийнятно для dev; для будь-якого non-localhost бінду
# порожній ключ усе одно відхиляється require_safe_api_bind.
_EPHEMERAL_KEY: str | None = None


def _jwt_key(settings: Any) -> str:
    """Ефективний JWT-ключ: з settings або ефемерний випадковий (dev)."""
    global _EPHEMERAL_KEY
    key = (settings.api_secret_key or "").strip()
    if key:
        return key
    if _EPHEMERAL_KEY is None:
        import secrets

        _EPHEMERAL_KEY = secrets.token_hex(32)
    return _EPHEMERAL_KEY


def create_access_token(data: dict[str, Any], expires_delta_hours: int = 24) -> str:
    """Створює JWT токен для API."""
    settings = get_settings()
    to_encode = data.copy()
    expire = time.time() + (expires_delta_hours * 3600)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, _jwt_key(settings), algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict[str, Any]:
    """Залежність FastAPI для перевірки Bearer токена."""
    settings = get_settings()
    token = credentials.credentials
    try:
        payload = jwt.decode(token, _jwt_key(settings), algorithms=[ALGORITHM])
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


def require_api_capability(
    *,
    mock_positions: bool = False,
    destructive: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Fail-closed: mock/destructive API лише за явним .env."""

    def _check(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
        settings = get_settings()
        if mock_positions and not settings.api_allow_mock_positions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Mock positions disabled (set API_ALLOW_MOCK_POSITIONS=true in .env)",
            )
        if destructive and not settings.api_allow_destructive_ops:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Destructive API disabled (set API_ALLOW_DESTRUCTIVE_OPS=true in .env)",
            )
        return token_payload

    return _check
