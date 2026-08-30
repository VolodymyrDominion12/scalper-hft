"""Ідемпотентні client order id для live-ордерів (pairs ніколи market)."""

from __future__ import annotations

import uuid


def next_client_order_id(prefix: str = "sh") -> str:
    """Короткий унікальний id: `{prefix}-{16 hex}`. Binance ≤ 36 символів."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"
