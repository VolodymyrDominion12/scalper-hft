"""Грошова арифметика для paper/live обліку (Decimal, без float drift)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# 8 знаків після коми — достатньо для USDT-M crypto sizing/fees
_QUANT = Decimal("0.00000001")


def to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Конвертувати у Decimal через str (без двійкового drift float)."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def quantize(value: Decimal) -> Decimal:
    """Округлити до стандартної точності проєкту."""
    return value.quantize(_QUANT, rounding=ROUND_HALF_UP)


def money(value: float | int | str | Decimal) -> Decimal:
    """Нормалізована грошова/кількісна величина для обліку."""
    return quantize(to_decimal(value))


def as_float(value: Decimal) -> float:
    """Межа з Decimal → float (REST, CSV, snapshots)."""
    return float(value)
