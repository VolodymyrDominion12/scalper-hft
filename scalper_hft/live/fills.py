"""Модель філла post-only (maker) на OHLC-барі.

Лімітний ордер стоїть на last close. Філл на наступному барі, якщо ціна
торкнулась рівня: buy — low ≤ limit, sell — high ≥ limit. Ціна філла = limit
(maker, без slippage). Якщо бар не торкнувся — unfilled (можна почекати
ще N барів, потім скасувати).

Обидві ноги пари перевіряються разом: філл лише all-or-none.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillDecision:
    filled: bool
    fill_price: float
    reason: str  # filled | unfilled_no_touch | timeout | pending


def post_only_touched(side: str, limit_price: float, high: float, low: float) -> bool:
    """Чи бар перетнув наш ліміт. side: buy | sell."""
    if side == "buy":
        return low <= limit_price
    return high >= limit_price


def decide_fill(side: str, limit_price: float, high: float, low: float) -> FillDecision:
    if post_only_touched(side, limit_price, high, low):
        return FillDecision(True, limit_price, "filled")
    return FillDecision(False, limit_price, "unfilled_no_touch")


def both_or_neither(d1: FillDecision, d2: FillDecision) -> tuple[FillDecision, FillDecision]:
    """Не допускаємо одноногу позицію: філл лише якщо обидві ноги торкнулись."""
    if d1.filled and d2.filled:
        return d1, d2
    reason = "unfilled_partial" if (d1.filled or d2.filled) else "unfilled_no_touch"
    return (
        FillDecision(False, d1.fill_price, reason),
        FillDecision(False, d2.fill_price, reason),
    )
