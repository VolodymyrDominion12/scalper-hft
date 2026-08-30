"""Модель філла post-only (maker) на OHLC-барі.

Лімітний ордер стоїть на last close. Філл на наступному барі, якщо ціна
торкнулась рівня: buy — low ≤ limit, sell — high ≥ limit. Ціна філла = limit
(maker, без slippage). Якщо бар не торкнувся — unfilled (можна почекати
ще N барів, потім скасувати).

Обидві ноги пари перевіряються разом: філл лише all-or-none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def fill_probability(
    side: str,
    limit_price: float,
    mid: float,
    k: float = 0.08,
) -> float:
    """P(fill | touch) від відстані ліміту до mid (bps). На mid → 1.0."""
    if mid <= 0:
        return 0.0
    if side == "buy":
        dist_bps = (mid - limit_price) / mid * 10_000.0
    else:
        dist_bps = (limit_price - mid) / mid * 10_000.0
    return float(min(max(np.exp(-k * max(dist_bps, 0.0)), 0.0), 1.0))


def decide_fill(
    side: str,
    limit_price: float,
    high: float,
    low: float,
    mid: float | None = None,
    rng: np.random.Generator | None = None,
    k: float = 0.08,
) -> FillDecision:
    """OHLC-touch + ймовірність філу від distance-to-mid.

    Без mid (або mid=limit) P=1 після touch — сумісно з наявними тестами.
    """
    if not post_only_touched(side, limit_price, high, low):
        return FillDecision(False, limit_price, "unfilled_no_touch")
    mid_px = float(mid) if mid is not None else float(limit_price)
    p = fill_probability(side, limit_price, mid_px, k=k)
    draw = 1.0 if rng is None else float(rng.random())
    if draw <= p:
        return FillDecision(True, limit_price, "filled")
    return FillDecision(False, limit_price, "unfilled_prob")


def both_or_neither(d1: FillDecision, d2: FillDecision) -> tuple[FillDecision, FillDecision]:
    """Не допускаємо одноногу позицію: філл лише якщо обидві ноги торкнулись."""
    if d1.filled and d2.filled:
        return d1, d2
    reason = "unfilled_partial" if (d1.filled or d2.filled) else "unfilled_no_touch"
    return (
        FillDecision(False, d1.fill_price, reason),
        FillDecision(False, d2.fill_price, reason),
    )
