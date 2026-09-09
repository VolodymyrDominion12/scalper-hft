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
    if d1.filled or d2.filled:
        reason = "unfilled_partial"
    elif d1.reason == "unfilled_prob" or d2.reason == "unfilled_prob":
        reason = "unfilled_prob"
    else:
        reason = "unfilled_no_touch"
    return (
        FillDecision(False, d1.fill_price, reason),
        FillDecision(False, d2.fill_price, reason),
    )


def maker_fill_rng(seed: int | None = None) -> np.random.Generator:
    """Детермінований Generator для paper і backtest (той самий MAKER_FILL_SEED)."""
    if seed is None:
        from scalper_hft.config import get_settings

        seed = get_settings().maker_fill_seed
    return np.random.default_rng(int(seed))


def _jsonable_rng_state(obj: object) -> object:
    """JSON-нативні типи: numpy uint64 інакше ламає json.dumps або стає str."""
    if isinstance(obj, dict):
        return {str(k): _jsonable_rng_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_rng_state(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def dump_fill_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """Стан bit_generator, придатний для SQLite JSON snapshot."""
    dumped = _jsonable_rng_state(rng.bit_generator.state)
    if not isinstance(dumped, dict):
        raise TypeError("bit_generator.state має бути dict")
    return dumped


def load_fill_rng_state(state: dict[str, Any]) -> np.random.Generator:
    """Відновити Generator з dump_fill_rng_state (або json.loads того знімка)."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def vector_post_only_touched(side: str, limit: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Векторизований post_only_touched для масивів OHLC."""
    limit = np.asarray(limit, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if side == "buy":
        return low <= limit
    return high >= limit


def vector_fill_probability(
    side: str,
    limit: np.ndarray,
    mid: np.ndarray,
    k: float = 0.08,
) -> np.ndarray:
    """Векторизований fill_probability (NaN → 0)."""
    limit = np.asarray(limit, dtype=float)
    mid = np.asarray(mid, dtype=float)
    safe = np.where(mid > 0, mid, np.nan)
    if side == "buy":
        dist_bps = (mid - limit) / safe * 10_000.0
    else:
        dist_bps = (limit - mid) / safe * 10_000.0
    p = np.exp(-k * np.maximum(dist_bps, 0.0))
    return np.clip(p, 0.0, 1.0)


def vector_both_legs_filled(
    s1: str,
    s2: str,
    *,
    t1_buy: np.ndarray,
    t1_sell: np.ndarray,
    t2_buy: np.ndarray,
    t2_sell: np.ndarray,
    p1_buy: np.ndarray,
    p1_sell: np.ndarray,
    p2_buy: np.ndarray,
    p2_sell: np.ndarray,
    draws1: np.ndarray,
    draws2: np.ndarray,
) -> np.ndarray:
    """All-or-none fill mask для двох ніг (як both_or_neither + decide_fill)."""
    touch1 = t1_buy if s1 == "buy" else t1_sell
    touch2 = t2_buy if s2 == "buy" else t2_sell
    prob1 = p1_buy if s1 == "buy" else p1_sell
    prob2 = p2_buy if s2 == "buy" else p2_sell
    f1 = touch1 & (draws1 <= prob1)
    f2 = touch2 & (draws2 <= prob2)
    return f1 & f2


@dataclass(frozen=True)
class LeggingResolution:
    """Результат аналізу розсинхронізації виконання двох ніг."""

    action: str  # both_filled | neither | chase_leg2 | chase_leg1 | unwind_leg1 | unwind_leg2
    d1: FillDecision
    d2: FillDecision
    drift_bps: float = 0.0
    reason: str = ""
    leg1_maker: bool = True
    leg2_maker: bool = True


def resolve_legging(
    d1: FillDecision,
    d2: FillDecision,
    side1: str,
    side2: str,
    limit1: float,
    limit2: float,
    mid1: float,
    mid2: float,
    max_drift_bps: float = 10.0,
    mode: str = "chase",
) -> LeggingResolution:
    """Вирішує розсинхронізацію ніг при частковому виконанні лімітних ордерів.

    Якщо одна нога виконалась як maker, а інша зависла:
        - mode="chase": якщо відхилення ринку <= max_drift_bps, доганяємо другу ногу
          ринковим ордером (taker/IOC) по поточній mid-ціні;
        - mode="unwind" або drift > max_drift_bps: негайно закриваємо першу ногу.
    """
    if d1.filled and d2.filled:
        return LeggingResolution("both_filled", d1, d2, reason="both_filled", leg1_maker=True, leg2_maker=True)
    if (not d1.filled) and (not d2.filled):
        return LeggingResolution("neither", d1, d2, reason="neither_filled")

    # Leg 1 виконалась, Leg 2 — ні
    if d1.filled and not d2.filled:
        drift = (mid2 - limit2) / limit2 * 10_000.0 if side2 == "buy" else (limit2 - mid2) / limit2 * 10_000.0
        drift = float(max(drift, 0.0))
        if mode == "chase" and drift <= max_drift_bps:
            chased_d2 = FillDecision(True, mid2, f"chase_taker_drift_{drift:.1f}bps")
            return LeggingResolution(
                "chase_leg2",
                d1,
                chased_d2,
                drift_bps=drift,
                reason="chased_leg2",
                leg1_maker=True,
                leg2_maker=False,
            )
        return LeggingResolution(
            "unwind_leg1",
            d1,
            d2,
            drift_bps=drift,
            reason="drift_exceeded_or_unwind",
            leg1_maker=True,
            leg2_maker=True,
        )

    # Leg 2 виконалась, Leg 1 — ні
    drift = (mid1 - limit1) / limit1 * 10_000.0 if side1 == "buy" else (limit1 - mid1) / limit1 * 10_000.0
    drift = float(max(drift, 0.0))
    if mode == "chase" and drift <= max_drift_bps:
        chased_d1 = FillDecision(True, mid1, f"chase_taker_drift_{drift:.1f}bps")
        return LeggingResolution(
            "chase_leg1",
            chased_d1,
            d2,
            drift_bps=drift,
            reason="chased_leg1",
            leg1_maker=False,
            leg2_maker=True,
        )
    return LeggingResolution(
        "unwind_leg2",
        d1,
        d2,
        drift_bps=drift,
        reason="drift_exceeded_or_unwind",
        leg1_maker=True,
        leg2_maker=True,
    )
