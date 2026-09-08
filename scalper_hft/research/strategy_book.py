"""Книга стратегій для дашборду — консолідований дослідницький статус.

Джерело істини для висновків: `docs/STRATEGY_STATUS.md`. Цей модуль —
структурована копія для UI (не парсимо markdown у рантаймі).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StrategyLane = Literal["validated", "research", "rejected", "waiting"]

_LANE_LABEL: dict[StrategyLane, str] = {
    "validated": "валідована",
    "research": "дослідження",
    "rejected": "відхилена",
    "waiting": "чекає даних",
}


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    """Один рядок книги: стратегія (опційно пара) і дослідницький вердикт."""

    name: str
    lane: StrategyLane
    headline: str
    conditions: str
    pair: str = ""


# Порядок = пріоритет сканування дослідником (валідовані зверху).
# Перевалідація 2026-09-07 (поточний код, 3y 1h maker, 49 WF-вікон):
# документовані 2026-08-30 цифри отримані до фіксів моделі виконання.
STRATEGY_BOOK: tuple[StrategyRecord, ...] = (
    StrategyRecord(
        "pairs_arb",
        "validated",
        "LINK/BTC 3y +55% (PF 1.95; z2/lb120: +86%, WF pos 71%)",
        "1h, maker, z=2.0 / lb=120–240",
        pair="LINKUSDT/BTCUSDT",
    ),
    StrategyRecord(
        "pairs_arb",
        "research",
        "XRP/BTC 3y <0 на поточному коді; позитивний лише у свіжому 400д-вікні",
        "моніторинг (повернути в ядро, якщо 3y знову позитивний)",
        pair="XRPUSDT/BTCUSDT",
    ),
    StrategyRecord(
        "pairs_arb",
        "research",
        "BTC/ETH та LINK/ETH на 3y від'ємні після фіксів виконання",
        "research, maker",
        pair="LINKUSDT/ETHUSDT",
    ),
    StrategyRecord(
        "sparse_basket",
        "research",
        "Lasso + PCA кошик; multi-asset контур не підключений до рушія — "
        "backtest = single-series z-score MR fallback (попередження в коді)",
        "мульти-актив, maker",
    ),
    StrategyRecord(
        "ensemble",
        "research",
        "Hedge / voting / regime-gating суб-моделей",
        "мета-модель",
    ),
    StrategyRecord(
        "ml_strategy",
        "research",
        "LightGBM meta-labeling; обов'язковий DSR",
        "triple-barrier + bet sizing",
    ),
    StrategyRecord(
        "supertrend",
        "rejected",
        "1d: OOS +0.50 на 3y, але −0.15 на 5y (2021–26) — edge період-специфічний; 1h/4h — від'ємні після комісій",
        "трендовий sleeve",
    ),
    StrategyRecord(
        "regime_supervisor",
        "research",
        "не б'є кращий сингл на 1h (iter1–3); корисний як regime-шар експозиції",
        "мета-модель, 1h+",
    ),
    StrategyRecord(
        "mean_reversion",
        "rejected",
        "OOS Sharpe < 0, DSR = 0",
        "regime-гейтована MR",
    ),
    StrategyRecord(
        "cvd_momentum",
        "rejected",
        "Fee-drag, PF 0.30 — надто часто на 1m",
        "1m taker-скальп",
    ),
    StrategyRecord(
        "funding_carry",
        "rejected",
        "Тертя > фандінг у режимі 2025–26",
        "перп funding",
    ),
    StrategyRecord(
        "funding_arb",
        "rejected",
        "Фандінг структурно низький (мало періодів >36% річних)",
        "перп + спот",
    ),
    StrategyRecord(
        "basis_reversion",
        "rejected",
        "Базис 1–2.4 bps проти 2-leg витрат",
        "перп vs спот",
    ),
    StrategyRecord(
        "hmm_reversion",
        "rejected",
        "Гейт майже вимикає входи; OOS SR < 0, DSR = 0",
        "HMM-гейтована MR",
    ),
    StrategyRecord(
        "ob_imbalance",
        "waiting",
        "Потрібен тривалий depth5 для depth-weighted imbalance",
        "L2",
    ),
    StrategyRecord(
        "market_maker",
        "waiting",
        "Потрібні L2 для черги та adverse selection",
        "passive quotes",
    ),
)


def lane_label(lane: StrategyLane) -> str:
    """Людська назва смуги статусу."""
    return _LANE_LABEL[lane]


def lane_for(name: str) -> StrategyLane:
    """Смуга стратегії; невідомі імена вважаємо дослідженням."""
    for rec in STRATEGY_BOOK:
        if rec.name == name:
            return rec.lane
    return "research"


def select_label(name: str) -> str:
    """Підпис для selectbox: ім'я + смуга без емодзі."""
    return f"{name} · {lane_label(lane_for(name))}"


def book_as_rows() -> list[dict[str, str]]:
    """Рядки для таблиці дашборду."""
    rows: list[dict[str, str]] = []
    for rec in STRATEGY_BOOK:
        rows.append(
            {
                "strategy": rec.name,
                "pair": rec.pair,
                "lane": lane_label(rec.lane),
                "lane_key": rec.lane,
                "headline": rec.headline,
                "conditions": rec.conditions,
            }
        )
    return rows
