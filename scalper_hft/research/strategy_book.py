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
STRATEGY_BOOK: tuple[StrategyRecord, ...] = (
    StrategyRecord(
        "pairs_arb",
        "validated",
        "+15.9%/рік, 28 угод, maxDD −6.5%, WF OOS>0",
        "1h, maker, z=2.0 / lb=480",
        pair="XRPUSDT/BTCUSDT",
    ),
    StrategyRecord(
        "pairs_arb",
        "validated",
        "+12.9%/рік, 46 угод, maxDD −4.6%",
        "1h, maker, z=2.0 / lb=240",
        pair="LINKUSDT/BTCUSDT",
    ),
    StrategyRecord(
        "pairs_arb",
        "validated",
        "+14.9%/рік, 51 угода, maxDD −5.7%",
        "1h, maker, z=2.0 / lb=240",
        pair="LINKUSDT/ETHUSDT",
    ),
    StrategyRecord(
        "pairs_arb",
        "validated",
        "+8.3%/рік, 47 угод, maxDD −4%",
        "1h, maker, z=2.0 / lb=240",
        pair="BTCUSDT/ETHUSDT",
    ),
    StrategyRecord(
        "sparse_basket",
        "research",
        "Lasso + PCA кошик; потрібен OOS на 5+ монетах",
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
