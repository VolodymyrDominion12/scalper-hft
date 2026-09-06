"""Політика клітинки (strategy × symbol × interval) — без pandas.

Narang: альфа лишається гіпотезою; витрати, час утримання і enablement —
окремий шар. Overlay не підганяє RSI/ATR під тікер.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExecutionMode = Literal["maker", "taker"]
EnabledWhen = Literal["always", "funding_regime"]


@dataclass(frozen=True, slots=True)
class CellMatch:
    """Критерій правила. Порожній frozenset + wildcard_* = «будь-що»."""

    strategy: str
    intervals: frozenset[str]
    symbols: frozenset[str]
    clusters: frozenset[str]
    wildcard_strategy: bool
    wildcard_interval: bool
    wildcard_symbol: bool


@dataclass(frozen=True, slots=True)
class CellPolicy:
    """Політика, яку overlay накладає на вже згенерований сигнал."""

    enabled: bool = True
    allow_short: bool = True
    execution: ExecutionMode = "maker"
    min_hold_hours: float = 0.0
    max_trades_per_day: float | None = None
    cost_gate: bool = False
    enabled_when: EnabledWhen = "always"
    funding_annual_min: float = 0.10
    size_mult: float = 1.0

    def strategy_kwargs(self) -> dict[str, bool]:
        """Параметри конструктора стратегії. cost_gate застосовує рушій, не __init__."""
        return {"allow_short": self.allow_short}


@dataclass(frozen=True, slots=True)
class CellRule:
    match: CellMatch
    policy: CellPolicy


@dataclass(frozen=True, slots=True)
class OverlayBook:
    """Набір кластерів + правила. Резолвер бере найспецифічніше збіг."""

    clusters: dict[str, frozenset[str]]
    defaults: CellPolicy
    rules: tuple[CellRule, ...]
