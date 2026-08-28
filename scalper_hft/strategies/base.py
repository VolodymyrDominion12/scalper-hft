"""Альфа-моделі (книга, гл. 3 — Alpha Models).

Єдиний інтерфейс: стратегія приймає DataFrame з фічами і повертає Series
бажаної позиції в [-1, 0, +1] (або цільову частку капіталу в [-1, 1]).

Конвенції:
    - сигнал обчислюється на закритті бару t;
    - виконання відбувається на відкритті бару t+1 (без lookahead);
    - 0 = поза ринком; +/-1 = лонг/шорт (або частка для часткових позицій).
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd


class Strategy(abc.ABC):
    """Базовий клас стратегії."""

    name: str = "base"
    # опис параметрів для оптимізації: {name: (lo, hi, step)} — для Optuna та grid
    param_space: dict[str, tuple[float, float, float]] = {}
    # чи потребує стратегія тікових даних (aggTrades) для фіч
    needs_trades: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    def get(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Повертає Series позицій, індексовану як df.index."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
