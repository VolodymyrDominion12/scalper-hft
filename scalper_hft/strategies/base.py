"""Альфа-моделі (книга, гл. 3 — Alpha Models).

Єдиний інтерфейс: стратегія приймає DataFrame з фічами і повертає Series
бажаної позиції в [-1, 0, +1] (або цільову частку капіталу в [-1, 1]).

Конвенції:
    - сигнал обчислюється на закритті бару t;
    - позиція діє з бару t+1 (рушій робить shift(1)): PnL бару t+1 =
      pos_t × close-to-close дохідність бару t+1; філ-модель рушія —
      ціна close бару t (див. backtest/engine.py);
    - 0 = поза ринком; +/-1 = лонг/шорт (або частка для часткових позицій).

Примітка: Strategy.exit_levels() — лише для ВІЗУАЛІЗАЦІЇ (графіки SL/TP);
рушій не симулює внутрішньобарове спрацювання стопів за цими рівнями.

Filter Tracing:
    generate_signals_traced() — розширена версія, яка повертає (signals, FilterTrace).
    За замовчуванням — заглушка (порожній трейс). Стратегії можуть перевизначити
    для детального аналізу, які фільтри скільки сигналів відкидають.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import pandas as pd

from scalper_hft.strategies.taxonomy import AlphaFamily, PreferredRegime

if TYPE_CHECKING:
    from scalper_hft.research.filter_trace import FilterTrace


class Strategy(abc.ABC):
    """Базовий клас стратегії."""

    name: str = "base"
    # опис параметрів для оптимізації: {name: (lo, hi, step)} — для Optuna та grid
    param_space: dict[str, tuple[float, float, float]] = {}
    # чи потребує стратегія тікових даних (aggTrades) для фіч
    needs_trades: bool = False
    # чи потребує стратегія історії фандінгу
    needs_funding: bool = False
    # якщо True — рушій вимикає сигнали, де очікуваний рух < round-trip витрат
    # (Narang гл. 5: edge має покривати транзакційні витрати)
    use_breakeven_gate: bool = False
    # сім'я альфи (Narang гл. 3); preferred_regimes — гіпотеза, не live-гейт.
    # порожній frozenset = усі режими (вага 1.0), поки OOS це не спростує.
    family: AlphaFamily = "meta"
    preferred_regimes: frozenset[PreferredRegime] = frozenset()

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        if "breakeven_gate" in params:
            self.use_breakeven_gate = bool(params["breakeven_gate"])

    def get(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    @abc.abstractmethod
    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Повертає Series позицій, індексовану як df.index."""

    def generate_signals_traced(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, FilterTrace]:
        """Розширена версія: повертає (signals, FilterTrace).

        За замовчуванням викликає generate_signals() і повертає порожній FilterTrace.
        Стратегії можуть перевизначити цей метод для запису деталей кожного
        заблокованого сигналу (причина, контекст). Не порушує lookahead-правило:
        трейс записується на закритому барі t, лише post-factum.
        """
        from scalper_hft.research.filter_trace import FilterTrace

        signals = self.generate_signals(df, trades=trades, funding=funding)
        return signals, FilterTrace()

    def exit_levels(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Опційні рівні SL/TP для візуалізації угод (ціни).

        Повертає DataFrame, індексований як df, з колонками:
            sl_long, tp_long, sl_short, tp_short — рівні стоп-лосу та
        тейк-профіту для лонга/шорта на кожному барі. Рушій бектесту бере
        рівні на барі входу угоди за її стороною і кладе у trades
        (`sl_price`/`tp_price`). None — стратегія не має явних рівнів,
        візуалізація просто не малює SL/TP.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
