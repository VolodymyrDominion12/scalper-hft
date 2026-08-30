"""Cross-sectional momentum стратегія (Narang гл. 3).

Логіка: ранжуємо N символів за їх поточним return за `lookback` барів.
Лонгуємо top-`top_pct` і шортуємо bottom-`top_pct`. Сигнал +1/-1/0.

Переваги для портфеля pairs_arb:
    - Низька кореляція: pairs_arb = mean-reversion, cross-momentum = trend
    - Диверсифікація альфи без додаткового ринкового ризику (long/short)

Обмеження (single-symbol CLI):
    df повинен містити колонку 'close' або кілька колонок '{symbol}_close'.
    При одному символі стратегія повертає сигнал за time-series momentum
    (порівняно з rolling median cross-section).

Використання:
    uv run python -m scalper_hft.cli backtest --strategy cross_momentum --symbol BTCUSDT --interval 1h --days 90
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy


class CrossMomentum(Strategy):
    """Cross-sectional momentum: long top performers, short bottom.

    У single-symbol режимі порівнює symbol з rolling percentile (time-series proxy).
    У multi-symbol режимі (df містить '{sym}_close' колонки) ранжує всі symbols.
    """

    name = "cross_momentum"

    param_space = {
        "lookback": (5.0, 60.0, 5.0),
        "top_pct": (0.1, 0.4, 0.1),
        "signal_smooth": (1.0, 5.0, 1.0),
    }

    def __init__(
        self,
        lookback: int = 20,
        top_pct: float = 0.2,
        signal_smooth: int = 1,
    ) -> None:
        super().__init__(
            lookback=int(lookback),
            top_pct=float(top_pct),
            signal_smooth=int(signal_smooth),
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Генерує сигнал cross-sectional momentum.

        Якщо df має >= 3 колонок '{sym}_close' — справжній cross-sectional режим.
        Інакше — time-series momentum (одна ціна).
        """
        lookback = int(self.get("lookback", 20))
        top_pct = float(self.get("top_pct", 0.2))
        smooth = int(self.get("signal_smooth", 1))

        close_cols = [c for c in df.columns if c.endswith("_close")]
        if len(close_cols) >= 3:
            return self._cross_sectional(df[close_cols], lookback, top_pct, smooth)

        price = df["close"] if "close" in df.columns else df.iloc[:, 0]
        return self._time_series_momentum(price, lookback, top_pct, smooth)

    @staticmethod
    def _cross_sectional(
        prices: pd.DataFrame, lookback: int, top_pct: float, smooth: int
    ) -> pd.Series:
        """Справжній cross-sectional: ранжуємо symbols за returns."""
        ret = prices.pct_change(lookback)
        ranks = ret.rank(axis=1, pct=True)
        sig_matrix = np.where(ranks.values >= 1.0 - top_pct, 1.0,
                     np.where(ranks.values <= top_pct, -1.0, 0.0))
        first = pd.Series(sig_matrix[:, 0], index=prices.index, dtype=float)
        if smooth > 1:
            first = first.ewm(span=smooth, adjust=False).mean().round()
        return first.fillna(0.0).shift(1).fillna(0.0).clip(-1, 1).astype(int)

    @staticmethod
    def _time_series_momentum(
        close: pd.Series, lookback: int, top_pct: float, smooth: int
    ) -> pd.Series:
        """Time-series momentum: ret vs rolling percentile як proxy cross-section."""
        ret = close.pct_change(lookback).fillna(0.0)
        high_thresh = ret.rolling(lookback * 3, min_periods=lookback).quantile(1.0 - top_pct)
        low_thresh = ret.rolling(lookback * 3, min_periods=lookback).quantile(top_pct)
        sig = pd.Series(0.0, index=close.index)
        sig[ret > high_thresh] = 1.0
        sig[ret < low_thresh] = -1.0
        if smooth > 1:
            sig = sig.ewm(span=smooth, adjust=False).mean()
            sig = sig.apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
        return sig.fillna(0.0).shift(1).fillna(0.0).clip(-1, 1).astype(int)


__all__ = ["CrossMomentum"]
