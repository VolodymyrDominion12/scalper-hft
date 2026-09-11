"""Buy-and-hold benchmark для порівняння зі стратегією (overfitting-audit skill)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_returns(close: pd.Series) -> pd.Series:
    """Дохідності buy-and-hold (long 1 unit, lag-1 execution)."""
    ret = close.pct_change().fillna(0.0)
    return ret.shift(1).fillna(0.0)


def sharpe_from_returns(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Annualized Sharpe з пербарних дохідностей."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=0))
    if sd <= 1e-12:
        return 0.0
    ann = float(periods_per_year) ** 0.5
    return float(r.mean() / sd * ann)


def buy_and_hold_sharpe(
    df: pd.DataFrame,
    *,
    periods_per_year: int = 8760,
) -> float:
    """Sharpe buy-and-hold на тому ж OHLCV-вікні."""
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")
    return sharpe_from_returns(buy_and_hold_returns(df["close"]), periods_per_year=periods_per_year)


def strategy_beats_benchmark(strategy_sharpe: float, benchmark_sharpe: float) -> bool:
    """Стратегія краща за B&H (обидва finite)."""
    if not np.isfinite(strategy_sharpe) or not np.isfinite(benchmark_sharpe):
        return False
    return strategy_sharpe > benchmark_sharpe
