"""Фічі та фільтри режиму ринку (книга, гл. 4 — Risk Models; гл. 10 — Regime Risk).

Скальпінг-стратегії чутливі до режиму: у тренді працює моментум, у флеті —
mean-reversion. Фільтри режиму відсікають несприятливі стани і тим самим
зменшують кількість збиткових трейдів (покращують win rate та PF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_regime(close: pd.Series, lookback: int = 60, percentile_window: int = 500) -> pd.Series:
    """Режим волатильності: low / normal / high за процентилем останньої реалізованої волатильності."""
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(lookback, min_periods=lookback // 2).std()
    pct = rv.rolling(percentile_window, min_periods=percentile_window // 2).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    regime = pd.Series("normal", index=close.index, dtype=object)
    regime[pct > 0.8] = "high"
    regime[pct < 0.2] = "low"
    return regime


def trend_strength(close: pd.Series, ema_fast: int = 9, ema_slow: int = 50) -> pd.Series:
    """Міра сили тренду в [0, 1]: нормалізована відстань між EMA."""
    f = close.ewm(span=ema_fast, adjust=False).mean()
    s = close.ewm(span=ema_slow, adjust=False).mean()
    dist = (f - s).abs() / close.rolling(ema_slow, min_periods=ema_slow).std().replace(0, np.nan)
    return dist.clip(upper=1.0).fillna(0.0)


def funding_filter(funding: pd.DataFrame, max_abs_rate: float = 0.001) -> pd.Series:
    """Маска, коли фандінг у межах норми (не перегрітий ринок перекосів)."""
    if funding is None or funding.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=bool)
    return funding["fundingRate"].abs() < max_abs_rate


def session_filter(index: pd.DatetimeIndex, start_hour: int = 0, end_hour: int = 24) -> pd.Series:
    """Маска активної сесії (години UTC). Скальпінг — лише в ліквідні години."""
    hours = index.hour
    if start_hour <= end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)
