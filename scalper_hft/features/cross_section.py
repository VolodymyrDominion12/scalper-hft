"""Cross-sectional feature matrix (Phase 2D).

Єдина universe-wide feature matrix для cross-sectional стратегій
(cross_momentum, sparse_basket): rank та z-score кожного символу відносно
інших на кожному барі. Каузально: фічі на барі t використовують лише дані ≤ t.

Раніше cross_momentum мав власний inline rank; тепер — єдина матриця, яку
можна переиспользовати і для factor stacking (2D).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_rank(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Ранг кожного символу у крос-секції на кожному барі (каузально).

    Args:
        prices: (T × N) ціни закриття N символів.
        lookback: вікно для розрахунку return.

    Returns:
        (T × N) DataFrame рангів у [0, 1] (pct-rank: 1 = найкращий).
    """
    ret = prices.pct_change(lookback)
    return ret.rank(axis=1, pct=True)


def cross_section_zscore(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Z-score кожного символу відносно крос-секції на кожному барі (каузально).

    z_{i,t} = (ret_{i,t} − mean_t) / std_t. Позитивне = сильніший за всесвіт.

    Returns:
        (T × N) DataFrame z-score.
    """
    ret = prices.pct_change(lookback)
    mu = ret.mean(axis=1)
    sd = ret.std(axis=1).replace(0.0, np.nan)
    z = ret.sub(mu, axis=0).div(sd, axis=0)
    return z.fillna(0.0)


def cross_section_features(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Єдина cross-sectional feature matrix: rank + z-score (2D).

    Повертає DataFrame з колонками-фічами для кожного символу:
        {sym}_rank, {sym}_zscore. Індексований як prices.
    Використовується cross_momentum/sparse_basket та factor stacking.
    """
    rank = cross_section_rank(prices, lookback)
    z = cross_section_zscore(prices, lookback)
    rank.columns = [f"{c}_rank" for c in rank.columns]
    z.columns = [f"{c}_zscore" for c in z.columns]
    return pd.concat([rank, z], axis=1)


__all__ = ["cross_section_rank", "cross_section_zscore", "cross_section_features"]
