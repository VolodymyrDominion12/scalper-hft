"""Спільні фікстури тестів: синтетичні OHLCV (жодної мережі/біржі).

Використання:
    def test_x(ohlcv): ...            # 300 барів 1m, детермінований seed
    def test_y(make_ohlcv): ...       # фабрика: make_ohlcv(500, freq="5min", seed=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def make_ohlcv():
    """Фабрика синтетичних OHLCV DataFrame (DatetimeIndex, UTC-naive)."""

    def _make(n: int = 300, freq: str = "1min", seed: int = 42, start: str = "2025-01-01") -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range(start, periods=n, freq=freq)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n))), index=idx)
        return pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.0005,
                "low": close * 0.9995,
                "close": close,
                "volume": rng.uniform(1.0, 100.0, n),
            },
            index=idx,
        )

    return _make


@pytest.fixture()
def ohlcv(make_ohlcv) -> pd.DataFrame:
    """300 барів 1m синтетики (детерміновано)."""
    return make_ohlcv(300)
