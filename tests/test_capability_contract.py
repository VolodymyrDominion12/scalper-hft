"""Phase 5.2: capability contract стратегій — fail-fast замість тихої деградації."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.strategies import get_strategy
from scalper_hft.strategies.base import MissingDataError


def _ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0},
        index=idx,
    )


def _trades(idx: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    return pd.DataFrame(
        {
            "price": 100.0,
            "amount": np.abs(rng.normal(1.0, 0.3, len(idx))),
            "side": np.where(rng.random(len(idx)) > 0.5, "buy", "sell"),
        },
        index=idx,
    )


def test_required_data_reflects_needs_flags() -> None:
    assert get_strategy("mean_reversion").required_data() == frozenset({"ohlcv"})
    assert get_strategy("cvd_momentum").required_data() == frozenset({"ohlcv", "trades"})
    assert get_strategy("funding_carry").required_data() == frozenset({"ohlcv", "funding"})


def test_missing_trades_fails_fast() -> None:
    """needs_trades стратегія без aggTrades — MissingDataError, не нульовий сигнал."""
    with pytest.raises(MissingDataError, match="trades"):
        run_backtest(_ohlcv(), get_strategy("cvd_momentum"))


def test_missing_funding_fails_fast() -> None:
    """needs_funding стратегія без funding — MissingDataError."""
    with pytest.raises(MissingDataError, match="funding"):
        run_backtest(_ohlcv(), get_strategy("funding_carry"))


def test_contract_satisfied_runs() -> None:
    df = _ohlcv()
    res = run_backtest(df, get_strategy("cvd_momentum"), trades=_trades(df.index))
    assert len(res.equity) == len(df)


def test_strict_data_false_is_conscious_escape_hatch() -> None:
    """strict_data=False — свідомий обхід (відтворення старих прогонів)."""
    res = run_backtest(_ohlcv(), get_strategy("cvd_momentum"), strict_data=False)
    assert len(res.equity) == 120


def test_precomputed_signals_skip_validation() -> None:
    """Walk-forward ML шлях передає готові сигнали — дані вже перевірені вище."""
    df = _ohlcv()
    signals = pd.Series(0, index=df.index)
    res = run_backtest(df, get_strategy("cvd_momentum"), signals=signals)
    assert len(res.equity) == len(df)
