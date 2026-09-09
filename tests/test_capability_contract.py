"""Phase 5.2: capability contract стратегій — fail-fast замість тихої деградації."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.strategies import get_strategy
from scalper_hft.strategies.base import CAPABILITIES, MissingDataError


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
    assert get_strategy("sparse_basket").required_data() == frozenset({"ohlcv", "basket"})
    assert get_strategy("cross_momentum").required_data() == frozenset({"ohlcv", "multi_symbol"})
    assert get_strategy("pairs_arb").required_data() == frozenset({"ohlcv", "multi_symbol"})
    assert get_strategy("market_maker").required_data() == frozenset({"ohlcv", "l2"})
    assert get_strategy("sparse_basket").requires <= CAPABILITIES
    assert get_strategy("cross_momentum").requires <= CAPABILITIES


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


def _wide_close(n: int = 120) -> pd.DataFrame:
    """OHLCV + три '{sym}_close' колонки для cross_momentum."""
    df = _ohlcv(n)
    df["AAA_close"] = df["close"]
    df["BBB_close"] = df["close"] * 0.5
    df["CCC_close"] = df["close"] * 1.2
    return df


def _basket(n: int = 120) -> pd.DataFrame:
    df = _ohlcv(n)
    return pd.DataFrame(
        {"A": df["close"], "B": df["close"] * 0.5, "C": df["close"] * 0.3},
        index=df.index,
    )


def test_sparse_basket_without_basket_fails_fast() -> None:
    """Без кошика — MissingDataError, не односерійний z-score MR."""
    with pytest.raises(MissingDataError, match="basket"):
        run_backtest(_ohlcv(), get_strategy("sparse_basket"))


def test_sparse_basket_with_basket_runs() -> None:
    df = _ohlcv()
    res = run_backtest(df, get_strategy("sparse_basket"), basket_df=_basket())
    assert len(res.equity) == len(df)


def test_sparse_basket_strict_data_false_keeps_fallback() -> None:
    """Свідомий обхід: fallback на single-series z-score лишається доступним."""
    res = run_backtest(_ohlcv(), get_strategy("sparse_basket"), strict_data=False)
    assert len(res.equity) == 120


def test_cross_momentum_single_symbol_fails_fast() -> None:
    """Один символ — MissingDataError, не time-series momentum proxy."""
    with pytest.raises(MissingDataError, match="multi_symbol"):
        run_backtest(_ohlcv(), get_strategy("cross_momentum"))


def test_cross_momentum_wide_frame_runs() -> None:
    df = _wide_close()
    res = run_backtest(df, get_strategy("cross_momentum", lookback=5, signal_smooth=1))
    assert len(res.equity) == len(df)


def test_pairs_arb_single_symbol_fails_fast() -> None:
    """pairs_arb без leg1/leg2 більше не повертає нулі як «результат»."""
    with pytest.raises(MissingDataError, match="multi_symbol"):
        run_backtest(_ohlcv(), get_strategy("pairs_arb"))


def test_pairs_arb_with_legs_runs() -> None:
    df = _ohlcv()
    df["leg1"] = df["close"]
    df["leg2"] = df["close"] * 0.05
    res = run_backtest(df, get_strategy("pairs_arb", lookback=30, regime_scale=False))
    assert len(res.equity) == len(df)


def test_market_maker_vector_path_requires_l2() -> None:
    """Векторний run_backtest(market_maker) без стакана — fail-fast (не нулі)."""
    with pytest.raises(MissingDataError, match="l2"):
        run_backtest(_ohlcv(), get_strategy("market_maker"))


def test_market_maker_with_imbalance_column_is_l2() -> None:
    df = _ohlcv()
    df["imbalance"] = 0.0
    res = run_backtest(df, get_strategy("market_maker"))
    assert len(res.equity) == len(df)


def test_supervisor_unions_child_requires() -> None:
    sup = get_strategy("regime_supervisor", strategies="cross_momentum,mean_reversion")
    assert "multi_symbol" in sup.requires
    with pytest.raises(MissingDataError, match="multi_symbol"):
        run_backtest(_ohlcv(), sup)
