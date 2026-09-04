"""Ensemble mode=regime: м'які ваги капіталу, не quorum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.features.regimes import market_structure
from scalper_hft.strategies import get_strategy
from scalper_hft.strategies.ensemble import regime_blend_signals, regime_blend_weights


def _uptrend_df(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = 100.0 + 0.5 * np.arange(n, dtype=float)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )


def test_regime_blend_matches_mean_when_all_weights_one() -> None:
    df = _uptrend_df()
    sig = pd.DataFrame({"a": 1.0, "b": -0.5}, index=df.index)
    out = regime_blend_signals(
        sig,
        df["close"],
        [frozenset(), frozenset()],
        unfavorable=0.25,
    )
    pd.testing.assert_series_equal(out, sig.mean(axis=1), check_names=False)


def test_regime_blend_favors_momentum_in_uptrend() -> None:
    df = _uptrend_df()
    sig = pd.DataFrame({"mr": 1.0, "mom": -1.0}, index=df.index)
    preferred = [frozenset({"range"}), frozenset({"trend_up", "trend_down"})]
    out = regime_blend_signals(sig, df["close"], preferred, unfavorable=0.25)
    structure = market_structure(df["close"])
    up = structure == "trend_up"
    assert up.iloc[80:].mean() > 0.8
    # (0.25*1 + 1*(-1)) / 2 = -0.375 — momentum dominates
    assert float(out.loc[up].median()) == pytest.approx(-0.375)


def test_regime_blend_favors_mean_reversion_in_range() -> None:
    n = 400
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 + rng.normal(0.0, 0.05, n), index=idx)
    sig = pd.DataFrame({"mr": 1.0, "mom": -1.0}, index=idx)
    preferred = [frozenset({"range"}), frozenset({"trend_up", "trend_down"})]
    out = regime_blend_signals(sig, close, preferred, unfavorable=0.25)
    structure = market_structure(close)
    flat = structure == "range"
    assert flat.iloc[50:].mean() > 0.7
    # (1*1 + 0.25*(-1)) / 2 = 0.375
    assert float(out.loc[flat].median()) == pytest.approx(0.375)


def test_unfavorable_is_soft_not_hard_zero() -> None:
    df = _uptrend_df()
    w = regime_blend_weights(
        df["close"],
        [frozenset({"range"})],
        unfavorable=0.25,
    )
    structure = market_structure(df["close"])
    up_w = w.loc[structure == "trend_up", "w0"]
    assert not up_w.empty
    assert (up_w == 0.25).all()
    assert (up_w != 0.0).all()


def test_ensemble_regime_mode_runs() -> None:
    df = _uptrend_df()
    strat = get_strategy(
        "ensemble",
        strategies="mean_reversion,supertrend",
        mode="regime",
        unfavorable_weight=0.25,
    )
    signals = strat.generate_signals(df)
    assert len(signals) == len(df)
    assert signals.abs().max() <= 1.0
    assert not signals.isna().any()


def test_regime_blend_rejects_unknown_preferred() -> None:
    df = _uptrend_df(n=60)
    sig = pd.DataFrame({"a": 1.0}, index=df.index)
    with pytest.raises(ValueError, match="unknown preferred"):
        regime_blend_signals(sig, df["close"], [frozenset({"bull"})])
