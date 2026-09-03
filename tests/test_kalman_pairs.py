"""Тести для 2D Kalman Filter dynamic hedge ratio та Ornstein-Uhlenbeck Half-Life."""

import numpy as np
import pandas as pd
import pytest

from scalper_hft.features.signal_processing import (
    KalmanHedgeRatio,
    dynamic_hedge_ratio,
    estimate_half_life,
)
from scalper_hft.strategies.pairs_arb import PairsArb


def test_kalman_hedge_ratio_convergence():
    """Тест: KalmanHedgeRatio збігається до справжнього бета (наприклад beta=1.75)."""
    np.random.seed(42)
    n = 500
    # True beta = 1.75, true alpha = 0.5
    x = np.linspace(5.0, 6.0, n) + np.random.normal(0, 0.05, n)
    true_beta = 1.75
    true_alpha = 0.5
    y = true_beta * x + true_alpha + np.random.normal(0, 0.01, n)

    kf = KalmanHedgeRatio(q_beta=1e-5, q_alpha=1e-5, r=1e-3, initial_beta=1.0)
    for i in range(n):
        beta, alpha, spread = kf.update(y[i], x[i])

    # Після 500 спостережень оцінка бета має бути дуже близькою до 1.75
    assert abs(beta - true_beta) < 0.05
    assert abs(alpha - true_alpha) < 0.3


def test_dynamic_hedge_ratio_dataframe():
    """Тест векторизованої обгортки dynamic_hedge_ratio."""
    idx = pd.date_range("2026-01-01", periods=100, freq="1h")
    s_x = pd.Series(np.linspace(10, 20, 100), index=idx)
    s_y = pd.Series(2.0 * np.linspace(10, 20, 100), index=idx)

    res = dynamic_hedge_ratio(s_y, s_x)
    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"beta", "alpha", "spread"}
    assert len(res) == 100
    # Бета прямує до 2.0
    assert abs(res["beta"].iloc[-1] - 2.0) < 0.1


def test_estimate_half_life():
    """Тест оцінки Ornstein-Uhlenbeck Half-Life."""
    np.random.seed(42)
    n = 300
    # Mean-reverting AR(1) процес: s_t = 0.85 * s_{t-1} + e_t
    # Half-life = -ln(2) / ln(0.85) ≈ 4.26 барів
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + np.random.normal(0, 0.1)

    s = pd.Series(spread)
    hl = estimate_half_life(s)
    assert np.isfinite(hl)
    assert 2.0 < hl < 10.0

    # Чистий Random Walk: не повинен мати повернення до середнього
    rw = pd.Series(np.cumsum(np.random.normal(0, 1.0, n)))
    hl_rw = estimate_half_life(rw)
    # Повертає inf або дуже велике число
    assert hl_rw > 50.0 or np.isinf(hl_rw)


def test_pairs_arb_with_kalman():
    """Тест PairsArb з прапорцем use_kalman=True."""
    idx = pd.date_range("2026-01-01", periods=200, freq="1h")
    leg1 = pd.Series(100.0 + np.random.normal(0, 2, 200), index=idx)
    leg2 = pd.Series(50.0 + np.random.normal(0, 1, 200), index=idx)
    df = pd.DataFrame({"leg1": leg1, "leg2": leg2}, index=idx)

    strat_default = PairsArb(lookback=60, use_kalman=False)
    sig_default = strat_default.generate_signals(df)

    strat_kalman = PairsArb(lookback=60, use_kalman=True, dynamic_half_life=True)
    sig_kalman = strat_kalman.generate_signals(df)

    assert len(sig_default) == 200
    assert len(sig_kalman) == 200
    assert set(sig_kalman.unique()).issubset({-1, 0, 1})
    assert strat_kalman.betas is not None
    assert len(strat_kalman.betas) == 200


def test_pairs_backtest_with_kalman():
    """Тест run_pairs_backtest з використанням Kalman стратегії."""
    from scalper_hft.backtest.pairs import run_pairs_backtest

    idx = pd.date_range("2026-01-01", periods=250, freq="1h")
    # Штучний коінтегрований ряд: leg1 = 1.5 * leg2 + noise
    p2 = 50.0 + np.cumsum(np.random.normal(0, 0.5, 250))
    p1 = 1.5 * p2 + 10.0 + np.random.normal(0, 1.0, 250)

    leg1 = pd.DataFrame({"close": p1, "high": p1 + 0.5, "low": p1 - 0.5}, index=idx)
    leg2 = pd.DataFrame({"close": p2, "high": p2 + 0.3, "low": p2 - 0.3}, index=idx)

    strat = PairsArb(lookback=60, use_kalman=True, dynamic_half_life=True)
    res = run_pairs_backtest(leg1, leg2, strat, maker_execution=True)

    assert res is not None
    assert len(res.equity) == 250
    assert res.metrics is not None

