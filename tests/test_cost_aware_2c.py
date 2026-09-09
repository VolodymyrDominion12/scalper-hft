"""Тести Phase 2C: net-PnL Hedge/Exp3 + turnover penalty + meta-DSR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.strategies.bandit import Exp3Bandit, exp3_select_signals
from scalper_hft.strategies.blend import hedge_blend_signals, hedge_weights
from scalper_hft.validation.deflated_sharpe import (
    deflated_sharpe_ratio_from_sharpe,
    meta_dsr,
)


def _sig_ret(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    sig = pd.DataFrame(
        {
            "a": rng.choice([-1, 0, 1], size=n),
            "b": rng.choice([-1, 0, 1], size=n),
        },
        index=idx,
    )
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.001, n)) * 100, index=idx)
    ret = close.pct_change().fillna(0.0)
    returns = sig.shift(1).fillna(0.0).mul(ret, axis=0)
    return sig, returns, close


def test_hedge_weights_turnover_penalty_changes_weights() -> None:
    sig, returns, _ = _sig_ret()
    w0 = hedge_weights(returns, eta=1.0)
    w1 = hedge_weights(returns, eta=1.0, turnover_penalty=0.5, signals=sig)
    # з turnover-штрафом ваги відрізняються
    assert not np.allclose(w0.values, w1.values)


def test_hedge_blend_signals_turnover_penalty_smoketest() -> None:
    sig, _, close = _sig_ret()
    s0 = hedge_blend_signals(sig, close, eta=1.0)
    s1 = hedge_blend_signals(sig, close, eta=1.0, turnover_penalty=0.3)
    assert len(s0) == len(s1)
    assert s1.abs().max() <= 1.0


def test_exp3_update_turnover_cost_reduces_reward() -> None:
    b = Exp3Bandit(n_arms=3, gamma=0.05, seed=1)
    b.update(0, 0.01, turnover_cost=0.005)
    assert b.history[-1]["net_reward"] == 0.01 - 0.005
    assert b.history[-1]["turnover_cost"] == 0.005


def test_exp3_select_signals_turnover_penalty_reproducible() -> None:
    sig, returns, _ = _sig_ret()
    s1 = exp3_select_signals(sig, returns, gamma=0.05, seed=7, turnover_penalty=0.1)
    s2 = exp3_select_signals(sig, returns, gamma=0.05, seed=7, turnover_penalty=0.1)
    pd.testing.assert_series_equal(s1, s2)


def test_meta_dsr_decreases_with_more_trials() -> None:
    # однаковий winner Sharpe, більше trials → нижче DSR
    sharpes = [1.5, 1.2, 0.9]
    dsr_few = meta_dsr(sharpes, n_meta_trials=3)
    dsr_many = meta_dsr(sharpes, n_meta_trials=100)
    assert dsr_many[0] <= dsr_few[0]
    assert dsr_few[1] == 3
    assert dsr_many[1] == 100


def test_meta_dsr_empty_returns_zero() -> None:
    dsr, n = meta_dsr([], n_meta_trials=5)
    assert dsr == 0.0
    assert n == 5


def test_deflated_sharpe_from_sharpe_basic() -> None:
    # без trials DSR ~ 1 для сильного Sharpe
    dsr = deflated_sharpe_ratio_from_sharpe(2.0, n_trials=1, n_obs=500)
    assert 0.0 <= dsr <= 1.0
    # більше trials → нижче
    dsr_many = deflated_sharpe_ratio_from_sharpe(2.0, n_trials=200, n_obs=500)
    assert dsr_many <= dsr
