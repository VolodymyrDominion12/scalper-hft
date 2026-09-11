"""Тести rolling ADF / half-life kill для paper pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.pair_health import check_cointegration_health


@pytest.fixture
def mock_spread():
    np.random.seed(42)
    # Звичайний random walk без дрейфу (не коінтегрований, p-value буде високим)
    s = pd.Series(np.cumsum(np.random.randn(200)))
    return s


@pytest.fixture
def mock_coint_spread():
    np.random.seed(42)
    # AR(1) процес з α=0.5 (mean-reverting, p-value буде низьким)
    s = np.zeros(200)
    for i in range(1, 200):
        s[i] = 0.5 * s[i - 1] + np.random.randn()
    return pd.Series(s)


def test_gate_rejects_non_coint(mock_spread):
    ok, reason = check_cointegration_health(mock_spread, min_obs=100)
    assert not ok
    assert "adf_kill" in reason or "hl_kill" in reason


def test_random_walk_blocks_entry() -> None:
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(0, 1.0, 250)))
    ok, reason = check_cointegration_health(s, min_obs=80)
    assert ok is False
    assert "kill" in reason
