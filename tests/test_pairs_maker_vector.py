"""Регресія: векторизований maker pairs == референсна реалізація."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.pairs import (
    _maker_pair_positions,
    _maker_pair_positions_reference,
)


def _synthetic_common(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    leg1 = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    leg2 = 50.0 + np.cumsum(rng.normal(0, 0.2, n))
    wobble1 = rng.uniform(0.2, 1.5, n)
    wobble2 = rng.uniform(0.1, 0.8, n)
    return pd.DataFrame(
        {
            "leg1": leg1,
            "leg2": leg2,
            "l1_high": leg1 + wobble1,
            "l1_low": leg1 - wobble1,
            "l2_high": leg2 + wobble2,
            "l2_low": leg2 - wobble2,
        },
        index=idx,
    )


def _synthetic_signals(index: pd.Index, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    raw = rng.integers(-1, 2, size=len(index))
    return pd.Series(raw.astype(float), index=index)


@pytest.mark.parametrize("position_pct", [0.05, 0.1])
def test_maker_pair_positions_matches_reference_deterministic(position_pct: float) -> None:
    common = _synthetic_common()
    signals = _synthetic_signals(common.index)
    fast = _maker_pair_positions(signals, common, position_pct, rng=None)
    ref = _maker_pair_positions_reference(signals, common, position_pct, rng=None)
    pd.testing.assert_series_equal(fast, ref)


def test_maker_pair_positions_matches_reference_with_rng() -> None:
    common = _synthetic_common(n=80, seed=3)
    signals = _synthetic_signals(common.index, seed=5)
    position_pct = 0.1
    rng_fast = np.random.default_rng(42)
    rng_ref = np.random.default_rng(42)
    fast = _maker_pair_positions(signals, common, position_pct, rng=rng_fast)
    ref = _maker_pair_positions_reference(signals, common, position_pct, rng=rng_ref)
    pd.testing.assert_series_equal(fast, ref)
