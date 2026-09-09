"""Тести Phase 2D: cross-sectional features + factor stacking + ERC/vol-target sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.features.cross_section import (
    cross_section_features,
    cross_section_rank,
    cross_section_zscore,
)
from scalper_hft.ml.factor_stack import factor_stack
from scalper_hft.portfolio.sizing import erc_vol_target_sizes, regime_scaled_size


def _prices(n: int = 200, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {f"s{i}": np.cumprod(1 + rng.normal(0, 0.002, n)) * 100 for i in range(k)},
        index=idx,
    )


def test_cross_section_rank_in_unit_interval() -> None:
    prices = _prices()
    rank = cross_section_rank(prices, lookback=10)
    assert rank.shape == prices.shape
    # ранги у [0,1] (після warmup; NaN на warmup припустимі)
    valid = rank.dropna()
    assert ((valid >= 0) & (valid <= 1)).all().all()


def test_cross_section_zscore_mean_zero_per_row() -> None:
    prices = _prices()
    z = cross_section_zscore(prices, lookback=10)
    # середнє по рядку ~ 0 (z-score центрований)
    row_means = z.mean(axis=1).iloc[20:]
    assert row_means.abs().max() < 1e-9


def test_cross_section_features_columns() -> None:
    prices = _prices(k=3)
    feats = cross_section_features(prices, lookback=10)
    assert "s0_rank" in feats.columns
    assert "s0_zscore" in feats.columns
    assert len(feats.columns) == 6


def test_factor_stack_oos_signal_shape() -> None:
    prices = _prices(n=300, k=4)
    feats = cross_section_features(prices, lookback=10)
    fwd = prices.pct_change().shift(-1).mean(axis=1).fillna(0.0)
    res = factor_stack(feats, fwd, n_folds=5, purge_bars=2, embargo_bars=2)
    assert len(res.oos_signal) == len(feats)
    assert res.oos_signal.abs().max() <= 1.0
    assert res.n_folds > 0
    assert len(res.feature_importance) == feats.shape[1]


def test_factor_stack_purge_no_overlap() -> None:
    # переконуємось, що purge/embargo не дає lookahead: train_end < test_start
    from scalper_hft.ml.factor_stack import _purged_folds

    folds = _purged_folds(n=100, k=5, purge_bars=3, embargo_bars=5)
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_e <= te_s  # train закінчується до (або з embargo до) test
        assert te_e > te_s


def test_erc_vol_target_sizes_causal_and_bounded() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=200, freq="1h")
    ret = pd.DataFrame(rng.normal(0, 0.001, (200, 4)), index=idx)
    w = erc_vol_target_sizes(ret, target_vol=0.01, max_weight=0.5)
    assert w.shape == ret.shape
    # ваги невід'ємні
    assert (w.values >= 0).all()
    # warmup — рівні ваги
    assert np.allclose(w.iloc[0].values, 0.25)


def test_regime_scaled_size() -> None:
    s = regime_scaled_size(1.0, regime_confidence=0.8, vol_scale=1.2)
    assert abs(s - 0.96) < 1e-9
    # confidence=0 → 0
    assert regime_scaled_size(1.0, 0.0) == 0.0
    # max_scale clip
    s2 = regime_scaled_size(1.0, 1.0, vol_scale=2.0, max_scale=1.5)
    assert s2 == 1.5
