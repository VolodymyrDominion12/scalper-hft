"""Тести depth5-фіч (features/depth_features.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _depth_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-08 00:00:00", periods=n, freq="100ms")
    mid = 100.0
    rows = {}
    for i in range(1, 6):
        rows[f"bid{i}"] = mid - i * 0.01
        rows[f"ask{i}"] = mid + i * 0.01
        rows[f"bid{i}_qty"] = rng.uniform(1, 10, n)
        rows[f"ask{i}_qty"] = rng.uniform(1, 10, n)
    df = pd.DataFrame(rows, index=idx)
    # штучний перекіс: більше обсягу на біді
    df["bid1_qty"] += 5.0
    return df


def test_depth_imbalance_sign_and_range() -> None:
    from scalper_hft.features.depth_features import depth_imbalance

    d = _depth_df()
    imb = depth_imbalance(d)
    assert len(imb) == len(d)
    assert imb.between(-1.0, 1.0).all()
    # bid-перекіс → додатний imbalance
    assert imb.mean() > 0.0


def test_depth_imbalance_depth_vs_top() -> None:
    from scalper_hft.features.depth_features import depth_imbalance, top_of_book_imbalance

    d = _depth_df()
    dw = depth_imbalance(d, levels=5, weights="linear")
    top = top_of_book_imbalance(d)
    assert not np.allclose(dw.values, top.values)  # глибина змінює картину


def test_depth_imbalance_empty_levels_zero() -> None:
    """Відсутні рівні (NaN qty) не ламають розрахунок."""
    from scalper_hft.features.depth_features import depth_imbalance

    d = _depth_df(10)
    d.loc[d.index[0], "bid3_qty":"ask5_qty"] = np.nan
    imb = depth_imbalance(d, levels=5)
    assert len(imb) == len(d)
    assert imb.notna().all()


def test_depth_spread_bps() -> None:
    from scalper_hft.features.depth_features import depth_spread_bps

    d = _depth_df()
    sp = depth_spread_bps(d)
    # spread 2×0.01 на mid 100 → 2.0 bps
    np.testing.assert_allclose(sp.iloc[0], 2.0, rtol=1e-6)


def test_snapshot_quality() -> None:
    from scalper_hft.features.depth_features import snapshot_quality

    d = _depth_df(100)  # 100 снапшотів × 100ms = 10 с
    q = snapshot_quality(d)
    assert q["rows"] == 100
    assert q["rows_per_sec"] == pytest.approx(10.0, rel=0.1)
    assert q["duplicates"] == 0


def test_depth_features_causal() -> None:
    """Фічі рядка t не залежать від майбутніх снапшотів."""
    from scalper_hft.features.depth_features import depth_imbalance, depth_spread_bps

    d = _depth_df(200)
    cut = 100
    imb_half = depth_imbalance(d.iloc[:cut])
    imb_full = depth_imbalance(d).iloc[:cut]
    pd.testing.assert_series_equal(imb_half, imb_full)
    sp_half = depth_spread_bps(d.iloc[:cut])
    sp_full = depth_spread_bps(d).iloc[:cut]
    pd.testing.assert_series_equal(sp_half, sp_full)
