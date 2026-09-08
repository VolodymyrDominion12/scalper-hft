"""Тести depth/bookTicker фіч у ML-датасеті (Фаза 2.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.ml.features import _build_features


def _ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-09-08", periods=n, freq="1min")
    rng = np.random.default_rng(3)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": rng.uniform(100, 200, n),
        },
        index=idx,
    )


def _depth5(idx_source: pd.DatetimeIndex) -> pd.DataFrame:
    # снапшоти кожні 100мс усередині діапазону барів
    idx = pd.date_range(idx_source[0], idx_source[-1], freq="100ms")
    rng = np.random.default_rng(5)
    rows: dict[str, np.ndarray] = {}
    for i in range(1, 6):
        rows[f"bid{i}"] = np.full(len(idx), 100.0 - i * 0.01)
        rows[f"ask{i}"] = np.full(len(idx), 100.0 + i * 0.01)
        rows[f"bid{i}_qty"] = rng.uniform(1, 10, len(idx))
        rows[f"ask{i}_qty"] = rng.uniform(1, 10, len(idx))
    return pd.DataFrame(rows, index=idx)


def test_depth_columns_added_when_depth_given() -> None:
    df = _ohlcv()
    depth = _depth5(df.index)
    f = _build_features(df, None, add_frac=False, frac_d=0.4, add_micro=False, depth=depth)
    for col in ("depth_imb", "depth_imb_l1", "depth_spread_bps"):
        assert col in f.columns
        assert f[col].notna().all()
    assert f["depth_imb"].between(-1.0, 1.0).all()
    assert (f["depth_spread_bps"] >= 0).all()


def test_no_depth_no_columns() -> None:
    f = _build_features(_ohlcv(), None, add_frac=False, frac_d=0.4, add_micro=False)
    assert "depth_imb" not in f.columns


def test_bookticker_attach_columns() -> None:
    df = _ohlcv()
    df["imbalance"] = np.linspace(-0.5, 0.5, len(df))
    f = _build_features(df, None, add_frac=False, frac_d=0.4, add_micro=False)
    assert "book_imb" in f.columns and "book_imb_chg" in f.columns
    assert f["book_imb"].notna().all()


def test_depth_features_no_lookahead() -> None:
    """Зміна майбутніх снапшотів не впливає на фічі минулих барів."""
    df = _ohlcv()
    depth = _depth5(df.index)
    f1 = _build_features(df, None, add_frac=False, frac_d=0.4, add_micro=False, depth=depth)
    depth2 = depth.copy()
    depth2.iloc[len(depth2) // 2 :] = depth2.iloc[len(depth2) // 2 :] * 10.0
    f2 = _build_features(df, None, add_frac=False, frac_d=0.4, add_micro=False, depth=depth2)
    cutoff = len(df) // 2 - 1  # ffill може протягнути зміну на межі
    pd.testing.assert_series_equal(f1["depth_imb"].iloc[:cutoff], f2["depth_imb"].iloc[:cutoff])
