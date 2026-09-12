"""Каузальний іменований стан ринку: range/trend_up/trend_down × vol."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scalper_hft.features.regimes import (
    COMPOSITE_ORDER,
    STRUCTURE_LABELS,
    VOL_LABELS,
    apply_min_dwell,
    apply_regime_gates,
    htf_market_structure,
    market_structure,
    named_market_state,
    volatility_regime,
)


def _close_trend(n: int = 800, *, up: bool = True) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    drift = 0.5 if up else -0.5
    t = np.arange(n, dtype=float)
    values = 100.0 + drift * t
    return pd.Series(values, index=idx)


def _close_range(n: int = 800) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(2)
    values = 100.0 + rng.normal(0.0, 0.05, n)
    return pd.Series(values, index=idx)


def test_market_structure_labels_are_named() -> None:
    out = market_structure(_close_trend())
    assert set(out.unique()) <= STRUCTURE_LABELS


def test_strong_uptrend_is_mostly_trend_up() -> None:
    out = market_structure(_close_trend(up=True))
    tail = out.iloc[100:]
    assert (tail == "trend_up").mean() > 0.8
    assert (tail == "trend_down").sum() == 0


def test_strong_downtrend_is_mostly_trend_down() -> None:
    out = market_structure(_close_trend(up=False))
    tail = out.iloc[100:]
    assert (tail == "trend_down").mean() > 0.8


def test_oscillation_is_mostly_range() -> None:
    out = market_structure(_close_range())
    tail = out.iloc[100:]
    assert (tail == "range").mean() > 0.7


def test_named_market_state_columns_and_composite() -> None:
    close = _close_trend()
    state = named_market_state(close)
    assert list(state.columns) == ["structure", "vol", "label"]
    assert set(state["structure"].unique()) <= STRUCTURE_LABELS
    assert set(state["vol"].unique()) <= VOL_LABELS
    expected = state["structure"].astype(str) + "|" + state["vol"].astype(str)
    pd.testing.assert_series_equal(state["label"], expected, check_names=False)
    assert set(state["label"].unique()) <= set(COMPOSITE_ORDER)


def test_named_market_state_no_lookahead() -> None:
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n))), index=idx)
    state = named_market_state(close)
    mutated = close.copy()
    mutated.iloc[-30:] *= 1.8
    state2 = named_market_state(mutated)
    pd.testing.assert_frame_equal(state.iloc[:-30], state2.iloc[:-30])


def test_volatility_regime_still_three_labels() -> None:
    close = _close_range()
    vol = volatility_regime(close)
    assert set(vol.unique()) <= VOL_LABELS


def test_market_structure_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="trend_threshold"):
        market_structure(_close_range(n=60), trend_threshold=-0.1)


def test_apply_min_dwell_hysteresis() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="1h")
    series = pd.Series(["A", "A", "B", "B", "B", "A"], index=idx, dtype=object)
    assert list(apply_min_dwell(series, 2)) == ["A", "A", "A", "B", "B", "B"]
    # фліпи без підтвердження не проходять
    flappy = pd.Series(["A", "B", "A", "B", "A", "B"], index=idx, dtype=object)
    assert list(apply_min_dwell(flappy, 2).unique()) == ["A"]


def _regime_df(structure: list[str], vol: list[str]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(structure), freq="1h")
    return pd.DataFrame(
        {
            "structure": pd.Series(structure, index=idx, dtype=object),
            "vol": pd.Series(vol, index=idx, dtype=object),
        }
    )


def test_apply_regime_gates_high_vol_veto() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    sig = pd.Series([1.0, -1.0, 1.0], index=idx)
    regime = _regime_df(["range", "range", "range"], ["high", "normal", "normal"])
    out = apply_regime_gates(sig, regime, vol_high_veto=True)
    assert list(out) == [0.0, -1.0, 1.0]


def test_apply_regime_gates_trend_direction() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    sig = pd.Series([-1.0, 1.0, 1.0], index=idx)  # шорт у тренді вгору, лонг у тренді вниз
    regime = _regime_df(["trend_up", "trend_down", "range"], ["normal", "normal", "normal"])
    out = apply_regime_gates(sig, regime, trend_direction_gate=True)
    assert list(out) == [0.0, 0.0, 1.0]  # range — без обмежень


def test_apply_regime_gates_noop_when_disabled() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    sig = pd.Series([1.0, -1.0, 1.0], index=idx)
    regime = _regime_df(["trend_up", "high", "range"], ["normal", "high", "low"])
    out = apply_regime_gates(sig, regime)
    assert list(out) == [1.0, -1.0, 1.0]


def test_htf_market_structure_binance_1d_has_no_pandas4_warning() -> None:
    """Binance '1d' має мапитись на pandas '1D' — інакше Pandas4Warning."""
    n = 24 * 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.1, index=idx)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*'d' is deprecated.*")
        out = htf_market_structure(close, htf="1d")
    assert len(out) == len(close)


def test_htf_market_structure_direction_and_labels() -> None:
    """Стійкий денний аптренд → trend_up на 1h-ряду через htf='1d'."""
    n = 24 * 400  # 400 днів 1h
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    t = np.arange(n, dtype=float)
    close = pd.Series(100.0 * np.exp(0.0004 * t), index=idx)  # сильний рівний аптренд
    out = htf_market_structure(close, htf="1d", ema_fast=3, ema_slow=20)
    assert set(out.dropna().unique()) <= STRUCTURE_LABELS
    tail = out.iloc[int(n * 0.8) :]
    assert (tail == "trend_up").mean() > 0.9, f"htf мав би бачити trend_up, got {tail.value_counts().to_dict()}"


def test_htf_market_structure_causal() -> None:
    """Рішення на барі t не залежить від майбутніх цін (мутація хвоста не міняє prefix)."""
    rng = np.random.default_rng(7)
    n = 24 * 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)
    base = htf_market_structure(close, htf="1d")
    mutated = close.copy()
    mutated.iloc[-24 * 60 :] *= 3.0  # майбутній шок
    mut = htf_market_structure(mutated, htf="1d")
    cut = n - 24 * 60
    pd.testing.assert_series_equal(base.iloc[:cut], mut.iloc[:cut])
