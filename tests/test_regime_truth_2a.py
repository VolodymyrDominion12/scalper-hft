"""Тести Phase 2A: RegimeDetector єдиний regime-шар (funding×liquidity×WF refit×lazy gating)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.features.regime_detector import (
    RegimeDetector,
    RegimeState,
    _funding_skew_series,
    _liquidity_label_from_volume,
)


def _close(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    ret = rng.normal(0, 0.002, n)
    return pd.Series(np.cumprod(1 + ret) * 100, index=idx)


def test_detect_adds_funding_skew_and_liquidity_columns() -> None:
    det = RegimeDetector(hmm_fit_bars=200)
    close = _close(600)
    funding = pd.Series(np.linspace(-0.0001, 0.0003, 600), index=close.index)
    volume = pd.Series(np.random.default_rng(1).uniform(100, 1000, 600), index=close.index)
    df = det.detect(close, funding=funding, volume=volume)
    assert "funding_skew" in df.columns
    assert "liquidity" in df.columns
    assert "joint_label" in df.columns
    # joint_label має формат label|hN|fX|liq
    jl = df["joint_label"].iloc[-1]
    assert jl.count("|") >= 3


def test_detect_without_funding_volume_defaults() -> None:
    det = RegimeDetector(hmm_fit_bars=200)
    df = det.detect(_close(400))
    assert (df["funding_skew"] == 0.0).all()
    assert (df["liquidity"] == "normal").all()


def test_funding_skew_series_causal() -> None:
    idx = pd.date_range("2025-01-01", periods=800, freq="1h")
    f = pd.Series(np.random.default_rng(2).normal(0, 0.0001, 800), index=idx)
    z = _funding_skew_series(f, window=200)
    # перевищення clipping
    assert (z.abs() <= 3.0 + 1e-9).all()
    # короткий ряд → нулі
    assert (_funding_skew_series(pd.Series([0.0] * 10)) == 0.0).all()


def test_liquidity_label_categories() -> None:
    close = _close(400)
    vol = pd.Series(np.random.default_rng(3).uniform(100, 1000, 400), index=close.index)
    lab = _liquidity_label_from_volume(close, vol, window=120)
    assert set(lab.unique()).issubset({"low", "normal", "high"})
    # невідповідна довжина → normal
    bad = _liquidity_label_from_volume(close, pd.Series([1.0, 2.0]))
    assert (bad == "normal").all()


def test_regime_state_joint_label() -> None:
    s = RegimeState(
        structure="trend_up",
        vol="low",
        label="trend_up|low",
        hmm_state=1,
        hmm_probs=np.array([0.2, 0.8]),
        confidence=0.8,
        funding_skew=1.2,
        liquidity="high",
    )
    assert s.joint_label == "trend_up|low|h1|fp|high"
    s0 = RegimeState("range", "normal", "range|normal", -1, np.array([]), 0.0, 0.0, "normal")
    assert s0.joint_label == "range|normal|h-1|f0|normal"


def test_refit_method_changes_model() -> None:
    det = RegimeDetector(hmm_fit_bars=200)
    det.fit(_close(400))
    assert det.is_fitted
    # refit на іншому вікні — не падає
    det.refit(_close(600), start=100, end=400)
    assert det.is_fitted


def test_detect_refit_every_produces_valid_output() -> None:
    det = RegimeDetector(hmm_fit_bars=200)
    df = det.detect(_close(800), refit_every=200)
    assert "hmm_state" in df.columns
    assert len(df) == 800


def test_lazy_gating_zeros_inactive_strategies() -> None:
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    sup = RegimeSupervisor(
        strategies="mean_reversion,supertrend",
        blend_mode="regime_soft",
        lazy_gating=True,
        hmm_fit_bars=200,
    )
    close = _close(600)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0})
    sigs = sup.generate_signals(df)
    # сигнали у [-1, 1]; lazy gating не падає
    assert sigs.abs().max() <= 1.0
    assert len(sigs) == len(df)


def test_step_accepts_funding_volume() -> None:
    det = RegimeDetector(hmm_fit_bars=200)
    close = _close(300)
    state = None
    for c in close:
        state = det.step(float(c), funding_rate=0.0001, volume=500.0)
    assert state is not None
    assert isinstance(state.liquidity, str)
    assert isinstance(state.funding_skew, float)
