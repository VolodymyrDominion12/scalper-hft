"""Тести Phase 2B: validated regime→strategy map (OOS perf matrix, hard-off, policy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.validation.regime_map import (
    RegimePerfMatrix,
    RegimeStrategyMap,
    build_regime_strategy_map,
    compute_regime_perf_matrix,
)


def _data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # стратегія A заробляє в trend, B — в range, C — ніде
    labels = np.where(np.arange(n) % 2 == 0, "trend_up|normal", "range|normal")
    regime_df = pd.DataFrame({"label": labels}, index=idx)
    ret = pd.DataFrame(index=idx)
    ret["A"] = np.where(labels == "trend_up|normal", 0.001, -0.0002) + rng.normal(0, 0.0005, n)
    ret["B"] = np.where(labels == "range|normal", 0.001, -0.0002) + rng.normal(0, 0.0005, n)
    ret["C"] = rng.normal(0.0, 0.0005, n)  # нульовий edge
    return ret, regime_df


def test_compute_regime_perf_matrix_sharpe_signs() -> None:
    ret, regime = _data()
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    # A має додатній Sharpe в trend, B — в range
    assert m.sharpe("trend_up|normal", "A") > 0
    assert m.sharpe("range|normal", "B") > 0
    # C ~ 0 (може бути малий додатний/від'ємний)
    assert m.sharpe("trend_up|normal", "C") < 0.5


def test_build_regime_strategy_map_hard_off() -> None:
    ret, regime = _data()
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    rmap = build_regime_strategy_map(m, hard_off_sharpe=0.0, high_vol_flat=True)
    # C (Sharpe ~0) hard-off у всіх режимах
    assert rmap.weights_for("trend_up|normal").get("C", 0.0) == 0.0
    # A активна в trend, B активна в range
    assert rmap.active_strategies("trend_up|normal") == ["A"]
    assert rmap.active_strategies("range|normal") == ["B"]


def test_build_regime_strategy_map_high_vol_flat() -> None:
    idx = pd.date_range("2025-01-01", periods=200, freq="1h")
    regime = pd.DataFrame({"label": ["range|high"] * 200}, index=idx)
    ret = pd.DataFrame({"A": np.random.default_rng(1).normal(0.001, 0.001, 200)}, index=idx)
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    rmap = build_regime_strategy_map(m, high_vol_flat=True)
    assert rmap.is_flat("range|high")
    assert rmap.policy.get("range|high") == "flat"


def test_build_regime_strategy_map_soft_blend_when_close() -> None:
    # дві стратегії з близьким Sharpe → soft
    idx = pd.date_range("2025-01-01", periods=300, freq="1h")
    regime = pd.DataFrame({"label": ["trend_up|normal"] * 300}, index=idx)
    rng = np.random.default_rng(2)
    ret = pd.DataFrame({"A": 0.001 + rng.normal(0, 0.0005, 300), "B": 0.001 + rng.normal(0, 0.0005, 300)}, index=idx)
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    rmap = build_regime_strategy_map(m, hard_off_sharpe=0.0, best_prior_min_gap=0.5)
    assert rmap.policy.get("trend_up|normal") == "soft"
    w = rmap.weights_for("trend_up|normal")
    assert w["A"] > 0 and w["B"] > 0


def test_regime_map_json_roundtrip(tmp_path: Path) -> None:
    rmap = RegimeStrategyMap(
        weights={"trend_up|normal": {"A": 1.0, "B": 0.0}},
        policy={"trend_up|normal": "best_prior"},
    )
    p = tmp_path / "rmap.json"
    rmap.to_json(p)
    loaded = RegimeStrategyMap.from_json(p)
    assert loaded.weights == rmap.weights
    assert loaded.policy == rmap.policy


def test_perf_matrix_json_roundtrip(tmp_path: Path) -> None:
    ret, regime = _data()
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    p = tmp_path / "pm.json"
    m.to_json(p)
    loaded = RegimePerfMatrix.from_json(p)
    assert loaded.sharpe("trend_up|normal", "A") == m.sharpe("trend_up|normal", "A")


def test_supervisor_uses_regime_map(tmp_path: Path) -> None:
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    # будуємо map: A активна в trend, B в range
    rmap = RegimeStrategyMap(
        weights={
            "trend_up|normal": {"mean_reversion": 0.0, "supertrend": 1.0},
            "range|normal": {"mean_reversion": 1.0, "supertrend": 0.0},
        },
        policy={"trend_up|normal": "best_prior", "range|normal": "best_prior"},
    )
    p = tmp_path / "rmap.json"
    rmap.to_json(p)
    sup = RegimeSupervisor(
        strategies="mean_reversion,supertrend",
        blend_mode="regime_soft",
        regime_map_path=str(p),
        hmm_fit_bars=200,
    )
    r = sup._load_regime_map()
    assert r is not None
    assert r.active_strategies("trend_up|normal") == ["supertrend"]
    assert r.active_strategies("range|normal") == ["mean_reversion"]
