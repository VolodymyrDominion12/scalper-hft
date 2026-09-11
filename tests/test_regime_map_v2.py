"""Regime-карта v2.0: fit на IS-періоді + авторитетність у RegimeSupervisor.

Що перевіряється:
    1. пулінг Sharpe по символах (карта не підганяється під один інструмент);
    2. save/load карти разом із метаданими fit-вікна (no-lookahead аудит);
    3. regime_soft і best_prior беруть ваги З КАРТИ, а не з taxonomy-пріорів;
    4. flat-комірка карти (усі ваги 0) дає позицію 0, а не «першу стратегію»;
    5. no-lookahead: мутація майбутніх барів не змінює минулі сигнали;
    6. попередження, якщо карта містить стратегії поза пулом дітей.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.regime_supervisor import DEFAULT_CHILDREN, RegimeSupervisor
from scalper_hft.validation.regime_fit import RegimeFitResult, format_fit_summary
from scalper_hft.validation.regime_map import (
    RegimePerfMatrix,
    RegimeStrategyMap,
    load_regime_map_meta,
    pool_regime_perf_matrix,
    save_regime_map,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ── допоміжні заглушки ─────────────────────────────────────────────────────


class _Const(Strategy):
    """Стратегія з постійним сигналом — детермінований перемикач для тестів."""

    def __init__(self, name: str, value: float) -> None:
        super().__init__()
        self.name = name
        self._value = value

    def generate_signals(self, df: pd.DataFrame, trades=None, funding=None) -> pd.Series:
        return pd.Series(self._value, index=df.index, dtype=float)


def _ohlcv(n: int = 400, trend_break: int = 200, seed: int = 7) -> pd.DataFrame:
    """Ряд із двох половин: спочатку аптренд, потім флет (для режимних міток)."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(trend_break, 0.004), np.zeros(n - trend_break)])
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.0008, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )


def _map_file(tmp_path: Path, weights: dict[str, dict[str, float]], policy: dict[str, str] | None = None) -> Path:
    rmap = RegimeStrategyMap(weights=weights, policy=policy or {})
    p = tmp_path / "regime_map.json"
    save_regime_map(rmap, p, meta={"fit_end": "2026-01-01T00:00:00+00:00", "fit_symbols": ["BTCUSDT"]})
    return p


def _supervisor(tmp_path: Path, weights, *, blend_mode: str, policy=None) -> RegimeSupervisor:
    path = _map_file(tmp_path, weights, policy)
    sup = RegimeSupervisor(
        strategies="supertrend,stoch_rsi",
        blend_mode=blend_mode,
        regime_map_path=str(path),
        hmm_fit_bars=80,
    )
    # Детерміновані діти: суперетrend → +1, stoch_rsi → −1.
    sup._strats = [_Const("supertrend", 1.0), _Const("stoch_rsi", -1.0)]
    sup._strat_names = ["supertrend", "stoch_rsi"]
    return sup


def _sig_and_regime(n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    sig = pd.DataFrame({"supertrend": np.ones(n), "stoch_rsi": -np.ones(n)}, index=idx)
    structure = ["trend_up"] * (n // 2) + ["range"] * (n - n // 2)
    vol = ["normal"] * n
    regime = pd.DataFrame(
        {
            "structure": structure,
            "vol": vol,
            "label": [f"{s}|{v}" for s, v in zip(structure, vol, strict=False)],
        },
        index=idx,
    )
    return sig, regime


# ── 1. пулінг по символах ──────────────────────────────────────────────────


def test_pool_regime_perf_matrix_unions_symbols() -> None:
    """Два символи з різними режимами → спільна матриця містить обидва режими."""
    idx = pd.date_range("2025-01-01", periods=200, freq="1h")
    half = 100
    labels_a = ["trend_up|normal"] * half + ["range|normal"] * half
    labels_b = ["trend_down|normal"] * half + ["range|normal"] * half
    rng = np.random.default_rng(11)
    noise = lambda: rng.normal(0.0, 0.0002, len(idx))  # noqa: E731
    # supertrend заробляє у trend_up, funding_carry — у trend_down і range
    ret_a = pd.DataFrame(
        {
            "supertrend": np.where(np.array(labels_a) == "trend_up|normal", 0.002, -0.001) + noise(),
            "funding_carry": np.where(np.array(labels_a) == "range|normal", 0.001, -0.001) + noise(),
        },
        index=idx,
    )
    ret_b = pd.DataFrame(
        {
            "supertrend": np.where(np.array(labels_b) == "trend_up|normal", 0.002, -0.001) + noise(),
            "funding_carry": np.where(np.array(labels_b) == "trend_down|normal", 0.002, 0.0005) + noise(),
        },
        index=idx,
    )
    per_symbol = {
        "AAAUSDT": (ret_a, pd.DataFrame({"label": labels_a}, index=idx)),
        "BBBUSDT": (ret_b, pd.DataFrame({"label": labels_b}, index=idx)),
    }
    m = pool_regime_perf_matrix(per_symbol, min_bars=20)

    assert {"trend_up|normal", "trend_down|normal", "range|normal"} <= set(m.regimes())
    assert m.sharpe("trend_up|normal", "supertrend") > m.sharpe("range|normal", "supertrend")
    assert m.sharpe("trend_down|normal", "funding_carry") > 0
    # n_bars пуляться (AAA+BBB у range|normal)
    assert m.data["range|normal"]["funding_carry"]["n_bars"] == 200


def test_pool_regime_perf_matrix_skips_short_or_empty() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    per_symbol = {"AAAUSDT": (pd.DataFrame({"a": [0.0] * 5}, index=idx), pd.DataFrame({"label": ["range|normal"] * 5}, index=idx))}
    m = pool_regime_perf_matrix(per_symbol, min_bars=30)
    assert m.regimes() == []


# ── 2. save/load із метаданими ─────────────────────────────────────────────


def test_save_and_load_regime_map_with_meta(tmp_path: Path) -> None:
    path = _map_file(tmp_path, {"range|normal": {"funding_carry": 1.0}}, {"range|normal": "best_prior"})
    rmap = RegimeStrategyMap.from_json(path)
    assert rmap.weights_for("range|normal") == {"funding_carry": 1.0}
    assert rmap.policy["range|normal"] == "best_prior"
    meta = load_regime_map_meta(path)
    assert meta["fit_end"].startswith("2026-01-01")
    assert meta["fit_symbols"] == ["BTCUSDT"]


def test_regime_map_without_meta_is_compatible(tmp_path: Path) -> None:
    p = tmp_path / "plain.json"
    p.write_text(json.dumps({"weights": {"range|normal": {"a": 1.0}}, "policy": {}}), encoding="utf-8")
    rmap = RegimeStrategyMap.from_json(p)
    assert rmap.active_strategies("range|normal") == ["a"]
    assert load_regime_map_meta(p) == {}
    assert load_regime_map_meta(tmp_path / "missing.json") == {}


def test_format_fit_summary_has_no_lookahead_warning() -> None:
    rmap = RegimeStrategyMap(weights={"trend_up|normal": {"supertrend": 1.0}}, policy={"trend_up|normal": "best_prior"})
    result = RegimeFitResult(
        map=rmap,
        matrix=RegimePerfMatrix(),
        meta={
            "fit_start": "2025-01-01T00:00:00+00:00",
            "fit_end": "2026-01-01T00:00:00+00:00",
            "fit_interval": "1h",
            "fit_symbols": ["BTCUSDT"],
            "fit_strategies": ["supertrend"],
            "cells": [{"regime": "trend_up|normal", "policy": "best_prior", "active": "supertrend"}],
        },
    )
    text = format_fit_summary(result)
    assert "2026-01-01" in text
    assert "supertrend" in text
    assert "ПІСЛЯ fit_end" in text


# ── 3. карта авторитетна для regime_soft / best_prior ──────────────────────


@pytest.mark.parametrize("blend_mode", ["regime_soft", "best_prior"])
def test_map_drives_switch(tmp_path: Path, blend_mode: str) -> None:
    weights = {
        "trend_up|normal": {"supertrend": 1.0, "stoch_rsi": 0.0},
        "range|normal": {"supertrend": 0.0, "stoch_rsi": 1.0},
    }
    sup = _supervisor(tmp_path, weights, blend_mode=blend_mode)
    sig, regime = _sig_and_regime()
    if blend_mode == "regime_soft":
        res = sup._blend_regime_soft(sig, regime)
    else:
        res = sup._blend_best_prior(sig, regime)
    half = len(sig) // 2
    assert (res.iloc[:half] == 1.0).all()  # trend_up → supertrend (+1)
    assert (res.iloc[half:] == -1.0).all()  # range → stoch_rsi (−1)


def test_map_is_authoritative_over_taxonomy(tmp_path: Path) -> None:
    """Карта суперечить taxonomy — перемагає карта."""
    weights = {
        "trend_up|normal": {"stoch_rsi": 1.0, "supertrend": 0.0},
        "range|normal": {"supertrend": 1.0, "stoch_rsi": 0.0},
    }
    sup = _supervisor(tmp_path, weights, blend_mode="best_prior")
    # taxonomy каже протилежне (stoch_rsi створений як MR, supertrend — тренд)
    sup._strats[0].preferred_regimes = frozenset({"trend_up"})  # supertrend
    sup._strats[1].preferred_regimes = frozenset({"range"})  # stoch_rsi
    sig, regime = _sig_and_regime()
    res = sup._blend_best_prior(sig, regime)
    half = len(sig) // 2
    assert (res.iloc[:half] == -1.0).all()  # карта: stoch_rsi
    assert (res.iloc[half:] == 1.0).all()  # карта: supertrend


def test_map_flat_cell_returns_zero(tmp_path: Path) -> None:
    """Комірка з усіма нульовими вагами (політика flat) → позиція 0."""
    weights = {
        "trend_up|normal": {"supertrend": 0.0, "stoch_rsi": 0.0},
        "range|normal": {"supertrend": 0.0, "stoch_rsi": 1.0},
    }
    for mode in ("regime_soft", "best_prior"):
        sup = _supervisor(tmp_path, weights, blend_mode=mode)
        sig, regime = _sig_and_regime()
        res = sup._blend_regime_soft(sig, regime) if mode == "regime_soft" else sup._blend_best_prior(sig, regime)
        half = len(sig) // 2
        assert (res.iloc[:half] == 0.0).all(), mode
        assert (res.iloc[half:] == -1.0).all(), mode


def test_map_cell_missing_falls_back_to_taxonomy(tmp_path: Path) -> None:
    """Без карти ваги беруться з taxonomy — regресійна перевірка сумісності."""
    sup = RegimeSupervisor(strategies="supertrend,stoch_rsi", blend_mode="best_prior", hmm_fit_bars=80)
    sup._strats = [_Const("supertrend", 1.0), _Const("stoch_rsi", -1.0)]
    sup._strats[0].preferred_regimes = frozenset({"trend_up"})
    sup._strats[1].preferred_regimes = frozenset({"range"})
    sig, regime = _sig_and_regime()
    res = sup._blend_best_prior(sig, regime)
    half = len(sig) // 2
    assert (res.iloc[:half] == 1.0).all()
    assert (res.iloc[half:] == -1.0).all()


def test_map_unknown_strategy_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    weights = {"trend_up|normal": {"ghost_strategy": 1.0, "supertrend": 0.0}}
    path = _map_file(tmp_path, weights)
    with caplog.at_level(logging.WARNING, logger="scalper_hft.strategies.regime_supervisor"):
        RegimeSupervisor(
            strategies="supertrend,stoch_rsi",
            blend_mode="best_prior",
            regime_map_path=str(path),
            hmm_fit_bars=80,
        )
    assert any("поза пулом дітей" in rec.message for rec in caplog.records)


def test_regime_map_meta_exposed_on_instance(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, {"range|normal": {"stoch_rsi": 1.0}}, blend_mode="regime_soft")
    assert sup.regime_map_meta["fit_end"].startswith("2026-01-01")


# ── 4. no-lookahead ────────────────────────────────────────────────────────


def test_supervisor_with_map_no_lookahead(tmp_path: Path) -> None:
    """Мутація майбутніх барів не змінює сигнали минулих."""
    df = _ohlcv(600)
    weights = {
        "trend_up|normal": {"supertrend": 1.0, "stoch_rsi": 0.0},
        "trend_down|normal": {"supertrend": 0.0, "stoch_rsi": 1.0},
        "range|normal": {"supertrend": 0.0, "stoch_rsi": 1.0},
    }
    path = _map_file(tmp_path, weights)
    sup = RegimeSupervisor(
        strategies="supertrend,stoch_rsi",
        blend_mode="best_prior",
        regime_map_path=str(path),
        hmm_fit_bars=120,
    )
    base = sup.generate_signals(df, None, None)

    mutated = df.copy()
    cut = 500
    mutated.iloc[cut:, mutated.columns.get_loc("close")] *= 1.5
    mutated.iloc[cut:, mutated.columns.get_loc("high")] *= 1.5
    mutated.iloc[cut:, mutated.columns.get_loc("low")] *= 1.5
    after = sup.generate_signals(mutated, None, None)

    pd.testing.assert_series_equal(base.iloc[:cut], after.iloc[:cut])


def test_default_children_include_carry_sleeve() -> None:
    """Дефолт v2.0 має carry-рукав (range/trend_down), якого не було в v1.4."""
    sup = RegimeSupervisor()
    assert sup.sub_strategies == DEFAULT_CHILDREN.split(",")
    assert "funding_carry" in sup.sub_strategies
    assert sup.needs_funding is True
    assert sup.blend_mode == "regime_soft"
