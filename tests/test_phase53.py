"""Phase 5.3: OOS preferred_regimes, haircut roster, ML CV, cost stress."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.ml.trainer import _oof_folds, cpcv_validate_returns
from scalper_hft.strategies import get_strategy
from scalper_hft.strategies.regime_supervisor import RegimeSupervisor
from scalper_hft.validation.holdout import DEFAULT_OPTUNA_HOLDOUT_PCT, resolve_optuna_holdout_pct
from scalper_hft.validation.optimize import evaluate_holdout
from scalper_hft.validation.regime_map import (
    apply_oos_preferred_regimes,
    compute_regime_perf_matrix,
    preferred_regimes_book_from_matrix,
    preferred_regimes_from_matrix,
)
from scalper_hft.validation.stress import cost_concentration_stress, drop_top_trade_returns
from scalper_hft.validation.sweep import haircut_roster


def _ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0},
        index=idx,
    )


def _perf_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    labels = np.where(np.arange(n) % 2 == 0, "trend_up|normal", "range|low")
    regime_df = pd.DataFrame({"label": labels}, index=idx)
    ret = pd.DataFrame(index=idx)
    ret["mean_reversion"] = np.where(labels == "range|low", 0.001, -0.0002) + rng.normal(0, 0.0005, n)
    ret["supertrend"] = np.where(labels == "trend_up|normal", 0.001, -0.0002) + rng.normal(0, 0.0005, n)
    return ret, regime_df


def test_preferred_regimes_from_oos_matrix() -> None:
    ret, regime = _perf_data()
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    mr = preferred_regimes_from_matrix(m, "mean_reversion", min_sharpe=0.0)
    st = preferred_regimes_from_matrix(m, "supertrend", min_sharpe=0.0)
    assert "range" in mr and "low" in mr
    assert "trend_up" in st and "normal" in st
    book = preferred_regimes_book_from_matrix(m)
    assert set(book) >= {"mean_reversion", "supertrend"}


def test_apply_oos_preferred_does_not_wipe_empty() -> None:
    """Немає додатної комірки — гіпотеза класу лишається."""
    strat = get_strategy("pairs_arb")
    before = frozenset(strat.preferred_regimes)
    empty = compute_regime_perf_matrix(
        pd.DataFrame({"pairs_arb": [0.0] * 40}, index=pd.date_range("2025-01-01", periods=40, freq="1h")),
        pd.DataFrame({"label": ["range|normal"] * 40}, index=pd.date_range("2025-01-01", periods=40, freq="1h")),
        min_bars=100,
    )
    apply_oos_preferred_regimes([strat], empty)
    assert frozenset(strat.preferred_regimes) == before


def test_supervisor_applies_perf_matrix(tmp_path: Path) -> None:
    ret, regime = _perf_data()
    m = compute_regime_perf_matrix(ret, regime, min_bars=20)
    p = tmp_path / "perf.json"
    m.to_json(p)
    sup = RegimeSupervisor(
        strategies="mean_reversion,supertrend",
        blend_mode="regime_soft",
        hmm_fit_bars=80,
        perf_matrix_path=str(p),
    )
    names = {s.name: frozenset(s.preferred_regimes) for s in sup._strats}
    assert "range" in names["mean_reversion"]
    assert "trend_up" in names["supertrend"]


def test_haircut_roster_orders_survivors() -> None:
    w = pd.DataFrame(
        [
            {"winner_strategy": "a", "deflated_sharpe": 0.4, "survives_haircut": True},
            {"winner_strategy": "b", "deflated_sharpe": 1.2, "survives_haircut": True},
            {"winner_strategy": "c", "deflated_sharpe": 9.0, "survives_haircut": False},
            {"winner_strategy": "b", "deflated_sharpe": 0.8, "survives_haircut": True},
        ]
    )
    assert haircut_roster(w, max_size=5) == ["b", "a"]
    assert haircut_roster(w, max_size=1) == ["b"]
    assert haircut_roster(pd.DataFrame()) == []


def test_supervisor_from_haircut_roster() -> None:
    w = pd.DataFrame(
        [
            {"winner_strategy": "supertrend", "deflated_sharpe": 0.9, "survives_haircut": True},
            {"winner_strategy": "mean_reversion", "deflated_sharpe": 0.4, "survives_haircut": True},
            {"winner_strategy": "cvd_momentum", "deflated_sharpe": 2.0, "survives_haircut": False},
        ]
    )
    sup = RegimeSupervisor.from_haircut_roster(w, blend_mode="regime_soft", hmm_fit_bars=80)
    assert sup._strat_names == ["supertrend", "mean_reversion"]
    with pytest.raises(ValueError, match="порожній"):
        RegimeSupervisor.from_haircut_roster(
            pd.DataFrame([{"winner_strategy": "x", "survives_haircut": False, "deflated_sharpe": 1.0}])
        )


def test_oof_folds_purge_t1_overlap() -> None:
    n = 90
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    t1 = pd.Series(idx + pd.Timedelta("3h"), index=idx)
    folds = _oof_folds(X, t1=t1, n_splits=3, embargo_pct=0.01)
    assert len(folds) >= 2
    tr, te = folds[-1]
    assert len(set(tr) & set(te)) == 0
    t_test_start = X.index[int(te[0])]
    assert all(t1.iloc[int(i)] <= t_test_start for i in tr)
    ts = _oof_folds(X, t1=None, n_splits=3, gap=2)
    assert len(ts) >= 1


def test_cpcv_validate_returns_matches_pbo() -> None:
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0004, 0.01, (6, 800))
    res = cpcv_validate_returns(rets, n_blocks=6, max_combos=50)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_variants == 6


def test_seq_bootstrap_seed_reproducible() -> None:
    from scalper_hft.ml.sample_weights import seq_bootstrap

    idx = pd.date_range("2025-01-01", periods=40, freq="1h")
    t1 = pd.Series(idx[:12] + pd.Timedelta("2h"), index=idx[:12])
    from scalper_hft.ml.sample_weights import get_ind_matrix

    ind = get_ind_matrix(idx, t1)
    a = seq_bootstrap(ind, s_length=8, seed=7)
    b = seq_bootstrap(ind, s_length=8, seed=7)
    c = seq_bootstrap(ind, s_length=8, seed=8)
    assert a == b
    assert a != c


def test_resolve_optuna_holdout_pct() -> None:
    assert resolve_optuna_holdout_pct(0.0) == DEFAULT_OPTUNA_HOLDOUT_PCT
    assert resolve_optuna_holdout_pct(0.3) == pytest.approx(0.3)


def test_evaluate_holdout_short_is_nan() -> None:
    df = _ohlcv(10)
    assert np.isnan(evaluate_holdout(df, type(get_strategy("mean_reversion")), {}))


def test_evaluate_holdout_runs() -> None:
    class _Long(type(get_strategy("mean_reversion"))):
        name = "always_long"

        def generate_signals(self, df, trades=None, funding=None):
            return pd.Series(1.0, index=df.index)

    df = _ohlcv(80)
    s = evaluate_holdout(df, _Long, {})
    assert np.isfinite(s)


def test_drop_top_trades_and_cost_stress() -> None:
    trades = pd.DataFrame({"ret": [0.10, 0.01, -0.02, 0.03, 0.08, 0.00]})
    dropped = drop_top_trade_returns(trades, n=2)
    assert len(dropped) == 4
    assert 0.10 not in set(dropped) and 0.08 not in set(dropped)

    df = _ohlcv(150)
    strat = get_strategy("mean_reversion")
    cost = CostModel()
    tbl = cost_concentration_stress(df, strat, cost, position_pct=0.01, n_drop=5)
    assert list(tbl.index) == ["baseline", "fees_slippage_x2", "drop_top_5_trades"]
    assert "sharpe" in tbl.columns


def test_cost_stress_fees_x2_not_better_than_baseline() -> None:
    """Дорожче виконання не має покращувати total_return."""
    df = _ohlcv(150)
    tbl = cost_concentration_stress(df, get_strategy("mean_reversion"), CostModel(), position_pct=0.01)
    assert tbl.loc["fees_slippage_x2", "total_return"] <= tbl.loc["baseline", "total_return"] + 1e-12


def test_cmd_report_stress_section_in_markdown(monkeypatch, tmp_path: Path) -> None:
    """cmd_report завжди додає секцію стрес (fees×2 / top-5)."""
    from argparse import Namespace

    from scalper_hft.cli.research_audit import cmd_report

    df = _ohlcv(250)
    monkeypatch.setattr("scalper_hft.cli._load_klines", lambda *a, **k: df)
    monkeypatch.setattr("scalper_hft.cli._common._load_optional_streams", lambda *a, **k: (None, None))
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        strategy="mean_reversion",
        symbol="BTCUSDT",
        interval="1h",
        days=10,
        train=80,
        test=40,
        trials=1,
        param_dict={},
        base=None,
        derive=True,
    )
    cmd_report(args)
    text = (tmp_path / "docs/reports/mean_reversion_BTCUSDT_1h.md").read_text(encoding="utf-8")
    assert "fees×2" in text
    assert "топ-5" in text
