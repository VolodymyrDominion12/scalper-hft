"""Тести Спринту 1 (book_approaches_synthesis.md):

1. Bet sizing із імовірностей (AFML Ch.10.3) — ml/bet_sizing.py
2. Мета-лейблінг + probabilities у walk-forward — ml/trainer.py, strategies/ml_strategy.py
3. Breakeven-гейт (Narang гл. 5) — backtest/execution.py, backtest/engine.py
4. Cohort analysis (Predictive Marketing) — validation/cohort.py
5. Децильний lift (Predictive Marketing) — validation/lift.py
6. Hedge-блендінг (Gofer Ch.2) — strategies/blend.py, strategies/ensemble.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_trades(n_per_month: int = 10, n_months: int = 6, trend: float = -0.01) -> pd.DataFrame:
    """Синтетичні угоди зі спадним PnL по місяцях (для cohort-тестів)."""
    rows = []
    start = pd.Timestamp("2025-01-01")
    for m in range(n_months):
        month_start = start + pd.DateOffset(months=m)
        base = 0.05 + trend * m
        rng = np.random.default_rng(m)
        for i in range(n_per_month):
            ts = month_start + pd.Timedelta(days=i + 1)
            rows.append(
                {
                    "entry_ts": ts,
                    "exit_ts": ts + pd.Timedelta(hours=1),
                    "side": 1,
                    "ret": base + rng.normal(0, 0.005),
                }
            )
    return pd.DataFrame(rows)


# ── 1. Bet sizing ─────────────────────────────────────────────────────────────


class TestBetSizing:
    def test_prob_to_size_zero_at_half(self):
        from scalper_hft.ml.bet_sizing import prob_to_size

        m = prob_to_size(np.array([0.5]))
        assert abs(float(m[0])) < 1e-9

    def test_prob_to_size_monotone_bounded(self):
        from scalper_hft.ml.bet_sizing import prob_to_size

        p = np.linspace(0.05, 0.95, 20)
        m = prob_to_size(p)
        assert np.all(np.diff(m) > 0), "мапа має бути монотонною"
        assert np.all(m > -1.0) and np.all(m < 1.0)

    def test_prob_to_size_symmetric(self):
        from scalper_hft.ml.bet_sizing import prob_to_size

        assert abs(float(prob_to_size(np.array([0.3]))[0] + prob_to_size(np.array([0.7]))[0])) < 1e-9

    def test_meta_size_skips_below_half(self):
        from scalper_hft.ml.bet_sizing import meta_size

        m = meta_size(np.array([0.4, 0.5, 0.6, 0.9]))
        assert m[0] == 0.0 and m[1] == 0.0
        assert m[2] > 0.0 and m[3] > m[2]
        assert np.all(m <= 1.0)

    def test_discretize(self):
        from scalper_hft.ml.bet_sizing import discretize

        d = discretize(np.array([0.13, 0.27, 0.51]), step=0.2)
        assert np.allclose(d, [0.2, 0.2, 0.6])


# ── 2. Мета-лейблінг + probabilities ─────────────────────────────────────────


class TestWalkForwardProbabilities:
    def test_train_walk_forward_returns_probabilities(self):
        pytest.importorskip("lightgbm")
        from scalper_hft.ml.features import build_labeled_dataset
        from scalper_hft.ml.trainer import train_walk_forward

        df = _make_df(800)
        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8)
        res = train_walk_forward(X, y, train_size=150, test_size=60, sample_weights=w, close=df["close"])
        assert res.probabilities is not None
        assert len(res.probabilities) == len(res.predictions)
        assert res.probabilities.between(0.0, 1.0).all()

    def test_meta_pipeline_runs_and_bounded(self):
        pytest.importorskip("lightgbm")
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy("ml_strategy", train_bars=150, test_bars=60, holding_bars=8, meta_filter=True)
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert signals.index.equals(df.index)
        assert signals.abs().max() <= 1.0

    def test_meta_probabilities_not_degenerate(self):
        """OOF-мета-мітки (без IS-оптимізму) → p_meta має реальний розкид."""
        pytest.importorskip("lightgbm")
        from scalper_hft.ml.features import build_labeled_dataset
        from scalper_hft.ml.trainer import train_walk_forward_meta

        df = _make_df(800)
        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8)
        side, p_meta = train_walk_forward_meta(X, y, train_size=150, test_size=60, sample_weights=w)
        assert len(p_meta) == len(side)
        assert p_meta.between(0.0, 1.0).all()
        assert p_meta.std() > 1e-3, "p_meta дегенеративний (~0.5 всюди) — мета нічого не вчить"

    def test_confidence_filter_works(self):
        """Без мета і без sizing: з confidence_thr сигнали лишаються {-1, 0, +1}."""
        pytest.importorskip("lightgbm")
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy(
            "ml_strategy",
            train_bars=150,
            test_bars=60,
            holding_bars=8,
            confidence_thr=0.55,
            prob_size=False,
            meta_filter=False,
        )
        signals = strat.generate_signals(df)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_prob_size_mode_continuous(self):
        pytest.importorskip("lightgbm")
        from scalper_hft.strategies import get_strategy

        df = _make_df(800)
        strat = get_strategy("ml_strategy", train_bars=150, test_bars=60, holding_bars=8, prob_size=True)
        signals = strat.generate_signals(df)
        assert signals.abs().max() <= 1.0
        assert any(abs(v) > 0 and abs(v) < 1.0 for v in signals.unique()), "prob_size має давати безперервні розміри"


# ── 3. Breakeven-гейт ─────────────────────────────────────────────────────────


class TestBreakevenGate:
    def test_gate_zeroes_weak_moves(self):
        from scalper_hft.backtest.execution import CostModel, apply_breakeven_gate

        df = _make_df(100)
        df["atr_14"] = df["close"] * 0.00005  # 0.5 bps — менше round-trip taker (14 bps)
        signals = pd.Series(1.0, index=df.index)
        out = apply_breakeven_gate(signals, df, CostModel(), is_maker=False)
        assert (out == 0.0).all()

    def test_gate_keeps_strong_moves(self):
        from scalper_hft.backtest.execution import CostModel, apply_breakeven_gate

        df = _make_df(100)
        df["atr_14"] = df["close"] * 0.01  # 1% — значно більше round-trip
        signals = pd.Series(1.0, index=df.index)
        out = apply_breakeven_gate(signals, df, CostModel(), is_maker=False)
        assert (out == 1.0).all()

    def test_engine_hook(self):
        from scalper_hft.backtest.engine import run_backtest
        from scalper_hft.backtest.execution import CostModel
        from scalper_hft.strategies.base import Strategy

        class _Flat(Strategy):
            name = "flat_bt"

            def generate_signals(self, df, trades=None, funding=None):
                return pd.Series(1.0, index=df.index)

        # дуже низька волатильність: ATR ~ 1e-5 << round-trip taker (14 bps)
        rng = np.random.default_rng(11)
        idx = pd.date_range("2025-01-01", periods=100, freq="1min")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.00001, 100)))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.0001,
                "low": close * 0.9999,
                "close": close,
                "volume": np.full(100, 50.0),
            },
            index=idx,
        )
        strat = _Flat()
        # без гейту — є сигнали
        res = run_backtest(df, strat, cost=CostModel())
        assert (res.positions != 0).any()
        # з гейтом — ATR фолбек малий → все вимкнено
        strat.use_breakeven_gate = True
        res2 = run_backtest(df, strat, cost=CostModel())
        assert (res2.positions == 0).all()


# ── 4. Cohort analysis ────────────────────────────────────────────────────────


class TestCohort:
    def test_cohort_metrics_structure(self):
        from scalper_hft.validation.cohort import cohort_metrics

        trades = _make_trades()
        c = cohort_metrics(trades, freq="ME")
        assert len(c) == 6
        for col in ["n_trades", "pnl_per_trade", "win_rate", "cum_pnl", "sharpe"]:
            assert col in c.columns
        assert (c["n_trades"] == 10).all()

    def test_cohort_decay_detects_downtrend(self):
        from scalper_hft.validation.cohort import cohort_decay, cohort_metrics

        trades = _make_trades(trend=-0.02)
        c = cohort_metrics(trades, freq="ME")
        d = cohort_decay(c, metric="pnl_per_trade")
        assert d["slope"] < 0
        assert d["decaying"] is True

    def test_cohort_decay_small_sample_no_false_positive(self):
        from scalper_hft.validation.cohort import cohort_decay

        c = pd.DataFrame(
            {"pnl_per_trade": [0.01, 0.02]},
            index=["2025-01", "2025-02"],
        )
        d = cohort_decay(c)
        assert d["decaying"] is False  # замало когорт


# ── 5. Децильний lift ─────────────────────────────────────────────────────────


class TestLift:
    def test_decile_lift_strong_feature(self):
        from scalper_hft.validation.lift import decile_lift

        rng = np.random.default_rng(1)
        n = 200
        feat = rng.normal(0, 1, n)
        pnl = 0.02 * feat + rng.normal(0, 0.001, n)  # сильно корелює
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        lift = decile_lift(pd.Series(pnl, index=idx), pd.Series(feat, index=idx), n_bins=10)
        assert len(lift) >= 5
        assert lift["lift"].abs().max() > 0.005  # lift значущий

    def test_decile_lift_noise_feature(self):
        from scalper_hft.validation.lift import decile_lift

        rng = np.random.default_rng(2)
        n = 200
        feat = rng.normal(0, 1, n)
        pnl = rng.normal(0, 0.001, n)  # незалежна
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        lift = decile_lift(pd.Series(pnl, index=idx), pd.Series(feat, index=idx), n_bins=10)
        assert lift["lift"].abs().max() < 0.005

    def test_feature_lift_report(self):
        from scalper_hft.validation.lift import feature_lift_report, lift_summary

        trades = _make_trades(n_per_month=20, n_months=3, trend=0.0)
        n = len(trades)
        rng = np.random.default_rng(3)
        good = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n)
        trades["ret"] = 0.01 * good + rng.normal(0, 0.001, n)  # ret корелює з good
        feats = pd.DataFrame({"good": good, "noise": noise}, index=trades["entry_ts"])
        report = feature_lift_report(trades, feats, pnl_col="ret", n_bins=5)
        assert set(report.keys()) == {"good", "noise"}
        summary = lift_summary(report)
        assert len(summary) == 2
        good_lift = summary.loc[summary["feature"] == "good", "max_abs_lift"].iloc[0]
        noise_lift = summary.loc[summary["feature"] == "noise", "max_abs_lift"].iloc[0]
        assert good_lift > noise_lift


# ── 6. Hedge-блендінг ─────────────────────────────────────────────────────────


class TestHedge:
    def test_weights_rows_sum_to_one(self):
        from scalper_hft.strategies.blend import hedge_weights

        rng = np.random.default_rng(5)
        r = rng.normal(0.0005, 0.01, (200, 3))
        w = hedge_weights(r)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert np.allclose(w.iloc[0].values, 1.0 / 3)  # перший рядок — рівні ваги

    def test_vectorized_matches_incremental(self):
        from scalper_hft.strategies.blend import HedgeBlend, hedge_weights

        rng = np.random.default_rng(6)
        r = rng.normal(0.0003, 0.01, (150, 2))
        w_vec = hedge_weights(r, eta=0.5)
        hb = HedgeBlend(n_experts=2, eta=0.5)
        w_inc = [hb.weights().copy()]
        for row in r:
            hb.step(row)
            w_inc.append(hb.weights().copy())
        w_inc = np.array(w_inc[:-1])  # вирівнюємо: w_inc[t] = ваги перед баром t
        assert np.allclose(w_vec.values, w_inc, atol=1e-6)

    def test_blend_favors_good_expert(self):
        from scalper_hft.strategies.blend import hedge_weights

        rng = np.random.default_rng(8)
        n = 400
        good = np.full(n, 0.001) + rng.normal(0, 0.001, n)
        bad = np.full(n, -0.001) + rng.normal(0, 0.001, n)
        w = hedge_weights(np.column_stack([good, bad]))
        assert w.iloc[-1, 0] > 0.9, "Hedge має сконцентруватися на хорошому експерті"

    def test_ensemble_hedge_mode(self):
        from scalper_hft.strategies import get_strategy

        df = _make_df(400)
        strat = get_strategy(
            "ensemble", strategies="mean_reversion,mean_reversion", mode="hedge", mean_reversion_period=20
        )
        # простіший тест: дві однакові стратегії → сигнал у межах
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert signals.abs().max() <= 1.0

    def test_hedge_blend_signals(self):
        from scalper_hft.strategies.blend import hedge_blend_signals

        df = _make_df(300)
        rng = np.random.default_rng(9)
        sig = pd.DataFrame(
            {
                "a": np.where(rng.random(len(df)) > 0.5, 1.0, -1.0),
                "b": -np.where(rng.random(len(df)) > 0.5, 1.0, -1.0),
            },
            index=df.index,
        )
        out = hedge_blend_signals(sig, df["close"])
        assert out.abs().max() <= 1.0
        assert len(out) == len(df)
