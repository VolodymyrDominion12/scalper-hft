"""Тести Спринту 3 (book_approaches_synthesis.md):

1. Стрес-тест + портфельний risk budget (Narang гл. 4/10)
2. Динамічний розмір/лімітна ціна через сигмоїду (AFML Ch.10.6)
3. Capacity-тест (PM Ch.4 «share of wallet»)
4. Survival/Kaplan–Meier аналіз (PM Ch.7/12/13)
5. Інтеграція micro/HMM/GARCH фіч у ml/features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 500, seed: int = 31) -> pd.DataFrame:
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


def _make_trades(n: int = 2000, seed: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="1s")
    price = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    amount = rng.uniform(0.001, 0.01, n)
    side = np.where(rng.random(n) > 0.5, "buy", "sell")
    return pd.DataFrame({"price": price, "amount": amount, "side": side}, index=ts)


# ── 1. Стрес-тест + risk budget ──────────────────────────────────────────────


class TestStress:
    def test_crash_amplifies_worst_window(self):
        from scalper_hft.validation.stress import apply_stress

        rng = np.random.default_rng(1)
        ret = pd.Series(rng.normal(0.0001, 0.005, 500))
        base_worst = ret.rolling(24).sum().min()
        stressed = apply_stress(ret, crash_mult=2.0, crash_bars=24)
        stressed_worst = stressed.rolling(24).sum().min()
        assert stressed_worst < base_worst * 1.5  # найгірше вікно посилене

    def test_liquidity_reduces_returns(self):
        from scalper_hft.validation.stress import apply_stress

        ret = pd.Series(np.full(300, 0.001))
        stressed = apply_stress(ret, cost_mult=10.0, cost_frac=0.0005)
        assert stressed.mean() < 0.001

    def test_stress_report_structure(self):
        from scalper_hft.validation.stress import stress_report

        rng = np.random.default_rng(2)
        ret = pd.Series(rng.normal(0.0001, 0.005, 400))
        rep = stress_report(ret, scenarios=["crash", "liquidity"])
        assert set(rep.index) == {"baseline", "crash", "liquidity"}
        for col in ["total_return", "sharpe", "max_drawdown", "worst_period"]:
            assert col in rep.columns

    def test_vol_target_scale(self):
        from scalper_hft.portfolio.risk_budget import vol_target_scale

        rng = np.random.default_rng(3)
        r = pd.DataFrame(rng.normal(0, 0.01, (1000, 3)))
        w = np.array([1 / 3] * 3)
        s = vol_target_scale(r, w, target_vol=0.005)
        port = (r.values @ w) * s
        assert abs(float(np.std(port)) - 0.005) < 0.0005

    def test_portfolio_var_and_loss_split(self):
        from scalper_hft.portfolio.risk_budget import loss_budget_split, portfolio_var

        rng = np.random.default_rng(4)
        r = pd.DataFrame(rng.normal(0, [0.01, 0.03], (1000, 2)), columns=["a", "b"])
        w = np.array([0.7, 0.3])
        var = portfolio_var(r, w, alpha=0.05)
        assert var > 0
        split = loss_budget_split(r, w, daily_loss_limit=0.03)
        assert abs(sum(split.values()) - 0.03) < 1e-9
        assert split["b"] > split["a"]  # ризикова нога отримує більшу частку ліміту


# ── 2. Сигмоїдний sizing + лімітна ціна (AFML Ch.10.6) ───────────────────────


class TestSigmoidSizing:
    def test_sigmoid_size_bounds_and_monotone(self):
        from scalper_hft.ml.bet_sizing import sigmoid_size

        x = np.linspace(-20, 20, 50)
        m = sigmoid_size(x, omega=0.5)
        assert np.all(m > -1.0) and np.all(m < 1.0)
        assert np.all(np.diff(m) > 0)
        assert abs(float(sigmoid_size(0.0, 0.5))) < 1e-9

    def test_calibrate_omega(self):
        from scalper_hft.ml.bet_sizing import calibrate_omega

        omega = calibrate_omega(x_star=10.0, m_star=0.95)
        assert 0.2 < omega < 1.0
        with pytest.raises(ValueError):
            calibrate_omega(x_star=0.0, m_star=0.5)

    def test_limit_price_is_inverse(self):
        """L = f − (1/ω)·ln((1+m)/(1−m)) — обернена до sigmoid_size."""
        from scalper_hft.ml.bet_sizing import calibrate_omega, limit_price, sigmoid_size

        omega = calibrate_omega(x_star=10.0, m_star=0.95)
        f, x = 100.0, 5.0
        m = sigmoid_size(x, omega)
        L = limit_price(omega, m, f)
        assert abs(float(L) - (f - x)) < 1e-6

    def test_target_size_integer(self):
        from scalper_hft.ml.bet_sizing import calibrate_omega, target_size

        omega = calibrate_omega(x_star=10.0, m_star=0.95)
        q = target_size(Q=100, omega=omega, fair_value=np.array([105.0]), price=np.array([100.0]))
        assert q.dtype.kind == "i"
        assert 0 < q[0] <= 100


# ── 3. Capacity-тест ─────────────────────────────────────────────────────────


class TestCapacity:
    def test_capacity_curve_structure(self):
        from scalper_hft.strategies.mean_reversion import MeanReversionScalper
        from scalper_hft.validation.capacity import capacity_curve

        df = _make_df(400)
        curve = capacity_curve(df, MeanReversionScalper(), scales=[1.0, 2.0])
        assert set(curve["scale"].tolist()) == {1.0, 2.0}
        for col in ["total_return", "sharpe", "max_drawdown", "impact_bps"]:
            assert col in curve.columns

    def test_saturation_scale(self):
        from scalper_hft.validation.capacity import saturation_scale

        curve = pd.DataFrame({"scale": [1.0, 2.0, 5.0, 10.0], "sharpe": [2.0, 1.8, 1.0, 0.4]})
        sat = saturation_scale(curve, drop_threshold=0.5)
        assert sat == 5.0  # scale 5 ще ≥ поріг (1.0), scale 10 вже нижче
        curve2 = pd.DataFrame({"scale": [1.0, 2.0], "sharpe": [2.0, 1.9]})
        assert saturation_scale(curve2) == 2.0  # не падає — останній масштаб


# ── 4. Survival / Kaplan–Meier ────────────────────────────────────────────────


class TestSurvival:
    def _trades(self):
        idx = pd.date_range("2025-01-01", periods=40, freq="1min")
        rows = []
        for i, t in enumerate(idx):
            rows.append({"entry_ts": t, "exit_ts": t + pd.Timedelta(minutes=1 + i % 5), "side": 1, "ret": 0.01})
        return pd.DataFrame(rows)

    def test_trade_durations(self):
        from scalper_hft.validation.survival import trade_durations

        dur = trade_durations(self._trades(), freq="1min")
        assert "duration" in dur.columns and "event" in dur.columns
        assert (dur["duration"] >= 1.0).all()
        assert dur["event"].isin([0, 1]).all()

    def test_kaplan_meier_monotone(self):
        from scalper_hft.validation.survival import kaplan_meier

        rng = np.random.default_rng(5)
        d = rng.exponential(10, 300)
        e = np.ones(300)
        km = kaplan_meier(d, e)
        assert km["survival"].is_monotonic_decreasing
        assert 0.0 <= km["survival"].iloc[-1] <= 1.0

    def test_median_survival_time(self):
        from scalper_hft.validation.survival import kaplan_meier, median_survival_time

        km = kaplan_meier(np.full(200, 5.0), np.ones(200))
        assert median_survival_time(km) == 5.0

    def test_survival_by_feature(self):
        from scalper_hft.validation.survival import survival_by_feature

        trades = self._trades()
        feat = pd.Series(np.random.default_rng(6).normal(0, 1, len(trades)), index=pd.to_datetime(trades["entry_ts"]))
        sb = survival_by_feature(trades, feat, n_bins=3, freq="1min")
        assert "median_hold" in sb.columns and len(sb) >= 2


# ── 5. Інтеграція micro/HMM/GARCH у ML-фічі ──────────────────────────────────


class TestMlFeatureIntegration:
    def test_garch_and_hmm_features_added(self):
        from scalper_hft.ml.features import build_labeled_dataset

        df = _make_df(600)
        X, y, w = build_labeled_dataset(
            df, mode="triple_barrier", holding_bars=8, add_hmm=True, add_garch=True, hmm_states=3
        )
        assert "garch_sigma" in X.columns
        assert "hmm_state" in X.columns
        assert any(c.startswith("hmm_p") for c in X.columns)
        assert X["garch_sigma"].notna().all()

    def test_micro_features_added_with_trades(self):
        from scalper_hft.ml.features import build_labeled_dataset

        df = _make_df(600)
        trades = _make_trades()
        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8, trades=trades)
        for col in ["vpin", "kyle_t", "roll_spread", "amihud", "parkinson_vol", "signed_flow_ac"]:
            assert col in X.columns, f"{col} не додано"
        assert X["vpin"].notna().all()

    def test_default_without_trades_unaffected(self):
        """Без trades і без нових прапорців — колишній набір фіч."""
        from scalper_hft.ml.features import build_labeled_dataset

        df = _make_df(600)
        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8)
        assert "vpin" not in X.columns
        assert "garch_sigma" not in X.columns
        assert "hmm_state" not in X.columns

    def test_no_lookahead_garch(self):
        """garch_sigma на барі t використовує лише дані до t (позитивний, скінченний)."""
        from scalper_hft.ml.features import build_labeled_dataset

        df = _make_df(400)
        X, y, w = build_labeled_dataset(df, mode="triple_barrier", holding_bars=8, add_garch=True)
        assert np.isfinite(X["garch_sigma"]).all()
        assert (X["garch_sigma"] > 0).all()
