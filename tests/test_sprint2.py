"""Тести Спринту 2 (book_approaches_synthesis.md):

1. Мікроструктурні фічі VPIN/Kyle λ/Roll/Amihud/Corwin-Schultz (AFML Ch.19)
2. HMM-режими (FSPML Ch.4.5, numpy-реалізація)
3. GARCH σ_{t+1} (FSPML Ch.7–8)
4. Емпіричний CostModel: vol-scaled slippage + Square-Root impact (Narang гл. 5)
5. ERC/risk-parity алокація з turnover tax (Narang гл. 6)
6. MDI/MDA/SFI feature importance (AFML Ch.8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_klines(n: int = 500, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_trades(n: int = 3000, seed: int = 22, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="1s")
    price = 100.0 + np.cumsum(rng.normal(drift, 0.01, n))
    amount = rng.uniform(0.001, 0.01, n)
    side = np.where(rng.random(n) > 0.5, "buy", "sell")
    return pd.DataFrame({"price": price, "amount": amount, "side": side}, index=ts)


# ── 1. Мікроструктура ────────────────────────────────────────────────────────

class TestMicrostructure:
    def test_volume_bars(self):
        from scalper_hft.features.microstructure import volume_bars

        trades = _make_trades()
        bars = volume_bars(trades, bar_volume=1.0)
        assert not bars.empty
        assert {"open", "high", "low", "close", "volume", "buy_volume", "sell_volume"} <= set(bars.columns)
        assert bars["volume"].between(0.9, 1.1).all()
        assert bars["buy_volume"].sum() > 0 and bars["sell_volume"].sum() > 0

    def test_vpin_in_range(self):
        from scalper_hft.features.microstructure import vpin

        v = vpin(_make_trades(), bar_volume=1.0, n=30)
        v = v.dropna()
        assert len(v) > 0
        assert v.between(0.0, 1.0).all()

    def test_kyle_lambda_positive(self):
        """Δp = 0.001·sv + шум → λ ≈ 0.001, t > 0."""
        from scalper_hft.features.microstructure import kyle_lambda

        rng = np.random.default_rng(3)
        n = 300
        idx = pd.date_range("2025-01-01", periods=n, freq="1min")
        sv = rng.normal(0, 1, n)
        dp = 0.001 * sv + rng.normal(0, 0.0001, n)
        close = 100.0 + np.cumsum(dp)
        lam, t = kyle_lambda(pd.Series(close, index=idx), pd.Series(sv, index=idx))
        assert 0.0005 < lam < 0.002
        assert t > 5

    def test_roll_spread_nonneg(self):
        from scalper_hft.features.microstructure import roll_spread

        s = roll_spread(_make_klines()["close"], window=20)
        assert s.dropna().ge(0).all()
        assert s.notna().sum() > 100

    def test_amihud_positive(self):
        from scalper_hft.features.microstructure import amihud

        df = _make_klines()
        a = amihud(df["close"].pct_change().fillna(0.0), df["volume"] * df["close"])
        assert a.dropna().ge(0).all()

    def test_corwin_schultz_and_parkinson(self):
        from scalper_hft.features.microstructure import corwin_schultz_spread, parkinson_vol

        df = _make_klines()
        cs = corwin_schultz_spread(df["high"], df["low"]).dropna()
        assert len(cs) > 0
        assert cs.ge(0).all()  # негативні оцінювачі кліпуються в 0 (стандартна практика)
        assert cs.sum() > 0
        pv = parkinson_vol(df["high"], df["low"]).dropna()
        assert pv.gt(0).all()

    def test_signed_flow_autocorr(self):
        from scalper_hft.features.microstructure import signed_flow_autocorr

        ac = signed_flow_autocorr(_make_trades(), resample="1min", lags=1, window=20).dropna()
        assert len(ac) > 0
        assert ac.between(-1.0, 1.0).all()

    def test_add_microstructure_features(self):
        from scalper_hft.features.microstructure import add_microstructure_features

        df = _make_klines()
        out = add_microstructure_features(df, _make_trades())
        for col in ["vpin", "kyle_t", "roll_spread", "amihud", "parkinson_vol", "signed_flow_ac"]:
            assert col in out.columns
        assert len(out) == len(df)


# ── 2. HMM-режими ────────────────────────────────────────────────────────────

class TestHmmRegime:
    def test_hmm_recovers_two_regimes(self):
        from scalper_hft.features.hmm_regime import GaussianHMM

        rng = np.random.default_rng(0)
        n = 1000
        blocks = np.repeat([0, 1], n // 2)
        mu = np.where(blocks == 0, -0.003, 0.003)
        obs = (mu + rng.normal(0, 0.005, n))[:, None]
        model = GaussianHMM(n_states=2, n_iter=100, seed=1).fit(obs)
        # стан 0 = нижчий mean (сортування за середнім першої фічі)
        corr = float(np.corrcoef(model.states_, blocks)[0, 1])
        assert abs(corr) > 0.7, f"HMM не відновив режими: corr={corr:.3f}"
        assert model.transmat_ is not None and model.means_ is not None

    def test_hmm_regime_features(self):
        from scalper_hft.features.hmm_regime import hmm_regime_features

        close = _make_klines(600)["close"]
        out = hmm_regime_features(close, n_states=3, seed=3)
        assert "hmm_state" in out.columns
        assert any(c.startswith("hmm_p") for c in out.columns)
        assert len(out) == len(close)
        pcols = [c for c in out.columns if c.startswith("hmm_p")]
        # після прогріву (20 барів) P(режим) мають сумуватися до 1
        assert out[pcols].iloc[30:].sum(axis=1).sub(1.0).abs().max() < 0.1


# ── 3. GARCH ─────────────────────────────────────────────────────────────────

class TestGarch:
    def _sim_garch(self, n: int = 2000, seed: int = 4, alpha: float = 0.08, beta: float = 0.9) -> np.ndarray:
        rng = np.random.default_rng(seed)
        omega = 1e-6
        var = np.zeros(n)
        r = np.zeros(n)
        var[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
            r[t] = np.sqrt(var[t]) * rng.normal()
        return r

    def test_fit_recovers_params(self):
        from scalper_hft.features.volatility import garch11_fit

        r = self._sim_garch()
        omega, alpha, beta = garch11_fit(r)
        assert 0.02 < alpha < 0.2
        assert 0.7 < beta < 0.99
        assert omega > 0

    def test_garch11_vol_positive(self):
        from scalper_hft.features.volatility import garch11_vol

        r = pd.Series(self._sim_garch(500))
        v = garch11_vol(r, omega=1e-6, alpha=0.08, beta=0.9)
        assert v.gt(0).all()

    def test_forecast_finite_and_aligned(self):
        from scalper_hft.features.volatility import garch_forecast

        r = pd.Series(self._sim_garch(600), index=pd.date_range("2025-01-01", periods=600, freq="1min"))
        f = garch_forecast(r, window=300, refit_every=50, warmup=50)
        assert f.notna().sum() > 400
        assert f.dropna().gt(0).all()

    def test_ewma_vol(self):
        from scalper_hft.features.volatility import ewma_vol

        v = ewma_vol(pd.Series(self._sim_garch(300)), span=20)
        assert v.iloc[10:].gt(0).all()

    def test_arch_lm(self):
        from scalper_hft.features.volatility import arch_lm_test

        lm, p_arch = arch_lm_test(pd.Series(self._sim_garch(1000)), lags=5)
        assert p_arch < 0.05, "ARCH-ефект не виявлено на GARCH-даних"
        rng = np.random.default_rng(5)
        _, p_noise = arch_lm_test(pd.Series(rng.normal(0, 1, 1000)), lags=5)
        assert p_noise > 0.05, "хибне виявлення ARCH на білому шумі"


# ── 4. Емпіричний CostModel ──────────────────────────────────────────────────

class TestCostModelSprint2:
    def test_vol_aware_slippage(self):
        from scalper_hft.backtest.execution import CostModel

        c = CostModel(slippage_frac=0.0002, vol_ref=0.01)
        assert c.vol_aware_slippage(0.01) == pytest.approx(0.0002)
        assert c.vol_aware_slippage(0.02) == pytest.approx(0.0004)
        assert c.vol_aware_slippage(0.005) == pytest.approx(0.0001)
        # без vol_ref — базовий slippage
        c0 = CostModel(slippage_frac=0.0002)
        assert c0.vol_aware_slippage(0.5) == pytest.approx(0.0002)

    def test_sqrt_law_impact(self):
        from scalper_hft.backtest.execution import CostModel

        c = CostModel(impact_k=0.1)
        assert c.sqrt_law_impact(qty_notional=0, adv_notional=1e6, sigma_frac=0.01) == 0.0
        i1 = c.sqrt_law_impact(qty_notional=1e4, adv_notional=1e6, sigma_frac=0.01)
        i2 = c.sqrt_law_impact(qty_notional=1e5, adv_notional=1e6, sigma_frac=0.01)
        assert 0 < i1 < i2
        assert i2 <= 0.1 * 0.01  # share ≤ 1

    def test_total_cost_combines(self):
        from scalper_hft.backtest.execution import CostModel

        c = CostModel(slippage_frac=0.0002, impact_k=0.1, vol_ref=0.01)
        base = c.total_cost_per_side(is_maker=True)
        scaled = c.total_cost_per_side(is_maker=True, vol_frac=0.02,
                                       qty_notional=1e4, adv_notional=1e6, sigma_frac=0.01)
        assert scaled > base

    def test_estimate_impact_k(self):
        from scalper_hft.backtest.execution import estimate_impact_k_from_bars

        k = estimate_impact_k_from_bars(_make_klines(300))
        assert np.isfinite(k) and k > 0

    def test_estimate_spread_from_bookticker(self):
        from scalper_hft.backtest.execution import estimate_spread_from_bookticker

        idx = pd.date_range("2025-01-01", periods=100, freq="1s")
        bt = pd.DataFrame({"bid": 100.0 + np.random.default_rng(1).normal(0, 0.001, 100),
                           "ask": 100.02 + np.random.default_rng(1).normal(0, 0.001, 100)}, index=idx)
        s = estimate_spread_from_bookticker(bt)
        assert 0.0001 < s < 0.001


# ── 5. ERC / risk-parity ─────────────────────────────────────────────────────

class TestErc:
    def _returns(self, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0, [0.01, 0.02, 0.03], (1500, 3))

    def test_erc_weights_properties(self):
        from scalper_hft.portfolio.erc import erc_weights

        w = erc_weights(self._returns(), max_weight=0.8)
        assert np.allclose(w.sum(), 1.0)
        assert (w > 0).all()
        assert (w <= 0.8).all()

    def test_erc_equal_risk_contributions(self):
        from scalper_hft.portfolio.erc import erc_weights, risk_contributions

        r = self._returns()
        w = erc_weights(r)
        cov = np.cov(r, rowvar=False)
        rc = risk_contributions(w, cov)
        shares = rc / rc.sum()
        assert np.allclose(shares, 1.0 / 3, atol=0.02), f"RC не рівні: {shares}"

    def test_erc_lower_vol_asset_gets_more_weight(self):
        from scalper_hft.portfolio.erc import erc_weights

        rng = np.random.default_rng(8)
        r = np.column_stack([rng.normal(0, 0.01, 1500), rng.normal(0, 0.05, 1500)])
        w = erc_weights(r, max_weight=0.9)
        assert w[0] > w[1], "нижчоризиковий актив має отримати більшу вагу"

    def test_allocate_portfolio_turnover_tax(self):
        from scalper_hft.portfolio.erc import allocate_portfolio

        rng = np.random.default_rng(9)
        idx = pd.date_range("2025-01-01", periods=400, freq="h")
        r = pd.DataFrame(rng.normal(0.001, 0.01, (400, 3)), index=idx)
        w = np.array([0.4, 0.3, 0.3])
        no_tax = allocate_portfolio(r, weights=w, turnover_rate=0.0, rebalance="ME")
        with_tax = allocate_portfolio(r, weights=w, turnover_rate=0.001, rebalance="ME")
        assert no_tax.sum() > with_tax.sum()

    def test_pairs_portfolio_erc_runs(self):
        """run_pairs_portfolio з method='erc' не падає на синтетиці."""
        from scalper_hft.backtest.execution import CostModel
        from scalper_hft.backtest.pairs_portfolio import run_pairs_portfolio
        from scalper_hft.strategies.pairs_arb import PairsArb

        data = {"A": _make_klines(300, seed=1), "B": _make_klines(300, seed=2),
                "C": _make_klines(300, seed=3)}
        cfg = [
            {"leg1": "A", "leg2": "B", "strategy": PairsArb(entry_z=2.0, exit_z=0.3, lookback=60)},
            {"leg1": "B", "leg2": "C", "strategy": PairsArb(entry_z=2.0, exit_z=0.3, lookback=60)},
        ]
        res = run_pairs_portfolio(data, cfg, position_pct=0.1, cost=CostModel(),
                                  method="erc", turnover_rate=0.0005)
        assert res.metrics is not None
        assert res.details["method"] == "erc"


# ── 6. Feature importance (AFML Ch.8) ────────────────────────────────────────

class TestFeatureImportance:
    def _data(self, n: int = 600, seed: int = 11):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2025-01-01", periods=n, freq="1min")
        x0 = rng.normal(0, 1, n)
        x1 = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n)
        y = np.sign(x0 + 0.8 * x1 + rng.normal(0, 0.5, n))
        X = pd.DataFrame({"good0": x0, "good1": x1, "noise": noise,
                          "noise2": rng.normal(0, 1, n)}, index=idx)
        return X, pd.Series(y, index=idx)

    def _factory(self):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier

        return lambda: LGBMClassifier(n_estimators=60, verbosity=-1, class_weight="balanced")

    def test_mdi_series(self):
        from scalper_hft.ml.feature_importance import mdi

        X, y = self._data()
        clf = self._factory()()
        clf.fit(X, y)
        imp = mdi(clf, X)
        assert set(imp.index) == set(X.columns)

    def test_mda_ranks_good_features(self):
        from scalper_hft.ml.feature_importance import mda
        from scalper_hft.validation.cv import PurgedKFold

        X, y = self._data()
        pkf = PurgedKFold(n_splits=3, embargo_pct=0.02)
        imp = mda(X, y, self._factory(), pkf, score="neg_log_loss")
        assert imp["good0"] > imp["noise"]
        assert imp["good1"] > imp["noise2"]

    def test_sfi_good_better_than_noise(self):
        from scalper_hft.ml.feature_importance import sfi
        from scalper_hft.validation.cv import PurgedKFold

        X, y = self._data()
        pkf = PurgedKFold(n_splits=3, embargo_pct=0.02)
        scores = sfi(X, y, self._factory(), pkf, score="neg_log_loss")
        # neg log loss: нижче = краще → good фічі мають вищі (менш від'ємні) значення
        assert scores["good0"] > scores["noise"]
        assert scores["good1"] > scores["noise2"]

    def test_pca_corr_in_range(self):
        from scalper_hft.ml.feature_importance import mdi, pca_importance_corr

        X, y = self._data()
        clf = self._factory()()
        clf.fit(X, y)
        imp = mdi(clf, X)
        tau = pca_importance_corr(imp.dropna(), X)
        assert -1.0 <= tau <= 1.0

    def test_report_structure(self):
        from scalper_hft.ml.feature_importance import feature_importance_report
        from scalper_hft.validation.cv import PurgedKFold

        X, y = self._data(400)
        pkf = PurgedKFold(n_splits=3, embargo_pct=0.02)
        rep = feature_importance_report(X, y, self._factory(), pkf, score="neg_log_loss")
        assert {"mda", "sfi", "mdi", "pca_tau"} <= set(rep.columns)
        assert rep.index[0] in {"good0", "good1"}  # топ за MDA — інформативні
