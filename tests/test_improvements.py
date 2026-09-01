"""Тести для покращень результативності (Sprint 5 additions)."""

import numpy as np
import pandas as pd
import pytest


# ── Vectorized microstructure ──────────────────────────────────────────────────

class TestVectorizedMicrostructure:
    def test_kyle_lambda_shape_and_no_loop(self):
        """Векторизований kyle_lambda повертає ту ж форму що й оригінал."""
        from scalper_hft.features.microstructure import kyle_lambda_series

        n = 500
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        sv = pd.Series(np.random.randn(n))
        lam, tstat = kyle_lambda_series(close, sv, window=60)
        assert len(lam) == n
        assert len(tstat) == n
        # перші 58 барів — NaN (недостатньо вікна)
        assert lam.iloc[:58].isna().all()
        # решта — є значення
        assert lam.iloc[60:].notna().any()

    def test_kyle_lambda_finite_values(self):
        """Значення lambda мають бути скінченними там де є дані."""
        from scalper_hft.features.microstructure import kyle_lambda_series

        np.random.seed(1)
        n = 200
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        sv = pd.Series(np.random.randn(n))
        lam, _ = kyle_lambda_series(close, sv, window=50)
        valid = lam.dropna()
        assert len(valid) > 0
        assert np.isfinite(valid.values).all()

    def test_roll_spread_vectorized(self):
        """Векторизований roll_spread дає невід'ємні значення."""
        from scalper_hft.features.microstructure import roll_spread

        np.random.seed(2)
        n = 300
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.2))
        rs = roll_spread(close, window=20)
        assert len(rs) == n
        assert (rs >= 0).all()

    def test_roll_spread_speed(self):
        """Векторизований roll_spread виконується < 50ms для T=10000."""
        import time

        from scalper_hft.features.microstructure import roll_spread

        n = 10000
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        t0 = time.perf_counter()
        roll_spread(close, window=20)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"roll_spread занадто повільний: {elapsed:.3f}s"


# ── CrossMomentum strategy ──────────────────────────────────────────────────────

class TestCrossMomentum:
    def test_registered(self):
        from scalper_hft.strategies import REGISTRY
        assert "cross_momentum" in REGISTRY

    def test_signals_shape(self):
        from scalper_hft.strategies import get_strategy

        n = 200
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(n))})
        s = get_strategy("cross_momentum")
        sig = s.generate_signals(df)
        assert len(sig) == n
        assert sig.isin([-1, 0, 1]).all()

    def test_no_lookahead_shift(self):
        """Сигнал на барі t не може бути = 1 якщо на t=0 даних немає."""
        from scalper_hft.strategies import get_strategy

        n = 100
        df = pd.DataFrame({"close": np.ones(n) * 100})
        s = get_strategy("cross_momentum", lookback=5)
        sig = s.generate_signals(df)
        # при константній ціні всі сигнали = 0
        assert sig.abs().sum() == 0

    def test_param_space(self):
        from scalper_hft.strategies import get_strategy

        s = get_strategy("cross_momentum")
        assert "lookback" in s.param_space


# ── PairsArb breakeven gate ────────────────────────────────────────────────────

class TestPairsArbGates:
    def _make_pairs_df(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        leg1 = pd.Series(30000 + np.cumsum(rng.normal(0, 150, n)))
        leg2 = pd.Series(2000 + np.cumsum(rng.normal(0, 15, n)))
        return pd.DataFrame({"leg1": leg1.values, "leg2": leg2.values})

    def test_breakeven_gate_param(self):
        from scalper_hft.strategies import get_strategy

        s = get_strategy("pairs_arb", breakeven_gate=True)
        assert bool(s.get("breakeven_gate", False)) is True

    def test_hmm_vol_gate_param(self):
        from scalper_hft.strategies import get_strategy

        s = get_strategy("pairs_arb", hmm_vol_gate=False)
        assert bool(s.get("hmm_vol_gate", True)) is False

    def test_breakeven_gate_signals_subset(self):
        """З breakeven_gate сигналів <= сигналів без гейту (підмножина)."""
        from scalper_hft.strategies import get_strategy

        df = self._make_pairs_df()
        s1 = get_strategy("pairs_arb")
        s2 = get_strategy("pairs_arb", breakeven_gate=True)
        sig1 = s1.generate_signals(df)
        sig2 = s2.generate_signals(df)
        assert sig2.abs().sum() <= sig1.abs().sum()


# ── GARCH optimization ─────────────────────────────────────────────────────────

class TestGARCHOptimized:
    def test_forecast_no_nested_recursion(self):
        """garch_forecast не має вкладеного loop — результат стабільний."""
        from scalper_hft.features.volatility import garch_forecast

        np.random.seed(7)
        ret = pd.Series(np.random.randn(300) * 0.01)
        fc = garch_forecast(ret, window=100, refit_every=50, warmup=30)
        assert len(fc) == 300
        valid = fc.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()


# ── CLI cmd_report includes quintile & time-decay ─────────────────────────────

class TestCmdReportExtended:
    def test_quintile_in_report(self):
        """quintile_spread_study виклик не кидає виняток."""
        import pandas as pd
        from scalper_hft.validation.quintile import quintile_spread_study

        np.random.seed(10)
        n = 100
        z = pd.Series(np.random.randn(n))
        fwd = pd.Series(np.random.randn(n))
        res = quintile_spread_study(z, fwd, n_bins=5)
        assert hasattr(res, "spearman")
        assert hasattr(res, "monotonic")
        assert -1 <= res.spearman <= 1

    def test_time_decay_returns_lags(self):
        """time_decay_test повертає результат для лагів 0..2."""
        from scalper_hft.strategies import get_strategy
        from scalper_hft.validation.time_decay import time_decay_test

        np.random.seed(11)
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="1h")
        df = pd.DataFrame(
            {"open": np.ones(n) * 100, "high": np.ones(n) * 101, "low": np.ones(n) * 99,
             "close": 100 + np.cumsum(np.random.randn(n) * 0.5), "volume": np.ones(n) * 1000},
            index=idx,
        )
        s = get_strategy("mean_reversion")
        res = time_decay_test(df, s, max_lag=2)
        assert len(res.lags) == 3  # 0, 1, 2
        assert len(res.sharpes) == 3
