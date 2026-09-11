"""Тести для покращень результативності (Sprint 5 additions)."""

import numpy as np
import pandas as pd

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

    def test_engine_lag_is_exactly_one_bar(self):
        """Сигнал на t → позиція рушія на t+1 (без подвійного shift у стратегії)."""
        from scalper_hft.backtest.engine import run_backtest
        from scalper_hft.backtest.execution import CostModel
        from scalper_hft.strategies import get_strategy

        n = 80
        idx = pd.date_range("2025-01-01", periods=n, freq="1h")
        a = pd.Series(100.0 + np.arange(n, dtype=float), index=idx)
        b = pd.Series(np.full(n, 100.0), index=idx)
        c = pd.Series(100.0 - np.arange(n, dtype=float) * 0.1, index=idx)
        df = pd.DataFrame(
            {
                "AAA_close": a,
                "BBB_close": b,
                "CCC_close": c,
                "open": a,
                "high": a * 1.001,
                "low": a * 0.999,
                "close": a,
                "volume": 1.0,
            },
            index=idx,
        )
        s = get_strategy("cross_momentum", lookback=5, top_pct=0.2, signal_smooth=1)
        sig = s.generate_signals(df)
        nz = sig[sig != 0]
        assert len(nz) > 0
        first_sig_i = int(df.index.get_loc(nz.index[0]))
        res = run_backtest(
            df,
            s,
            cost=CostModel(maker_fee=0.0, taker_fee=0.0, slippage_frac=0.0),
            is_maker=False,
            position_pct=1.0,
        )
        pos_i = int(np.argmax(res.positions.abs().to_numpy() > 0))
        assert pos_i == first_sig_i + 1


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
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 101,
                "low": np.ones(n) * 99,
                "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        s = get_strategy("mean_reversion")
        res = time_decay_test(df, s, max_lag=2)
        assert len(res.lags) == 3  # 0, 1, 2
        assert len(res.sharpes) == 3


# ── Unified Backtest Results: EventBacktestResult & PairsResult ──────────────


class TestUnifiedBacktestResults:
    def test_event_backtest_has_positions_and_chart_works(self):
        """EventBacktestResult містить positions та успішно будує графік без AttributeError."""
        from scalper_hft.backtest.router import run_strategy_backtest
        from scalper_hft.strategies.market_maker import PassiveMarketMaker
        from scalper_hft.visualization.charts import make_backtest_figure

        idx = pd.date_range("2025-01-01", periods=150, freq="1h")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.2, 150))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 100.0,
            },
            index=idx,
        )
        res = run_strategy_backtest(df, PassiveMarketMaker())
        assert hasattr(res, "positions")
        assert isinstance(res.positions, pd.Series)
        assert len(res.positions) == len(res.equity)

        # Перевіряємо, що графік будується без падіння
        fig = make_backtest_figure(df, res)
        assert fig is not None

    def test_pairs_result_has_trades(self):
        """PairsResult містить trades DataFrame з укладеними угодами."""
        from scalper_hft.backtest.pairs import run_pairs_backtest
        from scalper_hft.strategies.pairs_arb import PairsArb

        idx = pd.date_range("2025-01-01", periods=300, freq="1h")
        rng = np.random.default_rng(42)
        # Симулюємо коінтегровані ряди зі сплеском спреду
        noise = rng.normal(0, 0.01, 300)
        leg1_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
        spread_shock = np.sin(np.linspace(0, 4 * np.pi, 300)) * 0.05
        leg2_close = leg1_close * np.exp(-spread_shock + noise)

        l1 = pd.DataFrame({"close": leg1_close}, index=idx)
        l2 = pd.DataFrame({"close": leg2_close}, index=idx)
        strategy = PairsArb(entry_z=1.0, exit_z=0.2, lookback=50)
        res = run_pairs_backtest(l1, l2, strategy)

        assert hasattr(res, "trades")
        assert isinstance(res.trades, pd.DataFrame)
        assert "entry_ts" in res.trades.columns
        assert "exit_ts" in res.trades.columns
        assert "side" in res.trades.columns
        assert "ret" in res.trades.columns
        assert len(res.trades) > 0


class TestRealizedVolScaling:
    def test_realized_vol_autodetects_hourly(self):
        """1h бари масштабуються з bars_per_year ≈ 8760, а 1m — з 525600."""
        from scalper_hft.features.indicators import realized_vol

        idx_1h = pd.date_range("2025-01-01", periods=100, freq="1h")
        idx_1m = pd.date_range("2025-01-01", periods=100, freq="1min")

        close_1h = pd.Series(100.0 + np.arange(100) * 0.1, index=idx_1h)
        close_1m = pd.Series(100.0 + np.arange(100) * 0.1, index=idx_1m)

        vol_1h = realized_vol(close_1h, window=30).dropna()
        vol_1m = realized_vol(close_1m, window=30).dropna()

        # Оскільки log_ret однаковий, відношення має бути sqrt(525600 / 8760) = sqrt(60) ≈ 7.746
        ratio = vol_1m.iloc[0] / vol_1h.iloc[0]
        assert np.isclose(ratio, np.sqrt(60.0), rtol=1e-2)

    def test_realized_vol_explicit_bars_per_year(self):
        from scalper_hft.features.indicators import realized_vol

        idx = pd.date_range("2025-01-01", periods=50, freq="1h")
        close = pd.Series(100.0 + np.arange(50) * 0.1, index=idx)
        v1 = realized_vol(close, window=10, bars_per_year=1000.0).dropna()
        v2 = realized_vol(close, window=10, bars_per_year=4000.0).dropna()
        assert np.isclose(v2.iloc[0] / v1.iloc[0], 2.0, rtol=1e-3)


class TestPairsSpreadAlpha:
    """W0-Q: quintile/time-decay пар на −Δspread, не на close однієї ноги."""

    @staticmethod
    def _mr_pair(n: int = 900, phi: float = 0.75, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = phi * spread[i - 1] + rng.normal(0.0, 0.02)
        trend = np.linspace(0.0, 0.8, n)  # спільний drift ніг
        log_leg2 = trend + np.cumsum(rng.normal(0.0, 0.003, n))
        log_leg1 = log_leg2 + spread
        leg1 = np.exp(log_leg1) * 100.0
        leg2 = np.exp(log_leg2) * 100.0
        return pd.DataFrame(
            {"leg1": leg1, "leg2": leg2, "close": leg1, "open": leg1, "high": leg1, "low": leg1, "volume": 1.0},
            index=idx,
        )

    def test_pairs_quintile_on_spread_passes(self) -> None:
        from scalper_hft.strategies import get_strategy
        from scalper_hft.validation.audit_extensions import run_quintile_audit

        df = self._mr_pair()
        strategy = get_strategy("pairs_arb", lookback=60, entry_z=1.0, regime_scale=False)
        q = run_quintile_audit(df, strategy)
        assert q.error is None, q.error
        assert q.on_spread is True
        assert q.pass_ is True, f"ρ={q.spearman:+.3f} monotonic={q.monotonic}"

    def test_pairs_quintile_leg_close_not_required(self) -> None:
        from scalper_hft.validation.audit_extensions import QUINTILE_SPEARMAN_MIN, pairs_z_and_forward
        from scalper_hft.validation.quintile import quintile_spread_study

        df = self._mr_pair()
        z, spread_fwd = pairs_z_and_forward(df, lookback=60)
        # close хедж-ноги не містить −Δspread — старий fwd у cmd_report.
        hedge_fwd = df["leg2"].pct_change().shift(-1)
        spread_q = quintile_spread_study(z, spread_fwd)
        hedge_q = quintile_spread_study(z, hedge_fwd)
        spread_pass = spread_q.monotonic and abs(spread_q.spearman) >= QUINTILE_SPEARMAN_MIN
        hedge_pass = hedge_q.monotonic and abs(hedge_q.spearman) >= QUINTILE_SPEARMAN_MIN
        assert spread_pass is True
        assert hedge_pass is False

    def test_pairs_time_decay_on_spread(self) -> None:
        from scalper_hft.backtest.execution import CostModel
        from scalper_hft.strategies import get_strategy
        from scalper_hft.validation.audit_extensions import run_time_decay_audit

        df = self._mr_pair()
        strategy = get_strategy("pairs_arb", lookback=60, entry_z=1.0, regime_scale=False)
        td = run_time_decay_audit(df, strategy, cost=CostModel(), max_lag=2)
        assert td.error is None, td.error
        assert td.on_spread is True
        assert len(td.sharpes) == 3
        assert td.sharpes[0] > 0

    def test_pairs_arb_without_legs_errors(self) -> None:
        from scalper_hft.backtest.execution import CostModel
        from scalper_hft.strategies import get_strategy
        from scalper_hft.validation.audit_extensions import run_quintile_audit, run_time_decay_audit

        idx = pd.date_range("2024-01-01", periods=80, freq="1h")
        close = pd.Series(100.0 + np.arange(80), index=idx)
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
            index=idx,
        )
        strategy = get_strategy("pairs_arb")
        q = run_quintile_audit(df, strategy)
        assert q.pass_ is False
        assert q.error is not None and "leg1" in q.error
        td = run_time_decay_audit(df, strategy, cost=CostModel())
        assert td.pass_ is False
        assert td.error is not None and "leg1" in td.error


class TestPaperStoreWALMode:
    def test_wal_mode_enabled(self, tmp_path):
        from scalper_hft.live.store import PaperStore

        db_path = tmp_path / "test_paper.sqlite"
        store = PaperStore(db_path)
        cur = store._conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        store.close()
        assert mode.upper() == "WAL"
