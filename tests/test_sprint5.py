"""Тести для Спринту 5:
- Розширені бари: Dollar Imbalance (DIB), Tick Run (TRB), Dollar Run (DRB);
- Micro-Price, QueuePositionModel, ImplementationShortfallTracker;
- Sparse Mean-Reverting Basket (SDP/Box-Tiao, OU estimation);
- Exp3 Multi-Armed Bandit;
- Clustered Feature Importance (CFI);
- One-Way Trading Exit Ladders, Tail Dependence Copulas, Soft Penalty & Silent Attrition Kill Switch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.execution import CostModel, ImplementationShortfallTracker
from scalper_hft.backtest.micro_price import (
    QueuePositionModel,
    calculate_micro_price,
    estimate_order_book_imbalance,
)
from scalper_hft.data.bars import (
    create_dollar_imbalance_bars,
    create_dollar_run_bars,
    create_tick_run_bars,
)
from scalper_hft.live.exit_ladders import OneWayTradingLadder
from scalper_hft.live.trader import LiveTrader, SilentAttritionKillSwitch
from scalper_hft.ml.clustered_importance import (
    cluster_features,
    clustered_mda,
)
from scalper_hft.portfolio.risk_budget import estimate_tail_dependence
from scalper_hft.strategies.bandit import Exp3Bandit, exp3_select_signals
from scalper_hft.strategies.mean_reversion import MeanReversionScalper
from scalper_hft.strategies.sparse_basket import (
    SparseBasketArb,
    compute_sparse_basket_weights,
    estimate_ou_parameters,
)
from scalper_hft.validation.cv import PurgedKFold


def _synthetic_trades(n: int = 1500) -> pd.DataFrame:
    """Генерація синтетичного потоку трейдів."""
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    amounts = np.random.exponential(scale=1.5, size=n) + 0.1
    timestamps = pd.date_range("2026-01-01", periods=n, freq="200ms")
    return pd.DataFrame({"price": prices, "amount": amounts}, index=timestamps)


# ── 1. Data Layer: Information-Driven Bars ──────────────────────────────────
class TestInformationBars:
    def test_dollar_imbalance_bars(self):
        trades = _synthetic_trades(1200)
        dib = create_dollar_imbalance_bars(trades, expected_imbalance_window=100, ewma_window=30)
        assert not dib.empty
        assert "open" in dib.columns
        assert "high" in dib.columns
        assert "low" in dib.columns
        assert "close" in dib.columns
        assert (dib["high"] >= dib["low"]).all()
        assert (dib["high"] >= dib["open"]).all()
        assert (dib["high"] >= dib["close"]).all()

    def test_tick_and_dollar_run_bars(self):
        trades = _synthetic_trades(1200)
        trb = create_tick_run_bars(trades, expected_run_window=100, ewma_window=30)
        drb = create_dollar_run_bars(trades, expected_run_window=100, ewma_window=30)
        assert not trb.empty
        assert not drb.empty
        assert len(trb) > 0
        assert len(drb) > 0


# ── 2. Execution & Microstructure ───────────────────────────────────────────
class TestMicroPriceAndExecution:
    def test_micro_price_and_imbalance(self):
        bid_p, ask_p = 100.0, 100.2
        bid_q, ask_q = 10.0, 2.0  # сильний тиск покупців

        imbalance = estimate_order_book_imbalance(bid_q, ask_q)
        assert imbalance > 0.5  # (10-2)/12 = 0.667

        micro_p = calculate_micro_price(bid_p, ask_p, bid_q, ask_q)
        mid_p = (bid_p + ask_p) / 2.0
        assert micro_p > mid_p  # Micro-Price зсунута вгору до ask

    def test_queue_position_model(self):
        q_model = QueuePositionModel()
        # Близький ордер vs далекий ордер
        p_close = q_model.estimate_fill_prob(distance_bps=1.0, queue_ahead_ratio=0.1, vpin=0.3)
        p_far = q_model.estimate_fill_prob(distance_bps=10.0, queue_ahead_ratio=0.8, vpin=0.8)
        assert p_close > p_far
        assert 0.0 <= p_close <= 1.0
        assert 0.0 <= p_far <= 1.0

        # Adverse selection risk
        risk_buy_toxic = q_model.adverse_selection_risk(side=1, imbalance=-0.8, vpin=0.9)
        risk_buy_safe = q_model.adverse_selection_risk(side=1, imbalance=0.8, vpin=0.1)
        assert risk_buy_toxic > risk_buy_safe

    def test_implementation_shortfall_tracker(self):
        tracker = ImplementationShortfallTracker(alpha_decay=0.2)
        # Buy: Decision = 100.0, Fill = 100.05 -> Shortfall = +5 bps
        is_bps = tracker.record_execution(decision_price=100.0, fill_price=100.05, side=1, vol_frac=0.001)
        assert pytest.approx(is_bps, rel=1e-3) == 5.0
        assert pytest.approx(tracker.mean_slippage_bps, rel=1e-3) == 5.0

        # Sell: Decision = 100.0, Fill = 99.98 -> Shortfall = +2 bps
        tracker.record_execution(decision_price=100.0, fill_price=99.98, side=-1, vol_frac=0.001)
        assert tracker.mean_slippage_bps > 0

        calibrated_cost = tracker.calibrate_cost_model(CostModel(slippage_frac=0.0002))
        assert calibrated_cost.slippage_frac > 0


# ── 3. Alpha & Online Learning: Sparse Baskets & Exp3 ───────────────────────
class TestSparseBasketAndBandit:
    def test_ou_parameters_estimation(self):
        np.random.seed(42)
        # Синтетичний mean-reverting ряд AR(1)
        n = 300
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + np.random.randn() * 0.5
        s = pd.Series(x)

        ou = estimate_ou_parameters(s)
        assert ou["theta"] > 0
        assert ou["half_life"] > 0
        assert np.isfinite(ou["half_life"])

    def test_sparse_basket_weights(self):
        np.random.seed(42)
        n = 200
        p1 = 100 + np.cumsum(np.random.randn(n))
        p2 = 50 + 0.5 * p1 + np.random.randn(n) * 0.2  # коінтегований
        p3 = 20 + np.cumsum(np.random.randn(n))  # шумовий
        df = pd.DataFrame({"BTC": p1, "ETH": p2, "SOL": p3})

        w = compute_sparse_basket_weights(df, sparsity_k=2, l1_penalty=0.05)
        assert len(w) == 3
        assert np.sum(np.abs(w)) > 0
        assert np.count_nonzero(w) <= 2  # sparse k=2

    def test_sparse_basket_strategy_signal(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=100, freq="1h")
        df = pd.DataFrame({"close": 100 + np.random.randn(100) * 2}, index=dates)
        strat = SparseBasketArb(lookback=20, entry_z=1.5, exit_z=0.2)
        sig = strat.generate_signals(df)
        assert len(sig) == len(df)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_exp3_bandit(self):
        np.random.seed(42)
        bandit = Exp3Bandit(n_arms=3, gamma=0.1)
        # Симуляція: рука 1 постійно дає позитивний реворд, інші від'ємні
        for _ in range(50):
            arm = bandit.select_arm()
            reward = 0.5 if arm == 1 else -0.5
            bandit.update(arm, reward)

        probs = bandit.probabilities()
        assert probs[1] > probs[0]
        assert probs[1] > probs[2]

    def test_exp3_select_signals(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=50, freq="1min")

        sig_df = pd.DataFrame(
            {
                "s1": np.ones(50),
                "s2": -np.ones(50),
            },
            index=dates,
        )
        ret_df = pd.DataFrame(
            {
                "s1": np.full(50, 0.01),
                "s2": np.full(50, -0.01),
            },
            index=dates,
        )
        chosen = exp3_select_signals(sig_df, ret_df, gamma=0.1)
        assert len(chosen) == 50
        assert set(chosen.unique()).issubset({-1.0, 1.0})


# ── 4. ML: Clustered Feature Importance ─────────────────────────────────────
class TestClusteredFeatureImportance:
    def test_cluster_features(self):
        np.random.seed(42)
        n = 100
        f1 = np.random.randn(n)
        f2 = f1 + np.random.randn(n) * 0.05  # майже копія f1
        f3 = np.random.randn(n)
        X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})

        clusters, cl_dict = cluster_features(X, max_clusters=2)
        assert len(clusters) == 3
        # f1 і f2 мають потрапити в один кластер
        assert clusters["f1"] == clusters["f2"]

    def test_clustered_mda(self):
        np.random.seed(42)
        n = 120
        f1 = np.random.randn(n)
        f2 = f1 + np.random.randn(n) * 0.05
        f3 = np.random.randn(n)
        y = pd.Series(np.where(f1 > 0, 1, 0))
        X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})

        class DummyClf:
            def fit(self, X_tr, y_tr, sample_weight=None):
                self.classes_ = np.array([0, 1])
                return self

            def predict_proba(self, X_te):
                p = np.where(X_te.iloc[:, 0].values > 0, 0.8, 0.2)
                return np.column_stack([1 - p, p])

        cv = PurgedKFold(n_splits=3, embargo_pct=0.01)
        res = clustered_mda(X, y, lambda: DummyClf(), cv=cv, max_clusters=2)
        assert not res.clustered_mda.empty
        assert len(res.feature_mda) == 3


# ── 5. Risk, Exit Ladders & Kill Switch ─────────────────────────────────────
class TestRiskAndExitLadders:
    def test_one_way_trading_ladder(self):
        # Long ladder: entry = 100.0, step = 1%
        ladder = OneWayTradingLadder(entry_price=100.0, side=1, base_step_pct=0.01, num_levels=3)
        assert len(ladder.rungs) == 3
        assert ladder.remaining_fraction == 1.0
        assert not ladder.is_fully_closed

        # Ціна піднялася до 101.5 -> перший рівень закривається
        closed_1 = ladder.update_price(101.5)
        assert closed_1 > 0.0
        assert ladder.remaining_fraction < 1.0

        # Ціна виросла до 115.0 -> всі рівні закриті
        ladder.update_price(115.0)
        assert ladder.is_fully_closed
        assert ladder.remaining_fraction == 0.0

    def test_estimate_tail_dependence(self):
        np.random.seed(42)
        n = 200
        # Два ряди з високою нижньою залежністю (спільні крахи)
        shocks = np.random.randn(n)
        r1 = shocks + np.random.randn(n) * 0.1
        r2 = shocks + np.random.randn(n) * 0.1
        df = pd.DataFrame({"BTC": r1, "ETH": r2})

        tail_dep = estimate_tail_dependence(df, alpha=0.05)
        assert tail_dep.shape == (2, 2)
        assert tail_dep.loc["BTC", "ETH"] > 0.3  # висока tail dependence

    def test_soft_penalty_size_in_trader(self):
        strat = MeanReversionScalper()
        trader = LiveTrader(strategy=strat, symbol="BTCUSDT")

        # Сильний сигнал проти слабкого сигналу
        size_strong = trader.soft_penalty_size(base_size=10.0, signal_strength=5.0, limit_size=10.0, k=1.0)
        size_weak = trader.soft_penalty_size(base_size=10.0, signal_strength=0.5, limit_size=10.0, k=1.0)
        assert size_strong > size_weak
        assert size_strong <= 10.0

    def test_silent_attrition_kill_switch(self):
        ks = SilentAttritionKillSwitch(alpha_decay=0.2, min_trades=4, threshold_pnl=-0.01)
        assert not ks.tripped

        # Серія прибуткових угод
        for _ in range(5):
            assert not ks.record_trade(+0.005)

        # Серія сильних збитків -> згасання альфи
        for _ in range(10):
            ks.record_trade(-0.02)

        assert ks.tripped  # Kill switch спрацював!

        ks.reset()
        assert not ks.tripped
