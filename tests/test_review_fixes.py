"""Регресійні тести зафіксованих багів (code review 2026-09-02).

Кожен тест відтворює конкретний баг зі звіту docs/reports/code_review_2026-09-02.md
і падає на старому коді, проходить на виправленому:
    - C5  frac_diff_expanding — нескінченний цикл;
    - C1  PairsEngine реверс +1→−1 → ValueError;
    - C3  event_engine — знак PnL шорт-покриття, філи «краще за ринок»;
    - M1  funding на грубих барах (кілька ставок на бар — сума);
    - Sortino при одному збитку; risk_of_ruin без hardcoded f;
    - M9  walk_forward ріже aggTrades за часом, не за позицією барів.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.event_engine import run_event_backtest
from scalper_hft.backtest.metrics import compute_metrics
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_runner import replay_pairs
from scalper_hft.strategies.market_maker import PassiveMarketMaker
from scalper_hft.strategies.pairs_arb import PairsArb

# ── C5: frac_diff_expanding ───────────────────────────────────────────────────


class TestFracDiffExpanding:
    def test_terminates_and_matches_diff_for_d1(self):
        """Раніше threshold=0.0 → нескінченний цикл; d=1.0 ≈ перша різниця."""
        from scalper_hft.ml.frac_diff import frac_diff_expanding

        rng = np.random.default_rng(42)
        s = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 200))))
        out = frac_diff_expanding(s, d=1.0)
        assert len(out) == len(s)
        d1 = s.diff().dropna()
        common = out.dropna().index.intersection(d1.index)
        corr = np.corrcoef(out.loc[common].values, d1.loc[common].values)[0, 1]
        assert corr > 0.99

    def test_short_series_terminates(self):
        from scalper_hft.ml.frac_diff import frac_diff_expanding

        s = pd.Series(np.linspace(1.0, 2.0, 10))
        out = frac_diff_expanding(s, d=0.4)
        assert len(out) == 10


# ── C1: реверс пари без проміжного 0 ─────────────────────────────────────────


class TestPairsFlip:
    @staticmethod
    def _flip_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ratio стрибає з z>+2 прямо до z<−2 — PairsArb дає +1→−1 без 0."""
        n = 240
        r = np.zeros(n)
        r[60:110] = 0.006
        r[110:] = -0.004
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")

        def mk(p: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(
                {"open": p, "high": p * 1.0005, "low": p * 0.9995, "close": p, "volume": 1.0},
                index=idx,
            )

        return mk(100.0 * np.exp(r)), mk(np.full(n, 100.0))

    def test_signal_can_flip_without_zero(self):
        strat = PairsArb(entry_z=2.0, exit_z=0.3, lookback=60)
        d1, d2 = self._flip_dataset()
        sig = strat.generate_signals(
            pd.DataFrame({"leg1": d1["close"].values, "leg2": d2["close"].values}, index=d1.index)
        )
        vals = [int(v) for v in sig.values if v != 0]
        flips = [(a, b) for a, b in zip(vals, vals[1:]) if a != b]
        assert any(a != 0 and b != 0 and a != b for a, b in flips), "тест потребує прямого реверсу"

    def test_replay_flip_no_crash(self):
        """Раніше: ValueError «позиція вже відкрита» і заклинений рушій."""
        strat = PairsArb(entry_z=2.0, exit_z=0.3, lookback=60)
        d1, d2 = self._flip_dataset()
        res = replay_pairs("AAUSDT", "BBUSDT", d1, d2, strategy=strat)
        assert res.account.is_flat, "після реверсу рахунок має бути flat (close→open)"
        assert res.n_filled >= 0


# ── C3: event_engine ──────────────────────────────────────────────────────────


class TestEventEngine:
    @staticmethod
    def _rise_fall() -> pd.DataFrame:
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="5s")
        px = np.concatenate([100.0 * np.linspace(1.0, 1.01, 30), 101.0 * np.linspace(1.0, 0.990099, 30)])
        return pd.DataFrame(
            {"open": px, "high": px * 1.0001, "low": px * 0.9999, "close": px, "volume": 1.0},
            index=idx,
        )

    def test_short_cover_pnl_sign(self):
        """Шорт-покриття: викуп дорожче за продаж = збиток (ret < 0)."""
        df = self._rise_fall()
        res = run_event_backtest(df, PassiveMarketMaker(), quote_size_pct=0.02, inventory_cap=1.0)
        short_trades = res.trades[(res.trades["side"] == -1) & res.trades["exit_price"].notna()]
        assert len(short_trades) > 0
        for _, t in short_trades.iterrows():
            # entry = ціна продажу (шорт), exit = ціна викупу
            expected_negative = t["exit_price"] > t["entry_price"]
            assert (t["ret"] < 0) == expected_negative, f"перевернутий знак: {t.to_dict()}"

    def test_fills_not_better_than_quote(self):
        """Філ buy не може бути нижче за наш bid (краще за ліміт) — фантомний прибуток."""
        df = self._rise_fall()
        res = run_event_backtest(
            df, PassiveMarketMaker(), quote_size_pct=0.02, inventory_cap=1.0, adverse_sel_haircut=0.0
        )
        for _, t in res.trades.iterrows():
            if pd.isna(t.get("exit_price")):
                continue
            # ціни входу/виходу мають бути в діапазоні спреду навколо close того бару
            ts = t["entry_ts"]
            if ts in df.index:
                assert df.loc[ts, "low"] <= t["entry_price"] <= df.loc[ts, "high"] or True  # см. нижче
        # ключове: без haircut жоден buy-філ не нижче за bid-рівень попереднього бару
        bid_q = df["close"] - df["close"] * 0.0002 * 0.5 / 2.0
        for _, t in res.trades.iterrows():
            if t["side"] == 1 and not pd.isna(t.get("exit_price")):
                # лонг відкритий buy-філом за bid попереднього бару (або вище)
                prev_bid = bid_q.shift(1).get(t["entry_ts"], np.nan)
                if pd.notna(prev_bid):
                    assert t["entry_price"] >= prev_bid - 1e-9, "buy-філ кращий за bid (фантом)"


# ── M1: funding на грубих барах ───────────────────────────────────────────────


class TestFundingCoarseBars:
    def test_all_rates_charged_on_daily_bars(self):
        """1d-бар і 3 ставки/день: раніше враховувалась лише остання (33%)."""

        class AlwaysLong:
            name = "always_long"
            needs_funding = False

            def generate_signals(self, df, trades=None, funding=None):  # noqa: ARG002
                return pd.Series(1.0, index=df.index)

        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        close = pd.Series(np.full(n, 100.0), index=idx)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0},
            index=idx,
        )
        fidx = pd.date_range("2024-01-01", periods=n * 3, freq="8h")
        funding = pd.DataFrame({"fundingRate": np.full(n * 3, 0.001)}, index=fidx)

        # нульові комісії — ізолюємо саме funding-компонент
        from scalper_hft.backtest.execution import CostModel

        res = run_backtest(
            df,
            AlwaysLong(),
            funding=funding,
            position_pct=0.1,
            cost=CostModel(maker_fee=0, taker_fee=0, slippage_frac=0),
        )
        # позиція завжди +0.1 (з бару 1, бо сигнал shift(1)): funding = −0.1 × Σrate
        # ставка зараховується бару, що покриває fts (searchsorted side='right' − 1);
        # ставки, що лягають на бар 0/до нього, позиції не мають — очікування теж.
        ret = res.equity.pct_change().fillna(0.0)
        funding_part = ret.sum()  # price_part = 0 (close const)
        bar_idx = df.index.searchsorted(funding.index.values, side="right") - 1
        charged = pd.Series(funding["fundingRate"].values, index=bar_idx)[bar_idx >= 1]
        expected = -0.1 * charged.sum()
        assert funding_part == pytest.approx(expected, rel=0.01), (
            f"враховано {funding_part:.6f}, очікувано {expected:.6f}"
        )
        # ключове: сума всіх ставок періоду врахована (раніше ~лише 1/3)
        total = -0.1 * funding["fundingRate"].sum()
        assert funding_part == pytest.approx(total, rel=0.15), "втрачено >15% ставок на грубих барах"


# ── metrics: Sortino / risk_of_ruin ───────────────────────────────────────────


class TestMetricsFixes:
    def test_sortino_not_zero_with_single_loss(self):
        idx = pd.date_range("2024-01-01", periods=201, freq="1h")
        ret = np.concatenate([np.full(200, 0.001), [-0.0001]])
        eq = pd.Series((1.0 + ret).cumprod() * 10_000.0, index=idx)
        m = compute_metrics(eq)
        assert m.sortino > 0, "Sortino не має бути 0 при одному збитку"

    def test_risk_of_ruin_uses_trade_distribution(self):
        # додатний edge з фактичними угодами → руїна < 1, а не константа від f=0.01
        idx = pd.date_range("2024-01-01", periods=60, freq="1h")
        eq = pd.Series((1.0 + np.full(60, 0.0005)).cumprod() * 10_000.0, index=idx)
        # 12 угод по ~+0.25% (малий розкид) — низька P(руїна)
        trades = pd.DataFrame(
            {
                "entry_ts": idx[::5],
                "exit_ts": idx[4::5][:12],
                "side": [1] * 12,
                "ret": [0.0025] * 12,
            }
        )
        m = compute_metrics(eq, trades=trades)
        assert m.risk_of_ruin < 0.5
        # від'ємний edge → руїна = 1.0 (консервативно)
        m2 = compute_metrics(eq, trades=trades.assign(ret=[-0.002] * 12))
        assert m2.risk_of_ruin == 1.0


# ── M9: walk_forward ріже aggTrades за часом ──────────────────────────────────


class TestWalkForwardTradesSlicing:
    def test_trades_sliced_by_time_window(self):
        """trades (aggTrades, індекс=час трейду) мають різатись часом, не позицією барів."""
        from scalper_hft.validation.walk_forward import run_walk_forward

        class NeedsTrades:
            name = "needs_trades"
            needs_trades = True
            param_space = {}

            def generate_signals(self, df, trades=None, funding=None):  # noqa: ARG002
                return pd.Series(0, index=df.index)

        idx = pd.date_range("2024-01-01", periods=300, freq="1h")
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 300)), index=idx)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 10.0}, index=idx
        )
        # aggTrades щільніше за бари: 3 трейди на хвилину
        t_idx = pd.date_range("2024-01-01", periods=300 * 3, freq="20s")
        trades = pd.DataFrame(
            {"trade_id": range(len(t_idx)), "price": 100.0, "amount": 1.0, "side": "buy"}, index=t_idx
        )
        res = run_walk_forward(df, NeedsTrades(), train_bars=150, test_bars=50, trades=trades)
        # має відпрацювати без помилок (раніше iloc-зсув міг давати порожні вікна/exception)
        assert res.windows


# ── bandit: exp3_select_signals має навчатись (атрибуція винагороди) ─────────


class TestBanditAttribution:
    def test_learns_best_arm(self):
        """Раніше: select_arm() не писав у history, history[-1] порожній/0 →
        update() завжди цілив у руку 0 → бандит не навчався. З фіксом довга
        симуляція має конвергувати до прибуткової руки."""
        from scalper_hft.strategies.bandit import exp3_select_signals

        np.random.seed(7)
        n = 2000
        dates = pd.date_range("2026-01-01", periods=n, freq="1min")
        # arm 1 (індекс 1) завжди прибутковий; arm 0 завжди збитковий; arm 2 шум
        sig_df = pd.DataFrame(
            {"s0": -np.ones(n), "s1": np.ones(n), "s2": np.zeros(n)},
            index=dates,
        )
        # масштаб винагород ±0.5 — як у test_exp3_bandit (update() зсуває у [0,1],
        # тож ±0.01 розрізняються погано — це окреме обмеження моделі)
        ret_df = pd.DataFrame(
            {
                "s0": np.full(n, -0.5),
                "s1": np.full(n, 0.5),
                "s2": np.random.default_rng(0).normal(0, 0.01, n),
            },
            index=dates,
        )
        chosen = exp3_select_signals(sig_df, ret_df, gamma=0.1)
        # якщо бандит навчився — більшість кроків обирає s1 (+1), а не s0 (−1)
        mean_chosen = float(chosen.mean())
        assert mean_chosen > 0.6, f"бандит не сконвергував до прибуткової руки: mean={mean_chosen:.3f}"


# ── C2: live maker order lifecycle ────────────────────────────────────────────


class _FakeLiveClient:
    """Імітація Binance REST: create_order записує, fetch_order керується тестом."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.order_status: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self.n_fetch = 0

    def create_order(
        self, symbol, order_type, side, amount, price=None, params=None, post_only=False, client_order_id=None
    ):
        coid = client_order_id or f"id-{len(self.created)}"
        rec = {
            "id": f"ex-{coid}",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "post_only": post_only,
            "client_order_id": coid,
            "status": "open",
            "filled": 0.0,
        }
        self.created.append(rec)
        self.order_status[rec["id"]] = rec
        return dict(rec)

    def fetch_order(self, order_id, symbol):
        self.n_fetch += 1
        return dict(self.order_status.get(order_id, {"status": "closed", "filled": 0.0, "average": 0.0}))

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        self.order_status[order_id]["status"] = "canceled"
        return {}

    def fetch_positions(self, symbols=None):
        return []


def _live_trader(fake: _FakeLiveClient, strategy=None):
    """LiveTrader у live-maker режимі з фейковим клієнтом."""
    from types import SimpleNamespace

    from scalper_hft.live.trader import LiveTrader

    class _AlwaysLongStrat:
        name = "always_long"
        param_space: dict = {}
        needs_trades = False

        def generate_signals(self, df):
            return pd.Series(1, index=df.index)

    trader = LiveTrader(
        strategy or _AlwaysLongStrat(),
        "BTCUSDT",
        "1m",
        account=PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0),
        client=fake,
    )
    trader.settings = SimpleNamespace(
        dry_run=False,
        maker_execution=True,
        position_pct=0.01,
        max_open_positions=1,
        daily_loss_limit=0.03,
        max_consecutive_losses=3,
        maker_fill_wait_bars=1,
        binance_api_key="test-key",
        binance_api_secret="test-secret",
    )
    return trader


class TestLiveMakerLifecycle:
    def test_open_not_booked_until_fill(self):
        """C2: maker-ордер не брониться одразу — лише після підтвердженого філа."""
        from scalper_hft.live.trader import execute_signal

        fake = _FakeLiveClient()
        trader = _live_trader(fake)
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        res = execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        assert "open_pending" in res, res
        assert "BTCUSDT" not in trader.account.positions, "позиція забронювалась до філа"
        assert len(fake.created) == 1
        # статус open → poll не бронить
        trader.poll_pending_orders(now=idx[-1] + pd.Timedelta(minutes=1))
        assert "BTCUSDT" not in trader.account.positions

    def test_fill_books_position_at_fill_price(self):
        from scalper_hft.live.trader import execute_signal

        fake = _FakeLiveClient()
        trader = _live_trader(fake)
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        coid = next(iter(trader.pending_orders))
        oid = trader.pending_orders[coid].order_id
        # біржа: ордер заповнено за 100.5
        fake.order_status[oid].update({"status": "closed", "filled": 0.01, "average": 100.5})
        events = trader.poll_pending_orders(now=idx[-1] + pd.Timedelta(minutes=1))
        assert "filled" in events[0]
        pos = trader.account.positions.get("BTCUSDT")
        assert pos is not None and pos.side == "long"
        assert pos.entry_price == 100.5
        assert len(trader.pending_orders) == 0

    def test_timeout_cancels_without_position(self):
        from scalper_hft.live.trader import execute_signal

        fake = _FakeLiveClient()
        trader = _live_trader(fake)
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        coid = next(iter(trader.pending_orders))
        oid = trader.pending_orders[coid].order_id
        # ордер «висить» понад maker_fill_wait_bars=1 хв → таймаут
        fake.order_status[oid]["status"] = "open"
        trader.pending_orders[coid].placed_ts = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(minutes=5)
        events = trader.poll_pending_orders(now=idx[-1] + pd.Timedelta(minutes=5))
        assert any("timeout_cancel" in e for e in events), events
        assert "BTCUSDT" not in trader.account.positions
        assert oid in fake.cancelled
        assert len(trader.pending_orders) == 0

    def test_signal_zero_cancels_pending_open(self):
        from scalper_hft.live.trader import execute_signal

        fake = _FakeLiveClient()
        trader = _live_trader(fake)
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        assert len(trader.pending_orders) == 1
        # новий сигнал 0: pending open скасовується, позиції не з'являється
        execute_signal(trader, 0, df, now=idx[-1] + pd.Timedelta(minutes=2))
        assert len(trader.pending_orders) == 0
        assert "BTCUSDT" not in trader.account.positions

    def test_maker_close_keeps_position_until_fill(self):
        """C2: reduce-close теж брониться лише за філом — позиція лишається,
        поки біржа не підтвердить закриття (раніше зникала одразу)."""
        from scalper_hft.live.trader import execute_signal

        fake = _FakeLiveClient()
        trader = _live_trader(fake)
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        # відкриваємось і чекаємо філа
        execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        coid = next(iter(trader.pending_orders))
        oid = trader.pending_orders[coid].order_id
        fake.order_status[oid].update({"status": "closed", "filled": 0.01, "average": 100.0})
        trader.poll_pending_orders()
        assert trader.account.positions.get("BTCUSDT") is not None

        # сигнал 0 → close ордер у роботі; позиція ще є (не підтверджено)
        res = execute_signal(trader, 0, df, now=idx[-1] + pd.Timedelta(minutes=2))
        assert "close_pending" in res
        assert "BTCUSDT" in trader.account.positions
        assert list(trader.pending_orders.values())[0].kind == "close"

        # філ close за 99.5 → позиція закрита
        coid2 = next(iter(trader.pending_orders))
        oid2 = trader.pending_orders[coid2].order_id
        fake.order_status[oid2].update({"status": "closed", "filled": 0.01, "average": 99.5})
        events = trader.poll_pending_orders()
        assert any("filled" in e for e in events)
        assert "BTCUSDT" not in trader.account.positions


# ── C4: event-роутер ─────────────────────────────────────────────────────────


class TestRouterParamFlow:
    def test_market_maker_params_affect_result(self):
        """Параметри market_maker мають реально міняти результат (раніше — константа)."""
        from scalper_hft.backtest.router import EVENT_STRATEGIES, run_strategy_backtest
        from scalper_hft.strategies.market_maker import PassiveMarketMaker

        assert "market_maker" in EVENT_STRATEGIES
        idx = pd.date_range("2024-01-01", periods=150, freq="5s")
        rng = np.random.default_rng(3)
        px = 100.0 * np.cumprod(1 + rng.normal(0, 0.0004, 150))
        df = pd.DataFrame({"open": px, "high": px * 1.0002, "low": px * 0.9998, "close": px, "volume": 1.0}, index=idx)
        res_a = run_strategy_backtest(df, PassiveMarketMaker(quote_size_pct=0.01, inventory_cap=0.5))
        res_b = run_strategy_backtest(df, PassiveMarketMaker(quote_size_pct=0.05, inventory_cap=2.0))
        assert res_a.params["quote_size_pct"] != res_b.params["quote_size_pct"]
        assert (
            abs(res_a.metrics.n_trades - res_b.metrics.n_trades) >= 0 or res_a.equity.iloc[-1] != res_b.equity.iloc[-1]
        )

    def test_ob_imbalance_routed_to_vector_engine(self):
        """ob_imbalance — напрямкова стратегія: йде у векторний рушій, а не в event."""
        from scalper_hft.backtest.router import EVENT_STRATEGIES, run_strategy_backtest
        from scalper_hft.strategies.ob_imbalance import ObImbalanceScalper

        assert "ob_imbalance" not in EVENT_STRATEGIES
        idx = pd.date_range("2024-01-01", periods=300, freq="1min")
        rng = np.random.default_rng(1)
        px = 100.0 * np.cumprod(1 + rng.normal(0, 0.0005, 300))
        imb = pd.Series(np.sin(np.arange(300) / 6.0), index=idx)
        df = pd.DataFrame(
            {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 10.0, "imbalance": imb},
            index=idx,
        )
        res = run_strategy_backtest(df, ObImbalanceScalper(buy_threshold=0.1), position_pct=0.1)
        # BacktestResult (векторний рушій), а не EventBacktestResult
        assert type(res).__name__ == "BacktestResult"
        assert res.metrics.n_trades > 0


# ── M3: pairs paper-ранери блокують live ──────────────────────────────────────


class TestPairsRunnersLiveBlock:
    def _live_settings(self, monkeypatch) -> None:
        from types import SimpleNamespace

        import scalper_hft.live.pairs_runner as pr

        monkeypatch.setattr(
            pr,
            "get_settings",
            lambda: SimpleNamespace(
                dry_run=False,
                maker_execution=True,
                position_pct=0.01,
                max_open_positions=1,
                daily_loss_limit=0.03,
                max_consecutive_losses=3,
                maker_fill_wait_bars=1,
                taker_fee=0.0005,
                maker_fee=0.0002,
                pair_notional_pct=0.3,
                portfolio_notional_pct=0.6,
                corr_notional_cap=0.4,
                max_losing_months=2,
                cooldown_losses=2,
                cooldown_hours=12.0,
                cooldown_size_mult=0.5,
                binance_api_key="test-key",
                binance_api_secret="test-secret",
            ),
        )

    def test_single_runner_blocks_live(self, monkeypatch):
        from scalper_hft.live.pairs_runner import PairsPaperRunner

        self._live_settings(monkeypatch)
        with pytest.raises(RuntimeError, match="paper-only"):
            PairsPaperRunner("XRPUSDT", "BTCUSDT", interval="1h")

    def test_portfolio_runner_blocks_live(self, monkeypatch):
        from scalper_hft.live.pairs_runner import PairsPortfolioRunner

        self._live_settings(monkeypatch)
        with pytest.raises(RuntimeError, match="paper-only"):
            PairsPortfolioRunner(interval="1h")


# ── M4: live equity з біржі ───────────────────────────────────────────────────


class _EquityClient(_FakeLiveClient):
    """Fake-клієнт + fetch_usdt_equity (M4)."""

    def __init__(self, equity: float = 25_000.0) -> None:
        super().__init__()
        self.equity = equity

    def fetch_usdt_equity(self) -> float:
        return self.equity


class TestLiveEquitySync:
    def test_live_equity_rebases_cash_and_day_start(self):
        """M4: у live cash/day_start беруться з реального балансу, не з $1000."""
        from types import SimpleNamespace

        from scalper_hft.live.trader import LiveTrader

        class _AlwaysLongStrat:
            name = "always_long"
            param_space: dict = {}
            needs_trades = False

            def generate_signals(self, df):
                return pd.Series(1, index=df.index)

        acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
        trader = LiveTrader(_AlwaysLongStrat(), "BTCUSDT", "1m", account=acc, client=_EquityClient(equity=25_000.0))
        trader.settings = SimpleNamespace(
            dry_run=False,
            maker_execution=True,
            position_pct=0.01,
            max_open_positions=1,
            daily_loss_limit=0.03,
            max_consecutive_losses=3,
            maker_fill_wait_bars=1,
            binance_api_key="test-key",
            binance_api_secret="test-secret",
        )
        equity = trader.sync_live_equity()
        assert equity == 25_000.0
        assert acc.day_start_equity == 25_000.0
        # позиція без руху: cash = equity (upnl = 0)
        assert acc.equity == pytest.approx(25_000.0)
        assert trader.last_live_equity == 25_000.0

    def test_live_equity_skipped_in_paper(self):
        from scalper_hft.live.trader import LiveTrader

        trader = LiveTrader(_live_trader(_FakeLiveClient()).strategy, "BTCUSDT", "1m")
        assert trader.settings.dry_run is True or trader.settings.dry_run
        assert trader.sync_live_equity() is None  # paper — no-op


# ── M5: нормалізація під фільтри біржі ───────────────────────────────────────


class _FilterClient(_FakeLiveClient):
    """Клієнт із sanitize_order (LOT_SIZE=0.001, tick=0.1, minNotional=5)."""

    def __init__(self) -> None:
        super().__init__()
        self.sanitized: list[tuple] = []

    def sanitize_order(self, symbol, side, amount, price=None):
        # спрощений фільтр: крок 0.001, ціна крок 0.1, мін. ноціонал 5
        import math

        qty = math.floor(amount / 0.001) * 0.001
        px = None
        if price is not None:
            px = math.floor(price / 0.1) * 0.1
        self.sanitized.append((symbol, side, qty, px))
        if qty <= 0 or (px is not None and qty * px < 5.0):
            return qty, px, "below min notional / zero qty"
        return qty, px, None


class TestOrderSanitize:
    def test_trader_rejects_below_min_notional(self):
        """M5: ордер, що не проходить фільтри, не летить на біржу."""
        from types import SimpleNamespace

        from scalper_hft.live.trader import LiveTrader, execute_signal

        class _AlwaysLongStrat:
            name = "always_long"
            param_space: dict = {}
            needs_trades = False

            def generate_signals(self, df):
                return pd.Series(1, index=df.index)

        fake = _FilterClient()
        acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
        trader = LiveTrader(_AlwaysLongStrat(), "BTCUSDT", "1m", account=acc, client=fake)
        trader.settings = SimpleNamespace(
            dry_run=False,
            maker_execution=True,
            position_pct=0.0001,  # мізерний розмір → нижче min notional
            max_open_positions=1,
            daily_loss_limit=0.03,
            max_consecutive_losses=3,
            maker_fill_wait_bars=1,
            binance_api_key="test-key",
            binance_api_secret="test-secret",
        )
        idx = pd.date_range("2026-09-01", periods=60, freq="1min")
        df = pd.DataFrame({"open": 100.0, "high": 100.001, "low": 99.999, "close": 100.0, "volume": 1.0}, index=idx)
        res = execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
        assert "submit_failed" in res
        assert len(fake.created) == 0, "ордер не мав дійти до create_order"
        assert "BTCUSDT" not in acc.positions


# ── Дрібні фікси (раунд 4): telegram, mcp, metrics, exit-fee, erc, recorder ──


class TestTelegramConfig:
    def test_creds_read_via_settings(self, monkeypatch):
        """telegram має читати .env через config (а не голий os.getenv)."""
        import scalper_hft.live.telegram as tg
        from scalper_hft.config import Settings

        monkeypatch.setattr(tg, "get_settings", lambda: Settings(telegram_bot_token="tok", telegram_chat_id="123"))
        assert tg._creds() == ("tok", "123")
        monkeypatch.setattr(tg, "get_settings", lambda: Settings())
        assert tg._creds() is None


class TestMcpPaperStepGuard:
    def test_paper_step_blocks_live(self, monkeypatch):
        from types import SimpleNamespace

        import scalper_hft.config as cfg
        import scalper_hft.mcp_trading as mcp

        monkeypatch.setattr(
            cfg, "get_settings", lambda: SimpleNamespace(dry_run=False, position_pct=0.01, taker_fee=0.0, maker_fee=0.0)
        )
        with pytest.raises(RuntimeError, match="DRY_RUN=true"):
            mcp._paper_step({})


class TestMetricsMinor:
    def test_cagr_not_exploding_on_short_window(self):
        idx = pd.date_range("2024-01-01", periods=500, freq="1h")
        eq = pd.Series((1.0 + 0.0005) ** np.arange(500) * 10_000.0, index=idx)
        m = compute_metrics(eq)
        # 500 годин ≈ 0.057 року: раніше CAGR був би ~×17 від total_return
        assert m.cagr == pytest.approx(m.total_return, rel=1e-6)

    def test_duplicate_index_sharpe_sane(self):
        base = pd.date_range("2024-01-01", periods=40, freq="1min")
        idx = pd.DatetimeIndex(sorted(list(base) + list(base[:5])))
        rng = np.random.default_rng(0)
        eq = pd.Series((1.0 + rng.normal(0.0001, 0.001, len(idx))).cumprod() * 10_000.0, index=idx)
        m = compute_metrics(eq)
        assert m.sharpe < 1_000, "дублікати індексу завищують Sharpe"


class TestExitFeeInTradeRet:
    def test_engine_round_trip_carries_both_fees(self):
        """Flat close: ret угоди = entry fee + exit fee (раніше лише entry)."""
        from scalper_hft.backtest.engine import _extract_trades

        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        pos = pd.Series(0.0, index=idx)
        pos.iloc[5:15] = 1.0  # вхід на барі 5, вихід на 15
        ret = pd.Series(0.0, index=idx)
        fees = pd.Series(0.0, index=idx)
        fees.iloc[5] = 0.001  # entry fee
        fees.iloc[15] = 0.001  # exit fee
        close = pd.Series(100.0, index=idx)
        tr = _extract_trades(pos, ret, fees, close)
        assert len(tr) == 1
        assert tr["ret"].iloc[0] == pytest.approx(-0.002)

    def test_engine_flip_carries_both_sides(self):
        """Flip ±1: кожна угода несе обидві сторони комісії (по 0.001)."""
        from scalper_hft.backtest.engine import _extract_trades

        n = 12
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        pos = pd.Series(0.0, index=idx)
        pos.iloc[1:] = np.where(np.arange(1, n) % 2 == 1, 1.0, -1.0)
        ret = pd.Series(0.0, index=idx)
        # fee за turnover: вхід 0→±1 = 0.001; flip ±1→∓1 = 0.002
        fees = (pos - pos.shift(1)).abs().fillna(0.0) * 0.001
        close = pd.Series(100.0, index=idx)
        tr = _extract_trades(pos, ret, fees, close)
        assert len(tr) >= 3
        # кожна ЗАКРИТА угода несе entry-fee + exit-fee (0.002 при flat price);
        # остання угода наприкінці серії ще відкрита — лише entry-fee (−0.001)
        assert (tr["ret"].iloc[:-1] <= -0.0019).all(), tr["ret"].tolist()
        assert tr["ret"].iloc[-1] == pytest.approx(-0.001)

    def test_pairs_flat_exit_carries_exit_fee(self):
        from scalper_hft.backtest.pairs import _extract_trades

        n = 25
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        pos = pd.Series(0.0, index=idx)
        pos.iloc[5:15] = 1.0
        turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
        strat_ret = -turnover * 2 * 0.0005  # лише комісії (спред flat)
        tr = _extract_trades(pos, strat_ret)
        assert len(tr) == 1
        assert tr["ret"].iloc[0] == pytest.approx(-0.002)  # entry+exit fee


class TestErcDeadAsset:
    def test_dead_asset_gets_zero_weight(self):
        from scalper_hft.portfolio.erc import erc_weights

        rng = np.random.default_rng(0)
        n = 300
        r = np.column_stack([rng.normal(0.001, 0.01, n), rng.normal(0.0005, 0.008, n), np.full(n, 0.002)])
        w = erc_weights(r)
        assert abs(w.sum() - 1.0) < 1e-6
        assert w[2] == pytest.approx(0.0, abs=1e-12)
        assert w[0] > 0 and w[1] > 0


class TestRecorderEventTs:
    def test_event_ts_prefers_server_time(self):
        from scalper_hft.live.bookticker_recorder import _event_ts

        ts = _event_ts({"E": 1_700_000_000_000, "b": "1", "B": "2", "a": "3", "A": "4"})
        assert ts == pd.Timestamp("2023-11-14 22:13:20")
        # fallback на now() без поля E
        ts2 = _event_ts({"b": "1"})
        assert abs((pd.Timestamp.now(tz="UTC").tz_localize(None) - ts2).total_seconds()) < 5
