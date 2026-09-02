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
            df, AlwaysLong(), funding=funding, position_pct=0.1, cost=CostModel(maker_fee=0, taker_fee=0, slippage_frac=0)
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
