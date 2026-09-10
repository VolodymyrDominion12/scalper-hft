"""Тести візуалізації бектесту (модуль scalper_hft/visualization).

1. Рушій збагачує угоди цінами входу/виходу та рівнями SL/TP зі стратегії.
2. make_backtest_figure будує багатопанельну фігуру з маркерами угод.
3. Вікно (start/end) та даунсемплінг (max_bars) зберігають бари угод.
4. equity_figure / drawdown_series / trades_table коректні.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies import get_strategy
from scalper_hft.strategies.base import Strategy


def _make_df(n: int = 1500, seed: int = 7) -> pd.DataFrame:
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


class _NoLevelsStrategy(Strategy):
    """Стратегія без exit_levels — для перевірки graceful-поведінки."""

    name = "no_levels_test"

    def generate_signals(self, df, trades=None, funding=None):
        sig = pd.Series(0.0, index=df.index)
        sig.iloc[: len(df) // 2] = 1.0  # один довгий трейд
        return sig


def _empty_result(df: pd.DataFrame) -> BacktestResult:
    """BacktestResult без жодної угоди (для graceful-перевірок)."""
    from scalper_hft.backtest.metrics import compute_metrics

    equity = pd.Series(10_000.0, index=df.index)
    pos = pd.Series(0.0, index=df.index)
    trades = pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "ret", "entry_price", "exit_price"])
    return BacktestResult(
        equity=equity,
        positions=pos,
        trades=trades,
        metrics=compute_metrics(equity, trades=trades, exposure=0.0, turnover=0.0),
    )


# ── 1. Рушій: ціни та SL/TP у trades ─────────────────────────────────────────
class TestEngineEnrichment:
    def test_trades_have_entry_exit_prices(self):
        res = run_backtest(_make_df(), get_strategy("mean_reversion"), cost=CostModel())
        assert {"entry_price", "exit_price"} <= set(res.trades.columns)
        t = res.trades.dropna(subset=["entry_price", "exit_price"])
        assert len(t) >= 1
        assert np.all(t["entry_price"] > 0) and np.all(t["exit_price"] > 0)

    def test_mean_reversion_sl_tp_attached_and_sane(self):
        res = run_backtest(_make_df(), get_strategy("mean_reversion"), cost=CostModel())
        t = res.trades.dropna(subset=["sl_price", "tp_price"])
        assert len(t) >= 1
        for _, r in t.iterrows():
            if r.side == 1:
                assert r.sl_price <= r.tp_price  # рівні відносні середини BB
            else:
                assert r.sl_price >= r.tp_price
            assert np.isfinite(r.sl_price) and np.isfinite(r.tp_price)

    def test_strategy_without_levels_has_no_sl_tp(self):
        res = run_backtest(_make_df(), _NoLevelsStrategy(), cost=CostModel())
        assert "sl_price" not in res.trades.columns
        assert len(res.trades) >= 1  # угода є, але без рівнів

    def test_empty_trades_attach_noop(self):
        df = _make_df()
        res = run_backtest(df, _NoLevelsStrategy(), cost=CostModel())
        from scalper_hft.backtest.engine import _attach_exit_levels

        out = _attach_exit_levels(res.trades, None)
        assert out is res.trades


# ── 2. make_backtest_figure ──────────────────────────────────────────────────
class TestBacktestFigure:
    def _fig(self, **kwargs):
        from scalper_hft.visualization.charts import make_backtest_figure

        df = _make_df()
        res = run_backtest(df, get_strategy("mean_reversion"), cost=CostModel())
        return df, res, make_backtest_figure(df, res, symbol="TEST", **kwargs)

    def test_full_figure_has_all_panels(self):
        _, _, fig = self._fig()
        traces = [t.name for t in fig.data]
        # ціна (свічка + індикатори + маркери) + об'єм + позиція (2) + equity + просадка
        assert "OHLC" in traces
        assert "Equity" in traces
        assert "Просадка, %" in traces
        assert "Об'єм" in traces
        assert "Лонг" in traces and "Шорт" in traces
        assert any("вхід" in (t or "") for t in traces)
        assert any("вихід" in (t or "") for t in traces)

    def test_no_trades_figure_graceful(self):
        from scalper_hft.visualization.charts import make_backtest_figure

        df = _make_df()
        fig = make_backtest_figure(df, _empty_result(df))
        assert fig.data[0].name == "OHLC"

    def test_raises_on_empty_df(self):
        from scalper_hft.visualization.charts import make_backtest_figure

        with pytest.raises(ValueError):
            make_backtest_figure(pd.DataFrame(), _empty_result(_make_df()))

    def test_with_trades_false_hides_markers(self):
        _, _, fig = self._fig(with_trades=False)
        names = [t.name for t in fig.data]
        assert not any("вхід" in (n or "") for n in names)
        assert not any("вихід" in (n or "") for n in names)

    def test_downsample_keeps_trade_bars(self):
        _, res, fig = self._fig(max_bars=200)
        sampled = set(pd.DatetimeIndex(fig.data[0].x))
        for ts in res.trades["entry_ts"]:
            assert ts in sampled, f"бар входу {ts} загублено при даунсемплінгу"
        for ts in res.trades["exit_ts"]:
            assert ts in sampled, f"бар виходу {ts} загублено при даунсемплінгу"

    def test_window_slices_data(self):
        _, _, fig = self._fig(start="2025-01-02 00:00", end="2025-01-03 23:59")
        x = fig.data[0].x
        assert x[0] >= pd.Timestamp("2025-01-02 00:00")
        assert x[-1] <= pd.Timestamp("2025-01-03 23:59")

    def test_indicator_overlays_auto_detected(self):
        from scalper_hft.features.indicators import add_standard_features
        from scalper_hft.visualization.charts import make_backtest_figure

        df = _make_df()
        res = run_backtest(df, get_strategy("mean_reversion"), cost=CostModel())
        # фіча-колонки додаються ЛИШЕ для графіка (бектест отримує сирий OHLCV)
        fdf = add_standard_features(df)
        fig = make_backtest_figure(fdf, res)
        names = {t.name for t in fig.data}
        assert {"bb_mid", "bb_up", "bb_low", "ema_9", "vwap_20"} <= names


# ── 3. SL/TP сегменти ────────────────────────────────────────────────────────
class TestSlTpLevels:
    def test_sl_tp_segments_drawn(self):
        from scalper_hft.visualization.charts import make_backtest_figure

        df = _make_df()
        res = run_backtest(df, get_strategy("mean_reversion"), cost=CostModel())
        fig = make_backtest_figure(df, res)
        groups = {t.legendgroup for t in fig.data if t.legendgroup}
        assert "SL" in groups and "TP" in groups

    def test_sl_tp_cap_max_trades(self):
        from plotly.subplots import make_subplots
        from scalper_hft.visualization.charts import add_sl_tp_levels

        n = 600
        idx = pd.date_range("2025-01-01", periods=2 * n, freq="1min")
        trades = pd.DataFrame(
            {
                "entry_ts": idx[:n],
                "exit_ts": idx[n:],
                "side": 1,
                "sl_price": np.linspace(99.0, 100.0, n),
                "tp_price": np.linspace(101.0, 102.0, n),
            }
        )
        fig = make_subplots(rows=1, cols=1)
        add_sl_tp_levels(fig, trades, max_trades=100)
        sl_traces = [t for t in fig.data if t.legendgroup == "SL"]
        assert len(sl_traces) == 100

    def test_missing_sl_columns_noop(self):
        import plotly.graph_objects as go
        from scalper_hft.visualization.charts import add_sl_tp_levels

        fig = go.Figure()
        trades = pd.DataFrame({"entry_ts": [pd.Timestamp("2025-01-01")], "exit_ts": [pd.Timestamp("2025-01-02")]})
        add_sl_tp_levels(fig, trades)  # без sl_price/tp_price — не падає
        assert len(fig.data) == 0


# ── 4. Equity / drawdown / таблиця ───────────────────────────────────────────
class TestEquityHelpers:
    def test_drawdown_series(self):
        from scalper_hft.visualization.charts import drawdown_series

        eq = pd.Series([100.0, 120.0, 110.0, 130.0])
        dd = drawdown_series(eq)
        np.testing.assert_allclose(dd, [0.0, 0.0, -10.0 / 120.0, 0.0])

    def test_equity_figure_traces(self):
        from scalper_hft.visualization.charts import equity_figure

        df = _make_df()
        res = run_backtest(df, get_strategy("mean_reversion"), cost=CostModel())
        fig = equity_figure(res)
        names = {t.name for t in fig.data}
        assert "Equity" in names and "Просадка, %" in names

    def test_trades_table_columns_and_side(self):
        from scalper_hft.visualization.charts import trades_table

        df = _make_df()
        res = run_backtest(df, get_strategy("mean_reversion"), cost=CostModel())
        tb = trades_table(res, initial_capital=10_000.0)
        assert list(tb.columns) == [
            "Вхід",
            "Вихід",
            "Сторона",
            "Ціна входу",
            "Ціна виходу",
            "SL",
            "TP",
            "PnL, %",
            "PnL, $",
        ]
        assert set(tb["Сторона"].dropna().unique()) <= {"Лонг", "Шорт"}
        assert np.allclose(tb["PnL, $"], tb["PnL, %"] / 100.0 * 10_000.0)


# ── 5. exit_levels інших стратегій ───────────────────────────────────────────
class TestExitLevels:
    def test_hmm_reversion_delegates_to_mean_reversion(self):
        from scalper_hft.strategies.hmm_reversion import HmmReversionScalper

        df = _make_df()
        mr = get_strategy("mean_reversion", stop_atr_mult=2.5)
        hmm = HmmReversionScalper(stop_atr_mult=2.5)
        lv_mr = mr.exit_levels(df)
        lv_hmm = hmm.exit_levels(df)
        pd.testing.assert_frame_equal(lv_mr, lv_hmm)

    def test_ml_strategy_levels_are_close_relative(self):
        from scalper_hft.strategies.ml_strategy import MLStrategy

        df = _make_df()
        lv = MLStrategy(pt=1.5, sl=2.0).exit_levels(df)
        close = df["close"]
        assert set(lv.columns) == {"sl_long", "tp_long", "sl_short", "tp_short"}
        # лонг: tp вище close, sl нижче; шорт — дзеркально (після warmup σ)
        assert (lv["tp_long"] > close).iloc[60:].all() and (lv["sl_long"] < close).iloc[60:].all()
        assert (lv["tp_short"] < close).iloc[60:].all() and (lv["sl_short"] > close).iloc[60:].all()
        # симетрія з однаковою волатильністю σ: лонг/шорт TP і SL дзеркальні
        np.testing.assert_allclose(lv["tp_long"] - close, -(lv["tp_short"] - close), rtol=1e-12)
        np.testing.assert_allclose(lv["sl_long"] - close, -(lv["sl_short"] - close), rtol=1e-12)


# ── 6. Деталі угоди (клік по маркеру) ───────────────────────────────────────
class TestTradeDetail:
    def _res(self):
        return run_backtest(_make_df(), get_strategy("mean_reversion"), cost=CostModel())

    def test_find_trade_by_ts(self):
        from scalper_hft.visualization.charts import find_trade_by_ts

        res = self._res()
        t = res.trades.iloc[0]
        assert find_trade_by_ts(res.trades, t["entry_ts"])["entry_ts"] == t["entry_ts"]
        assert find_trade_by_ts(res.trades, t["exit_ts"])["entry_ts"] == t["entry_ts"]
        assert find_trade_by_ts(res.trades, pd.Timestamp("2020-01-01")) is None
        assert find_trade_by_ts(res.trades, "not-a-date") is None
        assert find_trade_by_ts(pd.DataFrame(), t["entry_ts"]) is None

    def test_trade_detail_figure(self):
        from scalper_hft.visualization.charts import trade_detail_figure

        df = _make_df()
        res = self._res()
        t = res.trades.iloc[0]
        fig = trade_detail_figure(df, res, t["entry_ts"])
        names = {tr.name for tr in fig.data}
        assert "OHLC" in names and "Equity" in names
        assert "Лонг" in names or "Шорт" in names

    def test_trade_detail_figure_unknown_trade_raises(self):
        from scalper_hft.visualization.charts import trade_detail_figure

        df = _make_df()
        res = self._res()
        with pytest.raises(ValueError):
            trade_detail_figure(df, res, pd.Timestamp("2020-01-01"))


class TestPairsFigure:
    def test_spread_line_and_markers(self) -> None:
        from scalper_hft.backtest.metrics import compute_metrics
        from scalper_hft.backtest.pairs import PairsResult
        from scalper_hft.visualization.charts import make_pairs_figure

        idx = pd.date_range("2025-01-01", periods=80, freq="1h")
        spread = pd.Series(np.linspace(-0.02, 0.02, 80), index=idx, name="spread")
        equity = pd.Series(10_000.0 + np.arange(80), index=idx, name="equity")
        pos = pd.Series(0.0, index=idx)
        pos.iloc[10:40] = 0.3
        trades = pd.DataFrame(
            {
                "entry_ts": [idx[10]],
                "exit_ts": [idx[39]],
                "side": [1],
                "ret": [0.01],
            }
        )
        res = PairsResult(
            equity=equity,
            positions=pos,
            spread=spread,
            funding_pnl=0.0,
            metrics=compute_metrics(equity, trades=trades, exposure=0.3, turnover=0.1),
            trades=trades,
        )
        fig = make_pairs_figure(res, symbol="XRP/BTC")
        names = {tr.name for tr in fig.data}
        assert "Спред" in names
        assert "Equity" in names
        assert "Лонг-вхід" in names

    def test_empty_spread_raises(self) -> None:
        from scalper_hft.backtest.metrics import compute_metrics
        from scalper_hft.backtest.pairs import PairsResult
        from scalper_hft.visualization.charts import make_pairs_figure

        idx = pd.date_range("2025-01-01", periods=3, freq="1h")
        equity = pd.Series([1.0, 1.0, 1.0], index=idx)
        res = PairsResult(
            equity=equity,
            positions=pd.Series(dtype=float),
            spread=pd.Series(dtype=float),
            funding_pnl=0.0,
            metrics=compute_metrics(equity, trades=pd.DataFrame(), exposure=0.0, turnover=0.0),
        )
        with pytest.raises(ValueError, match="spread"):
            make_pairs_figure(res)


# ── 7. Робоче вікно свічок ───────────────────────────────────────────────────
class TestPriceWindow:
    def test_default_starts_at_first_entry(self) -> None:
        from scalper_hft.visualization.charts import default_price_window

        idx = pd.date_range("2025-01-01", periods=1000, freq="1min")
        entry = idx[100]
        trades = pd.DataFrame({"entry_ts": [entry], "exit_ts": [idx[120]]})
        start, end = default_price_window(idx, trades, bars=50, pad_before=10)
        assert start == idx[90]
        assert end == idx[139]
        assert (end - start).value == (idx[49] - idx[0]).value or (
            idx.get_indexer([end])[0] - idx.get_indexer([start])[0] + 1
        ) == 50

    def test_default_without_trades_from_start(self) -> None:
        from scalper_hft.visualization.charts import default_price_window

        idx = pd.date_range("2025-01-01", periods=80, freq="1h")
        start, end = default_price_window(idx, None, bars=20)
        assert start == idx[0]
        assert end == idx[19]

    def test_default_empty_index_raises(self) -> None:
        from scalper_hft.visualization.charts import default_price_window

        with pytest.raises(ValueError, match="порожній"):
            default_price_window(pd.DatetimeIndex([]))

    def test_shift_does_not_leave_index(self) -> None:
        from scalper_hft.visualization.charts import shift_window

        idx = pd.date_range("2025-01-01", periods=100, freq="1min")
        start, end = shift_window(idx[0], idx[19], idx, 1)
        assert start == idx[10]
        assert end == idx[29]
        left_s, left_e = shift_window(idx[0], idx[19], idx, -1)
        assert left_s == idx[0]
        assert left_e == idx[19]
        right_s, right_e = shift_window(idx[80], idx[99], idx, 1)
        assert right_e == idx[99]
        assert right_s == idx[80]

    def test_window_around_trade_contains_entry(self) -> None:
        from scalper_hft.visualization.charts import window_around_trade

        idx = pd.date_range("2025-01-01", periods=200, freq="1min")
        entry = idx[50]
        start, end = window_around_trade(entry, idx, bars=40)
        assert start <= entry <= end
        assert idx.get_indexer([end])[0] - idx.get_indexer([start])[0] + 1 == 40

    def test_neighboring_entry_steps(self) -> None:
        from scalper_hft.visualization.charts import neighboring_entry_ts

        idx = pd.date_range("2025-01-01", periods=10, freq="1h")
        trades = pd.DataFrame({"entry_ts": [idx[1], idx[4], idx[7]]})
        assert neighboring_entry_ts(trades, None, step=1) == idx[1]
        assert neighboring_entry_ts(trades, idx[1], step=1) == idx[4]
        assert neighboring_entry_ts(trades, idx[4], step=-1) == idx[1]
        assert neighboring_entry_ts(trades, idx[7], step=1) is None
        assert neighboring_entry_ts(pd.DataFrame(), idx[1], step=1) is None
