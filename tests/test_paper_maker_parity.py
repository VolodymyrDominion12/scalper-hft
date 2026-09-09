"""Paper maker execution parity vs backtest ``_simulate_maker_fills``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings, set_settings
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.trader import LiveTrader, execute_signal
from scalper_hft.live.trader_bars import interval_seconds
from scalper_hft.strategies.base import Strategy


def _make_df(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=n, freq="1min")
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.002, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(index=idx)
    df["open"] = close
    df["close"] = close
    df["high"] = close * 1.002
    df["low"] = close * 0.998
    df["volume"] = 1000.0
    return df


class _StepLong(Strategy):
    """Long з бару 20 — як у realism-тестах рушія."""

    name = "step_long"
    param_space: dict = {}

    def generate_signals(self, df, trades=None, funding=None):
        s = pd.Series(0, index=df.index)
        s.iloc[20:] = 1
        return s


def _signed_position_frac(account: PaperAccount, symbol: str, mark: float) -> float:
    pos = account.positions.get(symbol)
    if pos is None:
        return 0.0
    sign = 1.0 if pos.side == "long" else -1.0
    eq = account.equity_at({symbol: mark})
    return sign * pos.size * mark / eq if eq > 0 else 0.0


def _paper_maker_step_loop(trader: LiveTrader, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Bar-by-bar paper maker з shift(1), як у backtest-рушії."""
    symbol = trader.symbol
    bar_sec = interval_seconds(trader.interval)
    fracs = np.zeros(len(df))
    for i in range(1, len(df)):
        ts = df.index[i]
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        chase = float(df["close"].iloc[i - 1])
        trader.resolve_pending_orders(ts, hi, lo, chase, i)

        window = df.iloc[:i]
        signal = int(signals.iloc[i - 1])
        bar_end_prev = df.index[i - 1] + pd.Timedelta(seconds=bar_sec)
        execute_signal(trader, signal, window, now=bar_end_prev)
        trader.resolve_pending_orders(ts, hi, lo, chase, i)

        mark = float(df["close"].iloc[i])
        trader.account.mark({symbol: mark})
        fracs[i] = _signed_position_frac(trader.account, symbol, mark)
    return pd.Series(fracs, index=df.index)


def test_submit_order_paper_maker_returns_pending() -> None:
    orig = get_settings()
    set_settings(dataclasses.replace(orig, dry_run=True, maker_execution=True))
    try:
        account = PaperAccount(initial_capital=10_000.0)
        trader = LiveTrader(_StepLong(), "BTCUSDT", "1m", account=account)
        ok, status = trader._submit_order("buy", 0.01, 100.0, kind="open", pos_side="long")
        assert ok is True
        assert status == "pending"
        assert len(trader.pending_orders) == 1
        assert "BTCUSDT" not in account.positions
    finally:
        set_settings(orig)


def test_paper_maker_matches_backtest_positions() -> None:
    df = _make_df(120)
    strat = _StepLong()
    position_pct = 0.5
    cost = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_frac=0.0)

    bt = run_backtest(
        df,
        strat,
        position_pct=position_pct,
        is_maker=True,
        cost=cost,
    )

    orig = get_settings()
    set_settings(
        dataclasses.replace(
            orig,
            dry_run=True,
            maker_execution=True,
            maker_fill_wait_bars=500,
            position_pct=position_pct,
            maker_fee=0.0,
            taker_fee=0.0,
            slippage_bps=0.0,
            max_open_positions=10,
            daily_loss_limit=1.0,
            max_consecutive_losses=999,
            per_symbol_notional_pct=1.0,
            max_leverage=10.0,
        )
    )
    try:
        account = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
        trader = LiveTrader(strat, "BTCUSDT", "1m", account=account)
        signals = strat.generate_signals(df)
        paper_pos = _paper_maker_step_loop(trader, df, signals)
    finally:
        set_settings(orig)

    # Порівняння з бару 21 (перший можливий fill після сигналу на 20).
    bt_vals = bt.positions.iloc[21:].to_numpy(dtype=float)
    paper_vals = paper_pos.iloc[21:].to_numpy(dtype=float)
    np.testing.assert_allclose(paper_vals, bt_vals, rtol=0.02, atol=0.02)


def test_resolve_pending_fills_on_touch() -> None:
    orig = get_settings()
    set_settings(dataclasses.replace(orig, dry_run=True, maker_execution=True, maker_fill_wait_bars=10))
    try:
        account = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
        trader = LiveTrader(_StepLong(), "BTCUSDT", "1m", account=account)
        ok, status = trader._submit_order("buy", 0.05, 100.0, kind="open", pos_side="long")
        assert ok and status == "pending"
        ts = pd.Timestamp("2024-06-01")
        events = trader.resolve_pending_orders(ts, high=101.0, low=99.0, chase_limit=100.0, bar_idx=5)
        assert any(e.startswith("filled:") for e in events)
        assert "BTCUSDT" in account.positions
    finally:
        set_settings(orig)
