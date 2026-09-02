"""Phase 0: paper-акаунт (ф'ючерсний облік), закритий бар, close-before-flip, REGISTRY."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.paper_replay import paper_replay
from scalper_hft.live.trader import LiveTrader, closed_klines, execute_signal
from scalper_hft.strategies import REGISTRY, get_strategy


class _AlwaysLong:
    name = "always_long"
    param_space: dict = {}
    needs_trades = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df):
        return pd.Series(1, index=df.index)


class _LongThenShort:
    name = "long_then_short"
    param_space: dict = {}
    needs_trades = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df):
        n = len(df)
        sig = np.ones(n)
        sig[n // 2 :] = -1
        return pd.Series(sig, index=df.index)


def _ohlc(idx: pd.DatetimeIndex, close: np.ndarray | list[float]) -> pd.DataFrame:
    close_s = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s,
            "high": close_s * 1.001,
            "low": close_s * 0.999,
            "close": close_s,
            "volume": 10.0,
        },
        index=idx,
    )


def test_pairs_arb_in_registry():
    assert "pairs_arb" in REGISTRY
    s = get_strategy("pairs_arb", entry_z=2.0, exit_z=0.3, lookback=240)
    assert s.name == "pairs_arb"
    assert s.get("lookback", 0) == 240


def test_paper_account_initial_capital_syncs_cash():
    acc = PaperAccount(initial_capital=5_000.0)
    assert acc.cash == 5_000.0
    assert acc.equity == 5_000.0
    assert acc.day_start_equity == 5_000.0


def test_paper_account_equity_no_double_count():
    """Ф'ючерс: cash не списує ноціонал; equity ≠ cash + realized (без подвоєння)."""
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("BTCUSDT", "long", 1.0, 100.0, ts, is_maker=True)
    assert acc.cash == 10_000.0
    assert acc.equity == 10_000.0
    assert abs(acc.equity_at({"BTCUSDT": 110.0}) - 10_010.0) < 1e-9

    acc.close_position("BTCUSDT", 110.0, ts, is_maker=True)
    assert abs(acc.cash - 10_010.0) < 1e-9
    assert abs(acc.equity - 10_010.0) < 1e-9
    assert abs(acc.realized_pnl - 10.0) < 1e-9
    # старий баг давав cash + realized = 10020
    assert abs((acc.cash + acc.realized_pnl) - 10_020.0) < 1e-9


def test_paper_account_fees_and_short():
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0005, maker_fee=0.0002)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("ETHUSDT", "short", 2.0, 50.0, ts, is_maker=True)
    entry_fee = 50.0 * 2.0 * 0.0002
    assert abs(acc.cash - (10_000.0 - entry_fee)) < 1e-9
    # ціна впала → шорт у плюсі
    assert abs(acc.equity_at({"ETHUSDT": 40.0}) - (acc.cash + 20.0)) < 1e-9
    acc.close_position("ETHUSDT", 40.0, ts, is_maker=True)
    exit_fee = 40.0 * 2.0 * 0.0002
    expected_cash = 10_000.0 - entry_fee + 20.0 - exit_fee
    assert abs(acc.cash - expected_cash) < 1e-9
    assert abs(acc.equity - expected_cash) < 1e-9
    assert abs(acc.realized_pnl - (20.0 - entry_fee - exit_fee)) < 1e-9


def test_paper_account_funding_hits_cash():
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("BTCUSDT", "long", 1.0, 100.0, ts)
    pnl = acc.apply_funding("BTCUSDT", 0.001, ts)
    assert abs(pnl - (-0.1)) < 1e-9
    assert abs(acc.cash - 9_999.9) < 1e-9
    assert abs(acc.equity - 9_999.9) < 1e-9


def test_open_twice_raises():
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    acc.open_position("BTCUSDT", "long", 1.0, 100.0, ts)
    try:
        acc.open_position("BTCUSDT", "short", 1.0, 100.0, ts)
    except ValueError:
        return
    raise AssertionError("очікували ValueError на повторний open без close")


def test_closed_klines_drops_forming_bar():
    now = pd.Timestamp("2026-08-28 12:00:30")
    idx = pd.date_range(end="2026-08-28 12:00:00", periods=10, freq="1min")
    df = _ohlc(idx, list(range(10)))
    closed = closed_klines(df, "1m", now=now)
    assert len(closed) == 9
    assert closed.index[-1] == idx[-2]


def test_closed_klines_keeps_fully_closed_history():
    idx = pd.date_range("2025-01-01", periods=10, freq="1min")
    df = _ohlc(idx, list(range(10)))
    closed = closed_klines(df, "1m", now=pd.Timestamp("2026-08-28"))
    assert len(closed) == 10


def test_execute_signal_uses_last_closed_price():
    idx = pd.date_range("2026-08-28 10:00", periods=5, freq="1min")
    df = _ohlc(idx, [100.0, 100.0, 100.0, 110.0, 999.0])
    now = pd.Timestamp("2026-08-28 10:04:30")
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    trader = LiveTrader(_AlwaysLong(), "BTCUSDT", "1m", account=acc)
    execute_signal(trader, 1, df, now=now)
    pos = acc.positions["BTCUSDT"]
    assert pos.entry_price == 110.0


def test_execute_signal_close_before_flip():
    idx = pd.date_range("2025-01-01", periods=60, freq="1min")
    df = _ohlc(idx, np.full(60, 100.0))
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    trader = LiveTrader(_AlwaysLong(), "BTCUSDT", "1m", account=acc)
    execute_signal(trader, 1, df)
    assert acc.positions["BTCUSDT"].side == "long"
    result = execute_signal(trader, -1, df)
    assert "closed" in result
    assert "opened short" in result
    assert acc.positions["BTCUSDT"].side == "short"
    closed_trades = [t for t in acc.trades if t.get("type") == "trade"]
    assert len(closed_trades) == 1
    assert closed_trades[0]["side"] == "long"
    # без подвоєння PnL (ціна не змінилась, комісії 0)
    assert abs(acc.equity - 10_000.0) < 1e-6


def test_submit_order_uses_limit_post_only():
    src = inspect.getsource(LiveTrader._submit_order)
    assert "post_only" in src and '"limit"' in src
    assert "reduceOnly" in src


def test_paper_replay_reverses_long_to_short():
    idx = pd.date_range("2025-01-01", periods=400, freq="1min")
    close = 100.0 + 0.001 * np.arange(400)
    df = _ohlc(idx, close)
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    paper_replay(df, _LongThenShort(), account=acc, position_pct=0.1, symbol="BTCUSDT")
    closed = [t for t in acc.trades if t.get("type") == "trade"]
    assert any(t["side"] == "long" for t in closed), "мав закрити лонг при реверсі"
    pos = acc.positions.get("BTCUSDT")
    assert pos is not None and pos.side == "short"


def test_reconcile_multi_pair_net_positions():
    """Перевірка коректної агрегації спільних символів у мультипарному портфелі."""
    from scalper_hft.live.reconcile import ExchangePosition, reconcile_positions

    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    ts = pd.Timestamp("2025-01-01")
    # Пара 1: XRP/BTC -> лонг BTC (0.1)
    acc.open_position("XRPUSDT/BTCUSDT:BTCUSDT", "long", 0.1, 50_000.0, ts)
    # Пара 2: BTC/ETH -> шорт BTC (0.04)
    acc.open_position("BTCUSDT/ETHUSDT:BTCUSDT", "short", 0.04, 50_000.0, ts)
    # Пара 3: LINK/BTC -> лонг BTC (0.01)
    acc.open_position("LINKUSDT/BTCUSDT:BTCUSDT", "long", 0.01, 50_000.0, ts)
    # Інша нога
    acc.open_position("BTCUSDT/ETHUSDT:ETHUSDT", "long", 0.5, 3_000.0, ts)

    # Нетто-позиція BTC = +0.1 - 0.04 + 0.01 = +0.07 long
    exchange_ok = {
        "BTCUSDT": ExchangePosition("BTCUSDT", "long", 0.07),
        "ETHUSDT": ExchangePosition("ETHUSDT", "long", 0.5),
    }
    ok, reason = reconcile_positions(acc, exchange_ok)
    assert ok, f"Має успішно звірити агреговані нетто-позиції: {reason}"

    # Розходження на біржі
    exchange_drift = {
        "BTCUSDT": ExchangePosition("BTCUSDT", "long", 0.05),
        "ETHUSDT": ExchangePosition("ETHUSDT", "long", 0.5),
    }
    ok_drift, reason_drift = reconcile_positions(acc, exchange_drift)
    assert not ok_drift
    assert "BTCUSDT: розмір 0.07 vs біржа 0.05" in reason_drift


def test_pairs_walk_forward_preserves_range():
    """Перевірка, що walk-forward для пар передає high/low і підтримує maker execution."""
    from scalper_hft.backtest.pairs import run_pairs_walk_forward
    from scalper_hft.strategies.pairs_arb import PairsArb

    idx = pd.date_range("2025-01-01", periods=2500, freq="1h")
    rng = np.random.default_rng(42)
    p1 = 100.0 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    p2 = 100.0 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    d1 = pd.DataFrame({"open": p1, "high": p1 * 1.01, "low": p1 * 0.99, "close": p1, "volume": 100.0}, index=idx)
    d2 = pd.DataFrame({"open": p2, "high": p2 * 1.01, "low": p2 * 0.99, "close": p2, "volume": 100.0}, index=idx)
    strat = PairsArb(lookback=60, entry_z=1.5, exit_z=0.2)
    res = run_pairs_walk_forward(d1, d2, strat, train_bars=1500, test_bars=500, maker_execution=True)
    assert res["n_windows"] == 2
    assert "avg_oos_sharpe" in res
