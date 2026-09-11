"""W0-R7: trader_loop control plane, DD flatten, fetch_positions fail-closed, H1 klines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.control import ControlState
from scalper_hft.live.trader import LiveTrader
from scalper_hft.live.trader_loop import run_trader_once
from scalper_hft.strategies.base import Strategy


class _AlwaysLong(Strategy):
    name = "always_long"
    param_space: dict[str, tuple[float, float, float]] = {}
    needs_trades = False
    needs_funding = False

    def generate_signals(self, df: pd.DataFrame, trades: object = None, funding: object = None) -> pd.Series:
        return pd.Series(1, index=df.index)


def _ohlc(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0},
        index=idx,
    )


def _paper_trader(equity: float = 10_000.0, max_dd: float = 0.05) -> LiveTrader:
    from scalper_hft.config import get_settings

    base = get_settings()
    trader = LiveTrader(_AlwaysLong(), "BTCUSDT", "1h", account=PaperAccount(equity))
    trader.settings = type(base)(**{**base.__dict__, "dry_run": True, "max_drawdown_pct": max_dd})
    trader.dd_breaker.max_dd_pct = max_dd
    trader.dd_breaker.high_water = equity
    trader.dd_breaker.triggered = False
    return trader


def test_trader_loop_pause_skips_signal() -> None:
    trader = _paper_trader()
    calls: list[str] = []
    real = trader.execute

    def _spy(decision: TradeDecision, price: float, ts: pd.Timestamp) -> str:
        calls.append("execute")
        return real(decision, price, ts)

    trader.execute = _spy  # type: ignore[method-assign]
    df = _ohlc()
    out = run_trader_once(trader, df, now=df.index[-1] + pd.Timedelta(seconds=30), control=ControlState(pause=True))
    assert out == "hold:paused"
    assert calls == []
    assert trader.account.is_flat


def test_trader_loop_dd_breaker_flatten() -> None:
    trader = _paper_trader()
    df = _ohlc()
    now = df.index[-1] + pd.Timedelta(seconds=30)
    opened = run_trader_once(trader, df, now=now, control=ControlState())
    assert "BTCUSDT" in trader.account.positions
    assert "dd_breaker" not in opened
    trader.dd_breaker.triggered = True
    out = run_trader_once(trader, df, now=now, control=ControlState())
    assert "dd_breaker:flatten" in out
    assert "BTCUSDT" not in trader.account.positions


def test_sync_fill_symbols_does_not_swallow(tmp_path: Path) -> None:
    from scalper_hft.live.store import PaperStore
    from scalper_hft.live.sync_engine import SyncEngine

    class _BoomExchange:
        def fetch_positions(self) -> list[object]:
            raise RuntimeError("fetch boom")

    engine = SyncEngine(PaperStore(tmp_path / "s.sqlite"), mode="live")
    with pytest.raises(RuntimeError, match="fetch boom"):
        engine._fill_symbols(_BoomExchange())


def test_paper_loop_klines_network_error_not_killswitch(tmp_path: Path) -> None:
    """H1: мережевий збій klines у paper-loop не ставить KillSwitch / pause."""
    import json

    import ccxt
    from scalper_hft.live.pairs_runner import _paper_loop

    acc = PaperAccount(10_000.0)
    calls = {"n": 0}
    ctrl = tmp_path / "control.json"

    def step() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ccxt.NetworkError("klines timeout")
        return "quoted want=1"

    res = _paper_loop(
        step,
        None,
        "1h",
        acc,
        "AAA/BBB",
        lambda: 0,
        lambda: 0,
        daemon=False,
        iterations=2,
        sleep_sec=0,
        install_signals=False,
        control_path=ctrl,
    )
    assert res.actions[0].startswith("error:")
    assert "killed:" not in res.actions[0]
    assert res.actions[1] == "quoted want=1"
    assert calls["n"] == 2
    if ctrl.exists():
        assert not json.loads(ctrl.read_text(encoding="utf-8")).get("pause")
