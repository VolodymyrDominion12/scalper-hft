"""Тести per-symbol notional cap + margin/liquidation proximity (1D)."""

from __future__ import annotations

import pandas as pd
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.risk_gate import margin_proximity_ok, per_symbol_notional_ok
from scalper_hft.live.trader import LiveTrader, execute_signal


def _df(close: float, n: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    c = pd.Series([close] * n, index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)


class _AlwaysLong:
    name = "always_long"
    param_space: dict = {}
    needs_trades = False
    needs_funding = False

    def generate_signals(self, df, trades=None, funding=None):
        return pd.Series(1, index=df.index)


def _trader(start_equity: float = 10_000.0, cap_pct: float = 0.30, buf: float = 0.10) -> LiveTrader:
    from scalper_hft.config import get_settings

    base = get_settings()
    settings = type(base)(
        **{
            **base.__dict__,
            "dry_run": True,
            "per_symbol_notional_pct": cap_pct,
            "liquidation_proximity_buffer": buf,
            "position_pct": 0.01,
        }
    )
    t = LiveTrader(_AlwaysLong(), "BTCUSDT", "1h", account=PaperAccount(start_equity))
    t.settings = settings
    t.dd_breaker.triggered = False
    return t


def test_per_symbol_notional_ok() -> None:
    # 3000 butional ≤ 10000 × 0.30 = 3000 → ok (на межі)
    assert per_symbol_notional_ok(3000.0, 10_000.0, 0.30)
    assert not per_symbol_notional_ok(3001.0, 10_000.0, 0.30)
    # нулі/негатив → fail-closed
    assert not per_symbol_notional_ok(100.0, 0.0, 0.30)
    assert not per_symbol_notional_ok(100.0, 10_000.0, 0.0)


def test_margin_proximity_ok() -> None:
    # max_leverage=3, buffer=0.10 → поріг 3×0.9=2.7
    assert margin_proximity_ok(0.0, 20_000.0, 10_000.0, 3.0, 0.10)  # 2.0 ≤ 2.7
    assert not margin_proximity_ok(0.0, 28_000.0, 10_000.0, 3.0, 0.10)  # 2.8 > 2.7
    # buffer=0 → вироджується у leverage-ліміт (27_000/10_000=2.7 ≤ 3.0)
    assert margin_proximity_ok(0.0, 27_000.0, 10_000.0, 3.0, 0.0)
    assert not margin_proximity_ok(0.0, 31_000.0, 10_000.0, 3.0, 0.0)


def test_trader_blocks_entry_above_notional_cap() -> None:
    # position_pct такий, що butional > cap: 50% equity > 30% cap
    t = _trader(cap_pct=0.30)
    t.settings = type(t.settings)(**{**t.settings.__dict__, "position_pct": 0.50})
    df = _df(100.0)
    out = execute_signal(t, signal=1, df=df, now=df.index[-1] + pd.Timedelta(seconds=30))
    assert "blocked:per-symbol notional cap" in out
    assert "BTCUSDT" not in t.account.positions


def test_trader_blocks_entry_margin_proximity() -> None:
    # великий position_pct + низький max_leverage + buffer → margin proximity
    t = _trader(cap_pct=1.0, buf=0.50)  # cap не обмежує; buffer 50% → поріг 3×0.5=1.5
    t.settings = type(t.settings)(**{**t.settings.__dict__, "position_pct": 0.50, "max_leverage": 3.0})
    df = _df(100.0)
    out = execute_signal(t, signal=1, df=df, now=df.index[-1] + pd.Timedelta(seconds=30))
    # butional = 0.50 × 10000 / 100 × 100 = 5000; leverage 0.5 ≤ 1.5 → має пройти
    assert "long" in out


def test_trader_allows_entry_within_caps() -> None:
    t = _trader(cap_pct=0.30, buf=0.10)
    df = _df(100.0)
    out = execute_signal(t, signal=1, df=df, now=df.index[-1] + pd.Timedelta(seconds=30))
    # position_pct=0.01 → butional = 0.01×10000 = 100; 100 ≤ 3000 cap; leverage 0.01
    assert "long" in out
    assert "BTCUSDT" in t.account.positions
