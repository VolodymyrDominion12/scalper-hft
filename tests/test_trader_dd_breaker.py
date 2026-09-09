"""Тести DrawdownBreaker на single-symbol LiveTrader (1D).

Перевіряє, що при спрацюванні breaker (triggered):
- нові входи блокуються (halt);
- відкрита позиція авто-flatten'иться;
- close/вихід (сигнал=0) не блокується.
"""

from __future__ import annotations

import pandas as pd
from scalper_hft.live.account import PaperAccount
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


def _trader(start_equity: float = 10_000.0, max_dd: float = 0.05) -> LiveTrader:
    from scalper_hft.config import get_settings

    base = get_settings()
    settings = type(base)(**{**base.__dict__, "dry_run": True, "max_drawdown_pct": max_dd})
    t = LiveTrader(_AlwaysLong(), "BTCUSDT", "1h", account=PaperAccount(start_equity))
    t.settings = settings
    t.dd_breaker.max_dd_pct = max_dd
    t.dd_breaker.high_water = start_equity
    t.dd_breaker.triggered = False
    return t


def test_dd_breaker_blocks_new_entry_when_triggered() -> None:
    t = _trader()
    t.dd_breaker.triggered = True  # імітуємо спрацювання
    df = _df(100.0)
    out = execute_signal(t, signal=1, df=df, now=df.index[-1] + pd.Timedelta(seconds=30))
    # позиції немає → halt (не відкриваємо нову)
    assert "dd_breaker:halt" in out
    assert "BTCUSDT" not in t.account.positions


def test_dd_breaker_auto_flattens_open_position() -> None:
    t = _trader()
    # спершу відкриваємо позицію (breaker не tripped)
    df0 = _df(100.0)
    out0 = execute_signal(t, signal=1, df=df0, now=df0.index[-1] + pd.Timedelta(seconds=30))
    assert "long" in out0
    assert "BTCUSDT" in t.account.positions
    # тепер breaker спрацював → наступний крок з сигналом=1 авто-flatten
    t.dd_breaker.triggered = True
    df1 = _df(101.0)
    out1 = execute_signal(t, signal=1, df=df1, now=df1.index[-1] + pd.Timedelta(seconds=30))
    assert "dd_breaker:flatten" in out1
    assert "BTCUSDT" not in t.account.positions


def test_dd_breaker_allows_exit_when_triggered() -> None:
    t = _trader()
    # відкриваємо позицію
    df0 = _df(100.0)
    execute_signal(t, signal=1, df=df0, now=df0.index[-1] + pd.Timedelta(seconds=30))
    assert "BTCUSDT" in t.account.positions
    # breaker tripped + сигнал=0 (вихід) — позиція закривається (вихід дозволений)
    t.dd_breaker.triggered = True
    df1 = _df(101.0)
    out = execute_signal(t, signal=0, df=df1, now=df1.index[-1] + pd.Timedelta(seconds=30))
    assert "BTCUSDT" not in t.account.positions
    # закриття відбувається через dd_breaker:flatten (вихід при triggered)
    assert "dd_breaker:flatten" in out
