"""P0: бар'єр у барах, maker-рушій, funding-кеш, fail-closed, денний roll."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.trader import LiveTrader, execute_signal
from scalper_hft.ml.labeling import add_vertical_barrier, label_from_ohlcv


class _AlwaysLong:
    name = "always_long"
    param_space: dict = {}
    needs_trades = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)


def _ohlc(n: int = 80, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )


def test_vertical_barrier_is_bars_not_calendar_days() -> None:
    close = _ohlc(200, freq="1min")["close"]
    t_events = close.index[:20]
    t1 = add_vertical_barrier(t_events, close, num_days=10)
    assert len(t1) == 20
    deltas = (t1.values - t1.index.values).astype("timedelta64[m]").astype(int)
    assert set(deltas.tolist()) == {10}


def test_label_from_ohlcv_holding_bars_on_1m() -> None:
    df = _ohlc(80, freq="1min")
    events = label_from_ohlcv(df, pt=10.0, sl=10.0, holding_bars=10, vol_span=20)
    valid = events.dropna(subset=["t1"])
    assert len(valid) > 5
    bars = [int((t1 - t0) / pd.Timedelta(minutes=1)) for t0, t1 in zip(valid.index, valid["t1"])]
    assert max(bars) <= 10
    assert min(bars) >= 1


def test_engine_maker_path_runs() -> None:
    df = _ohlc(80)
    res = run_backtest(
        df,
        _AlwaysLong(),
        cost=CostModel(maker_fee=0.0, taker_fee=0.0, slippage_frac=0.0),
        is_maker=True,
        position_pct=0.1,
    )
    assert len(res.equity) == len(df)
    assert res.params["is_maker"] is True
    assert (res.positions.abs() >= 0).all()


def test_download_funding_refreshes_stale_cache(monkeypatch, tmp_path) -> None:
    from scalper_hft.data import downloader as dl

    stale = pd.DataFrame(
        {"fundingRate": [0.0001, 0.0001]},
        index=pd.date_range("2025-01-01", periods=2, freq="8h"),
    )
    fresh = pd.DataFrame({"fundingRate": [0.0003]}, index=[pd.Timestamp("2026-08-30 12:00")])
    called = {"n": 0}

    class FakeDL:
        def funding(self, symbol: str, days: int) -> pd.DataFrame:
            called["n"] += 1
            return fresh

    monkeypatch.setattr(dl, "get_settings", lambda: SimpleNamespace(data_dir_abs=tmp_path))
    monkeypatch.setattr(dl, "load_funding", lambda path: stale)
    monkeypatch.setattr(dl, "Downloader", FakeDL)
    monkeypatch.setattr(dl, "_utc_now", lambda: pd.Timestamp("2026-08-30 12:00"))
    out = dl.download_funding("BTCUSDT", days=30)
    assert called["n"] == 1
    assert float(out["fundingRate"].iloc[-1]) == 0.0003


def test_download_funding_keeps_fresh_cache(monkeypatch, tmp_path) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2026-08-30 12:00")
    cached = pd.DataFrame(
        {"fundingRate": [0.0002] * 12},
        index=pd.date_range(end=now - pd.Timedelta(hours=4), periods=12, freq="8h"),
    )
    called = {"n": 0}

    class FakeDL:
        def funding(self, symbol: str, days: int) -> pd.DataFrame:
            called["n"] += 1
            return cached

    monkeypatch.setattr(dl, "get_settings", lambda: SimpleNamespace(data_dir_abs=tmp_path))
    monkeypatch.setattr(dl, "load_funding", lambda path: cached)
    monkeypatch.setattr(dl, "Downloader", FakeDL)
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    out = dl.download_funding("BTCUSDT", days=1)
    assert called["n"] == 0
    assert len(out) == 12


class _RejectClient:
    def create_order(self, *args: object, **kwargs: object) -> dict:
        raise RuntimeError("exchange reject")


def test_live_reject_does_not_open_local_position() -> None:
    idx = pd.date_range("2026-08-28 10:00", periods=60, freq="1min")
    df = _ohlc(60)
    df.index = idx
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    trader = LiveTrader(_AlwaysLong(), "BTCUSDT", "1m", account=acc, client=_RejectClient())
    trader.settings = SimpleNamespace(
        dry_run=False,
        maker_execution=True,
        position_pct=0.01,
        max_open_positions=1,
        daily_loss_limit=0.03,
        max_consecutive_losses=3,
    )
    result = execute_signal(trader, 1, df, now=idx[-1] + pd.Timedelta(seconds=30))
    assert "submit_failed" in result
    assert "BTCUSDT" not in acc.positions
    assert acc.cash == 10_000.0


def test_maybe_roll_day_resets_daily_loss() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    acc.day_start_equity = 9_000.0
    acc.consecutive_losses = 2
    trader = LiveTrader(_AlwaysLong(), "BTCUSDT", "1m", account=acc)
    day1 = pd.Timestamp("2026-08-28 23:00")
    day2 = pd.Timestamp("2026-08-29 00:05")
    assert trader.maybe_roll_day(day1) is False
    assert acc.day_start_equity == 9_000.0
    assert trader.maybe_roll_day(day2) is True
    assert acc.day_start_equity == acc.equity
    assert acc.consecutive_losses == 0
