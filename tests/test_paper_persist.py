"""Фаза 0 DEPLOY_PLAN: snapshot/restore, control.json, daemon sleep, same_bar."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.bar_clock import daemon_sleep_sec, seconds_until_bar_wake, seconds_until_next_close
from scalper_hft.live.control import ControlState, load_control
from scalper_hft.live.pairs_runner import (
    PairsEngine,
    PairsPaperRunner,
    _paper_loop,
    should_persist_action,
)
from scalper_hft.live.store import PaperStore
from scalper_hft.strategies.pairs_arb import PairsArb


def _ohlc(n: int = 80, close: float = 100.0, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    c = np.full(n, close, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 10.0},
        index=idx,
    )


def test_account_snapshot_roundtrip() -> None:
    ts = pd.Timestamp("2025-06-01 12:00")
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0, maker_fee=0.0)
    acc.open_position("AAA/BBB:AAA", "short", 2.0, 50.0, ts, is_maker=True)
    acc.cash = 9_990.0
    acc.realized_pnl = -10.0
    acc.consecutive_losses = 2
    acc.day_start_equity = 9_995.0
    acc.mark({"AAA/BBB:AAA": 51.0})

    restored = PaperAccount.from_snapshot(acc.to_snapshot())
    assert restored.cash == 9_990.0
    assert restored.realized_pnl == -10.0
    assert restored.consecutive_losses == 2
    assert restored.day_start_equity == 9_995.0
    pos = restored.positions["AAA/BBB:AAA"]
    assert pos.side == "short" and pos.size == 2.0 and pos.entry_price == 50.0
    assert restored._marks["AAA/BBB:AAA"] == 51.0


def test_store_runtime_roundtrip(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    assert store.load_runtime() is None
    store.save_runtime({"version": 1, "account": {"cash": 1.0}})
    got = store.load_runtime()
    assert got is not None and got["account"]["cash"] == 1.0
    store.close()


def test_engine_pending_survives_snapshot() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, is_maker=True)
    ts0 = pd.Timestamp("2025-01-01 00:00")
    assert "quoted" in eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    snap = eng.to_snapshot()
    acc2 = PaperAccount.from_snapshot(acc.to_snapshot())
    eng2 = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc2, wait_bars=1, is_maker=True)
    eng2.apply_snapshot(snap)
    assert eng2.pending is not None
    assert eng2.last_bar_ts == ts0
    ts1 = pd.Timestamp("2025-01-01 01:00")
    msg = eng2.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1)
    assert "filled" in msg
    assert len(acc2.positions) == 2


def test_control_missing_file_defaults(tmp_path: Path) -> None:
    assert load_control(tmp_path / "nope.json") == ControlState()


def test_control_flags(tmp_path: Path) -> None:
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"pause": True, "no_new_entries": True, "flatten": True}), encoding="utf-8")
    st = load_control(p)
    assert st.pause and st.no_new_entries and st.flatten


def test_control_flatten_only_if_true(tmp_path: Path) -> None:
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"flatten": 1}), encoding="utf-8")
    assert load_control(p).flatten is False


def test_control_invalid_json_pauses(tmp_path: Path) -> None:
    p = tmp_path / "control.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_control(p).pause is True


def test_no_new_entries_blocks_entry_not_exit() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1)
    eng.control_block_entries = True
    ts = pd.Timestamp("2025-03-01")
    msg = eng.on_bar(ts, 101, 99, 100, 51, 49, 50, signal=1)
    assert "control:no_new_entries" in msg
    assert acc.is_flat

    eng.control_block_entries = False
    eng.on_bar(ts, 101, 99, 100, 51, 49, 50, signal=1)
    eng.on_bar(ts + pd.Timedelta(hours=1), 101, 99, 100, 51, 49, 50, signal=1)
    assert not acc.is_flat
    eng.control_block_entries = True
    msg_exit = eng.on_bar(ts + pd.Timedelta(hours=2), 101, 99, 100, 51, 49, 50, signal=0)
    assert "quoted" in msg_exit
    assert "blocked" not in msg_exit


def test_pause_skips_fetch(tmp_path: Path, monkeypatch) -> None:
    ctrl = tmp_path / "control.json"
    ctrl.write_text(json.dumps({"pause": True}), encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("fetch не має викликатись під pause")

    monkeypatch.setattr("scalper_hft.live.pairs_runner._fetch_ohlcv", boom)
    runner = PairsPaperRunner("AAA", "BBB", store=PaperStore(tmp_path / "p.sqlite"), control_path=ctrl)
    assert runner.step() == "hold:paused"


def test_runner_restore_skips_same_bar(tmp_path: Path, monkeypatch) -> None:
    df1 = _ohlc(80, 100.0)
    df2 = _ohlc(80, 50.0)

    def fake_fetch(symbol: str, interval: str, limit: int = 800) -> pd.DataFrame:
        return df1.copy() if symbol == "AAA" else df2.copy()

    monkeypatch.setattr("scalper_hft.live.pairs_runner._fetch_ohlcv", fake_fetch)
    db = tmp_path / "p.sqlite"
    now = df1.index[-1] + pd.Timedelta(minutes=5)
    r1 = PairsPaperRunner("AAA", "BBB", store=PaperStore(db), control_path=tmp_path / "c.json")
    action = r1.step(now=now)
    assert "hold:same_bar" not in action
    cash = r1.account.cash
    last = r1._last_ts
    r1.store.close()

    r2 = PairsPaperRunner("AAA", "BBB", store=PaperStore(db), control_path=tmp_path / "c.json")
    assert r2._last_ts == last
    assert r2.account.cash == cash
    assert r2.step(now=now) == "hold:same_bar"


def test_should_persist_action() -> None:
    assert should_persist_action("quoted want=1")
    assert not should_persist_action("hold:paused")
    assert not should_persist_action("hold:same_bar")
    assert not should_persist_action("error:boom")
    assert not should_persist_action("hold:same_bar || hold:same_bar")
    assert should_persist_action("quoted want=1 || hold:same_bar")


def test_bar_wake_before_close() -> None:
    now = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    assert seconds_until_next_close("1h", now) == 3600.0
    assert seconds_until_bar_wake("1h", now, lead_sec=75.0) == 3525.0
    late = pd.Timestamp("2026-01-01 10:59:00", tz="UTC")
    wake = seconds_until_bar_wake("1h", late, lead_sec=75.0)
    assert 60.0 <= wake <= 65.0
    assert daemon_sleep_sec("1h", "hold:paused") == 5.0


def test_paper_loop_daemon_stops(tmp_path: Path) -> None:
    stop = threading.Event()
    acc = PaperAccount(10_000.0)

    def step() -> str:
        stop.set()
        return "quoted want=1"

    res = _paper_loop(
        step,
        None,
        "1h",
        acc,
        "AAA/BBB",
        lambda: 0,
        lambda: 0,
        daemon=True,
        iterations=99,
        sleep_sec=0,
        stop=stop,
        install_signals=False,
    )
    assert res.actions == ["quoted want=1"]
    assert len(res.equity) == 1
