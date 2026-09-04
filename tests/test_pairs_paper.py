"""Phase 1: maker fill, all-or-none ноги, paper pairs replay, SQLite, ризик портфеля."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import both_or_neither, decide_fill
from scalper_hft.live.pairs_runner import PairsEngine, pair_size_pct, replay_pairs
from scalper_hft.live.store import PaperStore
from scalper_hft.strategies.pairs_arb import PairsArb


def _ohlc(idx: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )


def test_post_only_fill_touch_and_gap():
    buy_hit = decide_fill("buy", 100.0, high=101.0, low=99.5)
    buy_miss = decide_fill("buy", 100.0, high=102.0, low=100.5)
    sell_hit = decide_fill("sell", 100.0, high=100.5, low=99.0)
    sell_miss = decide_fill("sell", 100.0, high=99.5, low=98.0)
    assert buy_hit.filled and buy_hit.fill_price == 100.0
    assert not buy_miss.filled
    assert sell_hit.filled
    assert not sell_miss.filled


def test_both_or_neither_rejects_partial():
    ok = decide_fill("buy", 100.0, 101.0, 99.0)
    miss = decide_fill("sell", 100.0, 99.0, 98.0)
    a, b = both_or_neither(ok, miss)
    assert not a.filled and not b.filled
    assert a.reason == "unfilled_partial"


def test_pair_size_pct_portfolio_cap():
    assert abs(pair_size_pct(1, 0.30, 0.60) - 0.30) < 1e-12
    assert abs(pair_size_pct(3, 0.30, 0.60) - 0.20) < 1e-12


def test_engine_fills_both_legs_or_none():
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, is_maker=True)
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    # сигнал +1 → quote на close 100/50
    a0 = eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    assert "quoted" in a0
    assert eng.pending is not None
    assert acc.is_flat
    # обидва бари торкаються → філл обох
    a1 = eng.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1)
    assert "filled" in a1
    assert eng.have == 1
    assert len(acc.positions) == 2
    keys = set(acc.positions)
    assert "AAA/BBB:AAA" in keys and "AAA/BBB:BBB" in keys
    assert acc.positions["AAA/BBB:AAA"].side == "short"
    assert acc.positions["AAA/BBB:BBB"].side == "long"


def test_engine_unfilled_when_gap_against():
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, is_maker=True)
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    # gap up: buy BBB не торкається (low 51 > 50), sell AAA теж може торкнутись — all-or-none
    a1 = eng.on_bar(ts1, 102, 101, 101.5, 53, 51, 52, signal=1)
    assert "unfilled" in a1
    assert acc.is_flat
    assert eng.n_unfilled == 1
    assert eng.n_filled == 0


def test_losing_months_block_new_opens():
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1)
    eng.losing_months = 2
    ts = pd.Timestamp("2025-03-01")
    msg = eng.on_bar(ts, 101, 99, 100, 51, 49, 50, signal=1)
    assert "blocked" in msg
    assert acc.is_flat


def test_store_roundtrip(tmp_path):
    store = PaperStore(tmp_path / "p.sqlite")
    ts = pd.Timestamp("2025-01-01")
    store.log_order(ts, "A/B", "A", "buy", 1.0, 10.0, "filled", "filled")
    store.log_order(ts, "A/B", "B", "sell", 1.0, 10.0, "unfilled", "unfilled_no_touch")
    store.log_equity(ts, "A/B", 10_000.0, 10_000.0, 0.0)
    stats = store.fill_stats("A/B")
    assert stats["filled"] == 1 and stats["unfilled"] == 1
    eq = store.recent_equity()
    assert len(eq) == 1
    store.close()


def test_replay_mean_reversion_opens_two_legs():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # сильний синус на log-ratio → z-score має входити
    z = 0.08 * np.sin(np.arange(n) / 8.0)
    c1 = 100.0 * np.exp(z)
    c2 = np.full(n, 100.0)
    df1 = _ohlc(idx, c1)
    df2 = _ohlc(idx, c2)
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    res = replay_pairs(
        "XRPUSDT",
        "BTCUSDT",
        df1,
        df2,
        strategy=PairsArb(entry_z=1.5, exit_z=0.2, lookback=24),
        account=acc,
        wait_bars=1,
        is_maker=True,
        interval="1h",
        now=pd.Timestamp("2026-01-01"),
    )
    assert res.n_filled >= 1
    assert any(k.startswith("XRPUSDT/BTCUSDT:") for k in acc.positions) or any(
        t.get("type") == "trade" for t in acc.trades
    )
    assert not res.equity.empty


def test_portfolio_block_entries_blocks_open() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, coint_kill=False)
    eng.portfolio_block_entries = True
    ts = pd.Timestamp("2025-03-01")
    msg = eng.on_bar(ts, 101, 99, 100, 51, 49, 50, signal=1)
    assert "blocked" in msg
    assert acc.is_flat


def test_markout_logged_on_next_bar() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, coint_kill=False)
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    ts2 = pd.Timestamp("2025-01-01 02:00")
    eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    eng.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1)
    assert eng.n_filled == 1
    eng.on_bar(ts2, 102, 100, 101, 52, 50, 51, signal=0)
    marked = [r for r in eng.is_journal.records if r.markout_bps is not None]
    assert len(marked) >= 1


def test_portfolio_should_halt_on_daily_loss() -> None:
    acc = PaperAccount(10_000.0)
    acc.cash = 9_600.0  # −4% vs 3% default daily_loss_limit
    from scalper_hft.live.pairs_runner import PairsPortfolioRunner

    r = object.__new__(PairsPortfolioRunner)
    r.account = acc
    r.daily_loss_limit = 0.03
    r.weekly_loss_limit = 0.07
    r.week_start_equity = 10_000.0
    assert PairsPortfolioRunner._should_halt_entries(r) is True
    acc.cash = 10_000.0
    assert PairsPortfolioRunner._should_halt_entries(r) is False


def test_pairs_engine_cancel_pending():
    acc = PaperAccount(10_000.0)
    eng = PairsEngine("AAA", "BBB", PairsArb(lookback=20), acc, wait_bars=1, coint_kill=False)
    ts0 = pd.Timestamp("2025-01-01 00:00")
    eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    assert eng.pending is not None
    assert eng.cancel_pending(reason="test") is True
    assert eng.pending is None
    assert eng.cancel_pending(reason="test") is False


def test_paper_loop_cleanup_on_stop():
    import threading

    from scalper_hft.live.pairs_runner import _paper_loop

    acc = PaperAccount(10_000.0)
    stop_event = threading.Event()
    on_stop_called = []

    def mock_step():
        stop_event.set()
        return "step_action"

    res = _paper_loop(
        step=mock_step,
        save=None,
        interval="1m",
        account=acc,
        pair="AAA/BBB",
        n_filled=lambda: 0,
        n_unfilled=lambda: 0,
        daemon=True,
        iterations=1,
        sleep_sec=1,
        stop=stop_event,
        install_signals=False,
        on_stop=lambda: on_stop_called.append(True),
    )
    assert len(on_stop_called) == 1
    assert "step_action" in res.actions

