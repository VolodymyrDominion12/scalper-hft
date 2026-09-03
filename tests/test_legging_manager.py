"""Тести для Legging Risk Manager (chase/unwind логіка розсинхронізації ніг)."""

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import FillDecision, resolve_legging
from scalper_hft.live.pairs_runner import PairsEngine, PendingOrder
from scalper_hft.strategies.pairs_arb import PairsArb


def test_resolve_legging_both_filled():
    d1 = FillDecision(True, 100.0, "filled")
    d2 = FillDecision(True, 50.0, "filled")
    res = resolve_legging(d1, d2, "buy", "sell", 100.0, 50.0, 100.0, 50.0)
    assert res.action == "both_filled"
    assert res.d1.filled and res.d2.filled


def test_resolve_legging_neither_filled():
    d1 = FillDecision(False, 100.0, "unfilled")
    d2 = FillDecision(False, 50.0, "unfilled")
    res = resolve_legging(d1, d2, "buy", "sell", 100.0, 50.0, 100.0, 50.0)
    assert res.action == "neither"
    assert not res.d1.filled and not res.d2.filled


def test_resolve_legging_chase_leg2():
    # Leg 1 buy filled @ 100.0; Leg 2 sell limit was 50.0, mid is 49.98 (drift = 4 bps)
    d1 = FillDecision(True, 100.0, "filled")
    d2 = FillDecision(False, 50.0, "unfilled_no_touch")
    res = resolve_legging(
        d1,
        d2,
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=49.98,
        max_drift_bps=10.0,
        mode="chase",
    )
    assert res.action == "chase_leg2"
    assert res.d1.filled is True
    assert res.d2.filled is True
    assert res.d2.fill_price == 49.98
    assert res.drift_bps <= 10.0
    assert res.leg1_maker is True
    assert res.leg2_maker is False


def test_resolve_legging_unwind_when_drift_exceeded():
    # Drift is 100 bps (> 10 bps limit) -> unwind
    d1 = FillDecision(True, 100.0, "filled")
    d2 = FillDecision(False, 50.0, "unfilled_no_touch")
    res = resolve_legging(
        d1,
        d2,
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=45.0,  # 10% drift
        max_drift_bps=10.0,
        mode="chase",
    )
    assert res.action == "unwind_leg1"


def test_pairs_engine_legging_chase_integration():
    acc = PaperAccount(initial_capital=10_000.0)
    strat = PairsArb(lookback=60)
    engine = PairsEngine("XRPUSDT", "BTCUSDT", strat, acc, legging_mode="chase", max_drift_bps=20.0)

    now = pd.Timestamp("2026-01-01 12:00:00")
    # Setup pending order: o1 buy @ 100, o2 sell @ 50
    engine.pending = (
        PendingOrder("XRPUSDT", "XRPUSDT/BTCUSDT:XRPUSDT", "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BTCUSDT", "XRPUSDT/BTCUSDT:BTCUSDT", "sell", "short", 0.1, 50.0, False, now),
    )

    # Next bar: o1 touches and fills (high=102, low=99 -> mid=100.5)
    # o2 does not touch (high=49.9, low=48.0 -> limit 50 was sell, high did not reach 50)
    # But mid2 is 48.95 (drift is small) -> chased!
    status = engine._resolve_pending(now, high1=102.0, low1=99.0, high2=49.99, low2=49.95)
    assert "chase" in status
    assert engine.n_filled == 1
    assert engine.pending is None
    k1 = engine._k("XRPUSDT")
    k2 = engine._k("BTCUSDT")
    assert k1 in acc.positions and k2 in acc.positions
    assert acc.positions[k1].entry_fee == pytest.approx(100.0 * 10.0 * acc.maker_fee)
    chased_px = acc.positions[k2].entry_price
    assert acc.positions[k2].entry_fee == pytest.approx(chased_px * 0.1 * acc.taker_fee)
    maker_both = 100.0 * 10.0 * acc.maker_fee + chased_px * 0.1 * acc.maker_fee
    actual_fees = acc.positions[k1].entry_fee + acc.positions[k2].entry_fee
    assert actual_fees > maker_both


def test_unwind_flattens_open_leg_as_taker() -> None:
    acc = PaperAccount(initial_capital=10_000.0)
    strat = PairsArb(lookback=60)
    engine = PairsEngine("XRPUSDT", "BTCUSDT", strat, acc, legging_mode="chase", max_drift_bps=1.0)
    now = pd.Timestamp("2026-01-01 12:00:00")
    key = engine._k("XRPUSDT")
    acc.open_position(key, "long", 10.0, 100.0, now, is_maker=True)
    cash_after_open = acc.cash
    engine.pending = (
        PendingOrder("XRPUSDT", key, "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BTCUSDT", engine._k("BTCUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    # Buy still touches (low ≤ 100), mid=95 < entry → flatten taker-ом у збиток.
    status = engine._resolve_pending(now, high1=100.0, low1=90.0, high2=45.0, low2=44.0)
    assert "unwind" in status
    assert key not in acc.positions
    assert acc.cash < cash_after_open
    exit_px = acc.trades[-1]["exit_price"]
    assert exit_px == pytest.approx(95.0)
    taker_fee = exit_px * 10.0 * acc.taker_fee
    assert cash_after_open - acc.cash >= taker_fee
    assert engine.pending is None


def test_strict_both_partial_does_not_open() -> None:
    acc = PaperAccount(initial_capital=10_000.0)
    strat = PairsArb(lookback=60)
    engine = PairsEngine("XRPUSDT", "BTCUSDT", strat, acc, legging_mode="strict_both", wait_bars=3)
    now = pd.Timestamp("2026-01-01 12:00:00")
    engine.pending = (
        PendingOrder("XRPUSDT", engine._k("XRPUSDT"), "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BTCUSDT", engine._k("BTCUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    status = engine._resolve_pending(now, high1=102.0, low1=99.0, high2=49.0, low2=48.0)
    assert status == "pending"
    assert acc.positions == {}
    assert engine.pending is not None
