"""L0: shadow-TCA chase, поки paper лишається strict_both."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.chase_shadow import (
    LIVE_CHASE_TCA_BUDGET_BPS,
    aggregate_shadow_events,
    evaluate_chase_shadow,
    extra_fee_bps,
    format_live_chase_gap,
)
from scalper_hft.live.fills import FillDecision
from scalper_hft.live.pairs_engine import PairsEngine, PendingOrder
from scalper_hft.live.store import SCHEMA_VERSION, PaperStore
from scalper_hft.strategies.pairs_arb import PairsArb


class _FillOnTouch:
    def random(self) -> float:
        return 0.0


def _event_chase_leg2() -> tuple[FillDecision, FillDecision]:
    return FillDecision(True, 100.0, "filled"), FillDecision(False, 50.0, "unfilled_no_touch")


def test_extra_fee_bps_binance_defaults() -> None:
    assert extra_fee_bps(0.0002, 0.0005) == pytest.approx(3.0)


def test_evaluate_none_when_both_or_neither() -> None:
    both = evaluate_chase_shadow(
        FillDecision(True, 100.0, "filled"),
        FillDecision(True, 50.0, "filled"),
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=50.0,
        symbol1="AAA",
        symbol2="BBB",
        max_drift_bps=10.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        arrival1=100.0,
        arrival2=50.0,
    )
    neither = evaluate_chase_shadow(
        FillDecision(False, 100.0, "unfilled_no_touch"),
        FillDecision(False, 50.0, "unfilled_no_touch"),
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=50.0,
        symbol1="AAA",
        symbol2="BBB",
        max_drift_bps=10.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        arrival1=100.0,
        arrival2=50.0,
    )
    assert both is None
    assert neither is None


def test_evaluate_chase_leg2_adds_taker_delta_and_is() -> None:
    d1, d2 = _event_chase_leg2()
    event = evaluate_chase_shadow(
        d1,
        d2,
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.5,
        mid2=49.98,
        symbol1="AAA",
        symbol2="BBB",
        max_drift_bps=20.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        arrival1=100.0,
        arrival2=50.0,
    )
    assert event is not None
    assert event.action == "chase_leg2"
    assert event.chased_symbol == "BBB"
    assert event.extra_fee_bps == pytest.approx(3.0)
    assert event.chase_is_bps > 0
    assert event.extra_cost_bps == pytest.approx(event.extra_fee_bps + event.chase_is_bps)


def test_evaluate_unwind_when_drift_exceeded() -> None:
    event = evaluate_chase_shadow(
        FillDecision(True, 100.0, "filled"),
        FillDecision(False, 50.0, "unfilled_no_touch"),
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=45.0,
        symbol1="AAA",
        symbol2="BBB",
        max_drift_bps=10.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        arrival1=100.0,
        arrival2=50.0,
    )
    assert event is not None
    assert event.action == "unwind_leg1"
    assert event.extra_fee_bps == pytest.approx(7.0)
    assert event.extra_cost_bps >= event.extra_fee_bps


def test_strict_both_does_not_open_but_logs_shadow(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper.sqlite")
    acc = PaperAccount(initial_capital=10_000.0, taker_fee=0.0005, maker_fee=0.0002)
    engine = PairsEngine(
        "AAAUSDT",
        "BBBUSDT",
        PairsArb(lookback=60),
        acc,
        store=store,
        legging_mode="strict_both",
        wait_bars=3,
        fill_rng=_FillOnTouch(),
    )
    now = pd.Timestamp("2026-01-01 12:00:00")
    engine.pending = (
        PendingOrder("AAAUSDT", engine._k("AAAUSDT"), "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BBBUSDT", engine._k("BBBUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    status = engine._resolve_pending(now, high1=102.0, low1=99.0, high2=49.99, low2=49.95)
    assert status == "pending"
    assert acc.positions == {}
    shadow = store.all_shadow_legging()
    assert len(shadow) == 1
    assert str(shadow.iloc[0]["action"]) == "chase_leg2"
    stats = store.fill_stats()
    assert stats.get("filled", 0) == 0


def test_shadow_logged_once_across_wait_bars(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper.sqlite")
    acc = PaperAccount(initial_capital=10_000.0)
    engine = PairsEngine(
        "AAAUSDT",
        "BBBUSDT",
        PairsArb(lookback=60),
        acc,
        store=store,
        legging_mode="strict_both",
        wait_bars=3,
        fill_rng=_FillOnTouch(),
    )
    now = pd.Timestamp("2026-01-01 12:00:00")
    engine.pending = (
        PendingOrder("AAAUSDT", engine._k("AAAUSDT"), "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BBBUSDT", engine._k("BBBUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    engine._resolve_pending(now, high1=102.0, low1=99.0, high2=49.99, low2=49.95)
    later = now + pd.Timedelta(hours=1)
    engine._resolve_pending(later, high1=102.0, low1=99.0, high2=49.99, low2=49.95)
    assert len(store.all_shadow_legging()) == 1
    assert engine.pending is not None
    assert engine.pending[0].shadow_logged is True


def test_snapshot_restores_shadow_logged() -> None:
    acc = PaperAccount(initial_capital=10_000.0)
    engine = PairsEngine("AAAUSDT", "BBBUSDT", PairsArb(lookback=60), acc, wait_bars=3)
    now = pd.Timestamp("2026-01-01 12:00:00")
    engine.pending = (
        PendingOrder(
            "AAAUSDT",
            engine._k("AAAUSDT"),
            "buy",
            "long",
            10.0,
            100.0,
            False,
            now,
            shadow_logged=True,
        ),
        PendingOrder("BBBUSDT", engine._k("BBBUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    snap = engine.to_snapshot()
    restored = PairsEngine("AAAUSDT", "BBBUSDT", PairsArb(lookback=60), acc, wait_bars=3)
    restored.apply_snapshot(snap)
    assert restored.pending is not None
    assert restored.pending[0].shadow_logged is True
    assert restored.pending[1].shadow_logged is False


def test_chase_mode_does_not_write_shadow(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "paper.sqlite")
    acc = PaperAccount(initial_capital=10_000.0)
    engine = PairsEngine(
        "AAAUSDT",
        "BBBUSDT",
        PairsArb(lookback=60),
        acc,
        store=store,
        legging_mode="chase",
        max_drift_bps=20.0,
        fill_rng=_FillOnTouch(),
    )
    now = pd.Timestamp("2026-01-01 12:00:00")
    engine.pending = (
        PendingOrder("AAAUSDT", engine._k("AAAUSDT"), "buy", "long", 10.0, 100.0, False, now),
        PendingOrder("BBBUSDT", engine._k("BBBUSDT"), "sell", "short", 0.1, 50.0, False, now),
    )
    status = engine._resolve_pending(now, high1=102.0, low1=99.0, high2=49.99, low2=49.95)
    assert "chase" in status
    assert store.all_shadow_legging().empty


def test_schema_v4_has_shadow_table(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "v4.sqlite")
    assert store.schema_version() == SCHEMA_VERSION
    assert store.all_shadow_legging().empty


def test_live_chase_hold_when_extra_breaches_budget() -> None:
    d1, d2 = _event_chase_leg2()
    event = evaluate_chase_shadow(
        d1,
        d2,
        side1="buy",
        side2="sell",
        limit1=100.0,
        limit2=50.0,
        mid1=100.0,
        mid2=49.90,
        symbol1="AAA",
        symbol2="BBB",
        max_drift_bps=20.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        arrival1=100.0,
        arrival2=50.0,
    )
    assert event is not None
    report = aggregate_shadow_events([event])
    assert report.would_breach_budget(2.5, budget_bps=LIVE_CHASE_TCA_BUDGET_BPS)
    text = format_live_chase_gap(report, paper_blended_tca_bps=2.5)
    assert "LIVE_CHASE_HOLD" in text


def test_paper_audit_includes_chase_shadow(tmp_path: Path) -> None:
    from scalper_hft.validation.paper_audit import audit_paper_store

    store = PaperStore(tmp_path / "paper.sqlite")
    ts = pd.Timestamp("2026-01-01")
    store.log_equity(ts, "A/B", 10_000.0, 10_000.0, 0.0)
    store.log_shadow_legging(
        ts,
        "A/B",
        "chase_leg2",
        drift_bps=4.0,
        chased_symbol="BBB",
        chased_side="sell",
        chase_fill_px=49.98,
        maker_symbol="AAA",
        extra_fee_bps=3.0,
        chase_is_bps=4.0,
        extra_cost_bps=7.0,
    )
    audit = audit_paper_store(store)
    assert audit.chase_shadow is not None
    assert audit.chase_shadow.n_chase == 1
    assert "Chase shadow" in audit.summary()
