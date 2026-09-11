"""W0-TCA: чесний Implementation Shortfall (mid ≠ limit) і opportunity cost."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.is_log import IsJournal, blended_tca_bps, shortfall_bps
from scalper_hft.live.is_report import (
    aggregate_journal,
    build_from_orders,
    calibrate_slippage_bps,
    records_from_orders,
)
from scalper_hft.live.pairs_engine import PairsEngine, PendingOrder
from scalper_hft.live.store import PaperStore
from scalper_hft.strategies.pairs_arb import PairsArb
from scalper_hft.validation.paper_audit import audit_paper_store


def test_buy_fill_nonzero_is_when_mid_differs_from_limit() -> None:
    """limit=fill=100, mid=99.8 → buy IS > 0 (не нуль від mid=limit)."""
    j = IsJournal()
    rec = j.log("t0", "A/B", "A", "buy", 99.8, 100.0, is_maker=True)
    assert rec.shortfall_bps == pytest.approx(shortfall_bps("buy", 99.8, 100.0))
    assert rec.shortfall_bps == pytest.approx(20.04008016, rel=1e-6)
    assert j.mean_bps() > 0


def test_unfilled_increases_blended_tca() -> None:
    j = IsJournal()
    j.log("t0", "A/B", "A", "buy", 99.8, 100.0, is_maker=True)
    fills_only = aggregate_journal(j, model_slippage_bps=2.0)
    j.log_unfilled("t1", "A/B", "A", "buy", arrival_mid=100.0, cancel_mid=100.5)
    with_miss = aggregate_journal(j, model_slippage_bps=2.0)
    assert with_miss.n_unfilled == 1
    assert with_miss.median_miss_bps > 0
    assert with_miss.blended_tca_bps > fills_only.blended_tca_bps
    assert with_miss.fill_rate == pytest.approx(0.5)


def test_blended_tca_formula() -> None:
    assert blended_tca_bps(fill_rate=0.7, fill_is_bps=2.0, miss_bps=10.0) == pytest.approx(4.4)


def test_build_from_orders_without_mid_skips_fills() -> None:
    orders = pd.DataFrame(
        {
            "ts": ["2025-01-01"],
            "pair": ["A/B"],
            "symbol": ["A"],
            "side": ["buy"],
            "price": [100.0],
            "status": ["filled"],
            "reason": ["filled"],
        }
    )
    report = build_from_orders(orders, model_slippage_bps=2.0)
    assert report.n_fills == 0
    assert report.coverage_ok is False
    assert report.median_is_bps == 0.0


def test_build_from_orders_uses_mid_not_limit() -> None:
    orders = pd.DataFrame(
        {
            "ts": ["2025-01-01"],
            "pair": ["A/B"],
            "symbol": ["A"],
            "side": ["buy"],
            "price": [100.0],
            "status": ["filled"],
            "reason": ["filled"],
            "mid": [99.8],
            "decision_mid": [100.0],
        }
    )
    report = build_from_orders(orders, model_slippage_bps=2.0)
    assert report.n_fills == 1
    assert report.mid_distinct is True
    assert report.median_is_bps == pytest.approx(20.04008016, rel=1e-6)
    assert report.coverage_ok is False  # < 20 fills


def test_build_from_orders_unfilled_miss() -> None:
    orders = pd.DataFrame(
        {
            "ts": ["2025-01-01", "2025-01-01"],
            "pair": ["A/B", "A/B"],
            "symbol": ["A", "A"],
            "side": ["buy", "buy"],
            "price": [100.0, 100.0],
            "status": ["filled", "unfilled"],
            "reason": ["filled", "unfilled_no_touch"],
            "mid": [99.8, 100.4],
            "decision_mid": [100.0, 100.0],
        }
    )
    recs = records_from_orders(orders)
    assert len(recs) == 2
    assert recs[0].filled is True
    assert recs[1].filled is False
    report = build_from_orders(orders, model_slippage_bps=2.0)
    assert report.n_unfilled == 1
    assert report.median_miss_bps == pytest.approx(40.0, rel=1e-6)
    assert report.blended_tca_bps > report.median_is_bps * report.fill_rate


def test_calibrate_requires_distinct_mid_and_coverage() -> None:
    j = IsJournal()
    for i in range(20):
        j.log(f"t{i}", "A/B", "A", "buy", 100.0, 100.0, is_maker=True)
    same = aggregate_journal(j, model_slippage_bps=2.0)
    assert same.n_fills == 20
    assert same.mid_distinct is False
    assert same.coverage_ok is False
    assert calibrate_slippage_bps(same) == 2.0

    j2 = IsJournal()
    for i in range(20):
        j2.log(f"t{i}", "A/B", "A", "buy", 99.8, 100.0, is_maker=True)
    honest = aggregate_journal(j2, model_slippage_bps=2.0)
    assert honest.coverage_ok is True
    assert calibrate_slippage_bps(honest) == pytest.approx(max(honest.median_is_bps, 0.1))


def test_calibrate_from_is_ignores_unfilled() -> None:
    from scalper_hft.backtest.execution import calibrate_from_is

    j = IsJournal()
    j.log("t0", "A/B", "A", "buy", 100.0, 100.2, is_maker=True)  # 20 bps
    j.log_unfilled("t1", "A/B", "A", "buy", 100.0, 101.0)  # 100 bps miss
    assert calibrate_from_is(j.records, kind="maker") == pytest.approx(0.002)


def test_pairs_engine_logs_fill_vs_bar_mid(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
        store=store,
    )
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    eng.on_bar(ts0, 101, 99, 100, 51, 49, 50, signal=1)
    assert eng.pending is not None
    # Філ майже на mid, але mid ≠ limit: AAA sell mid=100.025, BBB buy mid=49.975.
    # dist-to-mid ≤ 0 → P(fill)=1 після touch.
    eng.on_bar(ts1, 100.1, 99.95, 100, 50.05, 49.90, 50, signal=1)
    assert eng.n_filled == 1
    # want=+1 → short AAA / long BBB: buy-нога — BBB, limit=50, mid=49.975
    buys = [r for r in eng.is_journal.records if r.filled and r.side == "buy"]
    assert len(buys) == 1
    assert buys[0].symbol == "BBB"
    assert buys[0].fill_price == pytest.approx(50.0)
    assert buys[0].mid_at_decision == pytest.approx(49.975)
    assert buys[0].shortfall_bps > 0
    filled_rows = store.all_orders()[store.all_orders()["status"] == "filled"]
    assert filled_rows["mid"].notna().all()
    assert (filled_rows["mid"] != filled_rows["price"]).any()
    store.close()


def test_pairs_engine_timeout_logs_unfilled_miss() -> None:
    acc = PaperAccount(10_000.0, taker_fee=0.0, maker_fee=0.0)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20, regime_scale=False),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
    )
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    eng.pending = (
        PendingOrder("AAA", "AAA/BBB:AAA", "buy", "long", 1.0, 90.0, False, ts0, decision_mid=100.0),
        PendingOrder("BBB", "AAA/BBB:BBB", "sell", "short", 1.0, 60.0, False, ts0, decision_mid=50.0),
    )
    # Бар не торкається лімітів → unfilled_no_touch, mid1=100.5, mid2=50.5
    msg = eng.on_bar(ts1, 101, 100, 100.5, 51, 50, 50.5, signal=0)
    assert "unfilled" in msg
    misses = [r for r in eng.is_journal.records if not r.filled]
    assert len(misses) == 2
    buy_miss = next(r for r in misses if r.side == "buy")
    assert buy_miss.shortfall_bps == pytest.approx(shortfall_bps("buy", 100.0, 100.5))


def test_paper_audit_prints_is(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    ts = pd.Timestamp("2025-01-01")
    store.log_equity(ts, "A/B", 10_000.0, 10_000.0, 0.0)
    store.log_order(ts, "A/B", "A", "buy", 1.0, 100.0, "filled", "filled", mid=99.8, decision_mid=100.0)
    store.log_order(ts, "A/B", "B", "sell", 1.0, 50.0, "unfilled", "unfilled_no_touch", mid=50.4, decision_mid=50.0)
    audit = audit_paper_store(store)
    text = audit.summary()
    assert "IS Report" in text
    assert "Blended TCA" in text
    assert audit.is_report is not None
    assert audit.is_report.n_fills == 1
    assert audit.is_report.n_unfilled == 1
    store.close()


def test_store_persists_mid_columns(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "p.sqlite")
    ts = pd.Timestamp("2025-01-01")
    store.log_order(ts, "A/B", "A", "buy", 1.0, 100.0, "filled", "filled", mid=99.8, decision_mid=100.0)
    row = store.all_orders().iloc[0]
    assert float(row["mid"]) == pytest.approx(99.8)
    assert float(row["decision_mid"]) == pytest.approx(100.0)
    store.close()
