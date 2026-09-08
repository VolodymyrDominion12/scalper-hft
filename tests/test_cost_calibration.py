"""Калібровка CostModel з Implementation Shortfall."""

from __future__ import annotations

from scalper_hft.backtest.execution import CostModel, calibrate_from_is
from scalper_hft.live.is_log import IsJournal


def test_calibrate_from_is_median_maker_vs_chase() -> None:
    j = IsJournal()
    j.log("t1", "A/B", "A", "buy", 100.0, 100.2, is_maker=True)  # 20 bps
    j.log("t2", "A/B", "B", "sell", 50.0, 50.0, is_maker=True)  # 0
    j.log("t3", "A/B", "A", "buy", 100.0, 100.5, is_maker=False)  # 50 bps
    maker = calibrate_from_is(j.records, kind="maker")
    chase = calibrate_from_is(j.records, kind="chase")
    assert chase > maker
    cost = CostModel().with_is_slippage(j.records, kind="chase")
    assert cost.slippage_frac == chase


def test_calibrate_from_is_empty_fallback() -> None:
    assert calibrate_from_is([], fallback=0.0003) == 0.0003


def test_is_cost_hint_warns_when_chase_far_from_current() -> None:
    from scalper_hft.validation.paper_audit import format_cost_hint, is_cost_hint

    j = IsJournal()
    j.log("t1", "A/B", "A", "buy", 100.0, 100.2, is_maker=True)  # 20 bps
    j.log("t2", "A/B", "A", "buy", 100.0, 100.5, is_maker=False)  # 50 bps
    hint = is_cost_hint(j.records, current_slippage_frac=0.0002, warn_bps=1.0)
    assert hint["chase_slippage_frac"] > hint["maker_slippage_frac"]
    assert hint["warn"] is True
    text = format_cost_hint(hint)
    assert "chase=" in text and "warn" in text
