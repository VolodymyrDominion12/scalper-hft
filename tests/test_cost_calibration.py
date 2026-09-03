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
