"""Implementation Shortfall агрегатор (Phase B2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scalper_hft.live.is_log import IsJournal, IsRecord, shortfall_bps


@dataclass
class ISReport:
    n_fills: int
    median_is_bps: float
    maker_is_bps: float
    taker_is_bps: float
    model_slippage_bps: float
    coverage_ok: bool  # ≥ 20 fills для калібровки

    def summary(self) -> str:
        delta = self.median_is_bps - self.model_slippage_bps
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
        return (
            f"IS Report: {self.n_fills} fills | "
            f"Realized={self.median_is_bps:.1f} bps {arrow} Model={self.model_slippage_bps:.1f} bps\n"
            f"  Maker={self.maker_is_bps:.1f} bps | Taker={self.taker_is_bps:.1f} bps"
        )


def _median_bps(records: list[IsRecord]) -> float:
    if not records:
        return 0.0
    vals = sorted(r.shortfall_bps for r in records)
    return float(vals[len(vals) // 2])


def aggregate_journal(journal: IsJournal, *, model_slippage_bps: float) -> ISReport:
    """Агрегувати IS з IsJournal (pairs engine)."""
    makers, takers = journal.split_by_maker()
    all_recs = journal.records
    return ISReport(
        n_fills=len(all_recs),
        median_is_bps=_median_bps(all_recs),
        maker_is_bps=_median_bps(makers),
        taker_is_bps=_median_bps(takers) if takers else 0.0,
        model_slippage_bps=model_slippage_bps,
        coverage_ok=len(all_recs) >= 20,
    )


def build_from_orders(orders: pd.DataFrame, *, model_slippage_bps: float) -> ISReport:
    """Оцінка IS з SQLite orders (limit як decision, price як fill)."""
    if orders is None or orders.empty:
        return ISReport(0, 0.0, 0.0, 0.0, model_slippage_bps, False)
    filled = orders[orders["status"].astype(str).str.lower().isin({"filled", "fill", "ok"})].copy()
    if filled.empty:
        filled = orders.copy()
    records: list[IsRecord] = []
    for _, row in filled.iterrows():
        side = str(row.get("side", "buy")).lower()
        limit_px = float(row.get("price", 0.0) or 0.0)
        fill_px = float(row.get("price", 0.0) or 0.0)
        if limit_px <= 0:
            continue
        reason = str(row.get("reason", "")).lower()
        is_maker = "taker" not in reason and "chase" not in reason
        records.append(
            IsRecord(
                ts=row.get("ts"),
                pair=str(row.get("pair", "")),
                symbol=str(row.get("symbol", "")),
                side=side,
                mid_at_decision=limit_px,
                fill_price=fill_px,
                shortfall_bps=shortfall_bps(side, limit_px, fill_px),
                is_maker=is_maker,
            )
        )
    journal = IsJournal(records=records)
    return aggregate_journal(journal, model_slippage_bps=model_slippage_bps)


def calibrate_slippage_bps(report: ISReport) -> float:
    """Рекомендований slippage_bps для CostModel з IS (якщо coverage_ok)."""
    if not report.coverage_ok:
        return report.model_slippage_bps
    return max(report.median_is_bps, 0.1)


def aggregate_is_bps(values: list[float]) -> float:
    """Медіана IS (bps) зі списку."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.median(arr))
