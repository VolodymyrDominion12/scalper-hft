"""Implementation Shortfall агрегатор (Phase B2 / W0-TCA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scalper_hft.live.is_log import IsJournal, IsRecord, blended_tca_bps, shortfall_bps

_MID_EPS = 1e-12
_FILLED = frozenset({"filled", "fill", "ok"})
_UNFILLED = frozenset({"unfilled", "canceled", "cancelled", "timeout", "expired"})


@dataclass
class ISReport:
    n_fills: int
    n_unfilled: int
    median_is_bps: float
    maker_is_bps: float
    taker_is_bps: float
    median_miss_bps: float
    fill_rate: float
    blended_tca_bps: float
    model_slippage_bps: float
    coverage_ok: bool  # ≥ 20 fills і mid ≠ fill хоча б на одному філі
    mid_distinct: bool

    def summary(self) -> str:
        delta = self.median_is_bps - self.model_slippage_bps
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
        mid_flag = "mid≠fill" if self.mid_distinct else "mid=fill (IS ненадійний)"
        return (
            f"IS Report: {self.n_fills} fills / {self.n_unfilled} unfilled "
            f"(fill-rate={self.fill_rate:.0%}, {mid_flag}) | "
            f"Realized={self.median_is_bps:.1f} bps {arrow} Model={self.model_slippage_bps:.1f} bps\n"
            f"  Maker={self.maker_is_bps:.1f} bps | Taker={self.taker_is_bps:.1f} bps | "
            f"Miss={self.median_miss_bps:.1f} bps | Blended TCA={self.blended_tca_bps:.1f} bps"
        )


def _median_bps(records: list[IsRecord]) -> float:
    if not records:
        return 0.0
    vals = sorted(r.shortfall_bps for r in records)
    return float(vals[len(vals) // 2])


def _fills_have_distinct_mid(records: list[IsRecord]) -> bool:
    return any(abs(r.mid_at_decision - r.fill_price) > _MID_EPS for r in records if r.filled)


def aggregate_journal(journal: IsJournal, *, model_slippage_bps: float) -> ISReport:
    """Агрегувати IS з IsJournal (pairs engine)."""
    fills = journal.filled_records()
    misses = journal.unfilled_records()
    makers, takers = journal.split_by_maker()
    n_fill = len(fills)
    n_miss = len(misses)
    total = n_fill + n_miss
    fill_rate = n_fill / total if total else 0.0
    fill_is = _median_bps(fills)
    miss_bps = _median_bps(misses)
    mid_distinct = _fills_have_distinct_mid(fills)
    return ISReport(
        n_fills=n_fill,
        n_unfilled=n_miss,
        median_is_bps=fill_is,
        maker_is_bps=_median_bps(makers),
        taker_is_bps=_median_bps(takers) if takers else 0.0,
        median_miss_bps=miss_bps,
        fill_rate=fill_rate,
        blended_tca_bps=blended_tca_bps(fill_rate=fill_rate, fill_is_bps=fill_is, miss_bps=miss_bps),
        model_slippage_bps=model_slippage_bps,
        coverage_ok=n_fill >= 20 and mid_distinct,
        mid_distinct=mid_distinct,
    )


def _col_float(row: pd.Series, name: str) -> float | None:
    if name not in row.index:
        return None
    raw = row.get(name)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    return val


def records_from_orders(orders: pd.DataFrame) -> list[IsRecord]:
    """Зібрати IsRecord з таблиці orders. Без колонки mid філи не інферяться з limit."""
    if orders is None or orders.empty:
        return []
    records: list[IsRecord] = []
    for _, row in orders.iterrows():
        status = str(row.get("status", "")).lower()
        side = str(row.get("side", "buy")).lower()
        reason = str(row.get("reason", "")).lower()
        is_maker = "taker" not in reason and "chase" not in reason
        limit_px = _col_float(row, "price") or 0.0
        eval_mid = _col_float(row, "mid")
        arrival = _col_float(row, "decision_mid")
        ts = row.get("ts")
        pair = str(row.get("pair", ""))
        symbol = str(row.get("symbol", ""))
        if status in _FILLED:
            if eval_mid is None:
                continue
            fill_px = limit_px
            if fill_px <= 0:
                continue
            records.append(
                IsRecord(
                    ts=ts,
                    pair=pair,
                    symbol=symbol,
                    side=side,
                    mid_at_decision=eval_mid,
                    fill_price=fill_px,
                    shortfall_bps=shortfall_bps(side, eval_mid, fill_px),
                    is_maker=is_maker,
                    filled=True,
                    arrival_mid=arrival,
                )
            )
        elif status in _UNFILLED:
            if arrival is None or eval_mid is None:
                continue
            records.append(
                IsRecord(
                    ts=ts,
                    pair=pair,
                    symbol=symbol,
                    side=side,
                    mid_at_decision=arrival,
                    fill_price=eval_mid,
                    shortfall_bps=shortfall_bps(side, arrival, eval_mid),
                    is_maker=is_maker,
                    filled=False,
                    arrival_mid=arrival,
                )
            )
    return records


def build_from_orders(orders: pd.DataFrame, *, model_slippage_bps: float) -> ISReport:
    """Оцінка IS з SQLite orders. mid обов'язковий; limit≠mid."""
    if orders is None or orders.empty:
        return aggregate_journal(IsJournal(), model_slippage_bps=model_slippage_bps)
    journal = IsJournal(records=records_from_orders(orders))
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
