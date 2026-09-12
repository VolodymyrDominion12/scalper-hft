"""Shadow-TCA: що коштував би live chase, поки paper лишається strict_both.

Paper Gate крутить maker all-or-none. Live `PairsLiveRunner` chase-ить другу
ногу taker-ом, якщо перша вже філилась. Цей модуль рахує контрфактичний
t-cost **без** зміни філів і PnL paper.

Один івент на pending-пару, лише на першому XOR-барі (live б chase/unwind
одразу). Не логувати both_filled / neither.
"""

from __future__ import annotations

from dataclasses import dataclass

from scalper_hft.live.fills import FillDecision, resolve_legging
from scalper_hft.live.is_log import shortfall_bps

LIVE_CHASE_TCA_BUDGET_BPS = 3.0


@dataclass(frozen=True, slots=True)
class ChaseShadowEvent:
    """Контрфактичний chase/unwind на сирих decide_fill (до both_or_neither)."""

    action: str
    drift_bps: float
    chased_symbol: str
    chased_side: str
    chase_fill_px: float
    maker_symbol: str
    extra_fee_bps: float
    chase_is_bps: float
    extra_cost_bps: float


@dataclass(frozen=True, slots=True)
class ChaseShadowReport:
    n_events: int
    n_chase: int
    n_unwind: int
    median_extra_cost_bps: float
    median_chase_is_bps: float
    median_drift_bps: float

    def summary(self) -> str:
        if self.n_events == 0:
            return "Chase shadow: немає XOR-ніг (live chase не відрізнявся б від paper)"
        return (
            f"Chase shadow: {self.n_chase} chase / {self.n_unwind} unwind "
            f"(n={self.n_events}) | extra={self.median_extra_cost_bps:.1f} bps "
            f"IS={self.median_chase_is_bps:.1f} bps drift={self.median_drift_bps:.1f} bps"
        )

    def live_tca_estimate_bps(self, paper_blended_tca_bps: float) -> float:
        """Консервативно: paper TCA + медіанний extra chase (на ногу, що доганяють)."""
        return float(paper_blended_tca_bps) + self.median_extra_cost_bps

    def would_breach_budget(
        self,
        paper_blended_tca_bps: float,
        *,
        budget_bps: float = LIVE_CHASE_TCA_BUDGET_BPS,
    ) -> bool:
        if self.n_events == 0:
            return False
        return self.live_tca_estimate_bps(paper_blended_tca_bps) > budget_bps


def extra_fee_bps(maker_fee: float, taker_fee: float) -> float:
    """Різниця taker−maker у bps (типово 3.0 на Binance USDT-M)."""
    return max(float(taker_fee) - float(maker_fee), 0.0) * 10_000.0


def evaluate_chase_shadow(
    d1: FillDecision,
    d2: FillDecision,
    *,
    side1: str,
    side2: str,
    limit1: float,
    limit2: float,
    mid1: float,
    mid2: float,
    symbol1: str,
    symbol2: str,
    max_drift_bps: float,
    maker_fee: float,
    taker_fee: float,
    arrival1: float,
    arrival2: float,
) -> ChaseShadowEvent | None:
    """Якщо рівно одна нога торкнулась — що зробив би live chase.

    None, якщо обидві або жодна: paper і live збігаються.
    """
    if d1.filled == d2.filled:
        return None
    res = resolve_legging(
        d1,
        d2,
        side1,
        side2,
        limit1,
        limit2,
        mid1,
        mid2,
        max_drift_bps=max_drift_bps,
        mode="chase",
    )
    fee_bps = extra_fee_bps(maker_fee, taker_fee)
    if res.action == "chase_leg2":
        is_bps = shortfall_bps(side2, arrival2 if arrival2 > 0 else limit2, mid2)
        return ChaseShadowEvent(
            action=res.action,
            drift_bps=res.drift_bps,
            chased_symbol=symbol2,
            chased_side=side2,
            chase_fill_px=float(mid2),
            maker_symbol=symbol1,
            extra_fee_bps=fee_bps,
            chase_is_bps=is_bps,
            extra_cost_bps=fee_bps + is_bps,
        )
    if res.action == "chase_leg1":
        is_bps = shortfall_bps(side1, arrival1 if arrival1 > 0 else limit1, mid1)
        return ChaseShadowEvent(
            action=res.action,
            drift_bps=res.drift_bps,
            chased_symbol=symbol1,
            chased_side=side1,
            chase_fill_px=float(mid1),
            maker_symbol=symbol2,
            extra_fee_bps=fee_bps,
            chase_is_bps=is_bps,
            extra_cost_bps=fee_bps + is_bps,
        )
    if res.action == "unwind_leg1":
        is_bps = shortfall_bps(side1, arrival1 if arrival1 > 0 else limit1, mid1)
        flatten_fee = (float(maker_fee) + float(taker_fee)) * 10_000.0
        return ChaseShadowEvent(
            action=res.action,
            drift_bps=res.drift_bps,
            chased_symbol=symbol2,
            chased_side=side2,
            chase_fill_px=float(mid1),
            maker_symbol=symbol1,
            extra_fee_bps=flatten_fee,
            chase_is_bps=is_bps,
            extra_cost_bps=flatten_fee + is_bps,
        )
    if res.action == "unwind_leg2":
        is_bps = shortfall_bps(side2, arrival2 if arrival2 > 0 else limit2, mid2)
        flatten_fee = (float(maker_fee) + float(taker_fee)) * 10_000.0
        return ChaseShadowEvent(
            action=res.action,
            drift_bps=res.drift_bps,
            chased_symbol=symbol1,
            chased_side=side1,
            chase_fill_px=float(mid2),
            maker_symbol=symbol2,
            extra_fee_bps=flatten_fee,
            chase_is_bps=is_bps,
            extra_cost_bps=flatten_fee + is_bps,
        )
    return None


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    return float(ordered[len(ordered) // 2])


def aggregate_shadow_events(events: list[ChaseShadowEvent]) -> ChaseShadowReport:
    chase = [e for e in events if e.action.startswith("chase_")]
    unwind = [e for e in events if e.action.startswith("unwind_")]
    return ChaseShadowReport(
        n_events=len(events),
        n_chase=len(chase),
        n_unwind=len(unwind),
        median_extra_cost_bps=_median([e.extra_cost_bps for e in events]),
        median_chase_is_bps=_median([e.chase_is_bps for e in events]),
        median_drift_bps=_median([e.drift_bps for e in events]),
    )


def events_from_frame(frame: object) -> list[ChaseShadowEvent]:
    """Зібрати івенти з таблиці shadow_legging (pandas DataFrame)."""
    import pandas as pd

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    out: list[ChaseShadowEvent] = []
    for _, row in frame.iterrows():
        out.append(
            ChaseShadowEvent(
                action=str(row.get("action", "")),
                drift_bps=float(row.get("drift_bps") or 0.0),
                chased_symbol=str(row.get("chased_symbol", "")),
                chased_side=str(row.get("chased_side", "")),
                chase_fill_px=float(row.get("chase_fill_px") or 0.0),
                maker_symbol=str(row.get("maker_symbol", "")),
                extra_fee_bps=float(row.get("extra_fee_bps") or 0.0),
                chase_is_bps=float(row.get("chase_is_bps") or 0.0),
                extra_cost_bps=float(row.get("extra_cost_bps") or 0.0),
            )
        )
    return out


def build_from_store_frame(frame: object) -> ChaseShadowReport:
    return aggregate_shadow_events(events_from_frame(frame))


def format_live_chase_gap(
    report: ChaseShadowReport,
    *,
    paper_blended_tca_bps: float | None,
    budget_bps: float = LIVE_CHASE_TCA_BUDGET_BPS,
) -> str:
    """Рядок для paper-audit / is-report. Не валить Paper Gate (paper лишається maker)."""
    base = report.summary()
    if report.n_events == 0 or paper_blended_tca_bps is None:
        return base
    live_est = report.live_tca_estimate_bps(paper_blended_tca_bps)
    if report.would_breach_budget(paper_blended_tca_bps, budget_bps=budget_bps):
        return (
            f"{base}\n"
            f"  LIVE_CHASE_HOLD: paper TCA {paper_blended_tca_bps:.1f} + extra "
            f"{report.median_extra_cost_bps:.1f} = {live_est:.1f} bps > {budget_bps:.1f} "
            f"(не вмикати live chase без окремого PASS)"
        )
    return f"{base}\n  live TCA estimate={live_est:.1f} bps (budget {budget_bps:.1f}) — ok для soak після Gate"
