"""Implementation Shortfall: mid у момент рішення vs ціна філу (Narang гл. 7)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsRecord:
    ts: object
    pair: str
    symbol: str
    side: str  # buy | sell
    mid_at_decision: float
    fill_price: float
    shortfall_bps: float


def shortfall_bps(side: str, mid: float, fill: float) -> float:
    """Додатне = гірше за mid. Buy: (fill−mid)/mid; sell: (mid−fill)/mid."""
    if mid <= 0:
        return 0.0
    if side == "buy":
        return (fill - mid) / mid * 10_000.0
    return (mid - fill) / mid * 10_000.0


@dataclass
class IsJournal:
    records: list[IsRecord] = field(default_factory=list)

    def log(
        self,
        ts: object,
        pair: str,
        symbol: str,
        side: str,
        mid_at_decision: float,
        fill_price: float,
    ) -> IsRecord:
        rec = IsRecord(
            ts=ts,
            pair=pair,
            symbol=symbol,
            side=side,
            mid_at_decision=mid_at_decision,
            fill_price=fill_price,
            shortfall_bps=shortfall_bps(side, mid_at_decision, fill_price),
        )
        self.records.append(rec)
        return rec

    def mean_bps(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.shortfall_bps for r in self.records) / len(self.records)

    def summary(self) -> str:
        if not self.records:
            return "IS: немає філів"
        return (
            f"IS: n={len(self.records)} | mean={self.mean_bps():+.2f} bps "
            f"| max={max(r.shortfall_bps for r in self.records):+.2f} bps"
        )
