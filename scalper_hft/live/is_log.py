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
    is_maker: bool = True
    markout_bps: float | None = None


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
        is_maker: bool = True,
    ) -> IsRecord:
        rec = IsRecord(
            ts=ts,
            pair=pair,
            symbol=symbol,
            side=side,
            mid_at_decision=mid_at_decision,
            fill_price=fill_price,
            shortfall_bps=shortfall_bps(side, mid_at_decision, fill_price),
            is_maker=is_maker,
        )
        self.records.append(rec)
        return rec

    def split_by_maker(self) -> tuple[list[IsRecord], list[IsRecord]]:
        makers = [r for r in self.records if r.is_maker]
        chase = [r for r in self.records if not r.is_maker]
        return makers, chase

    def percentile_bps(self, records: list[IsRecord] | None = None, q: float = 0.5) -> float:
        rows = self.records if records is None else records
        if not rows:
            return 0.0
        vals = sorted(r.shortfall_bps for r in rows)
        idx = min(max(int(round(q * (len(vals) - 1))), 0), len(vals) - 1)
        return float(vals[idx])

    def aggregate(self) -> dict[str, float]:
        makers, chase = self.split_by_maker()
        return {
            "n": float(len(self.records)),
            "mean": self.mean_bps(),
            "p50": self.percentile_bps(q=0.5),
            "p90": self.percentile_bps(q=0.9),
            "maker_p50": self.percentile_bps(makers, 0.5),
            "chase_p50": self.percentile_bps(chase, 0.5),
            "mean_markout": self.mean_markout_bps(),
        }

    def mean_markout_bps(self) -> float:
        vals = [r.markout_bps for r in self.records if r.markout_bps is not None]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def apply_next_bar_markout(self, symbol: str, next_mid: float) -> None:
        """Markout: mid наступного бара vs fill. Додатне = ціна пішла проти нас."""
        for rec in reversed(self.records):
            if rec.symbol == symbol and rec.markout_bps is None:
                rec.markout_bps = shortfall_bps(rec.side, rec.fill_price, next_mid)
                return

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
