"""Implementation Shortfall: mid у момент оцінки vs ціна філу (Narang гл. 7)."""

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
    filled: bool = True
    arrival_mid: float | None = None


def shortfall_bps(side: str, mid: float, fill: float) -> float:
    """Додатне = гірше за mid. Buy: (fill−mid)/mid; sell: (mid−fill)/mid."""
    if mid <= 0:
        return 0.0
    if side == "buy":
        return (fill - mid) / mid * 10_000.0
    return (mid - fill) / mid * 10_000.0


def blended_tca_bps(*, fill_rate: float, fill_is_bps: float, miss_bps: float) -> float:
    """fill_rate × IS + (1 − fill_rate) × opportunity cost unfilled."""
    fr = min(max(float(fill_rate), 0.0), 1.0)
    return fr * float(fill_is_bps) + (1.0 - fr) * float(miss_bps)


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
        *,
        filled: bool = True,
        arrival_mid: float | None = None,
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
            filled=filled,
            arrival_mid=arrival_mid,
        )
        self.records.append(rec)
        return rec

    def log_unfilled(
        self,
        ts: object,
        pair: str,
        symbol: str,
        side: str,
        arrival_mid: float,
        cancel_mid: float,
        is_maker: bool = True,
    ) -> IsRecord:
        """Opportunity cost: mid рішення → mid скасування/таймауту."""
        rec = IsRecord(
            ts=ts,
            pair=pair,
            symbol=symbol,
            side=side,
            mid_at_decision=arrival_mid,
            fill_price=cancel_mid,
            shortfall_bps=shortfall_bps(side, arrival_mid, cancel_mid),
            is_maker=is_maker,
            filled=False,
            arrival_mid=arrival_mid,
        )
        self.records.append(rec)
        return rec

    def filled_records(self) -> list[IsRecord]:
        return [r for r in self.records if r.filled]

    def unfilled_records(self) -> list[IsRecord]:
        return [r for r in self.records if not r.filled]

    def split_by_maker(self) -> tuple[list[IsRecord], list[IsRecord]]:
        fills = self.filled_records()
        makers = [r for r in fills if r.is_maker]
        chase = [r for r in fills if not r.is_maker]
        return makers, chase

    def percentile_bps(self, records: list[IsRecord] | None = None, q: float = 0.5) -> float:
        rows = self.filled_records() if records is None else records
        if not rows:
            return 0.0
        vals = sorted(r.shortfall_bps for r in rows)
        idx = min(max(int(round(q * (len(vals) - 1))), 0), len(vals) - 1)
        return float(vals[idx])

    def aggregate(self) -> dict[str, float]:
        makers, chase = self.split_by_maker()
        fills = self.filled_records()
        misses = self.unfilled_records()
        n_fill = float(len(fills))
        n_miss = float(len(misses))
        total = n_fill + n_miss
        fill_rate = n_fill / total if total else 0.0
        fill_is = self.percentile_bps(fills, 0.5)
        miss = self.percentile_bps(misses, 0.5)
        return {
            "n": n_fill,
            "n_unfilled": n_miss,
            "fill_rate": fill_rate,
            "mean": self.mean_bps(),
            "p50": fill_is,
            "p90": self.percentile_bps(fills, 0.9),
            "maker_p50": self.percentile_bps(makers, 0.5),
            "chase_p50": self.percentile_bps(chase, 0.5),
            "mean_markout": self.mean_markout_bps(),
            "miss_p50": miss,
            "blended_tca": blended_tca_bps(fill_rate=fill_rate, fill_is_bps=fill_is, miss_bps=miss),
        }

    def mean_markout_bps(self) -> float:
        vals = [r.markout_bps for r in self.filled_records() if r.markout_bps is not None]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def apply_next_bar_markout(self, symbol: str, next_mid: float) -> None:
        """Markout: mid наступного бара vs fill. Додатне = ціна пішла проти нас."""
        for rec in reversed(self.records):
            if rec.filled and rec.symbol == symbol and rec.markout_bps is None:
                rec.markout_bps = shortfall_bps(rec.side, rec.fill_price, next_mid)
                return

    def mean_bps(self) -> float:
        rows = self.filled_records()
        if not rows:
            return 0.0
        return sum(r.shortfall_bps for r in rows) / len(rows)

    def summary(self) -> str:
        fills = self.filled_records()
        if not fills and not self.records:
            return "IS: немає філів"
        agg = self.aggregate()
        return (
            f"IS: n={int(agg['n'])} unfilled={int(agg['n_unfilled'])} | "
            f"mean={agg['mean']:+.2f} bps | blended={agg['blended_tca']:+.2f} bps"
        )
