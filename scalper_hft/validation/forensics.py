"""Loss forensics: MAE/MFE і сегментація збиткових угод.

Ідея з win/pipeline/loss_forensics.py: відділити проблеми входу
(угода ніколи не була на боці) від проблем виходу (був MFE, потім віддали).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real

import pandas as pd

WEEKDAY_NAMES = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд")
DURATION_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("<6h", 0.0, 360.0),
    ("6-24h", 360.0, 1440.0),
    ("1-3d", 1440.0, 4320.0),
    ("3-7d", 4320.0, 10080.0),
    (">7d", 10080.0, None),
)


@dataclass(frozen=True, slots=True)
class SegmentStats:
    count: int
    wins: int
    losses: int
    total_profit: float
    avg_profit: float
    winrate: float
    pct_of_trades: float
    pct_of_loss: float | None
    segment_loss: float


@dataclass(frozen=True, slots=True)
class MfeMaeSummary:
    winners_mfe_median: float | None
    mfe_threshold: float | None
    losers_mfe_median: float | None
    losers_mfe_mean: float | None
    losers_mae_mean: float | None
    roundtrip_exits: int
    entry_failures: int
    losers_with_mfe: int


@dataclass(frozen=True, slots=True)
class ForensicsReport:
    n_closed: int
    n_winners: int
    n_losers: int
    winrate: float | None
    total_profit: float
    expectancy: float | None
    max_losing_streak: int
    max_winning_streak: int
    mfe_mae: MfeMaeSummary
    by_pair: dict[str, SegmentStats] = field(default_factory=dict)
    by_weekday: dict[str, SegmentStats] = field(default_factory=dict)
    by_duration: dict[str, SegmentStats] = field(default_factory=dict)
    by_fill: dict[str, SegmentStats] = field(default_factory=dict)
    by_exit: dict[str, SegmentStats] = field(default_factory=dict)

    def summary(self) -> str:
        wr = "—" if self.winrate is None else f"{self.winrate:.1%}"
        exp = "—" if self.expectancy is None else f"{self.expectancy:+.4f}"
        mm = self.mfe_mae
        return (
            f"Forensics: closed={self.n_closed} winrate={wr} expectancy={exp}\n"
            f"  streaks: lose={self.max_losing_streak} win={self.max_winning_streak}\n"
            f"  entry-fail={mm.entry_failures} roundtrip-exit={mm.roundtrip_exits} "
            f"(losers with MFE={mm.losers_with_mfe})"
        )


def compute_mfe_mae(
    entry_price: float,
    extreme_high: float,
    extreme_low: float,
    *,
    is_short: bool,
) -> tuple[float | None, float | None]:
    """Fee-free MFE/MAE як частка ціни входу (обидва ≥ 0).

    MFE — max favorable excursion; MAE — max adverse excursion.
    None, якщо entry_price невалідний.
    """
    if entry_price <= 0 or entry_price != entry_price:
        return None, None
    if is_short:
        mfe = (entry_price - extreme_low) / entry_price
        mae = (extreme_high - entry_price) / entry_price
    else:
        mfe = (extreme_high - entry_price) / entry_price
        mae = (entry_price - extreme_low) / entry_price
    return float(mfe), float(mae)


def is_short_side(side: object) -> bool:
    if isinstance(side, str):
        return side.lower() in {"short", "sell", "-1"}
    if isinstance(side, bool) or not isinstance(side, Real):
        return False
    return float(side) < 0


def _trade_return(row: pd.Series) -> float:
    if "ret" in row.index and pd.notna(row["ret"]):
        return float(row["ret"])
    pnl = row["pnl"] if "pnl" in row.index else None
    size = row["size"] if "size" in row.index else None
    entry = row["entry_price"] if "entry_price" in row.index else None
    if pnl is None or pd.isna(pnl):
        return 0.0
    if size is not None and entry is not None and pd.notna(size) and pd.notna(entry):
        notional = abs(float(size) * float(entry))
        if notional > 0:
            return float(pnl) / notional
    return float(pnl)


def excursions_from_bars(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Додає колонки mae/mfe з high/low вікна утримання."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    if "high" not in bars.columns or "low" not in bars.columns:
        return trades.copy()
    rows: list[dict[str, object]] = []
    for _, trade in trades.iterrows():
        rec = trade.to_dict()
        entry_ts = pd.Timestamp(trade["entry_ts"])
        exit_ts = pd.Timestamp(trade.get("exit_ts", trade["entry_ts"]))
        entry_px = (
            float(trade["entry_price"])
            if "entry_price" in trade.index and pd.notna(trade["entry_price"])
            else float("nan")
        )
        if not (entry_px > 0):
            rows.append(rec)
            continue
        window = bars[(bars.index >= entry_ts) & (bars.index <= exit_ts)]
        if window.empty:
            rows.append(rec)
            continue
        mfe, mae = compute_mfe_mae(
            entry_px,
            float(window["high"].max()),
            float(window["low"].min()),
            is_short=is_short_side(trade.get("side", 1)),
        )
        rec["mfe"] = mfe
        rec["mae"] = mae
        rows.append(rec)
    return pd.DataFrame(rows)


def _max_streak(profits: list[float], *, losing: bool) -> int:
    best = current = 0
    for profit in profits:
        is_match = profit <= 0 if losing else profit > 0
        current = current + 1 if is_match else 0
        best = max(best, current)
    return best


def _segment_stats(
    entries: list[tuple[str, float]],
    total_loss: float,
    total_trades: int,
) -> dict[str, SegmentStats]:
    grouped: dict[str, list[float]] = {}
    for key, profit in entries:
        grouped.setdefault(key, []).append(profit)
    out: dict[str, SegmentStats] = {}
    for key, profits in grouped.items():
        wins = sum(1 for p in profits if p > 0)
        losses = len(profits) - wins
        segment_loss = sum(p for p in profits if p < 0)
        n = len(profits)
        out[key] = SegmentStats(
            count=n,
            wins=wins,
            losses=losses,
            total_profit=round(sum(profits), 6),
            avg_profit=round(sum(profits) / n, 6) if n else 0.0,
            winrate=round(wins / n, 4) if n else 0.0,
            pct_of_trades=round(n / total_trades, 4) if total_trades else 0.0,
            pct_of_loss=round(segment_loss / total_loss, 4) if total_loss < 0 else None,
            segment_loss=round(segment_loss, 6),
        )
    return out


def _duration_minutes(row: pd.Series) -> float | None:
    if "trade_duration" in row.index and pd.notna(row["trade_duration"]):
        return float(row["trade_duration"])
    if "entry_ts" not in row.index or "exit_ts" not in row.index:
        return None
    if pd.isna(row["entry_ts"]) or pd.isna(row["exit_ts"]):
        return None
    delta = pd.Timestamp(row["exit_ts"]) - pd.Timestamp(row["entry_ts"])
    return float(delta.total_seconds() / 60.0)


def _duration_bucket(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    for bucket, low, high in DURATION_BUCKETS:
        if minutes >= low and (high is None or minutes < high):
            return bucket
    return "unknown"


def _weekday_label(row: pd.Series) -> str:
    ts_col = "entry_ts" if "entry_ts" in row.index and pd.notna(row.get("entry_ts")) else "ts"
    if ts_col not in row.index or pd.isna(row.get(ts_col)):
        return "unknown"
    weekday = int(pd.Timestamp(row[ts_col]).dayofweek)
    if 0 <= weekday <= 6:
        return WEEKDAY_NAMES[weekday]
    return "unknown"


def analyze_trades(trades: pd.DataFrame | None) -> ForensicsReport:
    """Повний forensics-звіт з таблиці закритих угод."""
    empty_mfe = MfeMaeSummary(None, None, None, None, None, 0, 0, 0)
    if trades is None or trades.empty:
        return ForensicsReport(0, 0, 0, None, 0.0, None, 0, 0, empty_mfe)

    closed = trades.copy()
    if "kind" in closed.columns:
        closed = closed[closed["kind"].fillna("trade").astype(str) != "funding"]
    profits = [_trade_return(row) for _, row in closed.iterrows()]
    n = len(profits)
    winners = [p for p in profits if p > 0]
    loser_profits = [p for p in profits if p <= 0]
    total_loss = sum(loser_profits)
    total_profit = sum(profits)

    mfe_vals: list[float] = []
    mae_vals: list[float] = []
    loser_mfe: list[float] = []
    winner_mfe: list[float] = []
    if "mfe" in closed.columns:
        for (_, row), profit in zip(closed.iterrows(), profits):
            mfe = row["mfe"]
            mae = row["mae"] if "mae" in closed.columns else None
            if mfe is None or (isinstance(mfe, float) and mfe != mfe):
                continue
            mfe_f = float(mfe)
            mfe_vals.append(mfe_f)
            if mae is not None and not (isinstance(mae, float) and mae != mae):
                mae_vals.append(float(mae))
            if profit > 0:
                winner_mfe.append(mfe_f)
            else:
                loser_mfe.append(mfe_f)

    winner_median = statistics.median(winner_mfe) if winner_mfe else None
    mfe_threshold = round(winner_median * 0.5, 6) if winner_median is not None else None
    roundtrip = sum(1 for m in loser_mfe if mfe_threshold is not None and m >= mfe_threshold)
    entry_fail = sum(1 for m in loser_mfe if winner_median is not None and m < winner_median * 0.25)
    mfe_mae = MfeMaeSummary(
        winners_mfe_median=round(winner_median, 6) if winner_median is not None else None,
        mfe_threshold=mfe_threshold,
        losers_mfe_median=round(statistics.median(loser_mfe), 6) if loser_mfe else None,
        losers_mfe_mean=round(statistics.mean(loser_mfe), 6) if loser_mfe else None,
        losers_mae_mean=round(statistics.mean(mae_vals), 6) if mae_vals else None,
        roundtrip_exits=roundtrip,
        entry_failures=entry_fail,
        losers_with_mfe=len(loser_mfe),
    )

    def _entries(key_fn: Callable[[pd.Series], str]) -> list[tuple[str, float]]:
        return [(key_fn(row), profit) for (_, row), profit in zip(closed.iterrows(), profits)]

    def _pair_key(row: pd.Series) -> str:
        for col in ("pair", "symbol"):
            if col in row.index and pd.notna(row[col]) and str(row[col]):
                return str(row[col])
        return "no_pair"

    def _fill_key(row: pd.Series) -> str:
        if "fill_status" in row.index and pd.notna(row["fill_status"]):
            return str(row["fill_status"])
        return "filled"

    def _exit_key(row: pd.Series) -> str:
        if "exit_reason" in row.index and pd.notna(row["exit_reason"]):
            return str(row["exit_reason"])
        return "no_exit"

    return ForensicsReport(
        n_closed=n,
        n_winners=len(winners),
        n_losers=len(loser_profits),
        winrate=round(len(winners) / n, 4) if n else None,
        total_profit=round(total_profit, 6),
        expectancy=round(total_profit / n, 6) if n else None,
        max_losing_streak=_max_streak(profits, losing=True),
        max_winning_streak=_max_streak(profits, losing=False),
        mfe_mae=mfe_mae,
        by_pair=_segment_stats(_entries(_pair_key), total_loss, n),
        by_weekday=_segment_stats(_entries(_weekday_label), total_loss, n),
        by_duration=_segment_stats(
            _entries(lambda r: _duration_bucket(_duration_minutes(r))),
            total_loss,
            n,
        ),
        by_fill=_segment_stats(_entries(_fill_key), total_loss, n),
        by_exit=_segment_stats(_entries(_exit_key), total_loss, n),
    )


def trades_from_paper_frames(
    trades: pd.DataFrame,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Нормалізує SQLite paper-таблиці до формату analyze_trades."""
    rows: list[dict[str, object]] = []
    if trades is not None and not trades.empty:
        for _, row in trades.iterrows():
            kind = str(row["kind"]) if "kind" in row.index and pd.notna(row.get("kind")) else "trade"
            if kind == "funding":
                continue
            ts = row["ts"] if "ts" in row.index else row.get("exit_ts")
            rec: dict[str, object] = {
                "entry_ts": row["entry_ts"] if "entry_ts" in row.index and pd.notna(row.get("entry_ts")) else ts,
                "exit_ts": ts,
                "side": row.get("side", ""),
                "pair": row.get("pair", ""),
                "symbol": row.get("symbol", ""),
                "size": row.get("size"),
                "entry_price": row.get("entry_price"),
                "exit_price": row.get("exit_price"),
                "pnl": row.get("pnl"),
                "kind": kind,
                "fill_status": "filled",
                "exit_reason": kind,
            }
            rec["ret"] = _trade_return(pd.Series(rec))
            rows.append(rec)
    if orders is not None and not orders.empty:
        unfilled = orders[orders["status"].astype(str) == "unfilled"]
        for _, row in unfilled.iterrows():
            rows.append(
                {
                    "entry_ts": row.get("ts"),
                    "exit_ts": row.get("ts"),
                    "side": row.get("side", ""),
                    "pair": row.get("pair", ""),
                    "symbol": row.get("symbol", ""),
                    "size": row.get("size"),
                    "entry_price": row.get("price"),
                    "exit_price": None,
                    "pnl": 0.0,
                    "ret": 0.0,
                    "kind": "unfilled",
                    "fill_status": "unfilled",
                    "exit_reason": str(row.get("reason") or "unfilled"),
                }
            )
    return pd.DataFrame(rows)


def report_to_mapping(report: ForensicsReport) -> Mapping[str, object]:
    """Серіалізація для CLI/JSON без dataclasses.asdict рекурсії сегментів."""

    def _seg(items: dict[str, SegmentStats]) -> dict[str, dict[str, float | int | None]]:
        return {
            k: {
                "count": v.count,
                "wins": v.wins,
                "losses": v.losses,
                "total_profit": v.total_profit,
                "avg_profit": v.avg_profit,
                "winrate": v.winrate,
                "pct_of_trades": v.pct_of_trades,
                "pct_of_loss": v.pct_of_loss,
                "segment_loss": v.segment_loss,
            }
            for k, v in items.items()
        }

    mm = report.mfe_mae
    return {
        "n_closed": report.n_closed,
        "n_winners": report.n_winners,
        "n_losers": report.n_losers,
        "winrate": report.winrate,
        "total_profit": report.total_profit,
        "expectancy": report.expectancy,
        "max_losing_streak": report.max_losing_streak,
        "max_winning_streak": report.max_winning_streak,
        "mfe_mae": {
            "winners_mfe_median": mm.winners_mfe_median,
            "mfe_threshold": mm.mfe_threshold,
            "losers_mfe_median": mm.losers_mfe_median,
            "losers_mfe_mean": mm.losers_mfe_mean,
            "losers_mae_mean": mm.losers_mae_mean,
            "roundtrip_exits": mm.roundtrip_exits,
            "entry_failures": mm.entry_failures,
            "losers_with_mfe": mm.losers_with_mfe,
        },
        "by_pair": _seg(report.by_pair),
        "by_weekday": _seg(report.by_weekday),
        "by_duration": _seg(report.by_duration),
        "by_fill": _seg(report.by_fill),
        "by_exit": _seg(report.by_exit),
        "summary": report.summary(),
    }
