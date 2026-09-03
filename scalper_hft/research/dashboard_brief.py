"""Чисті хелпери дослідницького брифу дашборду (без Streamlit).

Свіжість кешу, paper KPI, hurdle комісій, підсвітка sweep — щоб сторінка
«Моніторинг» відповідала на питання дослідника, а не лише рахувала рядки.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

Freshness = Literal["fresh", "aging", "stale", "missing", "future"]

_SPOT_MARK = "spot"


@dataclass(frozen=True, slots=True)
class ParquetPeek:
    """Мінімальний зріз parquet: кількість рядків і часовий span індексу."""

    n_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class PaperKpis:
    """Агрегат paper-журналу: fill-rate і остання equity по парах."""

    filled: int
    unfilled: int
    pending: int
    fill_rate: float
    last_equity_by_pair: dict[str, float]
    sparkline: list[float]
    last_ts: pd.Timestamp | None
    n_equity_points: int


def peek_parquet(path: Path) -> ParquetPeek:
    """Row count і min/max timestamps без завантаження OHLCV-колонок."""
    if not path.exists():
        return ParquetPeek(0, None, None)
    try:
        pf = pq.ParquetFile(path)
    except Exception:  # noqa: BLE001 — битий файл = порожній зріз
        return ParquetPeek(0, None, None)
    n = int(pf.metadata.num_rows)
    if n <= 0:
        return ParquetPeek(0, None, None)
    names = list(pf.schema_arrow.names)
    ts_name: str | None = None
    for candidate in ("__index_level_0__", "timestamp", "ts", "time"):
        if candidate in names:
            ts_name = candidate
            break
    start, end = _span_from_statistics(pf, names, ts_name)
    if start is None and ts_name is not None and n <= 2_000_000:
        start, end = _span_from_column(path, ts_name)
    return ParquetPeek(n, start, end)


def _span_from_statistics(
    pf: pq.ParquetFile,
    names: list[str],
    ts_name: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if ts_name is None:
        return None, None
    col_idx = names.index(ts_name)
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(col_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            continue
        try:
            mins.append(pd.Timestamp(stats.min))
            maxs.append(pd.Timestamp(stats.max))
        except (ValueError, TypeError, OverflowError):
            return None, None
    if not mins:
        return None, None
    return min(mins), max(maxs)


def _span_from_column(path: Path, ts_name: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    try:
        series = pd.read_parquet(path, columns=[ts_name]).iloc[:, 0]
    except Exception:  # noqa: BLE001
        try:
            idx = pd.read_parquet(path, columns=[]).index
        except Exception:  # noqa: BLE001
            return None, None
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
            return None, None
        return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if ts.empty:
        return None, None
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def klines_interval_from_name(symbol: str, path: Path) -> str | None:
    """Інтервал з `{symbol}_{interval}_klines.parquet`; spot-файли ігноруємо."""
    name = path.name
    prefix = f"{symbol}_"
    suffix = "_klines.parquet"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    mid = name[len(prefix) : -len(suffix)]
    if not mid or _SPOT_MARK in mid:
        return None
    return mid


def classify_staleness(
    end: pd.Timestamp | None,
    now: pd.Timestamp,
    *,
    fresh_hours: float = 2.0,
    stale_hours: float = 24.0,
) -> Freshness:
    """Свіжий / старіючий / застарілий відносно `now` (naive UTC)."""
    if end is None or pd.isna(end):
        return "missing"
    age_h = (now - pd.Timestamp(end)).total_seconds() / 3600.0
    if age_h < 0:
        return "future"
    if age_h <= fresh_hours:
        return "fresh"
    if age_h <= stale_hours:
        return "aging"
    return "stale"


def age_hours(end: pd.Timestamp | None, now: pd.Timestamp) -> float:
    """Вік останнього бара в годинах; NaN якщо timestamps немає."""
    if end is None or pd.isna(end):
        return float("nan")
    return (now - pd.Timestamp(end)).total_seconds() / 3600.0


def format_age_hours(hours: float) -> str:
    """Короткий підпис віку для таблиці."""
    if hours != hours:  # NaN
        return "—"
    if hours < 0:
        return "майбутнє"
    if hours < 1:
        return f"{hours * 60:.0f} хв"
    if hours < 48:
        return f"{hours:.1f} год"
    return f"{hours / 24:.1f} дн"


def cache_inventory(
    data_dir: Path,
    symbols: Sequence[str],
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Один рядок на символ: 1m span, інші інтервали, funding/trades/bookTicker."""
    clock = now if now is not None else pd.Timestamp.now(tz="UTC").tz_convert(None)
    rows: list[dict[str, object]] = []
    for sym in symbols:
        k1m = peek_parquet(data_dir / f"{sym}_1m_klines.parquet")
        intervals = _discover_intervals(data_dir, sym)
        funding = peek_parquet(data_dir / f"{sym}_funding.parquet")
        trades = peek_parquet(data_dir / f"{sym}_aggTrades.parquet")
        book = peek_parquet(data_dir / f"{sym}_bookTicker.parquet")
        age = age_hours(k1m.end, clock)
        span_days = float("nan")
        if k1m.start is not None and k1m.end is not None:
            span_days = (k1m.end - k1m.start).total_seconds() / 86400.0
        rows.append(
            {
                "symbol": sym,
                "klines_1m": k1m.n_rows,
                "last_1m": k1m.end,
                "span_days": span_days,
                "age_hours": age,
                "age_label": format_age_hours(age),
                "freshness": classify_staleness(k1m.end, clock),
                "intervals": ",".join(intervals) if intervals else "—",
                "n_intervals": len(intervals),
                "funding": funding.n_rows,
                "funding_last": funding.end,
                "agg_trades": trades.n_rows,
                "book_ticker": book.n_rows,
            }
        )
    return pd.DataFrame(rows)


def _discover_intervals(data_dir: Path, symbol: str) -> list[str]:
    found: list[str] = []
    if not data_dir.exists():
        return found
    for path in sorted(data_dir.glob(f"{symbol}_*_klines.parquet")):
        interval = klines_interval_from_name(symbol, path)
        if interval is not None:
            found.append(interval)
    return found


def inventory_kpis(inv: pd.DataFrame) -> dict[str, int]:
    """Короткі лічильники для KPI-рядка."""
    if inv.empty:
        return {"symbols": 0, "with_1m": 0, "stale": 0, "missing": 0, "klines_1m": 0}
    fresh_col = inv["freshness"]
    return {
        "symbols": int(len(inv)),
        "with_1m": int((inv["klines_1m"] > 0).sum()),
        "stale": int((fresh_col == "stale").sum()),
        "missing": int((fresh_col == "missing").sum()),
        "klines_1m": int(inv["klines_1m"].sum()),
    }


def compute_paper_kpis(stats: dict[str, int], equity: pd.DataFrame) -> PaperKpis:
    """Fill-rate з ордерів і остання equity по парах."""
    filled = int(stats.get("filled", 0))
    unfilled = int(stats.get("unfilled", 0))
    pending = int(stats.get("pending", 0))
    denom = filled + unfilled
    fill_rate = filled / denom if denom else float("nan")
    last_by_pair: dict[str, float] = {}
    spark: list[float] = []
    last_ts: pd.Timestamp | None = None
    n_points = 0
    if equity is not None and not equity.empty and {"pair", "equity"} <= set(equity.columns):
        eq = equity.copy()
        if "ts" in eq.columns:
            eq["ts"] = pd.to_datetime(eq["ts"], utc=True, errors="coerce")
            eq["ts"] = eq["ts"].dt.tz_localize(None)
            eq = eq.dropna(subset=["ts"]).sort_values("ts")
            if not eq.empty:
                last_ts = pd.Timestamp(eq["ts"].iloc[-1])
        n_points = len(eq)
        for pair, group in eq.groupby("pair", sort=True):
            last_by_pair[str(pair)] = float(group["equity"].iloc[-1])
        # sparkline: сума equity по timestamp (портфель), інакше одна колонка
        if "ts" in eq.columns and not eq.empty:
            port = eq.groupby("ts", sort=True)["equity"].sum()
            spark = [float(v) for v in port.tail(48).tolist()]
        else:
            spark = [float(v) for v in eq["equity"].tail(48).tolist()]
    return PaperKpis(
        filled=filled,
        unfilled=unfilled,
        pending=pending,
        fill_rate=fill_rate,
        last_equity_by_pair=last_by_pair,
        sparkline=spark,
        last_ts=last_ts,
        n_equity_points=n_points,
    )


def paper_pair_table(equity: pd.DataFrame) -> pd.DataFrame:
    """Одна пара — рядок: start/last equity, дохідність, sparkline."""
    if equity is None or equity.empty or "pair" not in equity.columns:
        return pd.DataFrame(columns=["pair", "start", "last", "return", "spark", "n_points"])
    eq = equity.copy()
    if "ts" in eq.columns:
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True, errors="coerce")
        eq = eq.dropna(subset=["ts"]).sort_values("ts")
    rows: list[dict[str, object]] = []
    for pair, group in eq.groupby("pair", sort=True):
        start = float(group["equity"].iloc[0])
        last = float(group["equity"].iloc[-1])
        ret = last / start - 1.0 if start else float("nan")
        spark = [float(v) for v in group["equity"].tolist()[-40:]]
        rows.append(
            {
                "pair": str(pair),
                "start": start,
                "last": last,
                "return": ret,
                "spark": spark,
                "n_points": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def hurdle_note(avg_trade_net: float, round_trip: float, n_legs: int = 1) -> str:
    """Порівняння середньої угоди (вже після комісій) з maker round-trip hurdle."""
    hurdle = round_trip * n_legs
    if avg_trade_net != avg_trade_net:  # NaN
        return f"Hurdle maker round-trip × {n_legs} ніг: {hurdle:.2%}."
    net_bps = avg_trade_net * 10_000
    hurdle_bps = hurdle * 10_000
    if avg_trade_net > 0:
        verdict = "нетто додатне — edge покриває тертя на цій вибірці"
    elif avg_trade_net > -hurdle:
        verdict = "нетто біля нуля — edge слабкий відносно комісій"
    else:
        verdict = "fee-drag: середня угода не покриває round-trip"
    return (
        f"Hurdle (maker RT × {n_legs} ніг): {hurdle_bps:.1f} bps. "
        f"Середня угода вже після комісій: {net_bps:+.1f} bps — {verdict}."
    )


def dsr_verdict(dsr: float) -> Literal["significant", "weak", "none"]:
    """Bailey–López de Prado: >0.95 вважаємо значущим edge."""
    if dsr != dsr:  # NaN
        return "none"
    if dsr > 0.95:
        return "significant"
    if dsr > 0.50:
        return "weak"
    return "none"


def sweep_highlights(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Топ комбінацій за OOS Sharpe (fallback — in-sample Sharpe)."""
    if df is None or df.empty:
        return pd.DataFrame()
    view = df
    if "status" in view.columns:
        view = view[view["status"].fillna("ok") == "ok"]
    if view.empty:
        return pd.DataFrame()
    sort_col = "avg_oos_sharpe" if "avg_oos_sharpe" in view.columns else "sharpe"
    if sort_col not in view.columns:
        return pd.DataFrame()
    ranked = view.dropna(subset=[sort_col]).nlargest(int(top_n), sort_col)
    cols = [
        c
        for c in (
            "strategy",
            "symbol",
            "interval",
            "mode",
            "n_trades",
            "sharpe",
            "avg_oos_sharpe",
            "oos_positive_frac",
            "max_dd",
            "win_rate",
        )
        if c in ranked.columns
    ]
    return ranked[cols].reset_index(drop=True)


def fill_rate(filled: int, unfilled: int) -> float:
    """Частка filled серед завершених (filled+unfilled); NaN якщо ордерів немає."""
    denom = filled + unfilled
    if denom <= 0:
        return float("nan")
    return filled / denom
