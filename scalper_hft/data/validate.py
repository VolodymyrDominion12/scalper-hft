"""Валідація OHLCV-барів (UTC, OHLC-інваріанти, дірки, дублікати)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BarQualityReport:
    n_rows: int
    n_duplicates: int
    n_ohlc_violations: int
    n_gaps: int
    n_future: int
    monotonic: bool
    ok: bool
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "ok" if self.ok else "FAIL"
        return (
            f"bars {status}: n={self.n_rows} dup={self.n_duplicates} "
            f"ohlc={self.n_ohlc_violations} gaps={self.n_gaps} future={self.n_future}"
        )


def validate_bars(
    df: pd.DataFrame,
    *,
    interval: str | None = None,
    now: pd.Timestamp | None = None,
) -> BarQualityReport:
    """Перевіряє high/low, монотонний індекс, дублікати, майбутні мітки."""
    issues: list[str] = []
    if df is None or df.empty:
        return BarQualityReport(0, 0, 0, 0, 0, True, False, ["порожній датасет"])

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        issues.append("індекс не DatetimeIndex")
        monotonic = False
    else:
        monotonic = bool(idx.is_monotonic_increasing)
        if not monotonic:
            issues.append("індекс не монотонний")

    n_dup = int(idx.duplicated().sum()) if len(idx) else 0
    if n_dup:
        issues.append(f"дублікати: {n_dup}")

    n_ohlc = 0
    if {"open", "high", "low", "close"}.issubset(df.columns):
        hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
        bad = (hi < lo) | (hi < op) | (hi < cl) | (lo > op) | (lo > cl)
        n_ohlc = int(bad.sum())
        if n_ohlc:
            issues.append(f"OHLC-порушення: {n_ohlc}")

    n_future = 0
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        last = idx[-1]
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_convert("UTC").tz_localize(None)
        if last > now_ts:
            n_future = int((idx > now_ts).sum())
            issues.append(f"майбутні бари: {n_future}")

    n_gaps = 0
    if interval and isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        from scalper_hft.data.downloader import _interval_ms

        expected = pd.Timedelta(milliseconds=_interval_ms(interval))
        deltas = idx.to_series().diff().iloc[1:]
        n_gaps = int((deltas > expected * 1.5).sum())
        if n_gaps:
            issues.append(f"дірки: {n_gaps}")

    ok = not issues
    return BarQualityReport(len(df), n_dup, n_ohlc, n_gaps, n_future, monotonic, ok, issues)


@dataclass
class StreamQualityReport:
    """Якість потоку trades / funding / depth / bookTicker (Phase 5.4)."""

    kind: str
    n_rows: int
    n_duplicates: int
    n_invalid: int
    n_gaps: int
    n_future: int
    monotonic: bool
    ok: bool
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "ok" if self.ok else "FAIL"
        return (
            f"{self.kind} {status}: n={self.n_rows} dup={self.n_duplicates} "
            f"invalid={self.n_invalid} gaps={self.n_gaps} future={self.n_future}"
        )


_FUNDING_MAX_ABS = 0.05  # 5% / 8h — вище за будь-який нормальний Binance USDT-M
_DEPTH_BID_PX = [f"bid{i}" for i in range(1, 6)]
_DEPTH_ASK_PX = [f"ask{i}" for i in range(1, 6)]
_DEPTH_BID_QTY = [f"bid{i}_qty" for i in range(1, 6)]
_DEPTH_ASK_QTY = [f"ask{i}_qty" for i in range(1, 6)]


def _empty_stream(kind: str) -> StreamQualityReport:
    return StreamQualityReport(kind, 0, 0, 0, 0, 0, True, False, ["порожній датасет"])


def _now_naive(now: pd.Timestamp | None) -> pd.Timestamp:
    if now is not None:
        ts = now
    else:
        ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return ts
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _index_quality(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> tuple[bool, int, int, list[str]]:
    """Монотонність, дублікати, майбутні мітки. Не змінює validate_bars API."""
    issues: list[str] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        issues.append("індекс не DatetimeIndex")
        return False, 0, 0, issues
    monotonic = bool(idx.is_monotonic_increasing)
    if not monotonic:
        issues.append("індекс не монотонний")
    n_dup = int(idx.duplicated().sum()) if len(idx) else 0
    if n_dup:
        issues.append(f"дублікати: {n_dup}")
    n_future = 0
    now_ts = _now_naive(now)
    if len(idx):
        last = idx[-1]
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_convert("UTC").tz_localize(None)
        if last > now_ts:
            comparable = idx.tz_localize(None) if idx.tz is not None else idx
            n_future = int((comparable > now_ts).sum())
            issues.append(f"майбутні мітки: {n_future}")
    return monotonic, n_dup, n_future, issues


def _gap_count(idx: pd.DatetimeIndex, expected: pd.Timedelta) -> int:
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return 0
    deltas = idx.to_series().diff().iloc[1:]
    return int((deltas > expected * 1.5).sum())


def validate_trades(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> StreamQualityReport:
    """aggTrades: price/amount > 0, side ∈ {buy,sell}, монотонний час."""
    if df is None or df.empty:
        return _empty_stream("trades")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    n_invalid = 0
    price_col = "price" if "price" in df.columns else None
    amt_col = "amount" if "amount" in df.columns else ("qty" if "qty" in df.columns else None)
    if price_col is None or amt_col is None:
        issues.append("немає колонок price/amount")
        n_invalid = len(df)
    else:
        px = pd.to_numeric(df[price_col], errors="coerce")
        amt = pd.to_numeric(df[amt_col], errors="coerce")
        bad = px.isna() | amt.isna() | (px <= 0) | (amt <= 0)
        n_invalid += int(bad.sum())
        if n_invalid:
            issues.append(f"price/amount ≤0 або NaN: {n_invalid}")
    if "side" in df.columns:
        side_norm = df["side"].astype(str).str.lower()
        bad_side = ~side_norm.isin({"buy", "sell", "1", "-1"})
        n_side = int(bad_side.sum())
        if n_side:
            n_invalid += n_side
            issues.append(f"невідомий side: {n_side}")
    else:
        issues.append("немає колонки side")
        n_invalid += len(df)
    if "trade_id" in df.columns:
        n_id_dup = int(df["trade_id"].duplicated().sum())
        if n_id_dup:
            issues.append(f"дублікати trade_id: {n_id_dup}")
            n_invalid += n_id_dup
    ok = not issues
    return StreamQualityReport("trades", len(df), n_dup, n_invalid, 0, n_future, monotonic, ok, issues)


def validate_funding(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    interval: str = "8h",
) -> StreamQualityReport:
    """Funding: колонка fundingRate, |rate| не божевільний, крок ~8h."""
    if df is None or df.empty:
        return _empty_stream("funding")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    n_invalid = 0
    if "fundingRate" not in df.columns:
        issues.append("немає колонки fundingRate")
        n_invalid = len(df)
    else:
        rate = pd.to_numeric(df["fundingRate"], errors="coerce")
        n_nan = int(rate.isna().sum())
        n_crazy = int((rate.abs() > _FUNDING_MAX_ABS).sum())
        n_invalid = n_nan + n_crazy
        if n_nan:
            issues.append(f"NaN fundingRate: {n_nan}")
        if n_crazy:
            issues.append(f"|fundingRate| > {_FUNDING_MAX_ABS}: {n_crazy}")
    n_gaps = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        n_gaps = _gap_count(df.index, pd.Timedelta(interval))
        if n_gaps:
            issues.append(f"дірки (>1.5×{interval}): {n_gaps}")
    ok = not issues
    return StreamQualityReport("funding", len(df), n_dup, n_invalid, n_gaps, n_future, monotonic, ok, issues)


def validate_bookticker(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> StreamQualityReport:
    """L1 bookTicker: bid/ask > 0, ask ≥ bid, qty ≥ 0."""
    if df is None or df.empty:
        return _empty_stream("book")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    needed = {"bid", "ask", "bid_qty", "ask_qty"}
    n_invalid = 0
    if not needed.issubset(df.columns):
        issues.append(f"немає колонок {sorted(needed - set(df.columns))}")
        n_invalid = len(df)
    else:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        bq = pd.to_numeric(df["bid_qty"], errors="coerce")
        aq = pd.to_numeric(df["ask_qty"], errors="coerce")
        bad = (
            bid.isna()
            | ask.isna()
            | (bid <= 0)
            | (ask <= 0)
            | (ask < bid)
            | bq.isna()
            | aq.isna()
            | (bq < 0)
            | (aq < 0)
        )
        n_invalid = int(bad.sum())
        if n_invalid:
            issues.append(f"перехрещений/невалідний L1: {n_invalid}")
    ok = not issues
    return StreamQualityReport("book", len(df), n_dup, n_invalid, 0, n_future, monotonic, ok, issues)


def validate_depth(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    max_gap_sec: float = 1.0,
) -> StreamQualityReport:
    """depth5: px>0, qty≥0, ask1≥bid1, рівні впорядковані, гепи ts."""
    if df is None or df.empty:
        return _empty_stream("depth")
    monotonic, n_dup, n_future, issues = _index_quality(df, now=now)
    n_invalid = 0
    has_l1 = "bid1" in df.columns and "ask1" in df.columns
    if not has_l1:
        issues.append("немає bid1/ask1")
        n_invalid = len(df)
    else:
        bid1 = pd.to_numeric(df["bid1"], errors="coerce")
        ask1 = pd.to_numeric(df["ask1"], errors="coerce")
        crossed = bid1.isna() | ask1.isna() | (bid1 <= 0) | (ask1 <= 0) | (ask1 < bid1)
        n_cross = int(crossed.sum())
        if n_cross:
            n_invalid += n_cross
            issues.append(f"перехрещений/невалідний top-of-book: {n_cross}")
        # qty
        for col in _DEPTH_BID_QTY + _DEPTH_ASK_QTY:
            if col not in df.columns:
                continue
            q = pd.to_numeric(df[col], errors="coerce")
            n_bad_q = int((q.isna() | (q < 0)).sum())
            if n_bad_q:
                n_invalid += n_bad_q
                issues.append(f"{col} < 0 або NaN: {n_bad_q}")
        # рівні: bid спадний, ask зростаючий (де обидва валідні)
        for i in range(1, 5):
            b_lo, b_hi = f"bid{i}", f"bid{i + 1}"
            a_lo, a_hi = f"ask{i}", f"ask{i + 1}"
            if b_lo in df.columns and b_hi in df.columns:
                blo = pd.to_numeric(df[b_lo], errors="coerce")
                bhi = pd.to_numeric(df[b_hi], errors="coerce")
                n_ord = int(((bhi > blo) & blo.notna() & bhi.notna() & (bhi > 0)).sum())
                if n_ord:
                    n_invalid += n_ord
                    issues.append(f"bid рівні не спадні ({b_lo}<{b_hi}): {n_ord}")
            if a_lo in df.columns and a_hi in df.columns:
                alo = pd.to_numeric(df[a_lo], errors="coerce")
                ahi = pd.to_numeric(df[a_hi], errors="coerce")
                n_ord = int(((ahi < alo) & alo.notna() & ahi.notna() & (ahi > 0)).sum())
                if n_ord:
                    n_invalid += n_ord
                    issues.append(f"ask рівні не зростаючі ({a_lo}>{a_hi}): {n_ord}")
    n_gaps = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        n_gaps = _gap_count(df.index, pd.Timedelta(seconds=max_gap_sec))
        if n_gaps:
            issues.append(f"дірки ts (>{max_gap_sec}s×1.5): {n_gaps}")
    ok = not issues
    return StreamQualityReport("depth", len(df), n_dup, n_invalid, n_gaps, n_future, monotonic, ok, issues)


def stream_is_critical(report: StreamQualityReport) -> bool:
    """Критичні дефекти для fail-closed запису кешу (як у save_klines)."""
    if report.n_rows == 0:
        return False  # порожній checkpoint downloader-а не блокує
    return (not report.monotonic) or bool(report.n_duplicates) or bool(report.n_invalid) or bool(report.n_future)
