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
