"""Bar/time helpers для live-трейдера (no-lookahead closed klines)."""

from __future__ import annotations

import pandas as pd


def as_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def interval_seconds(interval: str) -> float:
    unit = interval[-1]
    num = int(interval[:-1])
    per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return float(num * per_unit)


def closed_klines(
    df: pd.DataFrame,
    interval: str | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Відкидає формуючий бар (останній рядок REST), щоб не було lookahead.

    Binance OHLCV включає поточну незакриту свічку як iloc[-1]. Сигнал і філл
    беруться лише з барів, чий close-час уже настав. Історичні ряди (останній
    бар у минулому) не змінюються.
    """
    if df is None or df.empty:
        return df
    from scalper_hft.data.downloader import _interval_ms, _utc_now

    interval = interval or "1m"
    now_ts = as_naive_utc(now if now is not None else _utc_now())
    last = as_naive_utc(df.index[-1])
    bar_end = last + pd.Timedelta(milliseconds=_interval_ms(interval))
    if now_ts < bar_end:
        return df.iloc[:-1]
    return df
