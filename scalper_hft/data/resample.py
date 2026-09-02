"""Ресемплінг OHLCV-барів: з 1m будуємо 5m/15m/30m/1h/4h/1d і будь-які інші.

Ідея: замість повторного запиту до Binance для кожного таймфрейму завантажуємо
один раз базовий (найчастіший) інтервал і агрегуємо решту локально:

    base  = download_klines("BTCUSDT", "1m", 90)   # один запит
    five  = resample_klines(base, "5m")            # миттєво, без API
    hour  = resample_klines(base, "1h")

Конвенції:
    - індекс — час ВІДКРИТТЯ бару (як у Binance);
    - неповний останній бар відкидається (щоб не малювати артефакти);
    - цільовий інтервал має бути кратним джерелу (1m → 5m ок; 1m → 3m ок; 5m → 1m — ні).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Інтервали у хвилинах (дробові для секундних)
INTERVAL_MINUTES: dict[str, float] = {
    "1s": 1 / 60,
    "5s": 5 / 60,
    "15s": 15 / 60,
    "30s": 30 / 60,
    "1m": 1,
    "2m": 2,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def interval_minutes(interval: str) -> float:
    """Тривалість інтервалу у хвилинах; підтримуються 'Xs', 'Xm', 'Xh', 'Xd'."""
    if interval in INTERVAL_MINUTES:
        return INTERVAL_MINUTES[interval]
    unit = interval[-1]
    num = int(interval[:-1])
    per_unit = {"s": 1 / 60, "m": 1.0, "h": 60.0, "d": 1440.0}[unit]
    return num * per_unit


def _rule(minutes: float) -> str:
    """pandas-rule для resample: '1min', '5min', '30min', '1h', '1D'."""
    if minutes >= 1440 and minutes % 1440 == 0:
        return f"{int(minutes // 1440)}D"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{int(minutes // 60)}h"
    if minutes >= 1 and minutes % 1 == 0:
        return f"{int(minutes)}min"
    return f"{int(minutes * 60)}s"


def infer_interval_minutes(index: pd.DatetimeIndex) -> float:
    """Визначити таймфрейм ряду за медіаною різниць сусідніх барів."""
    if len(index) < 2:
        raise ValueError("Замало барів для визначення інтервалу")
    diffs = pd.Series(index).diff().dropna()
    median = float(diffs.median().total_seconds()) / 60.0
    if median <= 0:
        raise ValueError("Індекс не є впорядкованим часовим рядом")
    return median


def check_target_valid(source_min: float, target_min: float) -> None:
    """Цільовий інтервал має бути >= джерела і кратним йому."""
    if target_min < source_min - 1e-9:
        raise ValueError(
            f"Цільовий інтервал ({target_min:g} хв) менший за джерело ({source_min:g} хв) — "
            f"ресемплінг лише «вгору»; завантажте частіші дані"
        )
    if abs(target_min / source_min - round(target_min / source_min)) > 1e-9:
        raise ValueError(
            f"Цільовий інтервал ({target_min:g} хв) не кратний джерелу ({source_min:g} хв) — "
            f"бари Binance не вирівняються по межах"
        )


def resample_klines(
    df: pd.DataFrame,
    target: str,
    *,
    source: str | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Агрегувати OHLCV-бари у старший таймфрейм.

    Args:
        df: DataFrame з колонками open/high/low/close/volume, DatetimeIndex UTC.
        target: цільовий інтервал, напр. '5m', '15m', '30m', '1h', '4h', '1d'.
        source: інтервал джерела (якщо None — визначається за індексом).
        drop_incomplete: відкинути останній бар, якщо він неповний
            (не вкритий барами джерела).
    """
    if df is None or df.empty:
        return df.copy()
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    df = df.copy()
    df.index = idx

    src_min = interval_minutes(source) if source else infer_interval_minutes(idx)
    tgt_min = interval_minutes(target)
    check_target_valid(src_min, tgt_min)

    out = df.resample(_rule(tgt_min), label="left", closed="left").agg(_AGG)
    out = out.dropna(subset=["open"])  # порожні біни (пропуски в даних)

    if drop_incomplete:
        src_end = idx[-1] + pd.Timedelta(minutes=src_min)  # кінець останнього бару джерела
        tgt_end = out.index[-1] + pd.Timedelta(minutes=tgt_min)  # кінець останнього цільового бару
        if tgt_end > src_end:
            out = out.iloc[:-1]
    if out.empty:
        logger.warning(
            "Ресемплінг %s → %s дав порожній результат (даних замало: %d барів джерела)",
            source or "?",
            target,
            len(df),
        )
    return out


def resample_series(s: pd.Series, target: str, *, source: str | None = None, agg: str = "last") -> pd.Series:
    """Ресемплінг довільної часової серії (funding, spreads, ...)."""
    src_min = interval_minutes(source) if source else infer_interval_minutes(pd.DatetimeIndex(s.index))
    tgt_min = interval_minutes(target)
    check_target_valid(src_min, tgt_min)
    return s.resample(_rule(tgt_min), label="left", closed="left").agg(agg).dropna()


__all__ = [
    "INTERVAL_MINUTES",
    "interval_minutes",
    "infer_interval_minutes",
    "check_target_valid",
    "resample_klines",
    "resample_series",
]
