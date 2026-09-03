"""Єдина точка доступу до klines із кешем + ресемплінгом.

`ensure_klines` — джерело істини для research:
    - базовий інтервал (1m) живе в кеші (parquet / Postgres);
    - старші таймфрейми будуються локально і **не** пишуться назад
      (щоб похідний 5m не роз'їхався з докачаним 1m);
    - мережа смикається лише для відсутніх вікон бази.
"""

from __future__ import annotations

import logging

import pandas as pd

from scalper_hft.data.resample import check_target_valid, interval_minutes, resample_klines
from scalper_hft.data.store import get_store

logger = logging.getLogger(__name__)

DEFAULT_BASE_INTERVAL = "1m"


def can_derive(interval: str, base_interval: str) -> bool:
    """Чи цільовий інтервал виводиться ресемплінгом із бази (старший і кратний)."""
    if interval == base_interval:
        return False
    try:
        check_target_valid(interval_minutes(base_interval), interval_minutes(interval))
    except (ValueError, KeyError):
        return False
    return True


def ensure_klines(
    symbol: str,
    interval: str,
    days: int,
    *,
    base_interval: str = DEFAULT_BASE_INTERVAL,
    derive: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Повернути klines (symbol, interval) за останні `days` днів.

    Args:
        derive: якщо True (за замовчуванням) — не питати Binance про цільовий
            інтервал, а виводити його ресемплінгом із base_interval.
        force: ігнорувати «свіжість» бази і докачати хвіст 1m.
    """
    from scalper_hft.data.downloader import download_klines

    if interval == base_interval or not derive or not can_derive(interval, base_interval):
        df = download_klines(symbol, interval, days, force=force)
    else:
        df = _derive_klines(symbol, interval, days, base_interval, force=force)
    return _tail_days(df, days)


def _tail_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Залишити лише останні `days` днів даних.

    Кеш може містити більше, ніж запитує клієнт (раніше качали 3 роки, а
    sweep просить 60–180 днів): без обрізання клітинки мовчки бектестили
    весь кеш, і результати не відповідали конфігу `--days`.
    """
    if df is None or df.empty or days <= 0:
        return df
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    trimmed = df[df.index >= cutoff]
    return trimmed if not trimmed.empty else df  # кеш коротший за запит — що є, те й повертаємо


def _derive_klines(symbol: str, interval: str, days: int, base_interval: str, *, force: bool) -> pd.DataFrame:
    """Завантажити базу (інкрементально) і ресемплінгом отримати цільовий інтервал.

    Похідний ряд не зберігається: джерело істини — лише base_interval.
    """
    from scalper_hft.data.downloader import download_klines

    base = download_klines(symbol, base_interval, days, force=force)
    if base is None or base.empty:
        raise RuntimeError(f"Немає базових даних {symbol} {base_interval} — не з чого ресемплити {interval}")
    out = resample_klines(base, interval, source=base_interval)
    logger.info(
        "Ресемплінг %s %s → %s (%d → %d барів, не кешуємо)", symbol, base_interval, interval, len(base), len(out)
    )
    return out


def klines_from_store(
    symbol: str,
    interval: str,
    days: int | None = None,
    *,
    base_interval: str = DEFAULT_BASE_INTERVAL,
) -> pd.DataFrame | None:
    """Прочитати кеш без мережі: 1m + ресемплінг, інакше нативний інтервал."""
    store = get_store()
    df: pd.DataFrame | None = None
    if interval != base_interval:
        base = store.load_klines(symbol, base_interval)
        if base is not None and not base.empty and can_derive(interval, base_interval):
            df = resample_klines(base, interval, source=base_interval)
    if df is None or df.empty:
        df = store.load_klines(symbol, interval)
    if df is None or df.empty:
        return None
    if days is not None:
        cutoff = df.index[-1] - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    return df


def warm_base_cache(
    symbols: list[str], days: int, base_interval: str = DEFAULT_BASE_INTERVAL
) -> dict[str, pd.DataFrame]:
    """Попереднє завантаження базового таймфрейму для набору символів.

    Викликається один раз на початку sweep: далі всі старші таймфрейми
    будуються локально без жодного звернення до Binance. Послідовно
    (спільний ccxt-клієнт пейсить запити глобально, 429-бекф).
    """
    from scalper_hft.data.downloader import download_klines

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = download_klines(sym, base_interval, days)
    return out


__all__ = ["ensure_klines", "klines_from_store", "warm_base_cache", "can_derive", "DEFAULT_BASE_INTERVAL"]
