"""Єдина точка доступу до klines із кешем + ресемплінгом.

`ensure_klines` — гнучке завантаження для гіпотез-сканувань:
    1. шукаємо готові бари потрібного таймфрейму в кеші (parquet або Postgres);
    2. якщо немає і `derive=True` — ресемплінг із базового інтервалу (1m),
       який качається з Binance лише один раз на символ;
    3. результат зберігається в кеш, тож наступний запит — нуль API-дзвінків.
"""

from __future__ import annotations

import logging

import pandas as pd

from scalper_hft.data.resample import resample_klines
from scalper_hft.data.store import get_store

logger = logging.getLogger(__name__)

DEFAULT_BASE_INTERVAL = "1m"


def ensure_klines(
    symbol: str,
    interval: str,
    days: int,
    *,
    base_interval: str = DEFAULT_BASE_INTERVAL,
    derive: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Повернути klines (symbol, interval) за останні `days` днів.

    Args:
        derive: якщо True — не питати Binance про цільовий інтервал взагалі,
            а виводити його ресемплінгом із base_interval (мінімум API-дзвінків).
        force: ігнорувати кеш і завантажувати заново.
    """
    from scalper_hft.data.downloader import _klines_fresh, download_klines

    if interval == base_interval:
        return download_klines(symbol, interval, days, force=force)

    store = get_store()
    if not force:
        cached = store.load_klines(symbol, interval)
        if cached is not None and not cached.empty:
            covers, fresh = _klines_fresh(cached, days, interval)
            if covers and fresh:
                logger.info("Кеш klines %s %s покриває період (ресемплінг/раніше): %d рядків", symbol, interval, len(cached))
                return cached
            logger.info("Кеш klines %s %s застарів — оновлюємо (covers=%s, fresh=%s)", symbol, interval, covers, fresh)

    if derive:
        return _derive_klines(symbol, interval, days, base_interval, force=force)

    # Спершу пробуємо прямий запит (стандартні інтервали Binance: 1m,5m,15m,1h,...)
    try:
        direct = download_klines(symbol, interval, days, force=force)
        if direct is not None and not direct.empty:
            return direct
    except Exception as exc:  # noqa: BLE001 — нестандартний інтервал тощо
        logger.info("Пряме завантаження %s %s не вдалось (%s) — ресемплінг з %s", symbol, interval, exc, base_interval)

    return _derive_klines(symbol, interval, days, base_interval, force=force)


def _derive_klines(symbol: str, interval: str, days: int, base_interval: str, *, force: bool) -> pd.DataFrame:
    """Завантажити базу (один раз) і ресемплінгом отримати цільовий інтервал."""
    from scalper_hft.data.downloader import download_klines

    base = download_klines(symbol, base_interval, days, force=force)
    if base is None or base.empty:
        raise RuntimeError(f"Немає базових даних {symbol} {base_interval} — не з чого ресемплити {interval}")
    out = resample_klines(base, interval, source=base_interval)
    logger.info("Ресемплінг %s %s → %s (%d → %d барів)", symbol, base_interval, interval, len(base), len(out))
    store = get_store()
    store.save_klines(symbol, interval, out)
    logger.info("Збережено ресемплінг %s %s: %d барів", symbol, interval, len(out))
    return out


def warm_base_cache(symbols: list[str], days: int, base_interval: str = DEFAULT_BASE_INTERVAL) -> dict[str, pd.DataFrame]:
    """Попереднє завантаження базового таймфрейму для набору символів.

    Викликається один раз на початку sweep: далі всі старші таймфрейми
    будуються локально без жодного звернення до Binance.

    Порядок джерел:
        1) data.binance.vision (архіви, ~1 запит на місяць замість ~44 REST-батчів);
        2) REST-дотяжка останніх днів (архіви публікуються із затримкою).
    """
    from scalper_hft.data.downloader import download_klines

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            _vision_warm(sym, days, base_interval)
        except Exception as exc:  # noqa: BLE001 — архів недоступний → REST
            logger.info("Vision klines %s недоступні (%s) — REST", sym, exc)
        out[sym] = download_klines(sym, base_interval, days)  # кеш або REST-дотяжка
    return out


def _vision_warm(symbol: str, days: int, interval: str) -> None:
    """Завантажити архівні klines з vision у кеш (без REST-батчів)."""
    from datetime import timedelta

    from scalper_hft.data.binance_vision import download_klines_vision

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = (now - pd.Timedelta(days=days)).date()
    end = (now - timedelta(days=2)).date()  # архіви відстають на 1-2 дні
    if start >= end:
        return
    df = download_klines_vision(symbol, interval, start, end)
    logger.info("Vision klines %s %s: %d барів (%s … %s)", symbol, interval, len(df), df.index[0], df.index[-1])


__all__ = ["ensure_klines", "warm_base_cache", "DEFAULT_BASE_INTERVAL"]
