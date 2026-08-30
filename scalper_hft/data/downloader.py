"""Завантаження історичних даних Binance у parquet-кеш.

Підтримувані типи даних (книга, гл. 8 — "Data"):
    - klines (OHLCV) — для свічкових стратегій;
    - aggTrades (трейди з buy/sell флагом) — для обчислення CVD та потоку заявок;
    - funding rate history — для фандінг-фільтрів.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.data.storage import (
    funding_path,
    klines_path,
    load_funding,
    load_klines,
    load_trades,
    save_funding,
    save_klines,
    save_trades,
    trades_path,
)

logger = logging.getLogger(__name__)

_MS = 1_000
_S = 60_000
_H = 3_600_000


def _utc_now() -> pd.Timestamp:
    """Поточний час як naive UTC Timestamp (збігається з індексами кешу)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _interval_ms(interval: str) -> int:
    """Тривалість інтервалу в мілісекундах: '1s'=1000, '1m'=60000, '1h'=3.6M."""
    unit = interval[-1]
    num = int(interval[:-1])
    per_unit = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return num * per_unit


class Downloader:
    """Ітеративне завантаження історії з повторними спробами та батчами."""

    def __init__(self, client: BinanceClient | None = None, retries: int = 3) -> None:
        settings = get_settings()
        self.client = client or BinanceClient(settings.binance_api_key, settings.binance_api_secret, settings.exchange)
        self.retries = retries

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — мережеві помилки різні
                last = exc
                logger.warning("Спроба %d/%d не вдалась: %s", attempt + 1, self.retries, exc)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Не вдалося завантажити дані після {self.retries} спроб: {last}")

    def klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Завантажити klines за останні `days` днів, доповнюючи кеш."""
        path = klines_path(get_settings().data_dir_abs, symbol, interval)
        existing = load_klines(path)
        start_ms = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            last_ts = int(existing.index[-1].value // 1_000_000)
            start_ms = min(start_ms, last_ts)

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        since = start_ms
        interval_ms = _interval_ms(interval)
        guard = 0
        while True:
            guard += 1
            if guard > 10_000:
                raise RuntimeError("Забагато батчів — ймовірно застрягли у циклі завантаження")
            batch = self._with_retry(self.client.fetch_klines, symbol, interval, since)
            if not batch:
                break
            df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")
            frames.append(df)
            since = int(df.index[-1].value // 1_000_000) + interval_ms
            if len(batch) < 1000:
                break
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        save_klines(path, out)
        return out

    def spot_klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Спотові klines (для delta-neutral арбітражу) у окремий кеш."""
        from scalper_hft.data.binance_client import BinanceClient
        from scalper_hft.data.storage import spot_klines_path

        path = spot_klines_path(get_settings().data_dir_abs, symbol, interval)
        existing = load_klines(path)
        start_ms = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            start_ms = min(start_ms, int(existing.index[-1].value // 1_000_000))

        spot_client = BinanceClient(market_type="spot")
        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        since = start_ms
        interval_ms = _interval_ms(interval)
        guard = 0
        while True:
            guard += 1
            if guard > 10_000:
                raise RuntimeError("Забагато батчів spot klines")
            batch = self._with_retry(spot_client.fetch_klines, symbol, interval, since)
            if not batch:
                break
            df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")
            frames.append(df)
            since = int(df.index[-1].value // 1_000_000) + interval_ms
            if len(batch) < 1000:
                break
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        save_klines(path, out)
        return out

    def agg_trades(self, symbol: str, days: int, start_ms: int | None = None) -> pd.DataFrame:
        """Завантажити історичні агреговані трейди (для CVD).

        ⚠ Binance futures REST обмежує вікно пошуку aggTrades останніми
        ~2 днями (код -4166). Для старіших трейдів — data.binance.vision dumps
        (див. docs/RESEARCH.md). Тут вікно автоматично обрізається до 2 днів.
        """
        path = trades_path(get_settings().data_dir_abs, symbol)
        existing = load_trades(path)
        end_ms = int(_utc_now().value // 1_000_000)
        max_window_ms = 2 * 24 * 3600 * 1000  # обмеження Binance: 2 доби
        begin_ms = start_ms or int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        begin_ms = max(begin_ms, end_ms - max_window_ms)
        if existing is not None and not existing.empty:
            begin_ms = min(begin_ms, max(existing.index[-1].value // 1_000_000, end_ms - max_window_ms))

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        since = begin_ms
        guard = 0
        total_rows = 0
        while since < end_ms:
            guard += 1
            if guard > 20_000:
                raise RuntimeError("Забагато батчів aggTrades")
            batch = self._with_retry(self.client.fetch_agg_trades, symbol, since)
            if not batch:
                break
            rows = [
                {
                    "ts": t["timestamp"],
                    "price": float(t["price"]),
                    "amount": float(t["amount"]),
                    "side": "buy"
                    if t.get("side") == "buy"
                    else (
                        "sell"
                        if t.get("side") == "sell"
                        else ("buy" if not t.get("info", {}).get("m", True) else "sell")
                    ),
                }
                for t in batch
            ]
            df = pd.DataFrame(rows)
            if df.empty:
                break
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts").sort_index()
            frames.append(df)
            total_rows += len(df)
            since = int(df.index[-1].value // 1_000_000) + 1
            if guard % 200 == 0:
                logger.info("aggTrades %s: %d батчів, %d трейдів (до %s)", symbol, guard, total_rows, df.index[-1])
            if len(batch) < 1000:
                break
        if not frames:
            return pd.DataFrame(columns=["price", "amount", "side"])
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        save_trades(path, out)
        return out

    def funding(self, symbol: str, days: int) -> pd.DataFrame:
        """Історія ставок фандінгу (зазвичай кожні 8 годин)."""
        path = funding_path(get_settings().data_dir_abs, symbol)
        existing = load_funding(path)
        start_ms = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            start_ms = min(start_ms, int(existing.index[-1].value // 1_000_000))

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        since = start_ms
        guard = 0
        while True:
            guard += 1
            if guard > 1000:
                break
            batch = self._with_retry(self.client.fetch_funding_rate_history, symbol, since)
            if not batch:
                break
            df = pd.DataFrame(batch)
            if df.empty:
                break
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("ts")
            frames.append(df[["fundingRate"]])
            since = int(df.index[-1].value // 1_000_000) + 1
            if len(batch) < 1000:
                break
        if not frames:
            return pd.DataFrame(columns=["fundingRate"])
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        save_funding(path, out)
        return out


# ── зручні функції верхнього рівня ───────────────────────────────────────────
def download_klines(symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
    """Завантажити/оновити klines. Кеш повертається лише якщо покриває період
    І свіжий (останній бар < 1 год тому); інакше — розширюється в обидва боки."""
    settings = get_settings()
    path = klines_path(settings.data_dir_abs, symbol, interval)
    cached = None if force else load_klines(path)
    if cached is not None and not cached.empty:
        now = _utc_now()
        oldest = cached.index[0]
        newest = cached.index[-1]
        needed_from = now - pd.Timedelta(days=days)
        # свіжість залежить від інтервалу: пропуск > 2 барів = stale (мін. 60 сек)
        staleness = max(2 * _interval_ms(interval), 60_000)
        stale = newest < now - pd.Timedelta(milliseconds=staleness)
        if oldest <= needed_from and not stale:
            logger.info(
                "Кеш klines %s %s покриває період і свіжий: %d рядків (до %s)", symbol, interval, len(cached), newest
            )
            return cached
        logger.info(
            "Розширення кешу klines %s %s: %d рядків (до %s, stale=%s)", symbol, interval, len(cached), newest, stale
        )
    logger.info("Завантаження klines %s %s за %d днів", symbol, interval, days)
    return Downloader().klines(symbol, interval, days)


def download_agg_trades(symbol: str, days: int, force: bool = False) -> pd.DataFrame:
    settings = get_settings()
    path = trades_path(settings.data_dir_abs, symbol)
    cached = None if force else load_trades(path)
    if cached is not None and not cached.empty:
        # REST може дістати лише останні ~2 доби — якщо кеш їх покриває,
        # повторне завантаження не потрібне (запобігає 10+ хв ре-фетчу)
        newest = cached.index[-1]
        now = _utc_now()
        if newest >= now - pd.Timedelta(days=2):
            logger.info("Кеш aggTrades %s актуальний (до %s): %d рядків", symbol, newest, len(cached))
            return cached
        logger.info("Оновлення aggTrades %s: було %d рядків до %s", symbol, len(cached), newest)
    logger.info("Завантаження aggTrades %s за %d днів", symbol, days)
    return Downloader().agg_trades(symbol, days)


def download_funding(symbol: str, days: int, force: bool = False) -> pd.DataFrame:
    """Кеш фандінгу: свіжий лише якщо покриває період і остання ставка < 16 год.

    Binance USDT-M нараховує фандінг кожні 8 год; 2 періоди без оновлення = stale.
    """
    settings = get_settings()
    path = funding_path(settings.data_dir_abs, symbol)
    cached = None if force else load_funding(path)
    if cached is not None and not cached.empty:
        now = _utc_now()
        oldest = cached.index[0]
        newest = cached.index[-1]
        needed_from = now - pd.Timedelta(days=days)
        stale = newest < now - pd.Timedelta(hours=16)
        if oldest <= needed_from and not stale:
            logger.info(
                "Кеш funding %s покриває період і свіжий: %d рядків (до %s)",
                symbol,
                len(cached),
                newest,
            )
            return cached
        logger.info(
            "Оновлення funding %s: %d рядків (до %s, stale=%s)",
            symbol,
            len(cached),
            newest,
            stale,
        )
    logger.info("Завантаження funding %s за %d днів", symbol, days)
    return Downloader().funding(symbol, days)


def download_spot_klines(symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
    """Спотові klines (для delta-neutral арбітражу) з окремим кешем."""
    from scalper_hft.data.storage import spot_klines_path

    settings = get_settings()
    path = spot_klines_path(settings.data_dir_abs, symbol, interval)
    cached = None if force else load_klines(path)
    if cached is not None and not cached.empty:
        now = _utc_now()
        newest = cached.index[-1]
        stale = newest < now - pd.Timedelta(milliseconds=max(2 * _interval_ms(interval), 60_000))
        if not stale:
            logger.info("Кеш spot klines %s %s свіжий: %d рядків", symbol, interval, len(cached))
            return cached
    logger.info("Завантаження spot klines %s %s за %d днів", symbol, interval, days)
    return Downloader().spot_klines(symbol, interval, days)
