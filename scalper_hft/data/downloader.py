"""Завантаження історичних даних Binance у кеш (parquet або PostgreSQL).

Підтримувані типи даних (книга, гл. 8 — "Data"):
    - klines (OHLCV) — для свічкових стратегій;
    - aggTrades (трейди з buy/sell флагом) — для обчислення CVD та потоку заявок;
    - funding rate history — для фандінг-фільтрів.

Кеш іде через scalper_hft.data.store.get_store(): DATA_BACKEND=parquet (файли
у data/) або DATA_BACKEND=postgres (PostgreSQL у Docker). Дані завантажуються
з Binance лише тоді, коли кеш не покриває період або застарів.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.data.store import get_store

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


def _trade_id(t: dict[str, Any], fallback: int) -> int:
    """aggTrade id з ccxt-словника; fallback — синтетичний лічильник."""
    tid = t.get("id")
    if tid is None:
        tid = t.get("info", {}).get("a")
    if tid is None:
        return fallback
    try:
        return int(tid)
    except (TypeError, ValueError):
        return fallback


_client: BinanceClient | None = None


def _default_client() -> BinanceClient:
    """Спільний клієнт на процес: ccxt enableRateLimit пейсить запити ГЛОБАЛЬНО.

    Без цього кожен Downloader створював свій ccxt-інстанс зі своїм лімітером —
    паралельні завантаження разом перевищували ліміт IP (429 -1003).
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = BinanceClient(settings.binance_api_key, settings.binance_api_secret, settings.exchange)
    return _client


class Downloader:
    """Ітеративне завантаження історії з повторними спробами та батчами."""

    def __init__(self, client: BinanceClient | None = None, retries: int = 3, store: Any = None) -> None:
        self.client = client or _default_client()
        self.retries = retries
        self.store = store if store is not None else get_store()

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — мережеві помилки різні
                last = exc
                # rate limit — чекаємо довше, ліміт IP 2400/хв спільний з іншими процесами
                if "429" in str(exc) or "Too many requests" in str(exc) or "RateLimitExceeded" in type(exc).__name__:
                    sleep_s = 15.0 * (attempt + 1)
                    logger.warning("Rate limit: сплю %.0f с (спроба %d/%d)", sleep_s, attempt + 1, self.retries)
                else:
                    sleep_s = 1.5 * (attempt + 1)
                logger.warning("Спроба %d/%d не вдалась: %s", attempt + 1, self.retries, exc)
                time.sleep(sleep_s)
        raise RuntimeError(f"Не вдалося завантажити дані після {self.retries} спроб: {last}")

    def klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Завантажити klines за останні `days` днів, доповнюючи кеш."""
        existing = self.store.load_klines(symbol, interval)
        requested_start = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            first_ts = int(existing.index[0].value // 1_000_000)
            last_ts = int(existing.index[-1].value // 1_000_000)
            if first_ts <= requested_start:
                # кеш покриває початок вікна — до-качуємо лише відсутній хвіст
                start_ms = last_ts
            else:
                # кеш починається пізніше — розширюємо назад
                start_ms = min(requested_start, first_ts)
        else:
            start_ms = requested_start

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
        self.store.save_klines(symbol, interval, out)
        return out

    def spot_klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Спотові klines (для delta-neutral арбітражу) у окремий кеш."""
        existing = self.store.load_spot_klines(symbol, interval)
        requested_start = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            first_ts = int(existing.index[0].value // 1_000_000)
            last_ts = int(existing.index[-1].value // 1_000_000)
            start_ms = last_ts if first_ts <= requested_start else min(requested_start, first_ts)
        else:
            start_ms = requested_start

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
        self.store.save_spot_klines(symbol, interval, out)
        return out

    def agg_trades(self, symbol: str, days: int, start_ms: int | None = None) -> pd.DataFrame:
        """Завантажити історичні агреговані трейди (для CVD).

        ⚠ Binance futures REST обмежує вікно пошуку aggTrades останніми
        ~2 днями (код -4166). Для старіших трейдів — data.binance.vision dumps
        (див. docs/RESEARCH.md). Тут вікно автоматично обрізається до 2 днів.
        """
        existing = self.store.load_trades(symbol)
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
        synthetic = 0
        while since < end_ms:
            guard += 1
            if guard > 20_000:
                raise RuntimeError("Забагато батчів aggTrades")
            batch = self._with_retry(self.client.fetch_agg_trades, symbol, since)
            if not batch:
                break
            rows = []
            for t in batch:
                synthetic -= 1  # негативні id-и резервуються для синтетичних
                rows.append(
                    {
                        "trade_id": _trade_id(t, synthetic),
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
                )
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
            return pd.DataFrame(columns=["trade_id", "price", "amount", "side"])
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        self.store.save_trades(symbol, out)
        return out

    def funding(self, symbol: str, days: int) -> pd.DataFrame:
        """Історія ставок фандінгу (зазвичай кожні 8 годин)."""
        existing = self.store.load_funding(symbol)
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
        self.store.save_funding(symbol, out)
        return out


# ── зручні функції верхнього рівня ───────────────────────────────────────────
def _klines_fresh(existing: pd.DataFrame, days: int, interval: str) -> tuple[bool, bool]:
    """(покриває період, свіжий) — спільна логіка для klines-кешів."""
    now = _utc_now()
    oldest = existing.index[0]
    newest = existing.index[-1]
    needed_from = now - pd.Timedelta(days=days)
    staleness = max(2 * _interval_ms(interval), 60_000)
    stale = newest < now - pd.Timedelta(milliseconds=staleness)
    return oldest <= needed_from, not stale


def download_klines(symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
    """Завантажити/оновити klines. Кеш повертається лише якщо покриває період
    І свіжий (останній бар < 1 год тому); інакше — розширюється в обидва боки."""
    store = get_store()
    cached = None if force else store.load_klines(symbol, interval)
    if cached is not None and not cached.empty:
        covers, fresh = _klines_fresh(cached, days, interval)
        if covers and fresh:
            logger.info("Кеш klines %s %s покриває період і свіжий: %d рядків (до %s)", symbol, interval, len(cached), cached.index[-1])
            return cached
        logger.info("Розширення кешу klines %s %s: %d рядків (до %s)", symbol, interval, len(cached), cached.index[-1])
    logger.info("Завантаження klines %s %s за %d днів", symbol, interval, days)
    return Downloader(store=store).klines(symbol, interval, days)


def download_agg_trades(symbol: str, days: int, force: bool = False) -> pd.DataFrame:
    store = get_store()
    cached = None if force else store.load_trades(symbol)
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
    return Downloader(store=store).agg_trades(symbol, days)


def download_funding(symbol: str, days: int, force: bool = False) -> pd.DataFrame:
    """Кеш фандінгу: свіжий лише якщо покриває період і остання ставка < 16 год.

    Binance USDT-M нараховує фандінг кожні 8 год; 2 періоди без оновлення = stale.
    """
    store = get_store()
    cached = None if force else store.load_funding(symbol)
    if cached is not None and not cached.empty:
        now = _utc_now()
        oldest = cached.index[0]
        newest = cached.index[-1]
        needed_from = now - pd.Timedelta(days=days)
        stale = newest < now - pd.Timedelta(hours=16)
        if oldest <= needed_from and not stale:
            logger.info("Кеш funding %s покриває період і свіжий: %d рядків (до %s)", symbol, len(cached), newest)
            return cached
        logger.info("Оновлення funding %s: %d рядків (до %s, stale=%s)", symbol, len(cached), newest, stale)
    logger.info("Завантаження funding %s за %d днів", symbol, days)
    return Downloader(store=store).funding(symbol, days)


def download_spot_klines(symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
    """Спотові klines (для delta-neutral арбітражу) з окремим кешем."""
    store = get_store()
    cached = None if force else store.load_spot_klines(symbol, interval)
    if cached is not None and not cached.empty:
        covers, fresh = _klines_fresh(cached, days, interval)
        if covers and fresh:
            logger.info("Кеш spot klines %s %s свіжий: %d рядків", symbol, interval, len(cached))
            return cached
    logger.info("Завантаження spot klines %s %s за %d днів", symbol, interval, days)
    return Downloader(store=store).spot_klines(symbol, interval, days)
