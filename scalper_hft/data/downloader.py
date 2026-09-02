"""Завантаження історичних даних Binance у кеш (parquet або PostgreSQL).

Підтримувані типи даних (книга, гл. 8 — "Data"):
    - klines (OHLCV) — для свічкових стратегій;
    - aggTrades (трейди з buy/sell флагом) — для обчислення CVD та потоку заявок;
    - funding rate history — для фандінг-фільтрів.

Кеш іде через scalper_hft.data.store.get_store(): DATA_BACKEND=parquet (файли
у data/) або DATA_BACKEND=postgres (PostgreSQL у Docker). Дані завантажуються
з Binance лише тоді, коли кеш не покриває період, має дірки або застарів.
Закриті бари не перекачуються: дописуються префікс, внутрішні пропуски і хвіст.
Підтримується стійкість до лімітів (429), блокувань (418), періодичний чекпоінтинг
та автоматичне збереження прогресу при збоях чи перериваннях.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.data.store import get_store

logger = logging.getLogger(__name__)

_MS = 1_000
_S = 60_000
_H = 3_600_000
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _utc_now() -> pd.Timestamp:
    """Поточний час як naive UTC Timestamp (збігається з індексами кешу)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _interval_ms(interval: str) -> int:
    """Тривалість інтервалу в мілісекундах: '1s'=1000, '1m'=60000, '1h'=3.6M."""
    unit = interval[-1]
    num = int(interval[:-1])
    per_unit = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return num * per_unit


def _to_ms(ts: pd.Timestamp) -> int:
    """Datetime64/Timestamp → epoch milliseconds (naive UTC)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int(t.value // 1_000_000)


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Злити перекривні/суміжні півінтервали [start, end)."""
    ordered = sorted((s, e) for s, e in windows if s < e)
    if not ordered:
        return []
    out: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_s, prev_e = out[-1]
        if start <= prev_e:
            out[-1] = (prev_s, max(prev_e, end))
        else:
            out.append((start, end))
    return out


def _index_ms(index: pd.DatetimeIndex) -> Any:
    """Epoch-ms для кожного бару. Pandas 3 тримає DatetimeIndex у us, не ns."""
    idx = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    if hasattr(idx, "as_unit"):
        return idx.as_unit("ms").asi8
    return idx.asi8 // 1_000_000


def _gap_windows(index: pd.DatetimeIndex, interval_ms: int, since_ms: int) -> list[tuple[int, int]]:
    """Внутрішні дірки, що перетинаються з [since_ms, …)."""
    if len(index) < 2:
        return []
    ms = _index_ms(index)
    diffs = ms[1:] - ms[:-1]
    threshold = interval_ms + interval_ms // 2  # 1.5× інтервалу, як validate_bars
    windows: list[tuple[int, int]] = []
    for i, delta in enumerate(diffs):
        if int(delta) <= threshold:
            continue
        start = max(int(ms[i]) + interval_ms, since_ms)
        end = int(ms[i + 1])
        if start < end:
            windows.append((start, end))
    return windows


def missing_klines_windows(
    existing: pd.DataFrame | None,
    requested_start_ms: int,
    now_ms: int,
    interval_ms: int,
) -> list[tuple[int, int]]:
    """Вікна, яких немає в кеші: префікс, внутрішні дірки, застарілий хвіст.

    Кожне вікно — півінтервал [start_ms, end_ms) у epoch-ms. Хвіст стартує
    з last_ts (включно), щоб оновити незакритий бар; середина не чіпається.
    """
    if existing is None or existing.empty:
        return [(requested_start_ms, now_ms + interval_ms)]

    idx = existing.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()

    first_ms = _to_ms(idx[0])
    last_ms = _to_ms(idx[-1])
    windows: list[tuple[int, int]] = []
    if requested_start_ms < first_ms:
        windows.append((requested_start_ms, first_ms))
    windows.extend(_gap_windows(idx, interval_ms, requested_start_ms))
    staleness_ms = max(2 * interval_ms, 60_000)
    if last_ms < now_ms - staleness_ms:
        windows.append((last_ms, now_ms + interval_ms))
    return merge_windows(windows)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLS)


def _batch_to_ohlcv(batch: list[list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(batch, columns=["ts", *_OHLCV_COLS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


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


def _calculate_backoff(attempt: int, exc: Exception, max_retries: int) -> tuple[float, str]:
    """Обчислити час очікування (секунди) та категорію помилки для логування.

    Категорії:
    1. IP Ban (HTTP 418, -1003, 'IP banned', 'banned until'):
       - Якщо є точний timestamp закінчення бану ('banned until 1708935600000'), спимо до нього + 2с.
       - Інакше експоненційно: 60s, 120s, 180s... з джиттером.
    2. Rate Limit (HTTP 429, RateLimitExceeded, 'Too many requests'):
       - 10s * (2 ** attempt) + jitter, обмежено 180s (10s, 20s, 40s, 80s, 160s...).
    3. Мережеві помилки (502, 503, 504, NetworkError, RequestTimeout, ExchangeNotAvailable):
       - 2s * (2 ** attempt) + jitter, обмежено 60s (2s, 4s, 8s, 16s, 32s...).
    """
    exc_str = str(exc)
    exc_type = type(exc).__name__
    combined = f"{exc_type} {exc_str}".lower()

    # 1. IP Ban / 418 / -1003 / "banned until"
    is_ip_ban = (
        "418" in combined
        or "-1003" in combined
        or "ip ban" in combined
        or "banned until" in combined
        or "ip has been auto-banned" in combined
    )
    if is_ip_ban:
        match = re.search(r"banned until (\d{10,13})", exc_str, flags=re.IGNORECASE)
        if match:
            try:
                ban_ts_raw = int(match.group(1))
                ban_ts_ms = ban_ts_raw if ban_ts_raw > 10_000_000_000 else ban_ts_raw * 1000
                now_ms = int(time.time() * 1000)
                diff_s = (ban_ts_ms - now_ms) / 1000.0
                if diff_s > 0:
                    return min(600.0, diff_s + 2.0), "ip_ban"
            except (ValueError, TypeError):
                pass
        base_ban = 60.0 * (attempt + 1) + random.uniform(1.0, 5.0)
        return min(300.0, base_ban), "ip_ban"

    # 2. Rate limit / 429 / RateLimitExceeded / DDoSProtection
    is_rate_limit = (
        "429" in combined
        or "too many requests" in combined
        or "ratelimitexceeded" in combined
        or "ddosprotection" in combined
        or "rate limit" in combined
    )
    if is_rate_limit:
        base_rl = 10.0 * (2**attempt) + random.uniform(1.0, 5.0)
        return min(180.0, base_rl), "rate_limit"

    # 3. Network / Server / Timeout / Other transient
    base_net = 2.0 * (2**attempt) + random.uniform(0.5, 2.0)
    return min(60.0, base_net), "network"


class Downloader:
    """Ітеративне завантаження історії з повторними спробами, батчами та чекпоінтами."""

    def __init__(
        self,
        client: BinanceClient | None = None,
        retries: int | None = None,
        store: Any = None,
        batch_delay: float | None = None,
        checkpoint_batches: int | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or _default_client()
        self.retries = int(retries if retries is not None else settings.download_retries)
        self.batch_delay = float(batch_delay if batch_delay is not None else settings.download_batch_delay)
        self.checkpoint_batches = int(
            checkpoint_batches if checkpoint_batches is not None else settings.download_checkpoint_batches
        )
        self.store = store if store is not None else get_store()

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — мережеві помилки різні
                last = exc
                if attempt == self.retries - 1:
                    logger.warning("Спроба %d/%d (остання) не вдалась: %s", attempt + 1, self.retries, exc)
                    break
                sleep_s, category = _calculate_backoff(attempt, exc, self.retries)
                if category == "ip_ban":
                    logger.warning(
                        "Binance IP тимчасово заблоковано (418 / -1003): очікую %.1f с (спроба %d/%d) [%s]",
                        sleep_s,
                        attempt + 1,
                        self.retries,
                        exc,
                    )
                elif category == "rate_limit":
                    logger.warning(
                        "Rate limit Binance (429): очікую %.1f с (спроба %d/%d) [%s]",
                        sleep_s,
                        attempt + 1,
                        self.retries,
                        exc,
                    )
                else:
                    logger.warning(
                        "Спроба %d/%d не вдалась: %s. Очікую %.1f с перед повтором...",
                        attempt + 1,
                        self.retries,
                        exc,
                        sleep_s,
                    )
                time.sleep(sleep_s)
        raise RuntimeError(f"Не вдалося завантажити дані після {self.retries} спроб: {last}")

    def _fetch_klines_windows(
        self,
        fetch_fn: Callable[..., list[list[Any]]],
        symbol: str,
        interval: str,
        windows: list[tuple[int, int]],
        interval_ms: int,
        on_batch: Callable[[pd.DataFrame, int], None] | None = None,
    ) -> list[pd.DataFrame]:
        """Качати лише задані вікна; бари з ts >= end_ms відкидаються (без overlap)."""
        frames: list[pd.DataFrame] = []
        batch_count = 0
        for start_ms, end_ms in windows:
            since = start_ms
            end_ts = pd.Timestamp(end_ms, unit="ms")
            guard = 0
            while since < end_ms:
                guard += 1
                if guard > 20_000:
                    raise RuntimeError("Забагато батчів — ймовірно застрягли у циклі завантаження")
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)
                batch = self._with_retry(fetch_fn, symbol, interval, since)
                if not batch:
                    break
                df = _batch_to_ohlcv(batch)
                df = df[df.index < end_ts]
                if df.empty:
                    break
                frames.append(df)
                batch_count += 1
                if on_batch is not None:
                    on_batch(df, batch_count)
                since = _to_ms(df.index[-1]) + interval_ms
                if len(batch) < 1000:
                    break
        return frames

    def _extend_klines(
        self,
        existing: pd.DataFrame | None,
        fetch_fn: Callable[..., list[list[Any]]],
        symbol: str,
        interval: str,
        days: int,
        *,
        force: bool = False,
        save_fn: Callable[[pd.DataFrame], None] | None = None,
    ) -> pd.DataFrame:
        """Долити в кеш лише відсутні вікна (префікс / дірки / хвіст) з чекпоінтами та graceful recovery."""
        interval_ms = _interval_ms(interval)
        requested_start = _to_ms(_utc_now() - pd.Timedelta(days=days))
        now_ms = _to_ms(_utc_now())
        windows = missing_klines_windows(existing, requested_start, now_ms, interval_ms)
        if force and existing is not None and not existing.empty:
            last_ms = _to_ms(existing.index[-1])
            windows = merge_windows([*windows, (last_ms, now_ms + interval_ms)])
        if not windows:
            return existing if existing is not None and not existing.empty else _empty_ohlcv()

        logger.info("Докачую klines %s %s: %d вікон %s", symbol, interval, len(windows), windows)
        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        new_frames: list[pd.DataFrame] = []

        def _make_merged(extra: list[pd.DataFrame]) -> pd.DataFrame:
            all_f = [*frames, *extra]
            if not all_f:
                return _empty_ohlcv()
            merged = pd.concat(all_f).sort_index()
            return merged[~merged.index.duplicated(keep="last")]

        def _on_batch(batch_df: pd.DataFrame, batch_idx: int) -> None:
            new_frames.append(batch_df)
            if self.checkpoint_batches > 0 and batch_idx % self.checkpoint_batches == 0:
                checkpoint_df = _make_merged(new_frames)
                if save_fn is not None:
                    save_fn(checkpoint_df)
                logger.info(
                    "klines %s %s: збережено чекпоінт %d свічок (батч %d, до %s)",
                    symbol,
                    interval,
                    len(checkpoint_df),
                    batch_idx,
                    batch_df.index[-1],
                )

        try:
            self._fetch_klines_windows(fetch_fn, symbol, interval, windows, interval_ms, on_batch=_on_batch)
        except BaseException as exc:
            if new_frames and save_fn is not None:
                try:
                    partial_df = _make_merged(new_frames)
                    save_fn(partial_df)
                    logger.warning(
                        "Збережено проміжний прогрес klines %s %s (%d свічок) перед перериванням: %s",
                        symbol,
                        interval,
                        len(partial_df),
                        exc,
                    )
                except Exception as save_err:  # noqa: BLE001
                    logger.error("Не вдалося зберегти аварійний чекпоінт: %s", save_err)
            raise

        out = _make_merged(new_frames)
        return out

    def klines(self, symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
        """Завантажити klines за останні `days` днів, доповнюючи кеш без перезапису середини."""
        existing = self.store.load_klines(symbol, interval)
        save_fn = lambda df: self.store.save_klines(symbol, interval, df)
        out = self._extend_klines(
            existing,
            self.client.fetch_klines,
            symbol,
            interval,
            days,
            force=force,
            save_fn=save_fn,
        )
        if not out.empty:
            self.store.save_klines(symbol, interval, out)
        return out

    def spot_klines(self, symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
        """Спотові klines (для delta-neutral арбітражу) у окремий кеш."""
        existing = self.store.load_spot_klines(symbol, interval)
        spot_client = BinanceClient(market_type="spot")
        save_fn = lambda df: self.store.save_spot_klines(symbol, interval, df)
        out = self._extend_klines(
            existing,
            spot_client.fetch_klines,
            symbol,
            interval,
            days,
            force=force,
            save_fn=save_fn,
        )
        if not out.empty:
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
        new_frames: list[pd.DataFrame] = []
        since = begin_ms
        guard = 0
        total_rows = 0
        synthetic = 0

        def _make_merged() -> pd.DataFrame:
            all_f = [*frames, *new_frames]
            if not all_f:
                return pd.DataFrame(columns=["trade_id", "price", "amount", "side"])
            merged = pd.concat(all_f).sort_index()
            return merged[~merged.index.duplicated(keep="last")]

        try:
            while since < end_ms:
                guard += 1
                if guard > 20_000:
                    raise RuntimeError("Забагато батчів aggTrades")
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)
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
                new_frames.append(df)
                total_rows += len(df)
                since = int(df.index[-1].value // 1_000_000) + 1
                if guard % 200 == 0:
                    logger.info("aggTrades %s: %d батчів, %d трейдів (до %s)", symbol, guard, total_rows, df.index[-1])
                if self.checkpoint_batches > 0 and guard % (self.checkpoint_batches * 2) == 0:
                    checkpoint_df = _make_merged()
                    self.store.save_trades(symbol, checkpoint_df)
                if len(batch) < 1000:
                    break
        except BaseException as exc:
            if new_frames:
                try:
                    partial_df = _make_merged()
                    self.store.save_trades(symbol, partial_df)
                    logger.warning(
                        "Збережено проміжний прогрес aggTrades %s (%d трейдів) перед перериванням: %s",
                        symbol,
                        len(partial_df),
                        exc,
                    )
                except Exception as save_err:  # noqa: BLE001
                    logger.error("Не вдалося зберегти чекпоінт aggTrades: %s", save_err)
            raise

        out = _make_merged()
        if not out.empty:
            self.store.save_trades(symbol, out)
        return out

    def funding(self, symbol: str, days: int) -> pd.DataFrame:
        """Історія ставок фандінгу (зазвичай кожні 8 годин)."""
        existing = self.store.load_funding(symbol)
        start_ms = int((_utc_now() - pd.Timedelta(days=days)).value // 1_000_000)
        if existing is not None and not existing.empty:
            start_ms = min(start_ms, int(existing.index[-1].value // 1_000_000))

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        new_frames: list[pd.DataFrame] = []
        since = start_ms
        guard = 0

        def _make_merged() -> pd.DataFrame:
            all_f = [*frames, *new_frames]
            if not all_f:
                return pd.DataFrame(columns=["fundingRate"])
            merged = pd.concat(all_f).sort_index()
            return merged[~merged.index.duplicated(keep="last")]

        try:
            while True:
                guard += 1
                if guard > 1000:
                    break
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)
                batch = self._with_retry(self.client.fetch_funding_rate_history, symbol, since)
                if not batch:
                    break
                df = pd.DataFrame(batch)
                if df.empty:
                    break
                df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("ts")
                new_frames.append(df[["fundingRate"]])
                since = int(df.index[-1].value // 1_000_000) + 1
                if len(batch) < 1000:
                    break
        except BaseException as exc:
            if new_frames:
                try:
                    partial_df = _make_merged()
                    self.store.save_funding(symbol, partial_df)
                    logger.warning(
                        "Збережено проміжний прогрес funding %s (%d записів) перед перериванням: %s",
                        symbol,
                        len(partial_df),
                        exc,
                    )
                except Exception as save_err:  # noqa: BLE001
                    logger.error("Не вдалося зберегти чекпоінт funding: %s", save_err)
            raise

        out = _make_merged()
        if not out.empty:
            self.store.save_funding(symbol, out)
        return out


# ── зручні функції верхнього рівня ───────────────────────────────────────────
def _klines_windows_for(existing: pd.DataFrame, days: int, interval: str) -> list[tuple[int, int]]:
    interval_ms = _interval_ms(interval)
    requested_start = _to_ms(_utc_now() - pd.Timedelta(days=days))
    return missing_klines_windows(existing, requested_start, _to_ms(_utc_now()), interval_ms)


def download_klines(
    symbol: str,
    interval: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
) -> pd.DataFrame:
    """Завантажити/оновити klines. Старі бари не видаляються: докачуються
    лише префікс, внутрішні дірки та застарілий хвіст."""
    store = get_store()
    cached = None if force else store.load_klines(symbol, interval)
    if cached is not None and not cached.empty and not force:
        windows = _klines_windows_for(cached, days, interval)
        if not windows:
            logger.info(
                "Кеш klines %s %s покриває період і свіжий: %d рядків (до %s)",
                symbol,
                interval,
                len(cached),
                cached.index[-1],
            )
            return cached
        logger.info(
            "Розширення/латання кешу klines %s %s: %d вікон, було %d рядків",
            symbol,
            interval,
            len(windows),
            len(cached),
        )
    logger.info("Завантаження klines %s %s за %d днів", symbol, interval, days)
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
    ).klines(symbol, interval, days, force=force)


def download_agg_trades(
    symbol: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
) -> pd.DataFrame:
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
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
    ).agg_trades(symbol, days)


def download_funding(
    symbol: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
) -> pd.DataFrame:
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
    return Downloader(store=store, retries=retries, batch_delay=batch_delay).funding(symbol, days)


def download_spot_klines(
    symbol: str,
    interval: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
) -> pd.DataFrame:
    """Спотові klines (для delta-neutral арбітражу) з окремим кешем."""
    store = get_store()
    cached = None if force else store.load_spot_klines(symbol, interval)
    if cached is not None and not cached.empty and not force:
        windows = _klines_windows_for(cached, days, interval)
        if not windows:
            logger.info("Кеш spot klines %s %s свіжий: %d рядків", symbol, interval, len(cached))
            return cached
    logger.info("Завантаження spot klines %s %s за %d днів", symbol, interval, days)
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
    ).spot_klines(symbol, interval, days, force=force)
