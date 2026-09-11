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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.client import ExchangeClient
from scalper_hft.data.storage import dedupe_trades
from scalper_hft.data.store import get_store

logger = logging.getLogger(__name__)

_MS = 1_000
_S = 60_000
_H = 3_600_000
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
# REST aggTrades на ф'ючерсах доступні лише за ~2 доби — кеш свіжий, якщо хвіст у цьому вікні.
_AGG_TRADES_REST_DAYS = 2
_COOLDOWN_FILE = ".binance_rest_cooldown"


def _utc_now() -> pd.Timestamp:
    """Поточний час як naive UTC Timestamp (збігається з індексами кешу)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


# Мінімальна частка очікуваних ставок фандінгу у вікні `days`. Нижче — кеш
# обрізаний (напр. testnet віддає лише ~13 міс історії, і funding_carry
# рахується зі 62% відсутніх нарахувань, лишаючись status="ok").
FUNDING_COVERAGE_MIN = 0.90


def funding_coverage_ratio(
    df: pd.DataFrame | None,
    days: int,
    *,
    now: pd.Timestamp | None = None,
) -> float:
    """Частка наявних ставок фандінгу від очікуваних за `days` днів (0..N).

    Крок ставки береться як медіана фактичних інтервалів у вікні (Binance
    переводив частину символів з 8h на 4h, тому «магічна» 8h некоректна).
    `days <= 0` → 1.0 (вікно не задано — не блокуємо).
    """
    if days is None or days <= 0:
        return 1.0
    if df is None or df.empty:
        return 0.0
    now_ts = now if now is not None else _utc_now()
    end = now_ts
    last = df.index[-1]
    if last < now_ts - pd.Timedelta(hours=16):
        # Історичний/тестовий кеш (фіксовані дати, реплей): міряємо вікно від
        # його власного кінця — інакше будь-які дані не «за сьогодні» дають
        # хибний FAIL. Обрізану історію це не маскує: свіжий кеш із testnet
        # закінчується «сьогодні», тому anchor = now і дефіцит початку видно.
        end = last
    window_start = end - pd.Timedelta(days=int(days))
    win = df[df.index >= window_start]
    win = win[win.index <= end]
    if win.empty:
        return 0.0
    step = win.index.to_series().diff().dropna().median() if len(win) > 1 else pd.NaT
    if step is pd.NaT or step is None or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=8)
    expected = max(int(pd.Timedelta(days=int(days)) / step), 1)
    return float(len(win)) / float(expected)


def _require_funding_coverage(
    symbol: str,
    df: pd.DataFrame | None,
    days: int,
    *,
    strict: bool,
    origin: str,
) -> None:
    """Fail-closed: неповне покриття funding → RuntimeError (або warning)."""
    ratio = funding_coverage_ratio(df, days)
    if ratio >= FUNDING_COVERAGE_MIN:
        return
    n = 0 if df is None or df.empty else len(df)
    first = "" if df is None or df.empty else str(df.index[0])
    msg = (
        f"funding {symbol}: покрито лише {ratio:.0%} очікуваних ставок за {days} днів "
        f"(рядків={n}, перша={first or '—'}). Схоже на обрізану історію "
        f"(testnet віддає ~13 міс; потрібен DATA_EXCHANGE=binanceusdm). "
        f"Поріг {FUNDING_COVERAGE_MIN:.0%}."
    )
    if strict:
        raise RuntimeError(f"{msg} [{origin}]")
    logger.warning("%s [%s, non-strict]", msg, origin)


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
        return getattr(idx.as_unit("ms"), "asi8")
    return getattr(idx, "asi8") // 1_000_000


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


def format_ms_windows(windows: list[tuple[int, int]]) -> str:
    """Людиночитний список вікон [start, end) з тривалістю."""
    if not windows:
        return "—"
    parts: list[str] = []
    for start_ms, end_ms in windows:
        start = pd.Timestamp(start_ms, unit="ms")
        end = pd.Timestamp(end_ms, unit="ms")
        parts.append(f"{start} → {end} ({end - start})")
    return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class CacheCoverage:
    """Що вже є в кеші і яких вікон бракує для запиту `--days`."""

    n_rows: int
    have_start: pd.Timestamp | None
    have_end: pd.Timestamp | None
    requested_start: pd.Timestamp
    now: pd.Timestamp
    windows: tuple[tuple[int, int], ...]

    @property
    def complete(self) -> bool:
        return len(self.windows) == 0

    def summary(self, symbol: str, interval: str) -> str:
        need = f"потрібно {self.requested_start} … {self.now}"
        if self.n_rows == 0:
            return f"{symbol} {interval}: кеш порожній; {need}; качаю {format_ms_windows(list(self.windows))}"
        have = f"у кеші {self.have_start} … {self.have_end} ({self.n_rows} барів)"
        if self.complete:
            return f"{symbol} {interval}: {have}; період покрито — нічого докачувати"
        n = len(self.windows)
        noun = "вікно" if n == 1 else "вікон"
        return f"{symbol} {interval}: {have}; {need}; бракує {n} {noun}: {format_ms_windows(list(self.windows))}"


def klines_coverage(
    existing: pd.DataFrame | None,
    days: int,
    interval: str,
    *,
    now: pd.Timestamp | None = None,
) -> CacheCoverage:
    """Інвентар кешу: діапазон наявних барів і вікна, які треба докачати."""
    now_ts = now if now is not None else _utc_now()
    requested_start = now_ts - pd.Timedelta(days=days)
    windows = missing_klines_windows(
        existing,
        _to_ms(requested_start),
        _to_ms(now_ts),
        _interval_ms(interval),
    )
    if existing is None or existing.empty:
        return CacheCoverage(0, None, None, requested_start, now_ts, tuple(windows))
    idx = existing.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    return CacheCoverage(
        len(existing), pd.Timestamp(idx[0]), pd.Timestamp(idx[-1]), requested_start, now_ts, tuple(windows)
    )


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLS)


def _batch_to_ohlcv(batch: list[list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(batch, columns=["ts", *_OHLCV_COLS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def _supports_from_id(client: Any) -> bool:
    """Чи приймає клієнт `fetch_agg_trades(..., from_id=…)`.

    `Downloader(client=…)` — публічний шов для кастомних/тестових клієнтів, тому
    перевіряємо сигнатуру, а не припускаємо.
    """
    import inspect

    try:
        params = inspect.signature(client.fetch_agg_trades).parameters
    except (TypeError, ValueError):
        return False
    if "from_id" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


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


_client: ExchangeClient | None = None


def _default_client(exchange_id: str | None = None) -> ExchangeClient:
    """Спільний клієнт на процес (глобальний кеш для кожного exchange).

    Біржа за замовчуванням — `settings.data_exchange` (live), НЕ `settings.exchange`
    (торгова). Fail-closed: testnet для ринкових даних заборонено
    (`require_live_data_exchange`), бо він віддає синтетичну історію.
    """
    global _client
    from scalper_hft.config import require_live_data_exchange

    settings = get_settings()
    ex = require_live_data_exchange(settings, exchange_id)
    if _client is None or getattr(_client, "exchange_id", None) != ex:
        _client = ExchangeClient(settings.binance_api_key, settings.binance_api_secret, ex)
    return _client


def _has_http_status(combined: str, code: int) -> bool:
    """Чи є HTTP-статус окремим токеном (не підрядок на кшталт 1418 у timestamp)."""
    return re.search(rf"(?:^|[^\d]){code}(?:[^\d]|$)", combined) is not None


def _calculate_backoff(attempt: int, exc: Exception, max_retries: int) -> tuple[float, str]:
    """Обчислити час очікування (секунди) та категорію помилки для логування.

    Категорії:
    1. IP Ban (HTTP 418, 'IP banned', 'banned until', 'IP has been auto-banned'):
       - Якщо є точний timestamp закінчення бану ('banned until 1708935600000'), спимо до нього + 2с.
       - Інакше експоненційно: 60s, 120s, 180s... з джиттером.
       ⚠ Код -1003 сам по собі НЕ є баном: Binance ставить його і на 429
       ("6000 requests per minute"), і на 418. Бан — лише 418 / banned-текст.
    2. Rate Limit (HTTP 429, -1003 без бану, RateLimitExceeded, 'Too many requests'):
       - 10s * (2 ** attempt) + jitter, обмежено 180s (10s, 20s, 40s, 80s, 160s...).
    3. Мережеві помилки (502, 503, 504, NetworkError, RequestTimeout, ExchangeNotAvailable):
       - 2s * (2 ** attempt) + jitter, обмежено 60s (2s, 4s, 8s, 16s, 32s...).
    """
    exc_str = str(exc)
    exc_type = type(exc).__name__
    combined = f"{exc_type} {exc_str}".lower()

    # 1. Справжній IP-ban: HTTP 418 або явний текст бану. Не -1003 окремо.
    is_ip_ban = (
        _has_http_status(combined, 418)
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

    # 2. Rate limit / 429 / -1003 (без бану) / RateLimitExceeded / DDoSProtection
    is_rate_limit = (
        _has_http_status(combined, 429)
        or "-1003" in combined
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


def agg_trades_cache_fresh(cached: pd.DataFrame | None, *, now: pd.Timestamp | None = None) -> bool:
    """Чи REST-хвіст aggTrades вже в кеші (Binance віддає лише ~2 доби)."""
    if cached is None or cached.empty:
        return False
    now_ts = now if now is not None else _utc_now()
    newest = cached.index[-1]
    return bool(newest >= now_ts - pd.Timedelta(days=_AGG_TRADES_REST_DAYS))


def _cooldown_path() -> Path:
    return get_settings().data_dir_abs / _COOLDOWN_FILE


def _wait_shared_cooldown() -> None:
    """Усі процеси шарять одну паузу після 429, щоб не бити IP одночасно."""
    path = _cooldown_path()
    try:
        until = float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    wait = until - time.time()
    if wait > 0:
        logger.warning("Спільна пауза Binance REST: ще %.1f с", wait)
        time.sleep(wait)


def _set_shared_cooldown(seconds: float) -> None:
    """Продовжити спільну паузу, якщо нова межа далі за вже записану."""
    if seconds <= 0:
        return
    until = time.time() + seconds
    path = _cooldown_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = float(path.read_text(encoding="utf-8").strip())
        if existing >= until:
            return
    except (OSError, ValueError):
        pass
    path.write_text(f"{until:.3f}", encoding="utf-8")


@contextmanager
def _exclusive_fetch(name: str) -> Iterator[None]:
    """Міжпроцесний flock: один REST-прохід на ключ (sweep ProcessPool / job workers)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        yield
        return
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]
    path = get_settings().data_dir_abs / f".lock_{safe}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class Downloader:
    """Ітеративне завантаження історії з повторними спробами, батчами та чекпоінтами."""

    def __init__(
        self,
        client: ExchangeClient | None = None,
        retries: int | None = None,
        store: Any = None,
        batch_delay: float | None = None,
        checkpoint_batches: int | None = None,
        exchange_id: str | None = None,
        strict_funding_coverage: bool = True,
    ) -> None:
        settings = get_settings()
        from scalper_hft.config import require_live_data_exchange

        # Fail-closed: покриття funding перевіряється після завантаження.
        self.strict_funding_coverage = bool(strict_funding_coverage)
        # Fail-closed: testnet/sandbox для ринкових даних заборонено (див.
        # `Settings.data_exchange`). `client=` (тести) обходить цю перевірку.
        self.exchange_id = require_live_data_exchange(settings, exchange_id) if client is None else (
            exchange_id or settings.data_exchange
        )
        self.client = client or _default_client(self.exchange_id)
        self.retries = int(retries if retries is not None else settings.download_retries)
        self.batch_delay = float(batch_delay if batch_delay is not None else settings.download_batch_delay)
        self.checkpoint_batches = int(
            checkpoint_batches if checkpoint_batches is not None else settings.download_checkpoint_batches
        )
        self.store = store if store is not None else get_store()

    def describe_source(self, client: Any = None) -> str:
        """Людиночитаний опис джерела даних з підтвердженням live/testnet."""
        c = client or self.client
        if hasattr(c, "describe_source") and callable(c.describe_source):
            return str(c.describe_source())
        is_testnet = "testnet" in str(self.exchange_id).lower()
        status = "УВАГА: TESTNET/SANDBOX ⚠ (синтетичні дані)" if is_testnet else "LIVE (НЕ testnet ✓)"
        client_name = type(c).__name__ if c is not None else "None"
        return f"{self.exchange_id} [{status}] (client={client_name})"

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            _wait_shared_cooldown()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — мережеві помилки різні
                last = exc
                if attempt == self.retries - 1:
                    logger.warning("Спроба %d/%d (остання) не вдалась: %s", attempt + 1, self.retries, exc)
                    break
                sleep_s, category = _calculate_backoff(attempt, exc, self.retries)
                if category in {"ip_ban", "rate_limit"}:
                    _set_shared_cooldown(sleep_s)
                if category == "ip_ban":
                    logger.warning(
                        "Binance IP заблоковано (418): очікую %.1f с (спроба %d/%d) [%s]",
                        sleep_s,
                        attempt + 1,
                        self.retries,
                        exc,
                    )
                elif category == "rate_limit":
                    logger.warning(
                        "Rate limit Binance (429 / -1003): очікую %.1f с (спроба %d/%d) [%s]",
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
        source_desc: str | None = None,
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

        src = source_desc or self.describe_source()
        logger.info("Докачую klines %s %s з джерела %s: %s", symbol, interval, src, format_ms_windows(windows))
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
        save_fn = partial(self.store.save_klines, symbol, interval)
        out = self._extend_klines(
            existing,
            self.client.fetch_klines,
            symbol,
            interval,
            days,
            force=force,
            save_fn=save_fn,
            source_desc=self.describe_source(),
        )
        if not out.empty:
            self.store.save_klines(symbol, interval, out)
        return out

    def spot_klines(self, symbol: str, interval: str, days: int, force: bool = False) -> pd.DataFrame:
        """Спотові klines (для delta-neutral арбітражу) у окремий кеш."""
        existing = self.store.load_spot_klines(symbol, interval)
        spot_client = ExchangeClient(market_type="spot")
        save_fn = partial(self.store.save_spot_klines, symbol, interval)
        out = self._extend_klines(
            existing,
            spot_client.fetch_klines,
            symbol,
            interval,
            days,
            force=force,
            save_fn=save_fn,
            source_desc=self.describe_source(spot_client),
        )
        if not out.empty:
            self.store.save_spot_klines(symbol, interval, out)
        return out

    def oi(self, symbol: str, days: int) -> pd.DataFrame:
        """Історія Open Interest (зазвичай 5m)."""
        existing = self.store.load_oi(symbol)
        needed_from_ms = _to_ms(_utc_now() - pd.Timedelta(days=days))
        if existing is None or existing.empty:
            start_ms = needed_from_ms
        else:
            oldest_ms = _to_ms(existing.index[0])
            newest_ms = _to_ms(existing.index[-1])
            start_ms = needed_from_ms if oldest_ms > needed_from_ms else newest_ms
            logger.info(
                "oi %s: у кеші %s … %s (%d); докачую з %s",
                symbol,
                existing.index[0],
                existing.index[-1],
                len(existing),
                pd.Timestamp(start_ms, unit="ms"),
            )

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        new_frames: list[pd.DataFrame] = []
        since = start_ms
        if _to_ms(_utc_now()) - start_ms > 60_000:
            logger.info(
                "oi %s: докачую з мережі [%s] (від %s)",
                symbol,
                self.describe_source(),
                pd.Timestamp(start_ms, unit="ms"),
            )
        guard = 0

        def _make_merged() -> pd.DataFrame:
            all_f = [*frames, *new_frames]
            if not all_f:
                return pd.DataFrame(columns=["openInterest"])
            merged = pd.concat(all_f).sort_index()
            return merged[~merged.index.duplicated(keep="last")]

        try:
            while True:
                guard += 1
                if guard > 5000:
                    break
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)
                batch = self._with_retry(self.client.fetch_open_interest_history, symbol, "5m", since, limit=500)
                if not batch:
                    break
                df = pd.DataFrame(batch)
                if df.empty:
                    break
                df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("ts")
                new_frames.append(df[["openInterestValue"]])
                since = int(df.index[-1].value // 1_000_000) + 1
                if len(batch) < 500:
                    break
        except BaseException as exc:
            if new_frames:
                try:
                    partial_df = _make_merged()
                    self.store.save_oi(symbol, partial_df)
                    logger.warning(
                        "Збережено проміжний прогрес oi %s (%d записів) перед перериванням: %s",
                        symbol,
                        len(partial_df),
                        exc,
                    )
                except Exception as save_err:
                    logger.error("Не вдалося зберегти чекпоінт oi: %s", save_err)
            raise

        out = _make_merged()
        if not out.empty:
            out.rename(columns={"openInterestValue": "oi"}, inplace=True)
            self.store.save_oi(symbol, out)
        return out

    def agg_trades(self, symbol: str, days: int, start_ms: int | None = None) -> pd.DataFrame:
        """Завантажити історичні агреговані трейди (для CVD).

        ⚠ Binance futures REST обмежує вікно пошуку aggTrades останніми
        ~2 днями (код -4166). Для старіших трейдів — data.binance.vision dumps
        (див. docs/RESEARCH.md). Тут вікно автоматично обрізається до 2 днів.
        """
        existing = self.store.load_trades(symbol)
        end_ms = _to_ms(_utc_now())
        max_window_ms = 2 * 24 * 3600 * 1000  # обмеження Binance: 2 доби
        begin_ms = start_ms if start_ms is not None else _to_ms(_utc_now() - pd.Timedelta(days=days))
        begin_ms = max(begin_ms, end_ms - max_window_ms)
        if existing is not None and not existing.empty:
            last_ms = _to_ms(existing.index[-1])
            if last_ms >= begin_ms:
                begin_ms = last_ms
            logger.info(
                "aggTrades %s: у кеші %s … %s (%d); REST з %s",
                symbol,
                existing.index[0],
                existing.index[-1],
                len(existing),
                pd.Timestamp(begin_ms, unit="ms"),
            )

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        new_frames: list[pd.DataFrame] = []
        since = begin_ms
        # Пагінація за id (fromId), якщо клієнт це підтримує і в кеші вже є
        # справжні aggTrade id — інакше лишається часовий курсор.
        next_from_id: int | None = None
        last_trade_id = -1
        if _supports_from_id(self.client):
            if existing is not None and not existing.empty and (existing["trade_id"] > 0).all():
                last_trade_id = int(existing["trade_id"].max())
                next_from_id = last_trade_id + 1
        else:
            logger.info("aggTrades %s: клієнт без from_id — пагінація за часом", symbol)
        if end_ms - since > 60_000:
            logger.info(
                "aggTrades %s: докачую з REST [%s] (від %s)",
                symbol,
                self.describe_source(),
                pd.Timestamp(since, unit="ms"),
            )
        guard = 0
        total_rows = 0
        synthetic = 0

        def _make_merged() -> pd.DataFrame:
            all_f = [*frames, *new_frames]
            if not all_f:
                return pd.DataFrame(columns=["trade_id", "price", "amount", "side"])
            # Дедуп за trade_id, НЕ за мілісекундним індексом (див. storage.dedupe_trades).
            return dedupe_trades(pd.concat(all_f))

        try:
            while since < end_ms:
                guard += 1
                if guard > 20_000:
                    raise RuntimeError("Забагато батчів aggTrades")
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)
                if next_from_id is not None:
                    batch = self._with_retry(self.client.fetch_agg_trades, symbol, None, from_id=next_from_id)
                else:
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
                # Курсор: за id, якщо API віддав справжні aggTrade id. Пагінація за
                # часом (`last_ts + 1`) пропускає угоди, що ділять мілісекунду на
                # межі батча, — саме так кеш BTCUSDT втратив ~60% потоку.
                batch_ids = df["trade_id"].to_numpy(dtype=np.int64)
                if (batch_ids > 0).all():
                    max_id = int(batch_ids.max())
                    if max_id <= last_trade_id:
                        break  # API не просунувся — не зациклюємось
                    last_trade_id = max_id
                    next_from_id = max_id + 1  # fromId ІНКЛЮЗИВНО
                last_ts_ms = int(df.index[-1].value // 1_000_000)
                since = last_ts_ms + 1
                if df.index[-1] >= pd.Timestamp(end_ms, unit="ms"):
                    break
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
        needed_from_ms = _to_ms(_utc_now() - pd.Timedelta(days=days))
        if existing is None or existing.empty:
            start_ms = needed_from_ms
        else:
            oldest_ms = _to_ms(existing.index[0])
            newest_ms = _to_ms(existing.index[-1])
            start_ms = needed_from_ms if oldest_ms > needed_from_ms else newest_ms
            logger.info(
                "funding %s: у кеші %s … %s (%d); докачую з %s",
                symbol,
                existing.index[0],
                existing.index[-1],
                len(existing),
                pd.Timestamp(start_ms, unit="ms"),
            )

        frames: list[pd.DataFrame] = [existing] if existing is not None and not existing.empty else []
        new_frames: list[pd.DataFrame] = []
        since = start_ms
        if _to_ms(_utc_now()) - start_ms > 60_000:
            logger.info(
                "funding %s: докачую з REST [%s] (від %s)",
                symbol,
                self.describe_source(),
                pd.Timestamp(start_ms, unit="ms"),
            )
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
        # Fail-closed: обрізана історія (testnet ~13 міс) робить funding_carry
        # беззмістовним, але раніше зберігалась як валідний кеш.
        _require_funding_coverage(symbol, out, days, strict=self.strict_funding_coverage, origin="downloader")
        if not out.empty:
            self.store.save_funding(symbol, out)
        return out


# ── зручні функції верхнього рівня ───────────────────────────────────────────
def download_klines(
    symbol: str,
    interval: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Завантажити/оновити klines. Старі бари не видаляються: докачуються
    лише префікс, внутрішні дірки та застарілий хвіст."""
    store = get_store()
    cached = store.load_klines(symbol, interval)
    coverage = klines_coverage(cached, days, interval)
    logger.info("%s", coverage.summary(symbol, interval))
    if coverage.complete and not force:
        return cached if cached is not None and not cached.empty else _empty_ohlcv()
    if force and coverage.complete:
        logger.info("force: оновлюю хвіст %s %s з джерела %s", symbol, interval, exchange_id or getattr(get_settings(), "data_exchange", "binanceusdm"))
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
        exchange_id=exchange_id,
    ).klines(symbol, interval, days, force=force)


def download_agg_trades(
    symbol: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    store = get_store()
    cached = None if force else store.load_trades(symbol)
    if not force and agg_trades_cache_fresh(cached):
        assert cached is not None
        logger.info("Кеш aggTrades %s актуальний (до %s): %d рядків", symbol, cached.index[-1], len(cached))
        return cached
    if cached is not None and not cached.empty:
        logger.info("Оновлення aggTrades %s: було %d рядків до %s", symbol, len(cached), cached.index[-1])
    logger.info("Завантаження aggTrades %s за %d днів (джерело: %s)", symbol, days, exchange_id or getattr(get_settings(), "data_exchange", "binanceusdm"))
    with _exclusive_fetch(f"aggtrades_{symbol}"):
        if not force:
            cached = store.load_trades(symbol)
            if agg_trades_cache_fresh(cached):
                assert cached is not None
                logger.info(
                    "Кеш aggTrades %s заповнений іншим воркером (до %s): %d рядків",
                    symbol,
                    cached.index[-1],
                    len(cached),
                )
                return cached
        return Downloader(
            store=store,
            retries=retries,
            batch_delay=batch_delay,
            checkpoint_batches=checkpoint_batches,
            exchange_id=exchange_id,
        ).agg_trades(symbol, days)


def download_funding(
    symbol: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
    exchange_id: str | None = None,
    strict_coverage: bool = True,
) -> pd.DataFrame:
    """Кеш фандінгу: свіжий лише якщо покриває період і остання ставка < 16 год.

    Binance USDT-M нараховує фандінг кожні 8 год; 2 періоди без оновлення = stale.
    `strict_coverage=True` (дефолт) — fail-closed на обрізаній історії:
    без цього `funding_carry` мовчки рахується з відсутніми нарахуваннями.
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
            _require_funding_coverage(symbol, cached, days, strict=strict_coverage, origin="cache")
            logger.info("Кеш funding %s покриває період і свіжий: %d рядків (до %s)", symbol, len(cached), newest)
            return cached
        logger.info("Оновлення funding %s: %d рядків (до %s, stale=%s)", symbol, len(cached), newest, stale)
    logger.info("Завантаження funding %s за %d днів (джерело: %s)", symbol, days, exchange_id or getattr(get_settings(), "data_exchange", "binanceusdm"))
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
        exchange_id=exchange_id,
        strict_funding_coverage=strict_coverage,
    ).funding(symbol, days)


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
    cached = store.load_spot_klines(symbol, interval)
    coverage = klines_coverage(cached, days, interval)
    logger.info("spot %s", coverage.summary(symbol, interval))
    if coverage.complete and not force:
        return cached if cached is not None and not cached.empty else _empty_ohlcv()
    if force and coverage.complete:
        logger.info("force: оновлюю хвіст spot %s %s з джерела binance spot [LIVE - НЕ testnet ✓]", symbol, interval)
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
    ).spot_klines(symbol, interval, days, force=force)


def download_oi(
    symbol: str,
    days: int,
    force: bool = False,
    retries: int | None = None,
    batch_delay: float | None = None,
    checkpoint_batches: int | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    store = get_store()
    logger.info("Завантаження oi %s за %d днів (джерело: %s)", symbol, days, exchange_id or getattr(get_settings(), "data_exchange", "binanceusdm"))
    return Downloader(
        store=store,
        retries=retries,
        batch_delay=batch_delay,
        checkpoint_batches=checkpoint_batches,
        exchange_id=exchange_id,
    ).oi(symbol, days)


def download_liquidations(
    symbol: str,
    days: int,
) -> pd.DataFrame:
    from datetime import date, timedelta

    from scalper_hft.data.binance_vision import download_liquidations_vision

    logger.info(
        "Завантаження ліквідацій %s за %d днів з офіційного архіву Binance Vision [LIVE - НЕ testnet ✓] (https://data.binance.vision/data/futures/um)",
        symbol,
        days,
    )
    start = date.today() - timedelta(days=days)
    return download_liquidations_vision(symbol, start=start, freq="daily")

