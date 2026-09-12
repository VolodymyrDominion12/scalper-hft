"""Завантаження історичних aggTrades з data.binance.vision (безкоштовні дампи).

Binance REST обмежує aggTrades останніми ~2 добами, тому для довгої
CVD-історії використовуємо офіційні архіви:
    https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM-DD.zip
    https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM.zip

Формат CSV (без заголовка): agg_trade_id, price, quantity, first_trade_id,
last_trade_id, transact_time (ms), is_buyer_maker.

⚠ Klines USDT-M ф'ючерсів у архівах НЕ публікуються (тільки aggTrades,
bookTicker, metrics) — klines завантажуються через REST (downloader).
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.binance.vision/data/futures/um"
_HEADERS = {"User-Agent": "scalper-hft/0.1"}
_COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
_MIN_COMPLETE_DAY = pd.Timedelta(hours=20)
# День вважається повним лише якщо в ньому є майже всі aggTrade id (крок 1).
# Сама лише довжина проміжку не годиться: кеш BTCUSDT після дедупу за
# мілісекундним індексом мав покриття 39.3%, але кожен день «виглядав повним»
# за часом — тому vision пропускав би його і втрата лишалась би назавжди.
_MIN_COMPLETE_COVERAGE = 0.99


def _has_full_id_coverage(day_ids: pd.Series) -> bool:
    """Чи покриває день aggTrade id майже без розривів.

    Синтетичні (від'ємні) або відсутні id оцінити неможливо — повертаємо True,
    щоб не перекачувати день щоразу (інакше цикл докачки ніколи не завершиться).
    """
    ids = pd.to_numeric(day_ids, errors="coerce")
    real = ids[ids > 0]
    if len(real) < 2:
        return True
    span = int(real.max() - real.min()) + 1
    if span <= 0:
        return True
    return len(real) / span >= _MIN_COMPLETE_COVERAGE


def _url_for(symbol: str, d: date, freq: str) -> str:
    if freq == "daily":
        fname = f"{symbol}-aggTrades-{d.isoformat()}.zip"
        folder = "daily"
    else:
        fname = f"{symbol}-aggTrades-{d.year:04d}-{d.month:02d}.zip"
        folder = "monthly"
    return f"{_BASE_URL}/{folder}/aggTrades/{symbol}/{fname}"


def _download_zip(url: str, retries: int = 3, timeout: int = 120) -> bytes | None:
    """Завантажити zip; None якщо 404 (файл ще не існує)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            logger.warning("Спроба %d/%d для %s: %s", attempt + 1, retries, url, exc)
            time.sleep(1.5 * (attempt + 1))
    return None


def _parse_zip(content: bytes) -> pd.DataFrame:
    """Розібрати vision-архів. Підтримує обидва формати: з заголовком і без.

    Старі дампи Binance публікуються БЕЗ заголовка (див. docstring модуля), і
    резервна гілка читала... сам ZIP: `pd.read_csv(io.BytesIO(content))` отримував
    стиснуті байти замість розпакованого CSV → `UnicodeDecodeError` і падіння
    всього завантаження символу. Тепер перечитуємо той самий член архіву.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = pd.read_csv(f)
            if "transact_time" not in raw.columns:
                f.seek(0)
                raw = pd.read_csv(f, header=None, names=_COLUMNS)
                raw["agg_trade_id"] = raw["agg_trade_id"].astype("int64")
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["trade_id"] = pd.to_numeric(df["agg_trade_id"], errors="coerce").fillna(0).astype("int64")
    df["side"] = df["is_buyer_maker"].map({True: "sell", False: "buy"})
    df["price"] = df["price"].astype(float)
    df["amount"] = df["quantity"].astype(float)
    return df.set_index("ts")[["trade_id", "price", "amount", "side"]].sort_index()


def _advance_period(d: date, freq: str) -> date:
    if freq == "monthly":
        if d.month == 12:
            return date(d.year + 1, 1, 1)
        return date(d.year, d.month + 1, 1)
    return d + timedelta(days=1)


def complete_trade_days(existing: pd.DataFrame | None, today: date) -> set[date]:
    """UTC-дні з кешу, які вже виглядають повними (span ≥ 20 год і всі id, не today).

    Обидві умови обов'язкові: довгий часовий проміжок сам по собі не означає
    повноти — кеш з втраченими aggTrades може мати повний span і 39% угод.
    """
    if existing is None or existing.empty:
        return set()
    idx = existing.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    ids = existing["trade_id"] if "trade_id" in existing.columns else None
    out: set[date] = set()
    for day_ts, stamps in idx.to_series().groupby(idx.floor("D")):
        day = pd.Timestamp(day_ts).date()
        if day >= today:
            continue
        if stamps.iloc[-1] - stamps.iloc[0] < _MIN_COMPLETE_DAY:
            continue
        if ids is not None and not _has_full_id_coverage(ids.loc[stamps.index]):
            logger.info("vision aggTrades: день %s неповний за aggTrade id — перекачаю", day)
            continue
        out.add(day)
    return out


def _month_days_needed(month_start: date, end: date, today: date) -> set[date]:
    start = date(month_start.year, month_start.month, 1)
    if start.month == 12:
        next_m = date(start.year + 1, 1, 1)
    else:
        next_m = date(start.year, start.month + 1, 1)
    last = min(end, today - timedelta(days=1), next_m - timedelta(days=1))
    if last < start:
        return set()
    days: set[date] = set()
    current = start
    while current <= last:
        days.add(current)
        current += timedelta(days=1)
    return days


def missing_vision_periods(
    existing: pd.DataFrame | None,
    start: date,
    end: date,
    freq: str,
    today: date,
) -> list[date]:
    """Дати/місяці архівів, яких немає в кеші (повні дні пропускаються)."""
    covered = complete_trade_days(existing, today)
    missing: list[date] = []
    current = start
    while current <= end:
        if freq == "monthly":
            needed = _month_days_needed(current, end, today)
            if needed and not needed.issubset(covered):
                missing.append(current)
        elif current not in covered:
            missing.append(current)
        current = _advance_period(current, freq)
    return missing


def download_agg_trades_vision(
    symbol: str,
    start: date,
    end: date | None = None,
    freq: str = "daily",
    *,
    checkpoint_every: int = 5,
) -> pd.DataFrame:
    """Завантажити aggTrades з архівів Binance за [start, end] і зберегти в кеш.

    freq: 'daily' (файл на день) або 'monthly' (файл на місяць).
    Повні календарні дні в кеші не перекачуються.

    checkpoint_every: після скількох архівів зливати накопичене в кеш.
    Раніше функція збирала ВСІ архіви в пам'ять і викликала `save_trades` один
    раз у самому кінці — тож обрив (Ctrl-C, OOM, битий архів) втрачав усе
    завантажене для цього символу. Тепер кожні N архівів дані вже на диску, і
    повторний запуск продовжує з місця обриву (повні дні пропускаються
    `missing_vision_periods`). `0` = стара поведінка (один запис у кінці).
    """
    from scalper_hft.data.storage import dedupe_trades
    from scalper_hft.data.store import get_store

    store = get_store()
    existing = store.load_trades(symbol)
    end = end or date.today()
    today = date.today()
    periods = missing_vision_periods(existing, start, end, freq, today)
    if existing is not None and not existing.empty:
        have = f"{existing.index[0].date()} … {existing.index[-1].date()} ({len(existing)} трейдів)"
    else:
        have = "порожній"
    preview = ", ".join(p.isoformat() for p in periods[:12])
    if len(periods) > 12:
        preview += "…"
    logger.info(
        "vision aggTrades %s: джерело Binance Vision Archives [LIVE (НЕ testnet ✓)] (%s) | у кеші %s; качаю %d %s періодів%s",
        symbol,
        _BASE_URL,
        have,
        len(periods),
        freq,
        f" ({preview})" if preview else "",
    )

    def _flush(batch: list[pd.DataFrame]) -> int:
        """Злити batch з поточним кешем і зберегти. Повертає кількість рядків.

        Кеш перечитується з диска, а не тримається в пам'яті: інакше checkpoint
        не обмежував би споживання RAM.
        """
        current = store.load_trades(symbol)
        parts = list(batch)
        if current is not None and not current.empty:
            parts.append(current[["trade_id", "price", "amount", "side"]])
        merged = dedupe_trades(pd.concat(parts))
        store.save_trades(symbol, merged)
        return len(merged)

    pending: list[pd.DataFrame] = []
    n_done = 0
    rows_saved = 0
    for current in periods:
        url = _url_for(symbol, current, freq)
        content = _download_zip(url)
        if content is None:
            logger.info("Файл не знайдено (404): %s", url)
            continue
        df = _parse_zip(content)
        pending.append(df)
        n_done += 1
        logger.info("%s: %d трейдів", url.split("/")[-1], len(df))
        if checkpoint_every > 0 and n_done % checkpoint_every == 0:
            rows_saved = _flush(pending)
            pending = []
            logger.info(
                "vision aggTrades %s: checkpoint після %d архівів — у кеші %d трейдів",
                symbol,
                n_done,
                rows_saved,
            )
        time.sleep(0.3)  # ввічливість до архіву

    # Дедуп за trade_id, а не за мілісекундним індексом (див. storage.dedupe_trades):
    # у дампах Binance Vision кілька aggTrades регулярно ділять одну мілісекунду,
    # і дедуп за індексом знищував би більшість потоку на активних символах.
    if pending:
        rows_saved = _flush(pending)

    if n_done == 0:
        if existing is not None and not existing.empty:
            logger.info("vision aggTrades %s: нічого докачувати", symbol)
            return existing
        raise FileNotFoundError(f"Жодного файлу не завантажено для {symbol} з {start} по {end}")

    out = store.load_trades(symbol)
    if out is None:
        raise FileNotFoundError(f"vision aggTrades {symbol}: кеш порожній після завантаження")
    logger.info("vision aggTrades %s: у кеші %d трейдів", symbol, len(out))
    return out


def _url_for_liquidations(symbol: str, d: date, freq: str) -> str:
    if freq == "daily":
        fname = f"{symbol}-liquidationSnapshot-{d.isoformat()}.zip"
        folder = "daily"
    else:
        fname = f"{symbol}-liquidationSnapshot-{d.year:04d}-{d.month:02d}.zip"
        folder = "monthly"
    return f"{_BASE_URL}/{folder}/liquidationSnapshot/{symbol}/{fname}"


def _parse_liquidations_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = pd.read_csv(f)
            if "time" not in raw.columns:
                # sometimes header is missing, we try to guess based on standard format
                # time,side,order_type,time_in_force,original_quantity,price,average_price,order_status,last_fill_quantity,accumulated_fill_quantity
                raw = pd.read_csv(
                    io.BytesIO(content),
                    header=None,
                    names=[
                        "time",
                        "side",
                        "order_type",
                        "time_in_force",
                        "original_quantity",
                        "price",
                        "average_price",
                        "order_status",
                        "last_fill_quantity",
                        "accumulated_fill_quantity",
                    ],
                )
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["time"], unit="ms")
    df["side"] = df["side"].str.lower()
    df["price"] = df["price"].astype(float)
    df["qty"] = df["original_quantity"].astype(float)
    return df.set_index("ts")[["price", "qty", "side"]].sort_index()


def download_liquidations_vision(
    symbol: str,
    start: date,
    end: date | None = None,
    freq: str = "daily",
) -> pd.DataFrame:
    """Завантажити ліквідації з архівів Binance за [start, end] і зберегти в кеш."""
    from scalper_hft.data.store import get_store

    store = get_store()
    existing = store.load_liquidations(symbol)
    end = end or date.today()
    today = date.today()
    periods = missing_vision_periods(existing, start, end, freq, today)
    if existing is not None and not existing.empty:
        have = f"{existing.index[0].date()} … {existing.index[-1].date()} ({len(existing)} записів)"
    else:
        have = "порожній"
    preview = ", ".join(p.isoformat() for p in periods[:12])
    if len(periods) > 12:
        preview += "…"
    logger.info(
        "vision liquidations %s: джерело Binance Vision Archives [LIVE (НЕ testnet ✓)] (%s) | у кеші %s; качаю %d %s періодів%s",
        symbol,
        _BASE_URL,
        have,
        len(periods),
        freq,
        f" ({preview})" if preview else "",
    )

    frames: list[pd.DataFrame] = []
    for current in periods:
        url = _url_for_liquidations(symbol, current, freq)
        content = _download_zip(url)
        if content is None:
            logger.info("Файл не знайдено (404): %s", url)
            continue
        df = _parse_liquidations_zip(content)
        frames.append(df)
        logger.info("%s: %d ліквідацій", url.split("/")[-1], len(df))
        time.sleep(0.3)

    if not frames:
        if existing is not None and not existing.empty:
            logger.info("vision liquidations %s: нічого докачувати", symbol)
            return existing
        return pd.DataFrame(columns=["price", "qty", "side"])

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if existing is not None and not existing.empty:
        out = pd.concat([out, existing[["price", "qty", "side"]]]).sort_index()
        out = out[~out.index.duplicated(keep="last")]

    store.save_liquidations(symbol, out)
    return out
