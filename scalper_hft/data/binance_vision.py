"""Завантаження історичних даних з data.binance.vision (безкоштовні дампи).

Binance REST обмежує aggTrades останніми ~2 добами, тому для довгої
CVD-історії використовуємо офіційні архіви:
    https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM-DD.zip
    https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM.zip

Формат CSV (без заголовка): agg_trade_id, price, quantity, first_trade_id,
last_trade_id, transact_time (ms), is_buyer_maker.

Klines-дампи (той самий CDN, 1 запит на місяць замість ~44 REST-батчів):
    https://data.binance.vision/data/futures/um/{daily|monthly}/klines/{SYMBOL}/{SYMBOL}-{interval}-YYYY-MM[-DD].zip
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


# ── klines ────────────────────────────────────────────────────────────────────
def _klines_url_for(symbol: str, interval: str, d: date, freq: str) -> str:
    if freq == "daily":
        fname = f"{symbol}-{interval}-{d.isoformat()}.zip"
        folder = "daily"
    else:
        fname = f"{symbol}-{interval}-{d.year:04d}-{d.month:02d}.zip"
        folder = "monthly"
    return f"{_BASE_URL}/{folder}/klines/{symbol}/{fname}"


_KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _parse_klines_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = pd.read_csv(f)
            if "open_time" not in raw.columns:
                raw = pd.read_csv(io.BytesIO(content), header=None, names=_KLINES_COLUMNS)
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    out = df.set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return out[~out.index.duplicated(keep="last")]


def download_klines_vision(
    symbol: str,
    interval: str,
    start: date,
    end: date | None = None,
    *,
    freq: str = "monthly",
    topup_days: int = 1,
) -> pd.DataFrame:
    """Завантажити klines з архівів Binance за [start, end] і зберегти в кеш.

    Спершу місячні дампи (1 запит на місяць), потім — денні для неповного
    поточного місяця. Останні `topup_days` днів архіви можуть не мати
    (публікуються із затримкою) — їх дотягне REST (downloader).
    """
    frames: list[pd.DataFrame] = []
    end = end or date.today() - timedelta(days=topup_days)

    # місячні дампи для повних місяців
    current = date(start.year, start.month, 1)
    while current <= end:
        if current.month == end.month and current.year == end.year:
            break  # поточний (неповний) місяць — денними
        url = _klines_url_for(symbol, interval, current, "monthly")
        content = _download_zip(url)
        if content is not None:
            df = _parse_klines_zip(content)
            frames.append(df)
            logger.info("%s: %d барів", url.split("/")[-1], len(df))
        else:
            logger.info("Файл не знайдено (404): %s", url)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
        time.sleep(0.3)

    # денні дампи для неповного місяця
    day = max(start, date(end.year, end.month, 1))
    while day <= end:
        url = _klines_url_for(symbol, interval, day, "daily")
        content = _download_zip(url)
        if content is not None:
            df = _parse_klines_zip(content)
            frames.append(df)
            logger.info("%s: %d барів", url.split("/")[-1], len(df))
        else:
            logger.info("Файл не знайдено (404): %s", url)
        day += timedelta(days=1)
        time.sleep(0.3)

    if not frames:
        raise FileNotFoundError(f"Жодного klines-файлу для {symbol} {interval} з {start} по {end}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    from scalper_hft.data.store import get_store

    store = get_store()
    existing = store.load_klines(symbol, interval)
    if existing is not None and not existing.empty:
        out = pd.concat([out, existing]).sort_index()
        out = out[~out.index.duplicated(keep="last")]

    store.save_klines(symbol, interval, out)
    return out


def _parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            # архіви мають заголовок; перевіряємо і пропускаємо при потребі
            raw = pd.read_csv(f)
            if "transact_time" not in raw.columns:
                raw = pd.read_csv(io.BytesIO(content), header=None, names=_COLUMNS)
                raw["agg_trade_id"] = raw["agg_trade_id"].astype("int64")
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["trade_id"] = pd.to_numeric(df["agg_trade_id"], errors="coerce").fillna(0).astype("int64")
    df["side"] = df["is_buyer_maker"].map({True: "sell", False: "buy"})
    df["price"] = df["price"].astype(float)
    df["amount"] = df["quantity"].astype(float)
    return df.set_index("ts")[["trade_id", "price", "amount", "side"]].sort_index()


def download_agg_trades_vision(
    symbol: str,
    start: date,
    end: date | None = None,
    freq: str = "daily",
) -> pd.DataFrame:
    """Завантажити aggTrades з архівів Binance за [start, end] і зберегти в кеш.

    freq: 'daily' (файл на день) або 'monthly' (файл на місяць).
    """
    frames: list[pd.DataFrame] = []
    end = end or date.today()

    def _advance(d: date, freq: str) -> date:
        if freq == "monthly":
            if d.month == 12:
                return date(d.year + 1, 1, 1)
            return date(d.year, d.month + 1, 1)
        return d + timedelta(days=1)

    current = start
    while current <= end:
        url = _url_for(symbol, current, freq)
        content = _download_zip(url)
        if content is None:
            logger.info("Файл не знайдено (404): %s", url)
            current = _advance(current, freq)
            continue
        df = _parse_zip(content)
        frames.append(df)
        logger.info("%s: %d трейдів", url.split("/")[-1], len(df))
        current = _advance(current, freq)
        time.sleep(0.3)  # ввічливість до архіву

    if not frames:
        raise FileNotFoundError(f"Жодного файлу не завантажено для {symbol} з {start} по {end}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # злиття з наявним кешем (REST-частина, останні 2 доби)
    from scalper_hft.data.store import get_store

    store = get_store()
    existing = store.load_trades(symbol)
    if existing is not None and not existing.empty:
        out = pd.concat([out, existing[["trade_id", "price", "amount", "side"]]]).sort_index()
        out = out[~out.index.duplicated(keep="last")]

    store.save_trades(symbol, out)
    return out
