"""Завантаження історичних aggTrades з data.binance.vision (безкоштовні дампи).

Binance REST обмежує aggTrades останніми ~2 добами, тому для довгої
CVD-історії використовуємо офіційні архіви:
    https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM-DD.zip
    https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-YYYY-MM.zip

Формат CSV (без заголовка): agg_trade_id, price, quantity, first_trade_id,
last_trade_id, transact_time (ms), is_buyer_maker.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from scalper_hft.config import get_settings
from scalper_hft.data.storage import save_trades, trades_path

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


def _parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            # архіви мають заголовок; перевіряємо і пропускаємо при потребі
            raw = pd.read_csv(f)
            if "transact_time" in raw.columns:
                df = raw
            else:
                raw = pd.read_csv(io.BytesIO(content), header=None, names=_COLUMNS)
                df = raw
    df["ts"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["side"] = df["is_buyer_maker"].map({True: "sell", False: "buy"})
    df["price"] = df["price"].astype(float)
    df["amount"] = df["quantity"].astype(float)
    return df.set_index("ts")[["price", "amount", "side"]].sort_index()


def download_agg_trades_vision(
    symbol: str,
    start: date,
    end: date | None = None,
    freq: str = "daily",
) -> pd.DataFrame:
    """Завантажити aggTrades з архівів Binance за [start, end] і зберегти в кеш.

    freq: 'daily' (файл на день) або 'monthly' (файл на місяць).
    """
    settings = get_settings()
    path = trades_path(settings.data_dir_abs, symbol)
    frames: list[pd.DataFrame] = []
    end = end or date.today()

    current = start
    while current <= end:
        url = _url_for(symbol, current, freq)
        content = _download_zip(url)
        if content is None:
            logger.info("Файл не знайдено (404): %s", url)
            if freq == "monthly":
                # пропускаємо місяць цілком
                if current.month == 12:
                    current = date(current.year + 1, 1, 1)
                else:
                    current = date(current.year, current.month + 1, 1)
                continue
        else:
            df = _parse_zip(content)
            frames.append(df)
            logger.info("%s: %d трейдів", url.split("/")[-1], len(df))
            if freq == "daily":
                current += timedelta(days=1)
            else:
                if current.month == 12:
                    current = date(current.year + 1, 1, 1)
                else:
                    current = date(current.year, current.month + 1, 1)
        time.sleep(0.3)  # ввічливість до архіву

    if not frames:
        raise FileNotFoundError(f"Жодного файлу не завантажено для {symbol} з {start} по {end}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # злиття з наявним кешем (REST-частина, останні 2 доби)
    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index)
        except Exception:  # noqa: BLE001
            existing = None
    if existing is not None and not existing.empty:
        out = pd.concat([out, existing[["price", "amount", "side"]]]).sort_index()
        out = out[~out.index.duplicated(keep="last")]

    save_trades(path, out)
    return out
