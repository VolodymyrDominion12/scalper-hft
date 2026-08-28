"""Parquet-кеш даних: klines, aggTrades, funding.

Формат імен файлів:
    data/{symbol}_{interval}_klines.parquet
    data/{symbol}_aggTrades.parquet
    data/{symbol}_funding.parquet
Індекс — datetime64 (UTC). Parquet + zstd — компактно і швидко.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_KLINES_COLUMNS = ["open", "high", "low", "close", "volume"]
_TRADES_COLUMNS = ["price", "amount", "side"]


def _safe_load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося прочитати кеш %s: %s — ігнорую", path, exc)
        return None


def klines_path(data_dir: Path, symbol: str, interval: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_{interval}_klines.parquet"


def trades_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_aggTrades.parquet"


def funding_path(data_dir: Path, symbol: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}_funding.parquet"


def load_klines(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    for col in _KLINES_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[_KLINES_COLUMNS].astype(float)


def load_trades(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    return df[_TRADES_COLUMNS]


def load_funding(path: Path) -> pd.DataFrame | None:
    df = _safe_load(path)
    if df is None or df.empty:
        return None
    return df[["fundingRate"]]


def save_klines(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[_KLINES_COLUMNS].astype(float).to_parquet(path, compression="zstd")
    logger.info("Збережено klines: %s (%d рядків)", path, len(df))


def save_trades(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[_TRADES_COLUMNS].to_parquet(path, compression="zstd")
    logger.info("Збережено aggTrades: %s (%d рядків)", path, len(df))


def save_funding(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["fundingRate"]].to_parquet(path, compression="zstd")
    logger.info("Збережено funding: %s (%d рядків)", path, len(df))
