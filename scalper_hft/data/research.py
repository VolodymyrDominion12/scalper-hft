"""Єдине завантаження research-даних: klines, trades, funding, L2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.validate import (
    BarQualityReport,
    StreamQualityReport,
    validate_bars,
    validate_bookticker,
    validate_depth,
    validate_funding,
    validate_trades,
)
from scalper_hft.strategies.base import Strategy


@dataclass
class MarketDataBundle:
    klines: pd.DataFrame
    trades: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None
    book: pd.DataFrame | None = None
    depth: pd.DataFrame | None = None
    quality: BarQualityReport | None = None
    quality_trades: StreamQualityReport | None = None
    quality_funding: StreamQualityReport | None = None
    quality_book: StreamQualityReport | None = None
    quality_depth: StreamQualityReport | None = None


def load_bookticker(data_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = data_dir / f"{symbol}_bookTicker.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        df = df.set_index("ts")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_depth5(data_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = data_dir / f"{symbol}_depth5.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        df = df.set_index("ts")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def attach_imbalance(klines: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """Додає колонки mid/imbalance з bookTicker на індекс klines (asof)."""
    out = klines.copy()
    if book is None or book.empty or "bid" not in book.columns:
        return out
    b = book.copy()
    mid = (b["bid"] + b["ask"]) / 2.0
    den = b["bid_qty"] + b["ask_qty"]
    imb = ((b["bid_qty"] - b["ask_qty"]) / den.replace(0, pd.NA)).astype(float)
    aux = pd.DataFrame({"mid": mid, "imbalance": imb}, index=b.index).sort_index()
    aligned = aux.reindex(out.index, method="ffill")
    out["imbalance"] = aligned["imbalance"]
    out["mid"] = aligned["mid"]
    return out


def load_research_data(
    symbol: str,
    interval: str,
    days: int,
    strategy: Strategy | None = None,
    *,
    force: bool = False,
    load_l2: bool = True,
    validate: bool = True,
    base: str = "1m",
    derive: bool = True,
    exchange_id: str | None = None,
) -> MarketDataBundle:
    """Один вхід для CLI/dashboard: klines + опційно trades/funding/L2.

    Старші таймфрейми за замовчуванням ресемпляться з `base` (1m) і не
    записуються в кеш — джерело істини лишається хвилинний ряд.
    """
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_agg_trades, download_funding

    klines = ensure_klines(
        symbol, interval, days, base_interval=base, derive=derive, force=force, exchange_id=exchange_id
    )
    trades = None
    funding = None
    if strategy is not None and getattr(strategy, "needs_trades", False):
        trades = download_agg_trades(symbol, days, force=force, exchange_id=exchange_id)
    if strategy is not None and getattr(strategy, "needs_funding", False):
        funding = download_funding(symbol, days, force=force, exchange_id=exchange_id)

    settings = get_settings()
    book = load_bookticker(settings.data_dir_abs, symbol) if load_l2 else None
    depth = load_depth5(settings.data_dir_abs, symbol) if load_l2 else None
    if book is not None and not book.empty:
        klines = attach_imbalance(klines, book)

    quality = validate_bars(klines, interval=interval) if validate else None
    quality_trades = validate_trades(trades) if validate and trades is not None else None
    quality_funding = validate_funding(funding) if validate and funding is not None else None
    quality_book = validate_bookticker(book) if validate and book is not None else None
    quality_depth = validate_depth(depth) if validate and depth is not None else None
    return MarketDataBundle(
        klines=klines,
        trades=trades,
        funding=funding,
        book=book,
        depth=depth,
        quality=quality,
        quality_trades=quality_trades,
        quality_funding=quality_funding,
        quality_book=quality_book,
        quality_depth=quality_depth,
    )
