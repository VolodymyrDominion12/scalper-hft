"""Шар даних: завантаження та кешування історичних даних Binance USDT-M."""

from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.data.downloader import Downloader, download_agg_trades, download_funding, download_klines
from scalper_hft.data.research import MarketDataBundle, load_research_data
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
from scalper_hft.data.validate import BarQualityReport, validate_bars

__all__ = [
    "BinanceClient",
    "Downloader",
    "download_klines",
    "download_agg_trades",
    "download_funding",
    "load_research_data",
    "MarketDataBundle",
    "validate_bars",
    "BarQualityReport",
    "klines_path",
    "trades_path",
    "funding_path",
    "load_klines",
    "load_trades",
    "load_funding",
    "save_klines",
    "save_trades",
    "save_funding",
]
