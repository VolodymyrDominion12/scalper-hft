"""Завантажувач Level 2 (стакан) даних з використанням Tardis.dev або Binance Vision.

Цей модуль відповідає за отримання, кешування та нормалізацію tick-level
та L2 даних, які є критичними для HFT маркет-мейкінгу та мікроструктурного аналізу.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

async def download_l2_snapshots(
    symbol: str, 
    start_date: str, 
    end_date: str, 
    exchange: str = "binance-futures",
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Завантажує L2 снапшоти з Tardis.dev.
    
    Args:
        symbol: Торговий інструмент (напр., 'BTCUSDT')
        start_date: Дата початку у форматі 'YYYY-MM-DD'
        end_date: Дата кінця у форматі 'YYYY-MM-DD'
        exchange: Біржа (дефолт 'binance-futures')
        cache_dir: Директорія для кешування (дефолт data/l2)
    """
    logger.info(f"Downloading L2 snapshots for {symbol} from {start_date} to {end_date}...")
    # TODO: Implement Tardis API client integration here
    # 1. Check local cache (parquet)
    # 2. Fetch from tardis-dev library
    # 3. Normalize into standard format (timestamp, bids[], asks[])
    # 4. Save to parquet
    
    raise NotImplementedError("Tardis.dev integration is pending API key configuration.")
