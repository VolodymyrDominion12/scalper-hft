"""Рекордер bookTicker (best bid/ask + об'єми) у реальному часі → parquet.

Призначення: накопичення даних стакана для OB-стратегій (order book imbalance,
spread dynamics). Історичні bookTicker НЕ доступні безкоштовно — записуємо самі.

Стрім: wss://fstream.binance.com/ws/{symbol}@bookTicker
Повідомлення: {"u":..,"s":"BTCUSDT","b":"bid","B":"bid_qty","a":"ask","A":"ask_qty"}

Використання:
    python -m scalper_hft.cli record-bookticker --symbol BTCUSDT --minutes 60
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore

    _HAS_WS = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore
    _HAS_WS = False

_WS_URL = "wss://fstream.binance.com/ws/{}@bookTicker"


async def _record_symbol(
    symbol: str,
    out_path: Path,
    duration_sec: int,
    flush_every: int = 500,
) -> int:
    """Записує bookTicker у parquet (батчами). Повертає кількість записів."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not _HAS_WS:
        raise ImportError("Встановіть websockets: uv pip install websockets")

    rows: list[dict] = []
    written = 0
    start = time.monotonic()
    url = _WS_URL.format(symbol.lower())
    logger.info("Підключення до %s на %d сек", url, duration_sec)

    try:
        async with websockets.connect(url, ping_interval=20) as ws:
            while time.monotonic() - start < duration_sec:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Таймаут стріму — продовжую")
                    continue
                except websockets.ConnectionClosed:
                    logger.warning("Стрім закрито сервером — перепідключення")
                    break
                import json

                data = json.loads(msg)
                rows.append(
                    {
                        "ts": pd.Timestamp.now(tz=None),
                        "bid": float(data["b"]),
                        "bid_qty": float(data["B"]),
                        "ask": float(data["a"]),
                        "ask_qty": float(data["A"]),
                    }
                )
                if len(rows) >= flush_every:
                    _flush(rows, out_path)
                    written += len(rows)
                    rows = []
                    logger.info("bookTicker %s: %d записів", symbol, written)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Помилка рекордингу: %s", exc)

    if rows:
        _flush(rows, out_path)
        written += len(rows)
    logger.info("Рекординг завершено: %d записів → %s", written, out_path)
    return written


def _flush(rows: list[dict], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        df = pd.concat([existing, df])
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")


def record_bookticker(symbol: str, minutes: int = 60, data_dir: Path | None = None) -> int:
    """Синхронна обгортка: запис bookTicker протягом `minutes` хвилин."""
    from scalper_hft.config import get_settings

    settings = get_settings()
    out_dir = data_dir or settings.data_dir_abs
    out_path = out_dir / f"{symbol}_bookTicker.parquet"
    return asyncio.run(_record_symbol(symbol, out_path, minutes * 60))
