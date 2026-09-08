"""Рекордер макро-даних (ліквідації та відкритий інтерес) у реальному часі.

Призначення: накопичення даних для RiskGate та створення фіч (oi_delta, liquidation_cascade).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.ws_urls import force_order_url

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore

    _HAS_WS = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore
    _HAS_WS = False


def _flush(rows: list[dict], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.set_index("ts").sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            if "ts" in existing.columns:
                existing = existing.set_index("ts")
            existing.index = pd.to_datetime(existing.index)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        except Exception:  # noqa: BLE001
            pass
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")


async def _record_liquidations(symbol: str, out_path: Path, duration_sec: int, flush_every: int = 50) -> int:
    """Записує стрім ліквідацій (forceOrder) у parquet."""
    if not _HAS_WS:
        raise ImportError("Встановіть websockets")

    rows: list[dict] = []
    written = 0
    start = time.monotonic()
    url = force_order_url(symbol)
    logger.info("Підключення до ліквідацій %s на %d сек", url, duration_sec)

    try:
        while time.monotonic() - start < duration_sec:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    while time.monotonic() - start < duration_sec:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        except TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            logger.warning("Стрім ліквідацій закрито — перепідключення")
                            break

                        data = json.loads(msg)
                        o_data = data.get("o", {})
                        if not o_data:
                            continue

                        # E - event time
                        try:
                            ts = pd.Timestamp(int(data.get("E", 0)), unit="ms", tz="UTC").tz_localize(None)
                        except (TypeError, ValueError):
                            ts = pd.Timestamp.now(tz="UTC").tz_localize(None)

                        rows.append(
                            {
                                "ts": ts,
                                "side": o_data.get("S"),
                                "price": float(o_data.get("p", 0)),
                                "qty": float(o_data.get("q", 0)),
                            }
                        )
                        if len(rows) >= flush_every:
                            _flush(rows, out_path)
                            written += len(rows)
                            rows = []
                            logger.info("Liquidations %s: %d записів", symbol, written)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Помилка ліквідацій: %s — перепідключення", exc)
            if time.monotonic() - start < duration_sec:
                await asyncio.sleep(2.0)
    finally:
        if rows:
            _flush(rows, out_path)
            written += len(rows)
    logger.info("Запис ліквідацій завершено: %d записів", written)
    return written


async def _record_open_interest(symbol: str, out_path: Path, duration_sec: int, interval_sec: int = 10) -> int:
    """Опитує REST API для запису Open Interest."""
    client = ExchangeClient(exchange_id="binance")
    if not hasattr(client.exchange, "fetch_open_interest"):
        logger.error("Біржа не підтримує fetch_open_interest")
        return 0

    rows: list[dict] = []
    written = 0
    start = time.monotonic()

    logger.info("Початок запису OI для %s (інтервал %d сек)", symbol, interval_sec)

    try:
        while time.monotonic() - start < duration_sec:
            try:
                # ccxt fetch_open_interest повертає dict з openInterest, symbol, datetime etc.
                oi_data = client.exchange.fetch_open_interest(symbol)
                ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
                if oi_data.get("timestamp"):
                    ts = pd.Timestamp(oi_data["timestamp"], unit="ms", tz="UTC").tz_localize(None)

                rows.append(
                    {
                        "ts": ts,
                        "open_interest": float(oi_data.get("openInterest", 0)),
                    }
                )

                if len(rows) >= 5:  # flush every ~50 sec
                    _flush(rows, out_path)
                    written += len(rows)
                    rows = []
                    logger.info("OI %s: %d записів", symbol, written)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Помилка отримання OI: %s", exc)

            await asyncio.sleep(interval_sec)
    finally:
        if rows:
            _flush(rows, out_path)
            written += len(rows)
    return written


async def _run_all(symbol: str, data_dir: Path, duration_sec: int) -> None:
    liq_path = data_dir / f"{symbol}_liquidations.parquet"
    oi_path = data_dir / f"{symbol}_oi.parquet"

    await asyncio.gather(
        _record_liquidations(symbol, liq_path, duration_sec),
        _record_open_interest(symbol, oi_path, duration_sec, interval_sec=10),
    )


def record_macro(symbol: str, minutes: int = 60, data_dir: Path | None = None) -> None:
    """Синхронна обгортка для запуску обох рекордерів."""
    from scalper_hft.config import get_settings

    settings = get_settings()
    out_dir = data_dir or settings.data_dir_abs
    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_all(symbol, out_dir, minutes * 60))
