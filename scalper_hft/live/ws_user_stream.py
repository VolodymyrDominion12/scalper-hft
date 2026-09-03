"""WebSocket User Data Stream для Binance Futures.

Отримує події виконання ордерів (ORDER_TRADE_UPDATE) та балансу (ACCOUNT_UPDATE)
у реальному часі без затримок REST-полінгу.

Підключення:
    1. Отримання listenKey через REST (POST /fapi/v1/listenKey)
    2. Підключення до wss://fstream.binance.com/ws/{listenKey}
    3. Keep-alive кожні 30-50 хвилин (PUT /fapi/v1/listenKey)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

_WS_BASE_URL = "wss://fstream.binance.com/ws/{}"
_WS_TESTNET_BASE_URL = "wss://stream.binancefuture.com/ws/{}"


@dataclass(frozen=True)
class OrderTradeEvent:
    """Нормалізована подія оновлення ордера з WebSocket стріму."""

    event_time: int
    symbol: str
    client_order_id: str
    order_id: int
    side: str  # BUY | SELL
    order_type: str  # LIMIT | MARKET
    status: str  # NEW | PARTIALLY_FILLED | FILLED | CANCELED | EXPIRED
    execution_type: str  # NEW | TRADE | CANCELED | EXPIRED
    last_filled_qty: float
    cumulative_filled_qty: float
    last_filled_price: float
    commission: float
    commission_asset: str
    is_maker: bool


def parse_order_trade_update(data: dict[str, Any]) -> OrderTradeEvent | None:
    """Парсить повідомлення Binance ORDER_TRADE_UPDATE."""
    if data.get("e") != "ORDER_TRADE_UPDATE":
        return None
    o = data.get("o", {})
    try:
        return OrderTradeEvent(
            event_time=int(data.get("E", 0)),
            symbol=str(o.get("s", "")),
            client_order_id=str(o.get("c", "")),
            order_id=int(o.get("i", 0)),
            side=str(o.get("S", "")),
            order_type=str(o.get("ot", o.get("o", ""))),
            status=str(o.get("X", "")),
            execution_type=str(o.get("x", "")),
            last_filled_qty=float(o.get("l", 0.0)),
            cumulative_filled_qty=float(o.get("z", 0.0)),
            last_filled_price=float(o.get("L", 0.0)),
            commission=float(o.get("n", 0.0)),
            commission_asset=str(o.get("N", "")),
            is_maker=bool(o.get("m", False)),
        )
    except Exception as err:
        logger.warning("Помилка парсингу ORDER_TRADE_UPDATE: %s", err)
        return None


class BinanceUserDataStream:
    """Асинхронний клієнт User Data Stream Binance USDT-M."""

    def __init__(
        self,
        listen_key: str,
        testnet: bool = False,
        on_order_update: Callable[[OrderTradeEvent], Any] | None = None,
    ) -> None:
        self.listen_key = listen_key
        self.testnet = testnet
        self.on_order_update = on_order_update
        self._running = False

    @property
    def ws_url(self) -> str:
        base = _WS_TESTNET_BASE_URL if self.testnet else _WS_BASE_URL
        return base.format(self.listen_key)

    def handle_raw_message(self, raw_msg: str | dict) -> OrderTradeEvent | None:
        """Обробляє сире повідомлення з вебсокета і викликає колбек."""
        data = json.loads(raw_msg) if isinstance(raw_msg, str) else raw_msg
        event = parse_order_trade_update(data)
        if event and self.on_order_update:
            try:
                self.on_order_update(event)
            except Exception as e:
                logger.error("Помилка в on_order_update колбеку: %s", e)
        return event

    async def run(self) -> None:
        """Основний асинхронний цикл отримання повідомлень з автоперепідключенням."""
        try:
            import websockets
        except ImportError:
            raise ImportError("Потрібно встановити websockets: uv pip install websockets")

        self._running = True
        logger.info("Підключення до User Data Stream: %s", self.ws_url)

        while self._running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    logger.info("WebSocket User Data Stream успішно з'єднано")
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            self.handle_raw_message(msg)
                        except TimeoutError:
                            # Періодичний пінг
                            continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Розрив WebSocket зв'язку (%s), перепідключення за 5 сек...", e)
                await asyncio.sleep(5.0)

    def stop(self) -> None:
        self._running = False
