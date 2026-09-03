"""WebSocket User Data Stream для Binance USDT-M.

Підключення (після 2026-04-23):
    1. REST POST /fapi/v1/listenKey
    2. wss://fstream.binance.com/private/ws?listenKey=…&events=…
    3. Keep-alive PUT /fapi/v1/listenKey кожні 30 хв (інакше ключ помирає за 60 хв)
    4. Регенерація listenKey на reconnect і події listenKeyExpired
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scalper_hft.live.ws_urls import KEEPALIVE_INTERVAL_SEC, private_user_stream_url

logger = logging.getLogger(__name__)


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


def is_listen_key_expired(data: dict[str, Any]) -> bool:
    return str(data.get("e", "")) == "listenKeyExpired"


class BinanceUserDataStream:
    """Асинхронний клієнт User Data Stream Binance USDT-M."""

    def __init__(
        self,
        listen_key: str,
        testnet: bool = False,
        on_order_update: Callable[[OrderTradeEvent], Any] | None = None,
        keepalive: Callable[[str], None] | None = None,
        refresh_listen_key: Callable[[], str] | None = None,
        keepalive_interval_sec: float = KEEPALIVE_INTERVAL_SEC,
    ) -> None:
        self.listen_key = listen_key
        self.testnet = testnet
        self.on_order_update = on_order_update
        self.keepalive = keepalive
        self.refresh_listen_key = refresh_listen_key
        self.keepalive_interval_sec = float(keepalive_interval_sec)
        self._running = False
        self._last_keepalive_mono: float = 0.0

    @property
    def ws_url(self) -> str:
        return private_user_stream_url(self.listen_key, testnet=self.testnet)

    def maybe_keepalive(self, now_mono: float | None = None) -> bool:
        """PUT listenKey якщо минув інтервал. Повертає True, якщо викликано."""
        now = time.monotonic() if now_mono is None else now_mono
        if self.keepalive is None:
            return False
        if self._last_keepalive_mono == 0.0:
            self._last_keepalive_mono = now
            return False
        if now - self._last_keepalive_mono < self.keepalive_interval_sec:
            return False
        self.keepalive(self.listen_key)
        self._last_keepalive_mono = now
        return True

    def regenerate_listen_key(self) -> bool:
        """Новий listenKey через інжектований REST-колбек. True якщо оновлено."""
        if self.refresh_listen_key is None:
            return False
        new_key = self.refresh_listen_key()
        if not new_key:
            return False
        self.listen_key = str(new_key)
        self._last_keepalive_mono = 0.0
        logger.info("listenKey регенеровано")
        return True

    def handle_raw_message(self, raw_msg: str | dict) -> OrderTradeEvent | None:
        """Обробляє сире повідомлення з вебсокета і викликає колбек."""
        data = json.loads(raw_msg) if isinstance(raw_msg, str) else raw_msg
        if is_listen_key_expired(data):
            logger.warning("listenKeyExpired — регенерація ключа")
            self.regenerate_listen_key()
            return None
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
        self._last_keepalive_mono = time.monotonic()
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
                            self.maybe_keepalive()
                            continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Розрив WebSocket зв'язку (%s), перепідключення за 5 сек...", e)
                self.regenerate_listen_key()
                await asyncio.sleep(5.0)

    def stop(self) -> None:
        self._running = False
