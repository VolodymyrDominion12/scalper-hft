"""Клієнт Binance USDT-M ф'ючерсів поверх ccxt (REST).

Використовується для:
    - завантаження історичних klines / aggTrades / funding (для бектестів)
    - отримання інформації про символи (мін. крок ціни, розмір, квота)
    - (у live-модулі) розміщення та керування ордерами

Live-стріми (WebSocket) винесені в scalper_hft.live — тут лише синхронний REST.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ccxt

logger = logging.getLogger(__name__)

# Обмеження ваги Binance: 2400 запитів/хв, не дратуємо — спимо між батчами
_RATE_LIMIT_SLEEP = 0.12


class BinanceClient:
    """Тонка обгортка над ccxt для USDT-M ф'ючерсів Binance."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        exchange_id: str = "binance-testnet",
        auth: bool = False,
        market_type: str = "future",
    ) -> None:
        """auth=True — лише коли потрібні приватні ендпоінти (торгівля).

        Для завантаження даних (публічні klines/trades/funding) ключі НЕ
        передаються: Binance валідує X-MBX-APIKEY навіть на публічних
        ендпоінтах і відхиляє невалідні ключі.
        market_type: 'future' (USDT-M ф'ючерси) або 'spot' — для delta-neutral
        арбітражу потрібні обидва ринки.
        """
        exchange_cls = getattr(ccxt, exchange_id) if hasattr(ccxt, exchange_id) else ccxt.binance
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        }
        if auth and api_key and api_secret:
            params.update({"apiKey": api_key, "secret": api_secret})
        self.exchange: ccxt.Exchange = exchange_cls(params)  # type: ignore[arg-type]
        self.market_type = market_type
        self._market_cache: dict[str, dict[str, Any]] = {}

    # ── метадані ──────────────────────────────────────────────────────────────
    def load_markets(self) -> dict[str, Any]:
        if not self._market_cache:
            self._market_cache = self.exchange.load_markets()
        return self._market_cache

    def market(self, symbol: str) -> dict[str, Any]:
        return self.load_markets()[symbol]

    def price_precision(self, symbol: str) -> float:
        """Мінімальний крок ціни (tick size) для символу."""
        m = self.market(symbol)
        return float(m.get("precision", {}).get("price", 0.01) or 0.01)

    def min_notional(self, symbol: str) -> float:
        m = self.market(symbol)
        limits = m.get("limits", {}).get("cost", {})
        return float(limits.get("min", 5.0) or 5.0)

    # ── дані ─────────────────────────────────────────────────────────────────
    def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> list[list[Any]]:
        """Один батч історичних свічок. Повертає сирі списки (у форматі ccxt)."""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded as exc:
            logger.warning("Rate limit: sleep 5s і повтор (%s)", exc)
            time.sleep(5.0)
            return self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)

    def fetch_agg_trades(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        """Історичні агреговані трейди (aggTrades) з флагом buy/sell."""
        try:
            return self.exchange.fetch_trades(symbol, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded as exc:
            logger.warning("Rate limit: sleep 5s і повтор (%s)", exc)
            time.sleep(5.0)
            return self.exchange.fetch_trades(symbol, since=since_ms, limit=limit)

    def fetch_funding_rate_history(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        """Історія ставок фандінгу."""
        try:
            return self.exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded as exc:
            logger.warning("Rate limit: sleep 5s і повтор (%s)", exc)
            time.sleep(5.0)
            return self.exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)

    # ── торгівля (використовується live-модулем) ─────────────────────────────
    def create_order(
        self,
        symbol: str,
        order_type: str,  # "limit" | "market"
        side: str,  # "buy" | "sell"
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
        post_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Створення ордера. post_only=True — лімітний maker-ордер (Binance:
        відхиляється, якщо перетнув би спред) — для збору maker-комісій."""
        params = dict(params or {})
        if post_only:
            params["postOnly"] = True
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self.exchange.create_order(symbol, order_type, side, amount, price, params)

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """Відкриті позиції USDT-M (для звірки з локальним рахунком)."""
        if symbols:
            return self.exchange.fetch_positions(symbols)
        return self.exchange.fetch_positions()
