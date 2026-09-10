"""Клієнт універсальної біржі поверх ccxt (REST).

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

from scalper_hft.data.exchange_registry import ExchangeRegistry

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Тонка обгортка над ccxt для підтримуваних бірж."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        exchange_id: str = "binance",
        auth: bool = False,
        market_type: str = "future",
    ) -> None:
        """auth=True — лише коли потрібні приватні ендпоінти (торгівля).

        Для завантаження даних (публічні klines/trades/funding) ключі НЕ
        передаються.
        """
        self.exchange_id = exchange_id
        meta = ExchangeRegistry.get(exchange_id)

        params: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        }
        if auth and api_key and api_secret:
            params.update({"apiKey": api_key, "secret": api_secret})

        self.exchange: ccxt.Exchange = meta.ccxt_class(params)  # type: ignore[arg-type]

        if str(exchange_id).lower().endswith("testnet") and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

        self.market_type = market_type
        self._market_cache: dict[str, dict[str, Any]] = {}

    @property
    def api_url(self) -> str:
        """Базовий URL REST API біржі."""
        urls = getattr(self.exchange, "urls", None)
        if not isinstance(urls, dict):
            return ""
        api_urls = urls.get("api")
        url = ""
        if isinstance(api_urls, dict):
            if getattr(self, "market_type", "") == "spot":
                for k in ("public", "v1", "sapi"):
                    if k in api_urls:
                        url = str(api_urls[k])
                        break
            if not url:
                for k in ("fapiPublic", "dapiPublic", "public", "fapiData", "dapiData", "v1", "sapi"):
                    if k in api_urls:
                        url = str(api_urls[k])
                        break
            if not url:
                try:
                    url = str(next(iter(api_urls.values())))
                except StopIteration:
                    url = ""
        elif isinstance(api_urls, str):
            url = api_urls

        if "{hostname}" in url:
            hostname = getattr(self.exchange, "hostname", "") or "binance.com"
            url = url.replace("{hostname}", hostname)
        return url

    @property
    def is_testnet(self) -> bool:
        """Чи веде клієнт на testnet / sandbox (синтетичні або тестові дані)."""
        ex_id = str(getattr(self, "exchange_id", "")).lower()
        if "testnet" in ex_id:
            return True
        exchange_obj = getattr(self, "exchange", None)
        if exchange_obj is not None:
            sandbox = getattr(exchange_obj, "sandbox", False) or getattr(exchange_obj, "is_sandbox_mode_enabled", False)
            if sandbox:
                return True
        url = self.api_url.lower()
        return "testnet" in url or "demo" in url

    def describe_source(self) -> str:
        """Опис джерела даних для логів: біржа, статус live/testnet та API endpoint."""
        url = self.api_url or "default URL"
        ex_id = getattr(self, "exchange_id", "exchange")
        if self.is_testnet:
            status = "УВАГА: TESTNET/SANDBOX ⚠ (синтетичні дані)"
        else:
            status = "LIVE (НЕ testnet ✓)"
        return f"{ex_id} [{status}] endpoint={url}"


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

    # ── M5: точність/ліміти біржі перед live-ордером ─────────────────────────

    def sanitize_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float | None = None,
    ) -> tuple[float, float | None, str | None]:
        """Округлити об'єм/ціну до кроків біржі та валідувати ліміти.

        Повертає (qty, price, error): qty — кратне LOT_SIZE, price — кратне
        tick (None для market), error — причина відхилення, якщо ордер не
        пройде фільтри (minQty/maxQty/MIN_NOTIONAL). Fail-closed: trader не
        відправляє ордер, доки error is not None.
        """
        m = self.market(symbol)
        try:
            qty = float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:  # noqa: BLE001
            qty = float(amount)
        px: float | None = None
        if price is not None:
            try:
                px = float(self.exchange.price_to_precision(symbol, price))
            except Exception:  # noqa: BLE001
                px = float(price)

        limits = m.get("limits", {}) or {}
        amt = limits.get("amount", {}) or {}
        min_qty = float(amt.get("min") or 0.0)
        max_qty = float(amt.get("max") or 0.0)
        min_notional = float((limits.get("cost", {}) or {}).get("min") or 0.0)

        err: str | None = None
        if min_qty > 0 and qty < min_qty:
            err = f"qty {qty:.8f} < minQty {min_qty}"
        elif max_qty > 0 and qty > max_qty:
            err = f"qty {qty:.8f} > maxQty {max_qty}"
        else:
            ref_px = px if px is not None else price
            notional = qty * (ref_px or 0.0)
            if min_notional > 0 and notional > 0 and notional < min_notional:
                err = f"notional {notional:.2f} < minNotional {min_notional}"
        return qty, px, err

    # ── дані ─────────────────────────────────────────────────────────────────
    def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> list[list[Any]]:
        """Один батч історичних свічок. Повертає сирі списки (у форматі ccxt)."""
        return self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)

    def fetch_agg_trades(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        """Історичні агреговані трейди (aggTrades) з флагом buy/sell."""
        return self.exchange.fetch_trades(symbol, since=since_ms, limit=limit)

    def fetch_funding_rate_history(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        """Історія ставок фандінгу."""
        return self.exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)

    def fetch_open_interest_history(
        self, symbol: str, timeframe: str, since_ms: int, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Історія Open Interest (ccxt)."""
        return self.exchange.fetch_open_interest_history(symbol, timeframe, since=since_ms, limit=limit)

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
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Створення ордера. post_only=True — лімітний maker-ордер (Binance:
        відхиляється, якщо перетнув би спред) — для збору maker-комісій.

        Rate-limit стійкість: ccxt.RateLimitExceeded (HTTP 418/429, код -1003)
        → експоненційний backoff (0.5s, 1s, 2s). Інші помилки — одразу вгору
        (не маскуємо реальні відхилення ордера).
        """
        params = dict(params or {})
        if post_only:
            params["postOnly"] = True
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        delay = 0.5
        for attempt in range(max_retries):
            try:
                return self.exchange.create_order(symbol, order_type, side, amount, price, params)
            except ccxt.RateLimitExceeded:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "Rate limit на create_order %s (спроба %d/%d) — backoff %.1fs",
                    symbol,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0
        raise RuntimeError("unreachable")  # для type-checker

    # ── Clock sync (live: timestamp-ордери потребують синхронного часу) ───────
    def fetch_server_time_ms(self) -> int | None:
        """Серверний час біржі (ms) або None, якщо біржа не підтримує."""
        fetch_time = getattr(self.exchange, "fetch_time", None)
        if not callable(fetch_time):
            return None
        try:
            return int(fetch_time())
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_time: %s", exc)
            return None

    def server_time_offset_ms(self) -> float | None:
        """Оцінка (server − local) з компенсацією половини RTT. None — н/д."""
        fetch_time = getattr(self.exchange, "fetch_time", None)
        if not callable(fetch_time):
            return None
        t0 = time.time() * 1000.0
        try:
            server = float(fetch_time())
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_time: %s", exc)
            return None
        t1 = time.time() * 1000.0
        local_mid = t0 + (t1 - t0) / 2.0
        return server - local_mid

    def assert_clock_synced(self, max_drift_ms: float = 1000.0) -> float:
        """Fail-closed для live: |offset| > max_drift_ms → RuntimeError.

        Розсинхрон часу ламає timestamp/recvWindow підписи запитів і логіку
        барових меж. Повертає виміряний offset (ms)."""
        offset = self.server_time_offset_ms()
        if offset is None:
            logger.warning("Біржа не підтримує fetch_time — clock sync не перевірено")
            return 0.0
        if abs(offset) > max_drift_ms:
            raise RuntimeError(
                f"Розсинхрон годинника з біржею: {offset:+.0f} ms > ±{max_drift_ms:.0f} ms. "
                "Увімкніть NTP (chrony/systemd-timesyncd) на хості."
            )
        return offset

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self.exchange.cancel_order(order_id, symbol)

    def cancel_all_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Скасувати всі робочі ордери на біржі (за символом або всі)."""
        res = self.exchange.cancel_all_orders(symbol)
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            return [res]
        return []

    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Статус ордера за id (для live maker-філів: заповнений/скасований)."""
        return self.exchange.fetch_order(order_id, symbol)

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Робочі (незаповнені) ордери — для звірки локального стану з біржею."""
        if symbol:
            return self.exchange.fetch_open_orders(symbol)
        return self.exchange.fetch_open_orders()

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_usdt_equity(self) -> float:
        """Реальний equity USDT-M (wallet balance + unrealized PnL).

        Використовується live-шаром для sizing/ризик-гейтів ЗАМІСТЬ
        фіктивного PaperAccount (M4). ccxt fetch_balance повертає
        {'total': {'USDT': wallet}, 'USDT': {...}, 'info': {...}}; для
        Binance futures unrealized PnL лежить в info.totalUnrealizedProfit.
        """
        bal = self.exchange.fetch_balance()
        info = bal.get("info") or {}
        wallet: float | None = None
        raw_wallet = info.get("totalWalletBalance")
        if raw_wallet is not None:
            try:
                wallet = float(raw_wallet)
            except (TypeError, ValueError):
                wallet = None
        if wallet is None:
            # fallback: ccxt total
            total_map = bal.get("total") or {}
            usdt_total = total_map.get("USDT")
            if usdt_total is None:
                usdt_ent = bal.get("USDT") or {}
                usdt_total = usdt_ent.get("total")
            if usdt_total is not None:
                try:
                    wallet = float(usdt_total)
                except (TypeError, ValueError):
                    wallet = 0.0
            else:
                wallet = 0.0
        unrealized = info.get("totalUnrealizedProfit")
        try:
            upnl = float(unrealized) if unrealized is not None else 0.0
        except (TypeError, ValueError):
            upnl = 0.0
        return wallet + upnl

    def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """Відкриті позиції USDT-M (для звірки з локальним рахунком)."""
        if symbols:
            return self.exchange.fetch_positions(symbols)
        return self.exchange.fetch_positions()

    # ── WebSocket listenKey (User Data Stream) ────────────────────────────────
    def create_listen_key(self) -> str:
        """Створити listenKey для Binance User Data Stream через ccxt."""
        for method_name in ("fapiPrivatePostListenKey", "dapiPrivatePostListenKey"):
            fn = getattr(self.exchange, method_name, None)
            if callable(fn):
                try:
                    resp = fn()
                    key = str((resp or {}).get("listenKey") or "")
                    if key:
                        return key
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", method_name, exc)
        return ""

    def keepalive_listen_key(self, listen_key: str) -> None:
        """Подовжити термін дії listenKey (кожні 30-50 хв)."""
        if not listen_key:
            return
        for method_name in ("fapiPrivatePutListenKey", "dapiPrivatePutListenKey"):
            fn = getattr(self.exchange, method_name, None)
            if callable(fn):
                try:
                    fn({"listenKey": listen_key})
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", method_name, exc)

    def close_listen_key(self, listen_key: str) -> None:
        """Закрити listenKey при завершенні роботи."""
        if not listen_key:
            return
        for method_name in ("fapiPrivateDeleteListenKey", "dapiPrivateDeleteListenKey"):
            fn = getattr(self.exchange, method_name, None)
            if callable(fn):
                try:
                    fn({"listenKey": listen_key})
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", method_name, exc)
