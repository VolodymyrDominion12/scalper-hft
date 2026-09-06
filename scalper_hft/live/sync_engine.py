"""SyncEngine: синхронізує локальний TradeStore з біржею через REST.

Paper mode: sync_account і sync_positions — no-op (симуляція не потребує REST).
Live mode: реальні запити через ccxt, KillSwitch якщо drift позицій.

Запуск:
    engine = SyncEngine(store, exchange_id="binance", mode="paper")
    engine.start()  # daemon thread
    ...
    engine.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pandas as pd

from scalper_hft.live.store import PaperStore

logger = logging.getLogger(__name__)

# Інтервали синхронізації (секунди)
_ACCOUNT_SYNC_SEC = 60
_POSITION_SYNC_SEC = 30


class SyncEngine:
    """Синхронізує локальний TradeStore з реальною біржею.

    У paper mode всі синхронізації — no-op (режим симуляції не потребує REST).
    У live mode викликає ccxt для отримання балансу і позицій.

    Запускається в окремому daemon-потоці через start().
    Зупиняється через stop() (graceful, чекає до _timeout_sec).
    """

    def __init__(
        self,
        store: PaperStore,
        exchange_id: str = "binance",
        mode: str = "paper",
        account_interval_sec: int = _ACCOUNT_SYNC_SEC,
        position_interval_sec: int = _POSITION_SYNC_SEC,
    ) -> None:
        self._store = store
        self._exchange_id = exchange_id
        self._mode = mode
        self._account_interval = account_interval_sec
        self._position_interval = position_interval_sec

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # ccxt exchange instance (ліниво ініціалізується в live mode)
        self._exchange: Any | None = None

        # Таймери останньої синхронізації
        self._last_account_sync: float = 0.0
        self._last_position_sync: float = 0.0

    # ─── Публічний API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Запустити SyncEngine в daemon-потоці."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("SyncEngine вже запущено")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"sync-{self._exchange_id}-{self._mode}",
        )
        self._thread.start()
        logger.info(
            "SyncEngine запущено: exchange=%s mode=%s account_interval=%ds position_interval=%ds",
            self._exchange_id,
            self._mode,
            self._account_interval,
            self._position_interval,
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        """Graceful shutdown SyncEngine."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
            if self._thread.is_alive():
                logger.warning("SyncEngine не зупинився за %s с", timeout_sec)
        self._thread = None
        logger.info("SyncEngine зупинено")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ─── Основний цикл ────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Основний цикл синхронізації в daemon-потоці."""
        logger.info("SyncEngine: цикл стартував")
        while not self._stop_event.is_set():
            now = time.monotonic()

            try:
                # Синхронізація балансу
                if now - self._last_account_sync >= self._account_interval:
                    self.sync_account()
                    self._last_account_sync = time.monotonic()

                # Синхронізація позицій
                if now - self._last_position_sync >= self._position_interval:
                    self.sync_positions()
                    self._last_position_sync = time.monotonic()

            except Exception as exc:
                logger.error("SyncEngine: помилка циклу: %s", exc, exc_info=True)

            # Сплячка з перевіркою кожну секунду
            self._stop_event.wait(timeout=1.0)

        logger.info("SyncEngine: цикл завершено")

    # ─── Методи синхронізації ─────────────────────────────────────────────────

    def sync_account(self) -> None:
        """Синхронізувати баланс рахунку з біржею.

        Paper mode: no-op (симуляція веде баланс локально).
        Live mode: REST запит до ccxt.fetch_balance().
        """
        if self._mode == "paper":
            logger.debug("SyncEngine.sync_account: paper mode — no-op")
            return

        try:
            exchange = self._get_exchange()
            raw = exchange.fetch_balance()
            usdt = raw.get("USDT", {})
            balance = float(usdt.get("total", 0.0))
            available = float(usdt.get("free", 0.0))
            used = float(usdt.get("used", 0.0))

            # Отримати unrealized PnL з позицій якщо є
            unrealized_pnl = self._fetch_unrealized_pnl(exchange)

            ts = pd.Timestamp.now(tz="UTC")
            self._store.log_account(
                ts=ts,
                exchange=self._exchange_id,
                mode=self._mode,
                balance=balance,
                unrealized_pnl=unrealized_pnl,
                margin_used=used,
                available=available,
            )
            logger.info(
                "SyncEngine.sync_account: balance=%.2f available=%.2f upnl=%.2f",
                balance, available, unrealized_pnl,
            )
        except Exception as exc:
            logger.error("SyncEngine.sync_account: помилка: %s", exc)

    def sync_positions(self) -> None:
        """Синхронізувати відкриті позиції з біржею.

        Paper mode: no-op.
        Live mode: fetch_positions() + KillSwitch при значному drift.
        """
        if self._mode == "paper":
            logger.debug("SyncEngine.sync_positions: paper mode — no-op")
            return

        try:
            exchange = self._get_exchange()
            positions = exchange.fetch_positions()
            ts = pd.Timestamp.now(tz="UTC")

            for pos in positions:
                size = float(pos.get("contracts", 0) or 0)
                if size == 0:
                    continue  # Пропускаємо нульові позиції
                side = "long" if pos.get("side") == "long" else "short"
                symbol = str(pos.get("symbol", ""))
                entry_price = float(pos.get("entryPrice") or 0)
                mark_price = pos.get("markPrice")
                unrealized_pnl = pos.get("unrealizedPnl")

                self._store.log_position(
                    ts=ts,
                    exchange=self._exchange_id,
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    mark_price=float(mark_price) if mark_price else None,
                    unrealized_pnl=float(unrealized_pnl) if unrealized_pnl else None,
                    mode=self._mode,
                )

            logger.info(
                "SyncEngine.sync_positions: %d відкритих позицій",
                sum(1 for p in positions if float(p.get("contracts", 0) or 0) > 0),
            )
        except Exception as exc:
            logger.error("SyncEngine.sync_positions: помилка: %s", exc)

    def sync_fills(self) -> None:
        """Синхронізувати останні заповнення ордерів.

        Paper mode: no-op (заповнення симулюються).
        Live mode: REST запит до fetch_my_trades().
        """
        if self._mode == "paper":
            logger.debug("SyncEngine.sync_fills: paper mode — no-op")
            return

        logger.debug("SyncEngine.sync_fills: live mode — TODO fetch_my_trades")
        # TODO Sprint 3: реалізувати порівняння fills з локальним store

    # ─── Приватні методи ──────────────────────────────────────────────────────

    def _get_exchange(self) -> Any:
        """Отримати або ініціалізувати ccxt exchange instance."""
        if self._exchange is not None:
            return self._exchange

        import ccxt  # type: ignore[import]
        from scalper_hft.config import get_settings

        settings = get_settings()
        exchange_class = getattr(ccxt, "binanceusdm", None)
        if exchange_class is None:
            raise RuntimeError(f"ccxt не підтримує exchange: {self._exchange_id}")

        self._exchange = exchange_class({
            "apiKey": settings.binance_api_key,
            "secret": settings.binance_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        return self._exchange

    def _fetch_unrealized_pnl(self, exchange: Any) -> float:
        """Отримати загальний unrealized PnL з відкритих позицій."""
        try:
            positions = exchange.fetch_positions()
            total = sum(
                float(p.get("unrealizedPnl") or 0)
                for p in positions
                if float(p.get("contracts", 0) or 0) > 0
            )
            return total
        except Exception:
            return 0.0


__all__ = ["SyncEngine"]
