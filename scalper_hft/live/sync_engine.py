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

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from scalper_hft.live.account import PaperAccount
from scalper_hft.live.reconcile import KillSwitch, halt_if_drift
from scalper_hft.live.store import PaperStore

logger = logging.getLogger(__name__)

# Інтервали синхронізації (секунди)
_ACCOUNT_SYNC_SEC = 60
_POSITION_SYNC_SEC = 30
_FILL_SYNC_SEC = 60

_META_KNOWN_TRADES = "sync_known_trade_ids"
_META_FILLS_SINCE_MS = "sync_fills_since_ms"


def _canonical_symbol(raw: str) -> str:
    """BTC/USDT:USDT → BTCUSDT для порівняння з локальним scope."""
    return raw.replace("/", "").replace(":USDT", "")


class SyncEngine:
    """Синхронізує локальний TradeStore з реальною біржею.

    У paper mode всі синхронізації — no-op (режим симуляції не потребує REST).
    У live mode викликає ccxt для отримання балансу, позицій і fills.

    Якщо передано ``account``, ``sync_positions`` викликає ``halt_if_drift``
    (як ``run_trader_once``). При KillSwitch викликається ``on_kill_switch``.
    """

    def __init__(
        self,
        store: PaperStore,
        exchange_id: str = "binance",
        mode: str = "paper",
        account_interval_sec: int = _ACCOUNT_SYNC_SEC,
        position_interval_sec: int = _POSITION_SYNC_SEC,
        fill_interval_sec: int = _FILL_SYNC_SEC,
        *,
        account: PaperAccount | None = None,
        scope: set[str] | None = None,
        dry_run: bool = True,
        on_kill_switch: Callable[[str], None] | None = None,
    ) -> None:
        self._store = store
        self._exchange_id = exchange_id
        self._mode = mode
        self._account_interval = account_interval_sec
        self._position_interval = position_interval_sec
        self._fill_interval = fill_interval_sec
        self._account = account
        self._scope = scope
        self._dry_run = dry_run
        self._on_kill_switch = on_kill_switch

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # ccxt exchange instance (ліниво ініціалізується в live mode)
        self._exchange: Any | None = None

        # Таймери останньої синхронізації
        self._last_account_sync: float = 0.0
        self._last_position_sync: float = 0.0
        self._last_fill_sync: float = 0.0

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
            "SyncEngine запущено: exchange=%s mode=%s account_interval=%ds position_interval=%ds fill_interval=%ds",
            self._exchange_id,
            self._mode,
            self._account_interval,
            self._position_interval,
            self._fill_interval,
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
                if now - self._last_account_sync >= self._account_interval:
                    self.sync_account()
                    self._last_account_sync = time.monotonic()

                if now - self._last_position_sync >= self._position_interval:
                    self.sync_positions()
                    self._last_position_sync = time.monotonic()

                if now - self._last_fill_sync >= self._fill_interval:
                    self.sync_fills()
                    self._last_fill_sync = time.monotonic()

            except KillSwitch as exc:
                logger.critical("SyncEngine KillSwitch: %s", exc)
                if self._on_kill_switch is not None:
                    try:
                        self._on_kill_switch(str(exc))
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.error("on_kill_switch callback failed: %s", cb_exc)
            except Exception as exc:
                logger.error("SyncEngine: помилка циклу: %s", exc, exc_info=True)

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
                balance,
                available,
                unrealized_pnl,
            )
        except Exception as exc:
            logger.error("SyncEngine.sync_account: помилка: %s", exc)

    def sync_positions(self) -> None:
        """Синхронізувати відкриті позиції з біржею.

        Paper mode: no-op.
        Live mode: fetch_positions() → store + halt_if_drift при наявному account.
        """
        if self._mode == "paper":
            logger.debug("SyncEngine.sync_positions: paper mode — no-op")
            return

        exchange = self._get_exchange()
        positions = exchange.fetch_positions()
        ts = pd.Timestamp.now(tz="UTC")

        for pos in positions:
            size = float(pos.get("contracts", 0) or 0)
            if size == 0:
                continue
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

        if self._account is not None and not self._dry_run:
            halt_if_drift(self._account, positions, dry_run=False, scope=self._scope)

    def sync_fills(self) -> None:
        """Синхронізувати останні заповнення ордерів з біржею.

        Paper mode: no-op (заповнення симулюються).
        Live mode: fetch_my_trades() → порівняння з локальним store, аудит-лог.
        """
        if self._mode == "paper":
            logger.debug("SyncEngine.sync_fills: paper mode — no-op")
            return

        try:
            exchange = self._get_exchange()
            since_ms = int(self._store.get_meta(_META_FILLS_SINCE_MS, "0") or "0")
            known = self._known_trade_ids()
            symbols = self._fill_symbols(exchange)
            new_trades: list[dict[str, Any]] = []
            max_ts = since_ms

            for sym in symbols:
                kwargs: dict[str, Any] = {"limit": 500}
                if since_ms > 0:
                    kwargs["since"] = since_ms
                try:
                    batch = exchange.fetch_my_trades(sym, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fetch_my_trades %s: %s", sym, exc)
                    continue
                if isinstance(batch, list):
                    new_trades.extend(batch)

            missing = 0
            for trade in new_trades:
                tid = str(trade.get("id") or "")
                ts_ms = int(trade.get("timestamp") or 0)
                max_ts = max(max_ts, ts_ms)
                if not tid or tid in known:
                    continue
                missing += 1
                self._log_exchange_fill(trade)
                known.add(tid)

            if missing:
                logger.warning(
                    "SyncEngine.sync_fills: %d fills на біржі відсутні в локальному store",
                    missing,
                )

            self._store.set_meta(_META_KNOWN_TRADES, json.dumps(sorted(known)))
            if max_ts > since_ms:
                self._store.set_meta(_META_FILLS_SINCE_MS, str(max_ts))
        except Exception as exc:
            logger.error("SyncEngine.sync_fills: помилка: %s", exc)

    # ─── Приватні методи ──────────────────────────────────────────────────────

    def _known_trade_ids(self) -> set[str]:
        raw = self._store.get_meta(_META_KNOWN_TRADES, "[]")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return {str(x) for x in data}
        except json.JSONDecodeError:
            pass
        return set()

    def _fill_symbols(self, exchange: Any) -> list[str]:
        """Символи для fetch_my_trades: scope, відкриті позиції або markets."""
        if self._scope:
            return [self._resolve_ccxt_symbol(exchange, s) for s in sorted(self._scope)]
        try:
            positions = exchange.fetch_positions()
            syms = [
                str(p.get("symbol", "")) for p in positions if float(p.get("contracts", 0) or 0) > 0 and p.get("symbol")
            ]
            if syms:
                return syms
        except Exception:
            pass
        return []

    def _resolve_ccxt_symbol(self, exchange: Any, canonical: str) -> str:
        """BTCUSDT → unified ccxt symbol (BTC/USDT:USDT)."""
        sym = canonical.replace("/", "").replace(":USDT", "")
        unified = f"{sym[:-4]}/USDT:USDT" if sym.endswith("USDT") and len(sym) > 4 else sym
        load = getattr(exchange, "load_markets", None)
        if callable(load):
            markets = load()
            if unified in markets:
                return unified
            if sym in markets:
                return sym
        return unified

    def _log_exchange_fill(self, trade: dict[str, Any]) -> None:
        """Записати біржовий fill у store для аудиту (status=exchange_sync)."""
        sym = _canonical_symbol(str(trade.get("symbol") or ""))
        side = str(trade.get("side") or "buy")
        size = float(trade.get("amount") or trade.get("qty") or 0.0)
        price = float(trade.get("price") or trade.get("average") or 0.0)
        ts = pd.Timestamp(int(trade.get("timestamp") or 0), unit="ms", tz="UTC")
        tid = str(trade.get("id") or "")
        self._store.log_order(
            ts=ts,
            pair=f"sync:{sym}",
            symbol=sym,
            side=side,
            size=size,
            price=price,
            status="exchange_sync",
            reason=f"trade_id={tid}",
            exchange=self._exchange_id,
            mode=self._mode,
        )

    def _get_exchange(self) -> Any:
        """Отримати або ініціалізувати ccxt exchange instance."""
        if self._exchange is not None:
            return self._exchange

        import ccxt

        from scalper_hft.config import get_settings

        settings = get_settings()
        exchange_class = getattr(ccxt, "binanceusdm", None)
        if exchange_class is None:
            raise RuntimeError(f"ccxt не підтримує exchange: {self._exchange_id}")

        self._exchange = exchange_class(
            {
                "apiKey": settings.binance_api_key,
                "secret": settings.binance_api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        return self._exchange

    def _fetch_unrealized_pnl(self, exchange: Any) -> float:
        """Отримати загальний unrealized PnL з відкритих позицій."""
        try:
            positions = exchange.fetch_positions()
            return sum(float(p.get("unrealizedPnl") or 0) for p in positions if float(p.get("contracts", 0) or 0) > 0)
        except Exception:
            return 0.0


__all__ = ["SyncEngine"]
