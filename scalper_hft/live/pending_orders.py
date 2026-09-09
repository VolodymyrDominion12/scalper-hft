"""Live maker pending-ордери: журнал, REST poll, WS fills, booking у PaperAccount."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import pandas as pd

from scalper_hft.live.account import PaperAccount
from scalper_hft.live.trader_bars import as_naive_utc, interval_seconds
from scalper_hft.live.ws_user_stream import OrderTradeEvent

logger = logging.getLogger(__name__)


def _as_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return as_naive_utc(ts)


def _interval_seconds(interval: str) -> float:
    return interval_seconds(interval)


@dataclass
class PendingOrder:
    """Live maker-ордер, що ще не заповнився (resting на біржі)."""

    client_order_id: str
    order_id: str
    symbol: str
    side: str  # buy | sell
    size: float
    price: float
    reduce_only: bool
    kind: str  # open | close
    pos_side: str = ""
    placed_ts: pd.Timestamp | None = None
    booked_qty: float = 0.0

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.size - self.booked_qty)


class PendingOrderManager:
    """Thread-safe журнал resting-ордерів і booking філів у локальний рахунок."""

    def __init__(
        self,
        symbol: str,
        account: PaperAccount,
        client: object,
        *,
        dry_run: bool,
        interval: str,
        maker_fill_wait_bars: int = 1,
        partial_fill_policy: str = "cancel",
    ) -> None:
        if partial_fill_policy not in ("cancel", "wait"):
            raise ValueError(f"partial_fill_policy має бути 'cancel' або 'wait', отримано {partial_fill_policy!r}")
        self.symbol = symbol
        self.account = account
        self.client = client
        self.dry_run = dry_run
        self.interval = interval
        self.maker_fill_wait_bars = maker_fill_wait_bars
        self.partial_fill_policy = partial_fill_policy
        self.pending_orders: dict[str, PendingOrder] = {}
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def find_for_symbol(self, symbol: str | None = None) -> PendingOrder | None:
        sym = symbol or self.symbol
        return next((po for po in self.pending_orders.values() if po.symbol == sym), None)

    def register(self, coid: str, po: PendingOrder) -> None:
        with self._lock:
            self.pending_orders[coid] = po

    def cancel(self, po: PendingOrder, reason: str = "") -> None:
        cancel = getattr(self.client, "cancel_order", None)
        if cancel is not None:
            try:
                cancel(po.order_id, po.symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cancel %s: %s", po.order_id, exc)
        with self._lock:
            self.pending_orders.pop(po.client_order_id, None)
        logger.info("Скасовано pending %s (%s)", po.client_order_id, reason or "n/a")

    def cancel_all(self, reason: str = "shutdown") -> int:
        with self._lock:
            pending = list(self.pending_orders.values())
        count = 0
        for po in pending:
            self.cancel(po, reason=reason)
            count += 1
        return count

    def poll(self, now: pd.Timestamp | None = None) -> list[str]:
        """REST poll resting-ордерів → book fills / cancel timeouts."""
        with self._lock:
            if self.dry_run or not self.pending_orders:
                return []
            pending_snapshot = list(self.pending_orders.items())
        fetch = getattr(self.client, "fetch_order", None)
        if fetch is None:
            return ["no_fetch_support"]
        cancel = getattr(self.client, "cancel_order", None)
        ts = _as_naive_utc(pd.Timestamp.now(tz="UTC"))
        timeout_s = max(_interval_seconds(self.interval), 1.0) * max(self.maker_fill_wait_bars, 1)
        events: list[str] = []
        for coid, po in pending_snapshot:
            try:
                info = fetch(po.order_id, po.symbol)
            except Exception as exc:  # noqa: BLE001
                logger.error("fetch_order %s: %s", po.order_id, exc)
                continue
            status = str((info or {}).get("status") or "").lower()
            filled = float((info or {}).get("filled") or 0.0)
            avg = float((info or {}).get("average") or (info or {}).get("price") or po.price)
            with self._lock:
                live_po = self.pending_orders.get(coid)
                if live_po is None:
                    continue
                if status in ("closed", "filled") or filled >= live_po.size - 1e-9:
                    self._book_fill_delta(live_po, live_po.size, avg, ts)
                    self.pending_orders.pop(coid, None)
                    events.append(f"filled:{coid}")
                    continue
                if filled > 1e-9:
                    self._book_fill_delta(live_po, filled, avg, ts)
                    if self.partial_fill_policy == "wait":
                        # Тримаємо ордер: дельта забукана, залишок resting —
                        # але лише до таймауту (maker_fill_wait_bars барів).
                        age_s = 0.0
                        if live_po.placed_ts is not None:
                            age_s = (ts - _as_naive_utc(live_po.placed_ts)).total_seconds()
                        if age_s > timeout_s:
                            if cancel is not None:
                                try:
                                    cancel(live_po.order_id, live_po.symbol)
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("cancel partial-timeout %s: %s", live_po.order_id, exc)
                            self.pending_orders.pop(coid, None)
                            events.append(f"timeout_cancel:{coid}")
                        else:
                            events.append(f"partial_wait:{coid}")
                        continue
                    if cancel is not None:
                        try:
                            cancel(live_po.order_id, live_po.symbol)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("cancel partial %s: %s", live_po.order_id, exc)
                    self.pending_orders.pop(coid, None)
                    events.append(f"partial:{coid}")
                    continue
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    self.pending_orders.pop(coid, None)
                    events.append(f"dead:{status}:{coid}")
                    continue
                age_s = 0.0
                if live_po.placed_ts is not None:
                    age_s = (ts - _as_naive_utc(live_po.placed_ts)).total_seconds()
                if age_s > timeout_s:
                    if cancel is not None:
                        try:
                            cancel(live_po.order_id, live_po.symbol)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("cancel timeout %s: %s", live_po.order_id, exc)
                    self.pending_orders.pop(coid, None)
                    events.append(f"timeout_cancel:{coid}")
        return events

    def on_ws_order_trade(self, event: OrderTradeEvent) -> None:
        coid = event.client_order_id
        if not coid:
            return
        with self._lock:
            if coid not in self.pending_orders:
                return
            po = self.pending_orders[coid]
            ts = _as_naive_utc(pd.Timestamp.now(tz="UTC"))
            status = event.status.upper()
            if status in ("FILLED", "CLOSED"):
                fill_px = event.last_filled_price if event.last_filled_price > 0 else po.price
                cumulative = event.cumulative_filled_qty if event.cumulative_filled_qty > 0 else po.size
                booked = self._book_fill_delta(po, max(cumulative, po.size), fill_px, ts)
                self.pending_orders.pop(coid, None)
                logger.info("WS fill booked: %s (delta=%.4f, px=%.4f)", coid, booked, fill_px)
            elif status == "PARTIALLY_FILLED":
                cumulative = event.cumulative_filled_qty
                if cumulative <= 1e-9:
                    cumulative = po.booked_qty + event.last_filled_qty
                if cumulative > po.booked_qty + 1e-9:
                    fill_px = event.last_filled_price if event.last_filled_price > 0 else po.price
                    booked = self._book_fill_delta(po, cumulative, fill_px, ts)
                    if po.remaining_qty <= 1e-9:
                        self.pending_orders.pop(coid, None)
                    logger.info(
                        "WS partial fill booked: %s (delta=%.4f, remaining=%.4f)",
                        coid,
                        booked,
                        po.remaining_qty,
                    )
            elif status in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
                self.pending_orders.pop(coid, None)
                logger.info("WS order dead: %s (%s)", coid, status)

    def _book_pending_fill(self, po: PendingOrder, size: float, price: float, ts: pd.Timestamp) -> None:
        if po.kind == "open":
            existing = self.account.positions.get(self.symbol)
            if existing is None:
                self.account.open_position(self.symbol, po.pos_side, size, price, ts, is_maker=True)
            elif existing.side == po.pos_side:
                self.account.add_to_position(self.symbol, size, price, ts, is_maker=True)
            else:
                logger.error(
                    "Філ %s (%s) при відкритій протилежній позиції %s — пропущено, звірка обов'язкова",
                    po.client_order_id,
                    po.pos_side,
                    existing.side,
                )
        elif self.symbol in self.account.positions:
            self.account.close_position(self.symbol, price, ts, is_maker=True, size=size)

    def _book_fill_delta(self, po: PendingOrder, cumulative_filled: float, price: float, ts: pd.Timestamp) -> float:
        delta = min(cumulative_filled, po.size) - po.booked_qty
        if delta <= 1e-9:
            return 0.0
        self._book_pending_fill(po, delta, price, ts)
        po.booked_qty += delta
        return delta
