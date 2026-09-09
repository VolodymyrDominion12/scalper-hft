"""Live-адаптер ніг pairs: реальні ордери, reconcile, KillSwitch (Фаза 3).

Композиція над PairsEngine: движок відповідає за сигнал/ризик/стан (have),
адаптер — за реальне розміщення/скасування/опитування ордерів ніг та звірку
з біржею.

Контракт (DEPLOY_PLAN.md, Фаза 3):
  - Ніколи market-ордери на вході (тільки post_only limit). Chase друга нога —
    окремо, після maker-філа першої (taker IOC), через resolve_legging.
  - Exchange = source of truth. На старті: fetch_positions + fetch_open_orders
    → гідратація; порожній локальний рахунок при наявності ніг на біржі =
    KillSwitch (fail-closed).
  - Drift під час роботи → KillSwitch, не торгувати.
  - Ідемпотентні client_order_id (next_client_order_id).
  - DRY_RUN=true → адаптер не викликається (paper-руннер як і раніше).
  - Немає тихого шляху ввімкнути live: явний DRY_RUN=false + ключі + адаптер.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import FillDecision, resolve_legging
from scalper_hft.live.intent_store import IntentStore
from scalper_hft.live.orders import next_client_order_id
from scalper_hft.live.pairs_engine import PairsEngine, PendingOrder
from scalper_hft.live.reconcile import KillSwitch, parse_exchange_positions, reconcile_positions
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class LiveLegOrder:
    """Реальний ордер ноги на біржі (живе в self._live_pending)."""

    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: str  # buy | sell
    size: float
    limit_price: float
    reduce_only: bool
    pos_side: str
    placed_ts: pd.Timestamp
    bars_waited: int = 0
    filled: bool = False
    fill_price: float = 0.0


class PairsLiveAdapter(PairsEngine):
    """PairsEngine + реальні ордери ніг на біржі.

    Підклас PairsEngine: перевизначає _quote (розміщення post_only лімітів) та
    _resolve_pending (опитування біржі замість OHLC-touch). Сигнал/ризик/стан
    (have) — від базового рушія. Reconcile на старті → гідратація have.
    """

    def __init__(
        self,
        leg1: str,
        leg2: str,
        strategy: Strategy,
        account: PaperAccount,
        client: Any,  # ExchangeClient з create_order/cancel_order/fetch_order/fetch_positions
        store: Any | None = None,
        n_pairs: int = 1,
        wait_bars: int | None = None,
        is_maker: bool = True,
        legging_mode: str = "chase",
        max_drift_bps: float = 10.0,
        coint_kill: bool = True,
        intent_store: IntentStore | None = None,
        vol_target_ann: float | None = None,
        vol_lookback: int = 168,
        bars_per_year: float = 8760.0,
    ) -> None:
        super().__init__(
            leg1,
            leg2,
            strategy,
            account,
            store=store,
            n_pairs=n_pairs,
            wait_bars=wait_bars,
            is_maker=is_maker,
            legging_mode=legging_mode,
            max_drift_bps=max_drift_bps,
            coint_kill=coint_kill,
            vol_target_ann=vol_target_ann,
            vol_lookback=vol_lookback,
            bars_per_year=bars_per_year,
        )
        if not is_maker:
            raise ValueError("PairsLiveAdapter: лише maker (post_only) на вході; taker заборонено")
        self.client = client
        # реальні ордери ніг (coid -> LiveLegOrder), паралельно до self.pending (paper)
        self._live_pending: tuple[LiveLegOrder, LiveLegOrder] | None = None
        self._reconciled = False
        self._intent_store = intent_store

    # ── Reconcile / гідратація на старті ──────────────────────────────────────

    def start_live(self) -> str:
        """На старті live: звірка позицій + відкритих ордерів, гідратація have.

        Exchange = source of truth. Повертає ('ok', '') або кидає KillSwitch.
        """
        if not hasattr(self.client, "fetch_positions"):
            raise RuntimeError("Live адаптер потребує клієнт з fetch_positions")
        # Clock sync: розсинхрон >1s ламає підписані запити (timestamp/recvWindow)
        check_clock = getattr(self.client, "assert_clock_synced", None)
        if callable(check_clock):
            check_clock(max_drift_ms=1000.0)  # RuntimeError при розсинхроні
        raw_pos = self.client.fetch_positions()
        ex_pos = parse_exchange_positions(raw_pos)
        scope = {self.leg1, self.leg2}
        ok, reason = reconcile_positions(self.account, ex_pos, scope=scope)
        if not ok:
            # порожній локальний + є ноги на біржі → гідратуємо; розбіжність → KillSwitch
            if not self.account.positions and ex_pos:
                self._hydrate_from_exchange(ex_pos)
            else:
                raise KillSwitch(f"reconcile на старті: {reason}")
        # скасувати сторонні відкриті ордери наших символів (залишок попереднього прогону)
        if hasattr(self.client, "cancel_all_orders"):
            for sym in scope:
                try:
                    self.client.cancel_all_orders(sym)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cancel_all_orders %s: %s", sym, exc)
        self._reconciled = True
        return "ok"

    def _hydrate_from_exchange(self, ex_pos: dict[str, Any]) -> None:
        """Гідратація локального PaperAccount з біржі (старт після рестарту)."""
        ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
        prices: dict[str, float] = {}
        for sym in ex_pos:
            prices[sym] = self._fetch_last_price(sym)
        for sym, ex in ex_pos.items():
            key = self._k(sym)
            self.account.open_position(key, ex.side, ex.size, prices[sym], ts, is_maker=True)
        # have = +1 якщо leg1 short / leg2 long (want=+1 → short leg1 / long leg2)
        leg1_pos = self.account.positions.get(self._k(self.leg1))
        if leg1_pos is None:
            self.have = 0
        elif leg1_pos.side == "short":
            self.have = 1
        else:
            self.have = -1
        logger.info("%s гідратовано з біржі: have=%d, позицій=%d", self.pid, self.have, len(self.account.positions))

    def _fetch_last_price(self, symbol: str) -> float:
        """Остання ціна символу для mark-ціни при гідратації. Немає ціни → KillSwitch."""
        try:
            if not hasattr(self.client, "fetch_klines"):
                raise KillSwitch(f"{self.pid}: немає mark-ціни {symbol} (клієнт без fetch_klines)")
            batch = self.client.fetch_klines(symbol, "1m", since_ms=0, limit=1)
        except KillSwitch:
            raise
        except Exception as exc:
            raise KillSwitch(f"{self.pid}: немає mark-ціни {symbol} ({exc})") from exc
        if not batch:
            raise KillSwitch(f"{self.pid}: немає mark-ціни {symbol} (порожня відповідь)")
        close = float(batch[-1][4])
        if close <= 0.0 or close != close:
            raise KillSwitch(f"{self.pid}: немає mark-ціни {symbol} (close={close})")
        return close

    # ── Розміщення ордерів (перевизначення _quote) ────────────────────────────

    def _quote(self, ts: pd.Timestamp, want: int, p1: float, p2: float, size_mult: float = 1.0) -> str:
        """Базовий _quote формує paper PendingOrder; адаптер ДОВІДНО розміщує
        реальні post_only ліміти обох ніг на біржі. Fail-closed: якщо хоч одна
        нога не розмістилась — друга скасовується, входу немає."""
        action = super()._quote(ts, want, p1, p2, size_mult=size_mult)
        if not action.startswith("quoted") or self.pending is None:
            return action
        o1, o2 = self.pending
        legs = self._place_legs(o1, o2)
        if legs is None:
            # не вдалося розмістити обидві ноги — не лишаємо ані paper-pending,
            # ані сирітського ордера на біржі (rollback)
            self.pending = None
            self._live_pending = None
            self.n_unfilled += 1
            return "quote_failed:place_error"
        self._live_pending = legs
        return f"quoted want={want}"

    def _place_legs(self, o1: PendingOrder, o2: PendingOrder) -> tuple[LiveLegOrder, LiveLegOrder] | None:
        """Розмістити обидві ноги post_only limit на біржі (ідемпотентні coid).

        Якщо перша нога розміщена, а друга впала — першу скасовуємо (rollback),
        щоб не лишити сирітський ордер. None = вхід скасовано."""
        lo1 = self._place_one_leg(o1)
        if lo1 is None:
            return None
        lo2 = self._place_one_leg(o2)
        if lo2 is None:
            logger.error("%s rollback: нога %s не розмістилась — скасовую %s", self.pid, o2.symbol, o1.symbol)
            self._cancel_live_leg(lo1)
            return None
        return lo1, lo2

    def _intent_key(self, kind: str, symbol: str, side: str, placed_ts: pd.Timestamp) -> str:
        """Стабільний ключ наміру між retry одного бара."""
        ts_iso = placed_ts.isoformat() if hasattr(placed_ts, "isoformat") else str(placed_ts)
        return f"{self.pid}:{kind}:{symbol}:{side}:{ts_iso}"

    def _coid_for(self, kind: str, symbol: str, side: str, placed_ts: pd.Timestamp, prefix: str) -> str:
        """Той самий clientOrderId на retry наміру; новий UUID лише вперше."""
        if self._intent_store is None:
            return next_client_order_id(prefix)
        key = self._intent_key(kind, symbol, side, placed_ts)
        existing = self._intent_store.get(key)
        if existing:
            return existing
        coid = next_client_order_id(prefix)
        self._intent_store.put(key, coid)
        return coid

    def _forget_intent(self, kind: str, symbol: str, side: str, placed_ts: pd.Timestamp) -> None:
        if self._intent_store is None:
            return
        self._intent_store.pop(self._intent_key(kind, symbol, side, placed_ts))

    def _sanitize_leg(
        self, symbol: str, side: str, size: float, price: float | None
    ) -> tuple[float, float | None] | None:
        """Нормалізація розміру/ціни під фільтри біржі. None = відхилено."""
        sanitize = getattr(self.client, "sanitize_order", None)
        if sanitize is None:
            return size, price
        try:
            qty, px, err = sanitize(symbol, side, size, price)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s sanitize_order %s: %s", self.pid, symbol, exc)
            return None
        if err is not None:
            logger.error("%s ордер відхилено фільтрами біржі: %s %s %s → %s", self.pid, side, symbol, size, err)
            return None
        return qty, px if px is not None else price

    def _place_one_leg(self, o: PendingOrder) -> LiveLegOrder | None:
        """Розмістити одну ногу; None при помилці (НЕ букуємо фейковий oid)."""
        sanitized = self._sanitize_leg(o.symbol, o.side, o.size, o.limit_price)
        if sanitized is None:
            return None
        size, limit_price = sanitized
        coid = self._coid_for("entry", o.symbol, o.side, o.placed_ts, "shp")
        try:
            resp = self.client.create_order(
                o.symbol,
                "limit",
                o.side,
                size,
                price=limit_price,
                params={"reduceOnly": True} if o.reduce_only else {},
                post_only=True,
                client_order_id=coid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s помилка розміщення %s %s: %s", self.pid, o.side, o.symbol, exc)
            return None
        oid = str((resp or {}).get("id") or "")
        if not oid:
            logger.error("%s біржа не повернула order id для %s — відхиляю", self.pid, o.symbol)
            return None
        logger.info(
            "%s розміщено %s %s %s @ %s coid=%s oid=%s",
            self.pid,
            o.side,
            size,
            o.symbol,
            limit_price,
            coid,
            oid,
        )
        return LiveLegOrder(
            client_order_id=coid,
            exchange_order_id=oid,
            symbol=o.symbol,
            side=o.side,
            size=size,
            limit_price=limit_price,
            reduce_only=o.reduce_only,
            pos_side=o.pos_side,
            placed_ts=o.placed_ts,
        )

    # ── Опитування / філи (перевизначення _resolve_pending) ──────────────────

    def _resolve_pending(self, ts: pd.Timestamp, high1: float, low1: float, high2: float, low2: float) -> str:
        """Опитування біржі замість OHLC-touch. All-or-none + chase/unwind."""
        if self._live_pending is None:
            return "no_pending"
        lo1, lo2 = self._live_pending
        d1 = self._poll_leg(lo1)
        d2 = self._poll_leg(lo2)

        if self.legging_mode == "strict_both":
            d1, d2 = self._both_or_neither_live(d1, d2)
            if d1.filled and d2.filled:
                self._apply_live_fills(ts, lo1, lo2, d1.fill_price, d2.fill_price, True, True)
                self._live_pending = None
                self.pending = None
                self.n_filled += 1
                return "filled"
        else:
            mid1 = 0.5 * (high1 + low1)
            mid2 = 0.5 * (high2 + low2)
            res = resolve_legging(
                d1,
                d2,
                lo1.side,
                lo2.side,
                lo1.limit_price,
                lo2.limit_price,
                mid1,
                mid2,
                max_drift_bps=self.max_drift_bps,
                mode=self.legging_mode,
            )
            if res.action == "both_filled":
                self._apply_live_fills(
                    ts, lo1, lo2, res.d1.fill_price, res.d2.fill_price, res.leg1_maker, res.leg2_maker
                )
                self._live_pending = None
                self.pending = None
                self.n_filled += 1
                return f"filled:{res.action}"
            if res.action in ("chase_leg1", "chase_leg2"):
                # Chase = РЕАЛЬНИЙ taker IOC на незаповнену ногу. Букування —
                # лише після підтвердженого філа біржі; невдача → реальний
                # unwind заповненої ноги (ніколи не "paper-на-live").
                chase_lo, maker_lo = (lo1, lo2) if res.action == "chase_leg1" else (lo2, lo1)
                chase_mid = mid1 if res.action == "chase_leg1" else mid2
                ok, chase_px = self._chase_leg_taker(chase_lo, chase_mid)
                if ok:
                    px1 = chase_px if res.action == "chase_leg1" else res.d1.fill_price
                    px2 = chase_px if res.action == "chase_leg2" else res.d2.fill_price
                    self._apply_live_fills(ts, lo1, lo2, px1, px2, res.leg1_maker, res.leg2_maker)
                    self._live_pending = None
                    self.pending = None
                    self.n_filled += 1
                    return f"filled:{res.action}"
                logger.error(
                    "%s chase %s не виконано біржею — unwind заповненої ноги %s",
                    self.pid,
                    chase_lo.symbol,
                    maker_lo.symbol,
                )
                maker_mid = mid1 if maker_lo is lo1 else mid2
                self._unwind_live_leg(ts, maker_lo, maker_mid)
                self._live_pending = None
                self.pending = None
                self.n_unfilled += 1
                return f"unfilled:chase_failed_{chase_lo.symbol}"
            if res.action in ("unwind_leg1", "unwind_leg2"):
                filled = lo1 if res.action == "unwind_leg1" else lo2
                px = mid1 if res.action == "unwind_leg1" else mid2
                self._unwind_live_leg(ts, filled, px)
                self._cancel_live_leg(lo2 if res.action == "unwind_leg1" else lo1)
                self._live_pending = None
                self.pending = None
                self.n_unfilled += 1
                logger.warning("%s legging risk %s drift=%.1f bps", self.pid, res.action, res.drift_bps)
                return f"unfilled:{res.action}"

        # neither filled — чекати або timeout
        lo1.bars_waited += 1
        lo2.bars_waited += 1
        if lo1.bars_waited >= self.wait_bars:
            self._cancel_live_leg(lo1)
            self._cancel_live_leg(lo2)
            self._live_pending = None
            self.pending = None
            self.n_unfilled += 1
            return "unfilled:timeout"
        return "pending"

    def _poll_leg(self, lo: LiveLegOrder) -> FillDecision:
        """Опитати статус ордера на біржі (REST). Повертає FillDecision."""
        try:
            resp = self.client.fetch_order(lo.exchange_order_id, lo.symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s fetch_order %s: %s", self.pid, lo.client_order_id, exc)
            return FillDecision(False, lo.limit_price, "poll_error")
        status = str((resp or {}).get("status") or "").lower()
        filled_qty = float((resp or {}).get("filled") or (resp or {}).get("filledQty") or 0.0)
        if status in ("filled", "closed") or filled_qty >= lo.size * 0.999:
            lo.filled = True
            lo.fill_price = float((resp or {}).get("average") or (resp or {}).get("price") or lo.limit_price)
            return FillDecision(True, lo.fill_price, "filled")
        if status in ("canceled", "cancelled", "expired", "rejected"):
            return FillDecision(False, lo.limit_price, f"order_{status}")
        return FillDecision(False, lo.limit_price, "pending")

    @staticmethod
    def _both_or_neither_live(d1: FillDecision, d2: FillDecision) -> tuple[FillDecision, FillDecision]:
        if d1.filled and d2.filled:
            return d1, d2
        reason = "unfilled_partial" if (d1.filled or d2.filled) else "unfilled_no_touch"
        return (FillDecision(False, d1.fill_price, reason), FillDecision(False, d2.fill_price, reason))

    def _apply_live_fills(
        self, ts: pd.Timestamp, lo1: LiveLegOrder, lo2: LiveLegOrder, px1: float, px2: float, m1: bool, m2: bool
    ) -> None:
        """Бронювати філи в локальний PaperAccount (як у базовому _apply_fills)."""
        # делегуємо бронювання базовому рушію через paper PendingOrder-обгортку
        o1 = PendingOrder(lo1.symbol, self._k(lo1.symbol), lo1.side, lo1.pos_side, lo1.size, px1, lo1.reduce_only, ts)
        o2 = PendingOrder(lo2.symbol, self._k(lo2.symbol), lo2.side, lo2.pos_side, lo2.size, px2, lo2.reduce_only, ts)
        self.pending = (o1, o2)
        self._apply_fills(ts, o1, o2, px1, px2, maker1=m1, maker2=m2)
        self._forget_intent("entry", lo1.symbol, lo1.side, lo1.placed_ts)
        self._forget_intent("entry", lo2.symbol, lo2.side, lo2.placed_ts)
        self._forget_intent("chase", lo1.symbol, lo1.side, lo1.placed_ts)
        self._forget_intent("chase", lo2.symbol, lo2.side, lo2.placed_ts)

    def _chase_leg_taker(self, lo: LiveLegOrder, mid: float) -> tuple[bool, float]:
        """Реальний taker-chase ноги: IOC limit з ціновим захистом.

        Ціна-кеп = mid ± max_drift_bps (захист від просковзування; drift уже
        перевірений resolve_legging). Повертає (ok, fill_price): ok=True лише
        при ПІДТВЕРДЖЕНОМУ повному філі біржі. Частковий/нульовий IOC —
        невдача (залишок IOC скасовується біржею сам).
        """
        drift_cap = self.max_drift_bps / 10_000.0
        cap_price = mid * (1.0 + drift_cap) if lo.side == "buy" else mid * (1.0 - drift_cap)
        sanitized = self._sanitize_leg(lo.symbol, lo.side, lo.size, cap_price)
        if sanitized is None:
            return False, 0.0
        size, chase_price = sanitized
        params: dict[str, Any] = {"timeInForce": "IOC"}
        if lo.reduce_only:
            params["reduceOnly"] = True
        try:
            resp = self.client.create_order(
                lo.symbol,
                "limit",
                lo.side,
                size,
                price=chase_price,
                params=params,
                client_order_id=self._coid_for("chase", lo.symbol, lo.side, lo.placed_ts, "shc"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s chase %s помилка ордера: %s", self.pid, lo.symbol, exc)
            return False, 0.0
        status = str((resp or {}).get("status") or "").lower()
        filled_qty = float((resp or {}).get("filled") or 0.0)
        if status in ("filled", "closed") or (0 < size * 0.999 <= filled_qty):
            px = float((resp or {}).get("average") or chase_price)
            logger.info("%s chase %s виконано @ %s", self.pid, lo.symbol, px)
            return True, px
        logger.warning("%s chase %s не заповнений (status=%s filled=%s)", self.pid, lo.symbol, status, filled_qty)
        return False, 0.0

    def _unwind_live_leg(self, ts: pd.Timestamp, lo: LiveLegOrder, price: float) -> None:
        """Закрити вже заповнену ногу РЕАЛЬНИМ reduce-only market ордером.

        Fail-closed: якщо біржа не підтвердила flatten — KillSwitch (на біржі
        лишилась однонога позиція, торгувати далі небезпечно).
        """
        close_side = "sell" if lo.side == "buy" else "buy"
        sanitized = self._sanitize_leg(lo.symbol, close_side, lo.size, None)
        if sanitized is None:
            raise KillSwitch(f"{self.pid}: unwind {lo.symbol} відхилено фільтрами біржі — однонога позиція!")
        size, _ = sanitized
        try:
            resp = self.client.create_order(
                lo.symbol,
                "market",
                close_side,
                size,
                params={"reduceOnly": True},
                client_order_id=self._coid_for("unwind", lo.symbol, close_side, lo.placed_ts, "shu"),
            )
            fill_px = float((resp or {}).get("average") or 0.0) or price
        except Exception as exc:  # noqa: BLE001
            raise KillSwitch(
                f"{self.pid}: unwind {lo.symbol} НЕ підтверджений біржею ({exc}) — однонога позиція!"
            ) from exc
        logger.warning("%s unwind %s %s @ %s (reduce-only)", self.pid, close_side, lo.symbol, fill_px)
        o = PendingOrder(lo.symbol, self._k(lo.symbol), lo.side, lo.pos_side, lo.size, fill_px, True, ts)
        self._unwind_filled_leg(ts, o, fill_px)

    def _cancel_live_leg(self, lo: LiveLegOrder) -> None:
        """Скасувати незаповнений реальний ордер ноги на біржі."""
        try:
            self.client.cancel_order(lo.exchange_order_id, lo.symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s cancel %s: %s", self.pid, lo.client_order_id, exc)

    # ── Reconcile під час роботи (drift → KillSwitch) ─────────────────────────

    def reconcile_runtime(self) -> None:
        """Періодична звірка позицій з біржею; drift → KillSwitch.

        Fail-closed: помилка fetch_positions — це теж KillSwitch (не знаємо
        стан біржі → торгувати небезпечно), а не log-and-continue."""
        if not self._reconciled:
            return
        try:
            raw = self.client.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            raise KillSwitch(f"{self.pid}: reconcile_runtime fetch_positions впав: {exc}") from exc
        ok, reason = reconcile_positions(self.account, parse_exchange_positions(raw), scope={self.leg1, self.leg2})
        if not ok:
            raise KillSwitch(f"drift під час роботи: {reason}")

    def cancel_pending(self, reason: str = "shutdown") -> bool:
        """Скасувати реальні pending ордери ніг (при зупинці)."""
        if self._live_pending is not None:
            for lo in self._live_pending:
                self._cancel_live_leg(lo)
            self._live_pending = None
        self.pending = None
        return super().cancel_pending(reason=reason)


__all__ = ["LiveLegOrder", "PairsLiveAdapter"]
