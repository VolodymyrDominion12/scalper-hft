"""Paper pairs: дві ноги, maker post-only, all-or-none філл, ризик портфеля.

Сигнал PairsArb: +1 = шорт leg1 / лонг leg2; −1 = дзеркально; 0 = флет.
Сигнал на закритті бару t → лімітки по close t → філл на барі t+1, якщо
обидві ноги торкнулись рівня. Інакше unfilled (чекаємо wait_bars, потім скасовуємо).

Позиції ключаться як `{pair}:{symbol}`, щоб BTC у кількох парах не злипався
(як у бектест-портфелі: ноги незалежні).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.bar_clock import daemon_sleep_sec
from scalper_hft.live.control import DEFAULT_CONTROL_PATH, ControlState, load_control
from scalper_hft.live.fills import both_or_neither, decide_fill
from scalper_hft.live.reconcile import reconcile_exchange_state
from scalper_hft.live.risk_gate import CooldownState, correlated_size_mult, decide_entry, open_pair_size_pcts
from scalper_hft.live.store import PaperStore
from scalper_hft.live.sync_engine import SyncEngine
from scalper_hft.live.trader import closed_klines
from scalper_hft.live.ws_user_stream import OrderTradeEvent
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.pairs_arb import PairsArb

logger = logging.getLogger(__name__)

VALIDATED_PAIRS: tuple[dict, ...] = (
    {"leg1": "XRPUSDT", "leg2": "BTCUSDT", "entry_z": 2.0, "exit_z": 0.3, "lookback": 480},
    {"leg1": "BTCUSDT", "leg2": "ETHUSDT", "entry_z": 2.0, "exit_z": 0.3, "lookback": 240},
    {"leg1": "LINKUSDT", "leg2": "BTCUSDT", "entry_z": 2.0, "exit_z": 0.3, "lookback": 240},
)

_RECENT_BARS = 800


@dataclass
class PendingOrder:
    symbol: str
    key: str
    side: str  # buy | sell
    pos_side: str  # long | short
    size: float
    limit_price: float
    reduce_only: bool
    placed_ts: pd.Timestamp
    bars_waited: int = 0

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "key": self.key,
            "side": self.side,
            "pos_side": self.pos_side,
            "size": float(self.size),
            "limit_price": float(self.limit_price),
            "reduce_only": bool(self.reduce_only),
            "placed_ts": str(self.placed_ts),
            "bars_waited": int(self.bars_waited),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> PendingOrder:
        return cls(
            symbol=str(data["symbol"]),
            key=str(data["key"]),
            side=str(data["side"]),
            pos_side=str(data["pos_side"]),
            size=float(data["size"]),
            limit_price=float(data["limit_price"]),
            reduce_only=bool(data["reduce_only"]),
            placed_ts=pd.Timestamp(data["placed_ts"]),
            bars_waited=int(data.get("bars_waited") or 0),
        )


@dataclass
class PairsPaperResult:
    equity: pd.Series
    actions: list[str]
    n_filled: int = 0
    n_unfilled: int = 0
    pair: str = ""
    account: PaperAccount | None = None

    def summary(self) -> str:
        if self.equity.empty or self.account is None:
            return f"Paper pairs {self.pair}: порожньо"
        ret = float(self.equity.iloc[-1] / self.equity.iloc[0] - 1) if self.equity.iloc[0] else 0.0
        total = self.n_filled + self.n_unfilled
        fill_pct = self.n_filled / total if total else 0.0
        return (
            f"Paper pairs {self.pair}\n"
            f"  кроків: {len(self.equity)} | капітал: {self.account.equity:.2f} ({ret:+.2%})\n"
            f"  філлів: {self.n_filled} | unfilled: {self.n_unfilled} | fill-rate: {fill_pct:.0%}\n"
            f"  угод (ніг): {len(self.account.trades)} | PnL: {self.account.realized_pnl:+.2f}"
        )


def pair_id(leg1: str, leg2: str) -> str:
    return f"{leg1}/{leg2}"


def pos_key(pid: str, symbol: str) -> str:
    return f"{pid}:{symbol}"


def align_ohlc(leg1: pd.DataFrame, leg2: pd.DataFrame) -> pd.DataFrame:
    a = leg1[["open", "high", "low", "close"]].add_prefix("l1_")
    b = leg2[["open", "high", "low", "close"]].add_prefix("l2_")
    return a.join(b, how="inner").dropna()


def legs_for_want(want: int) -> tuple[str, str]:
    """pos_side для (leg1, leg2). want=+1 → short/long."""
    if want > 0:
        return "short", "long"
    if want < 0:
        return "long", "short"
    return "", ""


def pair_size_pct(n_pairs: int, pair_cap: float, port_cap: float) -> float:
    """Ноціонал однієї ноги: min(ліміт пари, портфель / кількість пар)."""
    n = max(int(n_pairs), 1)
    return min(pair_cap, port_cap / n)


class PairsEngine:
    """Стан однієї пари на спільному PaperAccount."""

    def __init__(
        self,
        leg1: str,
        leg2: str,
        strategy: Strategy,
        account: PaperAccount,
        store: PaperStore | None = None,
        n_pairs: int = 1,
        wait_bars: int | None = None,
        is_maker: bool = True,
        legging_mode: str = "strict_both",
        max_drift_bps: float = 10.0,
        coint_kill: bool = True,
    ) -> None:
        settings = get_settings()
        self.leg1 = leg1
        self.leg2 = leg2
        self.pid = pair_id(leg1, leg2)
        self.strategy = strategy
        self.account = account
        self.store = store
        self.is_maker = is_maker
        self.legging_mode = legging_mode
        self.max_drift_bps = max_drift_bps
        self.coint_kill = coint_kill
        self.portfolio_block_entries = False
        self.control_block_entries = False
        self._spread_hist = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        self.wait_bars = wait_bars if wait_bars is not None else settings.maker_fill_wait_bars
        self.size_pct = pair_size_pct(n_pairs, settings.pair_notional_pct, settings.portfolio_notional_pct)
        self.max_losing_months = settings.max_losing_months
        self.daily_loss_limit = settings.daily_loss_limit
        self.weekly_loss_limit = settings.weekly_loss_limit
        self.cooldown_losses = settings.cooldown_losses
        self.max_consecutive_losses = settings.max_consecutive_losses
        self.cooldown_hours = settings.cooldown_hours
        self.cooldown_size_mult = settings.cooldown_size_mult
        self.corr_notional_cap = settings.corr_notional_cap
        self.cooldown = CooldownState()
        self.consecutive_pair_losses = 0
        self._entry_size_mult = 1.0
        self.week_start_equity = account.equity
        self._last_day: object | None = None
        self._last_week: tuple[int, int] | None = None
        from scalper_hft.live.is_log import IsJournal

        self.is_journal = IsJournal()
        self.pending: tuple[PendingOrder, PendingOrder] | None = None
        self.have = 0
        self.losing_months = 0
        self._month_key: str | None = None
        self._month_start_eq: float = account.equity
        self.n_filled = 0
        self.n_unfilled = 0
        self.last_bar_ts: pd.Timestamp | None = None
        self._ws_leg1_fill: float | None = None
        self._ws_leg2_fill: float | None = None

    def _k(self, symbol: str) -> str:
        return pos_key(self.pid, symbol)

    def _marks(self, p1: float, p2: float) -> dict[str, float]:
        return {self._k(self.leg1): p1, self._k(self.leg2): p2}

    def _roll_month(self, ts: pd.Timestamp, equity: float) -> None:
        month = f"{ts.year:04d}-{ts.month:02d}"
        if self._month_key is None:
            self._month_key = month
            self._month_start_eq = equity
            return
        if month == self._month_key:
            return
        pnl = equity - self._month_start_eq
        if self.store:
            self.store.log_month(self.pid, self._month_key, pnl)
        if pnl < 0:
            self.losing_months += 1
        else:
            self.losing_months = 0
        logger.info("%s місяць %s PnL=%+.2f | підряд збиткових=%d", self.pid, self._month_key, pnl, self.losing_months)
        self._month_key = month
        self._month_start_eq = equity

    def _can_open(self, ts: pd.Timestamp | None = None) -> tuple[bool, str]:
        if self.portfolio_block_entries:
            return False, "портфельний ліміт збитків"
        if self.control_block_entries:
            return False, "control:no_new_entries"
        if self.losing_months >= self.max_losing_months:
            return False, "два збиткові місяці — пауза"
        if self.coint_kill and not self._spread_hist.empty:
            from scalper_hft.live.pair_health import pair_entry_allowed

            lookback = int(self.strategy.get("lookback", 240))
            window = self._spread_hist.tail(max(lookback, 80))
            ok, reason = pair_entry_allowed(window)
            if not ok:
                return False, reason
        if self.account.equity <= self.account.day_start_equity * (1 - self.daily_loss_limit):
            return False, "денний ліміт збитків"
        if self.account.equity <= self.week_start_equity * (1 - self.weekly_loss_limit):
            return False, "тижневий ліміт збитків"
        now = ts if ts is not None else (self.last_bar_ts or pd.Timestamp.now(tz="UTC").tz_convert(None))
        decision = decide_entry(
            consecutive_losses=self.consecutive_pair_losses,
            now=now,
            cooldown=self.cooldown,
            cooldown_losses=self.cooldown_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            cooldown_hours=self.cooldown_hours,
            cooldown_size_mult=self.cooldown_size_mult,
            flattening=False,
        )
        self.cooldown = decision.cooldown
        self._entry_size_mult = decision.size_mult
        if decision.status == "reject":
            return False, decision.reason
        return True, decision.reason

    def _log_order(self, ts: pd.Timestamp, o: PendingOrder, status: str, reason: str) -> None:
        if self.store:
            self.store.log_order(ts, self.pid, o.symbol, o.side, o.size, o.limit_price, status, reason)

    def _resolve_pending(self, ts: pd.Timestamp, high1: float, low1: float, high2: float, low2: float) -> str:
        if self.pending is None:
            return "no_pending"
        o1, o2 = self.pending
        mid1 = 0.5 * (high1 + low1)
        mid2 = 0.5 * (high2 + low2)
        d1 = decide_fill(o1.side, o1.limit_price, high1, low1)
        d2 = decide_fill(o2.side, o2.limit_price, high2, low2)

        if self.legging_mode == "strict_both":
            d1, d2 = both_or_neither(d1, d2)
            if d1.filled and d2.filled:
                self._apply_fills(ts, o1, o2, d1.fill_price, d2.fill_price, True, True)
                self._log_order(ts, o1, "filled", "filled")
                self._log_order(ts, o2, "filled", "filled")
                self.pending = None
                self.n_filled += 1
                return "filled"
        else:
            from scalper_hft.live.fills import resolve_legging

            res = resolve_legging(
                d1,
                d2,
                o1.side,
                o2.side,
                o1.limit_price,
                o2.limit_price,
                mid1,
                mid2,
                max_drift_bps=self.max_drift_bps,
                mode=self.legging_mode,
            )
            if res.action in ("both_filled", "chase_leg1", "chase_leg2"):
                self._apply_fills(
                    ts,
                    o1,
                    o2,
                    res.d1.fill_price,
                    res.d2.fill_price,
                    res.leg1_maker,
                    res.leg2_maker,
                )
                self._log_order(ts, o1, "filled", res.d1.reason)
                self._log_order(ts, o2, "filled", res.d2.reason)
                self.pending = None
                self.n_filled += 1
                return f"filled:{res.action}"
            elif res.action in ("unwind_leg1", "unwind_leg2"):
                filled = o1 if res.action == "unwind_leg1" else o2
                px = mid1 if res.action == "unwind_leg1" else mid2
                self._unwind_filled_leg(ts, filled, px)
                self._log_order(ts, o1, "unfilled", f"legging_unwound_{res.action}")
                self._log_order(ts, o2, "unfilled", f"legging_unwound_{res.action}")
                self.pending = None
                self.n_unfilled += 1
                logger.warning("%s legging risk triggered %s, drift=%.1f bps", self.pid, res.action, res.drift_bps)
                return f"unfilled:{res.action}"

        o1.bars_waited += 1
        o2.bars_waited += 1
        if o1.bars_waited >= self.wait_bars:
            reason = d1.reason
            self._log_order(ts, o1, "unfilled", reason)
            self._log_order(ts, o2, "unfilled", reason)
            self.pending = None
            self.n_unfilled += 1
            logger.info("%s unfilled %s/%s: %s", self.pid, o1.symbol, o2.symbol, reason)
            return f"unfilled:{reason}"
        return "pending"

    def on_ws_order_trade(self, event: OrderTradeEvent, now: pd.Timestamp | None = None) -> str:
        """Обробка події ORDER_TRADE_UPDATE для парного трейдингу.

        Якщо одна з ніг заповнюється через maker WS, негайно
        виконує другу ногу (taker chase) або фіксує подвійний філ.
        """
        if self.pending is None:
            return "no_pending"
        o1, o2 = self.pending
        ts = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
        status = event.status.upper()
        if status not in ("FILLED", "CLOSED"):
            return "ignored"

        is_leg1 = event.symbol == o1.symbol or event.client_order_id == o1.key
        is_leg2 = event.symbol == o2.symbol or event.client_order_id == o2.key
        if not (is_leg1 or is_leg2):
            return "symbol_mismatch"

        fill_px = (
            event.last_filled_price if event.last_filled_price > 0 else (o1.limit_price if is_leg1 else o2.limit_price)
        )

        if self.legging_mode == "strict_both":
            if is_leg1:
                self._ws_leg1_fill = fill_px
            if is_leg2:
                self._ws_leg2_fill = fill_px
            px1 = getattr(self, "_ws_leg1_fill", None)
            px2 = getattr(self, "_ws_leg2_fill", None)
            if px1 is not None and px2 is not None:
                self._apply_fills(ts, o1, o2, px1, px2, True, True)
                self._log_order(ts, o1, "filled", "ws_strict_both")
                self._log_order(ts, o2, "filled", "ws_strict_both")
                self.pending = None
                self.n_filled += 1
                self._ws_leg1_fill = None
                self._ws_leg2_fill = None
                return "both_filled"
            return "waiting_other_leg"

        # У режимах з chase (taker chase другої ноги)
        if is_leg1:
            self._apply_fills(ts, o1, o2, fill_px, o2.limit_price, maker1=True, maker2=False)
            self._log_order(ts, o1, "filled", "ws_maker_leg1")
            self._log_order(ts, o2, "filled", "ws_chase_leg2")
            self.pending = None
            self.n_filled += 1
            return "chase_leg2"
        else:
            self._apply_fills(ts, o1, o2, o1.limit_price, fill_px, maker1=False, maker2=True)
            self._log_order(ts, o1, "filled", "ws_chase_leg1")
            self._log_order(ts, o2, "filled", "ws_maker_leg2")
            self.pending = None
            self.n_filled += 1
            return "chase_leg1"

    def _unwind_filled_leg(self, ts: pd.Timestamp, order: PendingOrder, price: float) -> None:
        """Flatten уже відкриту ногу taker-ом; pending без позиції — no-op."""
        if order.key not in self.account.positions:
            return
        tr = self.account.close_position(order.key, price, ts, is_maker=False)
        if self.store:
            self.store.log_trade(ts, self.pid, tr)
        self.have = 0

    def _apply_fills(
        self,
        ts: pd.Timestamp,
        o1: PendingOrder,
        o2: PendingOrder,
        px1: float,
        px2: float,
        maker1: bool | None = None,
        maker2: bool | None = None,
    ) -> None:
        m1 = self.is_maker if maker1 is None else maker1
        m2 = self.is_maker if maker2 is None else maker2
        if o1.reduce_only:
            closed: list[dict] = []
            for o, px, mk in ((o1, px1, m1), (o2, px2, m2)):
                if o.key in self.account.positions:
                    tr = self.account.close_position(o.key, px, ts, is_maker=mk)
                    closed.append(tr)
                    if self.store:
                        self.store.log_trade(ts, self.pid, tr)
            self.have = 0
            pair_pnl = sum(float(t.get("pnl") or 0.0) for t in closed)
            if closed:
                self.account.consecutive_losses = 1 if pair_pnl < 0 else 0
                if pair_pnl < 0:
                    self.consecutive_pair_losses += 1
                else:
                    self.consecutive_pair_losses = 0
            return
        self.account.open_position(o1.key, o1.pos_side, o1.size, px1, ts, is_maker=m1)
        self.account.open_position(o2.key, o2.pos_side, o2.size, px2, ts, is_maker=m2)
        self.have = 1 if o1.pos_side == "short" else -1
        if self.is_journal:
            self.is_journal.log(ts, self.pid, o1.symbol, o1.side, o1.limit_price, px1, is_maker=m1)
            self.is_journal.log(ts, self.pid, o2.symbol, o2.side, o2.limit_price, px2, is_maker=m2)

    def _quote(self, ts: pd.Timestamp, want: int, p1: float, p2: float) -> str:
        if want == self.have:
            return "hold"
        # Прямий реверс (want = -have): спершу закрити обидві ноги (reduce_only,
        # all-or-none), новий вхід — лише після підтвердженого закриття наступним
        # баром. Інакше котирували б відкриття поверх відкритої позиції →
        # ValueError у open_position (позиція вже існує) і вічно заклинений рушій.
        if want != 0 and self.have != 0 and want != self.have:
            want = 0  # реверс = спочатку flat; новий напрямок — коли сигнал ще активний
        if want != 0 and self.have == 0:
            ok, reason = self._can_open(ts)
            if not ok:
                return f"blocked:{reason}"
        equity = self.account.equity_at(self._marks(p1, p2))
        size_pct = self.size_pct
        if want != 0 and self.have == 0:
            size_pct *= self._entry_size_mult
            others = open_pair_size_pcts(self.account.positions, self.size_pct)
            others.pop(self.pid, None)
            size_pct *= correlated_size_mult(self.pid, self.size_pct, others, self.corr_notional_cap)
            if size_pct <= 0:
                return "blocked:корельований ноціонал"
        if want == 0:
            pos1 = self.account.positions.get(self._k(self.leg1))
            pos2 = self.account.positions.get(self._k(self.leg2))
            if pos1 is None or pos2 is None:
                self.have = 0
                return "hold"
            o1 = PendingOrder(
                self.leg1,
                pos1.symbol,
                "sell" if pos1.side == "long" else "buy",
                pos1.side,
                pos1.size,
                p1,
                True,
                ts,
            )
            o2 = PendingOrder(
                self.leg2,
                pos2.symbol,
                "sell" if pos2.side == "long" else "buy",
                pos2.side,
                pos2.size,
                p2,
                True,
                ts,
            )
        else:
            s1, s2 = legs_for_want(want)
            size1 = size_pct * equity / p1
            size2 = size_pct * equity / p2
            o1 = PendingOrder(
                self.leg1,
                self._k(self.leg1),
                "sell" if s1 == "short" else "buy",
                s1,
                size1,
                p1,
                False,
                ts,
            )
            o2 = PendingOrder(
                self.leg2,
                self._k(self.leg2),
                "sell" if s2 == "short" else "buy",
                s2,
                size2,
                p2,
                False,
                ts,
            )
        self.pending = (o1, o2)
        self._log_order(ts, o1, "pending", f"want={want}")
        self._log_order(ts, o2, "pending", f"want={want}")
        return f"quoted want={want}"

    def on_bar(
        self,
        ts: pd.Timestamp,
        high1: float,
        low1: float,
        close1: float,
        high2: float,
        low2: float,
        close2: float,
        signal: int,
        funding1: float | None = None,
        funding2: float | None = None,
    ) -> str:
        """Один закритий бар: спроба філла pending → новий сигнал → котирування."""
        marks = self._marks(close1, close2)
        self.account.mark(marks)
        parts: list[str] = []
        if funding1 is not None:
            pnl = self.account.apply_funding(self._k(self.leg1), funding1, ts)
            if pnl and self.store:
                self.store.log_trade(
                    ts, self.pid, {"type": "funding", "symbol": self.leg1, "pnl": pnl, "side": "", "size": 0}
                )
        if funding2 is not None:
            pnl = self.account.apply_funding(self._k(self.leg2), funding2, ts)
            if pnl and self.store:
                self.store.log_trade(
                    ts, self.pid, {"type": "funding", "symbol": self.leg2, "pnl": pnl, "side": "", "size": 0}
                )

        equity = self.account.equity_at(marks)
        day = ts.date()
        if getattr(self, "_last_day", None) is not None and day != self._last_day:
            self.account.roll_to_new_day(equity)
            # пауза/cooldown діє лише до кінця дня (як у LiveTrader): скидаємо
            # і парний лічильник серії збитків, інакше пара гальмує «назавжди»
            # після max_consecutive_losses збитків, розтягнутих на кілька днів.
            # Активне cooldown-вікно (CooldownState.until) зберігається, як і в
            # LiveTrader.risk_check — воно само згасне за часом.
            self.consecutive_pair_losses = 0
        self._last_day = day
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))
        if self._last_week is not None and week != self._last_week:
            self.week_start_equity = equity
        self._last_week = week
        self._roll_month(ts, equity)
        if close1 > 0 and close2 > 0:
            self._spread_hist.loc[pd.Timestamp(ts)] = float(np.log(close1) - np.log(close2))  # type: ignore[call-overload]
        if self.is_journal.records:
            mid1 = 0.5 * (high1 + low1)
            mid2 = 0.5 * (high2 + low2)
            self.is_journal.apply_next_bar_markout(self.leg1, mid1)
            self.is_journal.apply_next_bar_markout(self.leg2, mid2)
        parts.append(self._resolve_pending(ts, high1, low1, high2, low2))
        if self.pending is None:
            parts.append(self._quote(ts, int(signal), close1, close2))
        if self.store:
            self.store.log_equity(
                ts, self.pid, self.account.equity_at(marks), self.account.cash, self.account.realized_pnl
            )
        self.last_bar_ts = ts
        return " | ".join(parts)

    def cancel_pending(self, reason: str = "shutdown") -> bool:
        """Скасувати pending maker-ордери пари (при зупинці або скиданні)."""
        if self.pending is None:
            return False
        logger.info("Скасовано pending-ордери пари %s (%s)", self.pid, reason)
        self.pending = None
        return True

    def seed_spread_from_ohlc(self, common: pd.DataFrame) -> None:
        """Відновити ADF-вікно після рестарту з уже завантажених klines."""
        if common is None or common.empty:
            return
        if "l1_close" not in common.columns or "l2_close" not in common.columns:
            return
        l1 = common["l1_close"].astype(float)
        l2 = common["l2_close"].astype(float)
        ok = (l1 > 0) & (l2 > 0)
        self._spread_hist = pd.Series(np.log(l1[ok]) - np.log(l2[ok]), index=l1[ok].index, dtype=float)

    def to_snapshot(self) -> dict[str, Any]:
        pending: list[dict[str, Any]] | None = None
        if self.pending is not None:
            pending = [self.pending[0].to_snapshot(), self.pending[1].to_snapshot()]
        last_week = list(self._last_week) if self._last_week is not None else None
        until = self.cooldown.until
        return {
            "last_bar_ts": str(self.last_bar_ts) if self.last_bar_ts is not None else None,
            "have": int(self.have),
            "losing_months": int(self.losing_months),
            "month_key": self._month_key,
            "month_start_eq": float(self._month_start_eq),
            "n_filled": int(self.n_filled),
            "n_unfilled": int(self.n_unfilled),
            "consecutive_pair_losses": int(self.consecutive_pair_losses),
            "week_start_equity": float(self.week_start_equity),
            "last_day": str(self._last_day) if self._last_day is not None else None,
            "last_week": last_week,
            "pending": pending,
            "cooldown_until": str(until) if until is not None else None,
            "cooldown_reason": self.cooldown.reason,
        }

    def apply_snapshot(self, data: dict[str, Any]) -> None:
        raw_ts = data.get("last_bar_ts")
        self.last_bar_ts = pd.Timestamp(raw_ts) if raw_ts else None
        self.have = int(data.get("have") or 0)
        self.losing_months = int(data.get("losing_months") or 0)
        self._month_key = data.get("month_key")
        if data.get("month_start_eq") is not None:
            self._month_start_eq = float(data["month_start_eq"])
        self.n_filled = int(data.get("n_filled") or 0)
        self.n_unfilled = int(data.get("n_unfilled") or 0)
        self.consecutive_pair_losses = int(data.get("consecutive_pair_losses") or 0)
        if data.get("week_start_equity") is not None:
            self.week_start_equity = float(data["week_start_equity"])
        day = data.get("last_day")
        self._last_day = None if not day else pd.Timestamp(str(day)).date()
        lw = data.get("last_week")
        self._last_week = (int(lw[0]), int(lw[1])) if lw and len(lw) == 2 else None
        pending = data.get("pending")
        if pending and len(pending) == 2:
            self.pending = (PendingOrder.from_snapshot(pending[0]), PendingOrder.from_snapshot(pending[1]))
        else:
            self.pending = None
        until_raw = data.get("cooldown_until")
        self.cooldown = CooldownState(
            until=pd.Timestamp(until_raw) if until_raw else None,
            reason=str(data.get("cooldown_reason") or ""),
        )


def replay_pairs(
    leg1: str,
    leg2: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    strategy: Strategy | None = None,
    account: PaperAccount | None = None,
    store: PaperStore | None = None,
    funding1: pd.DataFrame | None = None,
    funding2: pd.DataFrame | None = None,
    n_pairs: int = 1,
    wait_bars: int | None = None,
    is_maker: bool = True,
    now: pd.Timestamp | None = None,
    interval: str = "1h",
) -> PairsPaperResult:
    """Історичний прогін з моделлю maker-філлів (сигнал t → філл t+1)."""
    settings = get_settings()
    c1 = closed_klines(df1, interval, now=now)
    c2 = closed_klines(df2, interval, now=now)
    common = align_ohlc(c1, c2)
    if len(common) < 50:
        raise ValueError("Замало спільних барів для пари")
    strategy = strategy or PairsArb()
    account = account or PaperAccount(
        initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
    )
    engine = PairsEngine(
        leg1, leg2, strategy, account, store=store, n_pairs=n_pairs, wait_bars=wait_bars, is_maker=is_maker
    )
    sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
    signals = strategy.generate_signals(sig_df)

    equity_pts: list[tuple[pd.Timestamp, float]] = []
    actions: list[str] = []
    prev: pd.Timestamp | None = None
    for ts_raw, row in common.iterrows():
        ts = pd.Timestamp(str(ts_raw))
        action = engine.on_bar(
            ts,
            float(row["l1_high"]),
            float(row["l1_low"]),
            float(row["l1_close"]),
            float(row["l2_high"]),
            float(row["l2_low"]),
            float(row["l2_close"]),
            int(signals.loc[ts]) if ts in signals.index else 0,
            funding1=_funding_between(funding1, prev, ts),
            funding2=_funding_between(funding2, prev, ts),
        )
        actions.append(action)
        equity_pts.append((ts, account.equity_at(engine._marks(float(row["l1_close"]), float(row["l2_close"])))))
        prev = ts

    eq = pd.Series({t: v for t, v in equity_pts}).sort_index()
    return PairsPaperResult(
        equity=eq,
        actions=actions,
        n_filled=engine.n_filled,
        n_unfilled=engine.n_unfilled,
        pair=engine.pid,
        account=account,
    )


def _funding_between(funding: pd.DataFrame | None, prev: pd.Timestamp | None, ts: pd.Timestamp) -> float | None:
    if funding is None or funding.empty or "fundingRate" not in funding.columns:
        return None
    idx = funding.index
    mask = idx <= ts if prev is None else (idx > prev) & (idx <= ts)
    rates = funding.loc[mask, "fundingRate"]
    if rates.empty:
        return None
    return float(rates.sum())


def _fetch_ohlcv(symbol: str, interval: str, limit: int = _RECENT_BARS) -> pd.DataFrame:
    client = ExchangeClient()
    batch = client.fetch_klines(symbol, interval, since_ms=0, limit=limit)
    if not batch:
        raise RuntimeError(f"Немає даних для {symbol}")
    df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts").sort_index()


def should_persist_action(action: str) -> bool:
    """Не писати знімок на pause / той самий бар / помилку."""
    if action.startswith("error:") or action.startswith("hold:paused"):
        return False
    stripped = action.replace("halt:portfolio_loss ", "")
    parts = [p.strip() for p in stripped.split("||")]
    if parts and all("hold:same_bar" in p for p in parts):
        return False
    return True


def _install_stop_signals(stop: threading.Event) -> None:
    def _handle(signum: int, _frame: object) -> None:
        logger.info("сигнал %s — зупиняю paper-демон", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _paper_loop(
    step: Callable[[], str],
    save: Callable[[], None] | None,
    interval: str,
    account: PaperAccount,
    pair: str,
    n_filled: Callable[[], int],
    n_unfilled: Callable[[], int],
    *,
    daemon: bool,
    iterations: int,
    sleep_sec: int,
    stop: threading.Event | None = None,
    install_signals: bool = True,
    on_stop: Callable[[], Any] | None = None,
) -> PairsPaperResult:
    halt = stop or threading.Event()
    if daemon and install_signals:
        _install_stop_signals(halt)
    actions: list[str] = []
    pts: list[tuple[pd.Timestamp, float]] = []
    i = 0
    try:
        while not halt.is_set():
            if not daemon and i >= iterations:
                break
            try:
                action = step()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Крок %d: %s", i, exc)
                action = f"error:{exc}"
            actions.append(action)
            pts.append((pd.Timestamp.now(tz="UTC").tz_convert(None), account.equity))
            i += 1
            if daemon:
                if halt.wait(timeout=daemon_sleep_sec(interval, action)):
                    break
            elif i < iterations:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        logger.info("Paper loop перервано користувачем (Ctrl+C)")
    finally:
        if on_stop is not None:
            try:
                on_stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Помилка on_stop cleanup: %s", exc)
        if save is not None:
            save()
    eq = pd.Series({t: v for t, v in pts}).sort_index() if pts else pd.Series(dtype=float)
    return PairsPaperResult(
        equity=eq,
        actions=actions,
        n_filled=n_filled(),
        n_unfilled=n_unfilled(),
        pair=pair,
        account=account,
    )


def merge_runtime_payload(
    existing: dict[str, Any] | None,
    account: PaperAccount,
    runners: dict[str, dict[str, Any]],
    portfolio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(existing or {})
    payload["version"] = 1
    payload["account"] = account.to_snapshot()
    merged = dict(payload.get("runners") or {})
    merged.update(runners)
    payload["runners"] = merged
    if portfolio is not None:
        payload["portfolio"] = portfolio
    return payload


class PairsPaperRunner:
    """Цикл paper на одній парі (REST → закритий бар → engine).

    ⚠ Paper-only: рушій СИМУЛЮЄ maker-філи на локальному PaperAccount і НЕ
    ставить реальні ордери. Тому DRY_RUN=false тут заборонено — звірка з
    реальною біржею була б безглуздою (локальні ноги ніколи не співпадуть з
    реальними позиціями) і закінчувалась KillSwitch. Live-pairs потребує
    окремого адаптера з реальними ордерами ніг (див. M3 у code review).
    """

    def __init__(
        self,
        leg1: str,
        leg2: str,
        interval: str = "1h",
        strategy: Strategy | None = None,
        account: PaperAccount | None = None,
        store: PaperStore | None = None,
        n_pairs: int = 1,
        is_maker: bool = True,
        client: object | None = None,
        restore: bool = True,
        control_path: Path | str | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError(
                "PairsPaperRunner — paper-only: реальні ордери ніг не реалізовані. "
                "Використовуйте DRY_RUN=true (paper); live-pairs потребує адаптера "
                "з реальними ордерами."
            )
        self._dry_run = settings.dry_run
        self.leg1 = leg1
        self.leg2 = leg2
        self.interval = interval or "1h"
        self.strategy = strategy or PairsArb()
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
        self.control_path = Path(control_path) if control_path is not None else DEFAULT_CONTROL_PATH
        payload: dict[str, Any] | None = None
        if restore and store is not None and account is None:
            payload = store.load_runtime()
            if payload and "account" in payload:
                account = PaperAccount.from_snapshot(payload["account"])
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        self.engine = PairsEngine(
            leg1, leg2, self.strategy, self.account, store=store, n_pairs=n_pairs, is_maker=is_maker
        )
        mode = "paper" if self._dry_run else "live"
        self.sync_engine = SyncEngine(store=store, exchange_id="binance", mode=mode) if store else None
        self._last_ts: pd.Timestamp | None = None
        if restore and store is not None:
            payload = payload or store.load_runtime()
            state = (payload or {}).get("runners", {}).get(self.engine.pid)
            if state:
                self.engine.apply_snapshot(state)
                self._last_ts = self.engine.last_bar_ts

    def save_runtime(self) -> None:
        if self.store is None:
            return
        existing = self.store.load_runtime()
        payload = merge_runtime_payload(existing, self.account, {self.engine.pid: self.engine.to_snapshot()})
        self.store.save_runtime(payload)

    def _persist_if_needed(self, action: str) -> None:
        if self.store is not None and should_persist_action(action):
            self.save_runtime()

    def step(
        self,
        now: pd.Timestamp | None = None,
        *,
        reconcile: bool = True,
        control: ControlState | None = None,
        persist: bool = True,
    ) -> str:
        ctrl = control if control is not None else load_control(self.control_path)
        if ctrl.pause:
            return "hold:paused"
        if reconcile:
            reconcile_exchange_state(
                self.account,
                self.client,
                dry_run=self._dry_run,
                scope={self.leg1, self.leg2},
            )
        self.engine.control_block_entries = ctrl.no_new_entries
        df1 = closed_klines(_fetch_ohlcv(self.leg1, self.interval), self.interval, now=now)
        df2 = closed_klines(_fetch_ohlcv(self.leg2, self.interval), self.interval, now=now)
        common = align_ohlc(df1, df2)
        if len(common) < 50:
            return "hold:мало барів"
        self.engine.seed_spread_from_ohlc(common)
        ts = common.index[-1]
        if self._last_ts is not None and ts == self._last_ts:
            return "hold:same_bar"
        sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
        signal = 0 if ctrl.flatten else int(self.strategy.generate_signals(sig_df).iloc[-1])
        row = common.iloc[-1]
        action = self.engine.on_bar(
            ts,
            float(row["l1_high"]),
            float(row["l1_low"]),
            float(row["l1_close"]),
            float(row["l2_high"]),
            float(row["l2_low"]),
            float(row["l2_close"]),
            signal,
        )
        self._last_ts = ts
        logger.info("%s %s | equity=%.2f", self.engine.pid, action, self.account.equity)
        if persist:
            self._persist_if_needed(action)
        return action

    def on_ws_order_trade(self, event: OrderTradeEvent, now: pd.Timestamp | None = None) -> str:
        """Передати подію WebSocket стріму до PairsEngine."""
        res = self.engine.on_ws_order_trade(event, now=now)
        if res not in ("no_pending", "symbol_mismatch", "ignored"):
            self._persist_if_needed(res)
        return res

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 300,
        *,
        daemon: bool = False,
        stop: threading.Event | None = None,
        install_signals: bool = True,
    ) -> PairsPaperResult:
        if self.sync_engine:
            self.sync_engine.start()

        def _on_stop():
            if self.sync_engine:
                self.sync_engine.stop()
            self.engine.cancel_pending(reason="shutdown")

        return _paper_loop(
            self.step,
            self.save_runtime if self.store is not None else None,
            self.interval,
            self.account,
            self.engine.pid,
            lambda: self.engine.n_filled,
            lambda: self.engine.n_unfilled,
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            stop=stop,
            install_signals=install_signals,
            on_stop=_on_stop,
        )


class PairsPortfolioRunner:
    """Кілька валідованих пар на спільному рахунку.

    ⚠ Paper-only: як і PairsPaperRunner, не ставить реальні ордери →
    DRY_RUN=false заборонено (див. M3 у code review).
    """

    def __init__(
        self,
        configs: list[dict] | None = None,
        interval: str = "1h",
        account: PaperAccount | None = None,
        store: PaperStore | None = None,
        is_maker: bool = True,
        client: object | None = None,
        restore: bool = True,
        control_path: Path | str | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.dry_run:
            raise RuntimeError(
                "PairsPortfolioRunner — paper-only: реальні ордери ніг не реалізовані. "
                "Використовуйте DRY_RUN=true (paper); live-pairs потребує адаптера "
                "з реальними ордерами."
            )
        self._dry_run = settings.dry_run
        self.configs = configs or [dict(p) for p in VALIDATED_PAIRS]
        self.interval = interval
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
        self.control_path = Path(control_path) if control_path is not None else DEFAULT_CONTROL_PATH
        payload: dict[str, Any] | None = None
        if restore and store is not None and account is None:
            payload = store.load_runtime()
            if payload and "account" in payload:
                account = PaperAccount.from_snapshot(payload["account"])
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        n = len(self.configs)
        self.runners: list[PairsPaperRunner] = []
        for cfg in self.configs:
            strat = PairsArb(
                entry_z=float(cfg.get("entry_z", 2.0)),
                exit_z=float(cfg.get("exit_z", 0.3)),
                lookback=int(cfg.get("lookback", 240)),
            )
            self.runners.append(
                PairsPaperRunner(
                    cfg["leg1"],
                    cfg["leg2"],
                    interval=interval,
                    strategy=strat,
                    account=self.account,
                    store=store,
                    n_pairs=n,
                    is_maker=is_maker,
                    client=self.client,
                    restore=False,
                    control_path=self.control_path,
                )
            )
        self.daily_loss_limit = settings.daily_loss_limit
        self.weekly_loss_limit = settings.weekly_loss_limit
        self.week_start_equity = self.account.equity
        self._last_week: tuple[int, int] | None = None
        self.enable_vol_target = False
        mode = "paper" if self._dry_run else "live"
        self.sync_engine = SyncEngine(store=store, exchange_id="binance", mode=mode) if store else None

        if payload:
            for r in self.runners:
                state = (payload.get("runners") or {}).get(r.engine.pid)
                if state:
                    r.engine.apply_snapshot(state)
                    r._last_ts = r.engine.last_bar_ts
            port = payload.get("portfolio") or {}
            if port.get("week_start_equity") is not None:
                self.week_start_equity = float(port["week_start_equity"])
            lw = port.get("last_week")
            if lw and len(lw) == 2:
                self._last_week = (int(lw[0]), int(lw[1]))

    def _roll_week(self, now: pd.Timestamp | None) -> None:
        ts = now if now is not None else pd.Timestamp.now(tz="UTC").tz_convert(None)
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))
        if self._last_week is not None and week != self._last_week:
            self.week_start_equity = self.account.equity
        self._last_week = week

    def _should_halt_entries(self) -> bool:
        eq = self.account.equity
        if eq <= self.account.day_start_equity * (1.0 - self.daily_loss_limit):
            return True
        if eq <= self.week_start_equity * (1.0 - self.weekly_loss_limit):
            return True
        return False

    def save_runtime(self) -> None:
        if self.store is None:
            return
        runners = {r.engine.pid: r.engine.to_snapshot() for r in self.runners}
        last_week = list(self._last_week) if self._last_week is not None else None
        payload = merge_runtime_payload(
            self.store.load_runtime(),
            self.account,
            runners,
            portfolio={"week_start_equity": self.week_start_equity, "last_week": last_week},
        )
        self.store.save_runtime(payload)

    def _persist_if_needed(self, action: str) -> None:
        if self.store is not None and should_persist_action(action):
            self.save_runtime()

    def step(self, now: pd.Timestamp | None = None) -> str:
        ctrl = load_control(self.control_path)
        if ctrl.pause:
            return "hold:paused"
        self._roll_week(now)
        scope = {sym for r in self.runners for sym in (r.leg1, r.leg2)}
        reconcile_exchange_state(self.account, self.client, dry_run=self._dry_run, scope=scope)
        halt = self._should_halt_entries()
        for r in self.runners:
            r.engine.portfolio_block_entries = halt
            r.engine.control_block_entries = ctrl.no_new_entries
        prefix = "halt:portfolio_loss " if halt else ""
        action = prefix + " || ".join(
            r.step(now=now, reconcile=False, control=ctrl, persist=False) for r in self.runners
        )
        self._persist_if_needed(action)
        return action

    def cancel_all_pending(self, reason: str = "shutdown") -> int:
        count = 0
        for r in self.runners:
            if r.engine.cancel_pending(reason=reason):
                count += 1
        return count

    def on_ws_order_trade(self, event: OrderTradeEvent, now: pd.Timestamp | None = None) -> list[str]:
        """Диспетчеризація WS подій до відповідного парного раннера."""
        results: list[str] = []
        for r in self.runners:
            res = r.on_ws_order_trade(event, now=now)
            if res not in ("no_pending", "symbol_mismatch", "ignored"):
                results.append(f"{r.engine.pid}:{res}")
        return results

    def run(
        self,
        iterations: int = 10,
        sleep_sec: int = 300,
        *,
        daemon: bool = False,
        stop: threading.Event | None = None,
        install_signals: bool = True,
    ) -> PairsPaperResult:
        if self.sync_engine:
            self.sync_engine.start()

        def _on_stop():
            if self.sync_engine:
                self.sync_engine.stop()
            self.cancel_all_pending(reason="shutdown")

        return _paper_loop(
            self.step,
            self.save_runtime if self.store is not None else None,
            self.interval,
            self.account,
            "portfolio",
            lambda: sum(r.engine.n_filled for r in self.runners),
            lambda: sum(r.engine.n_unfilled for r in self.runners),
            daemon=daemon,
            iterations=iterations,
            sleep_sec=sleep_sec,
            stop=stop,
            install_signals=install_signals,
            on_stop=_on_stop,
        )
