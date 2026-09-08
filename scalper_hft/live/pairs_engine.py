"""PairsEngine: maker all-or-none філи, ризик і snapshot однієї пари."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import both_or_neither, decide_fill
from scalper_hft.live.risk_gate import CooldownState, correlated_size_mult, decide_entry, open_pair_size_pcts
from scalper_hft.live.store import PaperStore
from scalper_hft.live.ws_user_stream import OrderTradeEvent
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


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
            closed: list[dict[str, Any]] = []
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

    def _quote(self, ts: pd.Timestamp, want: int, p1: float, p2: float, size_mult: float = 1.0) -> str:
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
            # regime_scale overlay: дробовий сигнал масштабує ноціонал входу.
            # Для цілих сигналів {-1,0,1} size_mult=1.0 — без зміни поведінки.
            size_pct *= max(0.0, min(size_mult, 1.0))
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
        signal: float,
        funding1: float | None = None,
        funding2: float | None = None,
    ) -> str:
        """Один закритий бар: спроба філла pending → новий сигнал → котирування.

        signal: дробовий у [-1,1] (regime_scale) або цілий {-1,0,1} (класичний).
        Напрямок = sign(signal), butціонал входу масштабується на |signal|
        (для цілих сигналів size_mult=1.0 — без зміни поведінки).
        """
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
            # Дробовий сигнал (regime_scale): напрямок = sign, butціонал ∝ |signal|.
            # Для цілих {-1,0,1} size_mult=1.0 — без зміни поведінки.
            sig_f = float(signal)
            want = int(np.sign(sig_f))
            size_mult = abs(sig_f) if sig_f != 0.0 else 1.0
            parts.append(self._quote(ts, want, close1, close2, size_mult=size_mult))
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
