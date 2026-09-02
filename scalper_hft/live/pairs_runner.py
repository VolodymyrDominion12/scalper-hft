"""Paper pairs: дві ноги, maker post-only, all-or-none філл, ризик портфеля.

Сигнал PairsArb: +1 = шорт leg1 / лонг leg2; −1 = дзеркально; 0 = флет.
Сигнал на закритті бару t → лімітки по close t → філл на барі t+1, якщо
обидві ноги торкнулись рівня. Інакше unfilled (чекаємо wait_bars, потім скасовуємо).

Позиції ключаться як `{pair}:{symbol}`, щоб BTC у кількох парах не злипався
(як у бектест-портфелі: ноги незалежні).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from scalper_hft.config import get_settings
from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.fills import both_or_neither, decide_fill
from scalper_hft.live.reconcile import reconcile_exchange_state
from scalper_hft.live.risk_gate import CooldownState, correlated_size_mult, decide_entry, open_pair_size_pcts
from scalper_hft.live.store import PaperStore
from scalper_hft.live.trader import closed_klines
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
    ) -> None:
        settings = get_settings()
        self.leg1 = leg1
        self.leg2 = leg2
        self.pid = pair_id(leg1, leg2)
        self.strategy = strategy
        self.account = account
        self.store = store
        self.is_maker = is_maker
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
        if self.losing_months >= self.max_losing_months:
            return False, "два збиткові місяці — пауза"
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
        d1 = decide_fill(o1.side, o1.limit_price, high1, low1)
        d2 = decide_fill(o2.side, o2.limit_price, high2, low2)
        d1, d2 = both_or_neither(d1, d2)
        if d1.filled and d2.filled:
            self._apply_fills(ts, o1, o2, d1.fill_price, d2.fill_price)
            self._log_order(ts, o1, "filled", "filled")
            self._log_order(ts, o2, "filled", "filled")
            self.pending = None
            self.n_filled += 1
            return "filled"
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

    def _apply_fills(self, ts: pd.Timestamp, o1: PendingOrder, o2: PendingOrder, px1: float, px2: float) -> None:
        if o1.reduce_only:
            closed: list[dict] = []
            for o, px in ((o1, px1), (o2, px2)):
                if o.key in self.account.positions:
                    tr = self.account.close_position(o.key, px, ts, is_maker=self.is_maker)
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
        self.account.open_position(o1.key, o1.pos_side, o1.size, px1, ts, is_maker=self.is_maker)
        self.account.open_position(o2.key, o2.pos_side, o2.size, px2, ts, is_maker=self.is_maker)
        self.have = 1 if o1.pos_side == "short" else -1
        if self.is_journal:
            self.is_journal.log(ts, self.pid, o1.symbol, o1.side, o1.limit_price, px1)
            self.is_journal.log(ts, self.pid, o2.symbol, o2.side, o2.limit_price, px2)

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
        parts.append(self._resolve_pending(ts, high1, low1, high2, low2))
        if self.pending is None:
            parts.append(self._quote(ts, int(signal), close1, close2))
        if self.store:
            self.store.log_equity(
                ts, self.pid, self.account.equity_at(marks), self.account.cash, self.account.realized_pnl
            )
        self.last_bar_ts = ts
        return " | ".join(parts)


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
    for ts, row in common.iterrows():
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
    client = BinanceClient()
    batch = client.fetch_klines(symbol, interval, since_ms=0, limit=limit)
    if not batch:
        raise RuntimeError(f"Немає даних для {symbol}")
    df = pd.DataFrame(batch, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts").sort_index()


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
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
        self.engine = PairsEngine(
            leg1, leg2, self.strategy, self.account, store=store, n_pairs=n_pairs, is_maker=is_maker
        )
        self._last_ts: pd.Timestamp | None = None

    def step(self, now: pd.Timestamp | None = None, *, reconcile: bool = True) -> str:
        if reconcile:
            reconcile_exchange_state(self.account, self.client, dry_run=self._dry_run)
        df1 = closed_klines(_fetch_ohlcv(self.leg1, self.interval), self.interval, now=now)
        df2 = closed_klines(_fetch_ohlcv(self.leg2, self.interval), self.interval, now=now)
        common = align_ohlc(df1, df2)
        if len(common) < 50:
            return "hold:мало барів"
        ts = common.index[-1]
        if self._last_ts is not None and ts == self._last_ts:
            return "hold:same_bar"
        sig_df = pd.DataFrame({"leg1": common["l1_close"], "leg2": common["l2_close"]}, index=common.index)
        signal = int(self.strategy.generate_signals(sig_df).iloc[-1])
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
        return action

    def run(self, iterations: int = 10, sleep_sec: int = 300) -> PairsPaperResult:
        actions: list[str] = []
        pts: list[tuple[pd.Timestamp, float]] = []
        for i in range(iterations):
            try:
                action = self.step()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Крок %d: %s", i, exc)
                action = f"error:{exc}"
            actions.append(action)
            pts.append((pd.Timestamp.utcnow().tz_localize(None), self.account.equity))
            if i < iterations - 1:
                time.sleep(sleep_sec)
        eq = pd.Series({t: v for t, v in pts}).sort_index()
        return PairsPaperResult(
            equity=eq,
            actions=actions,
            n_filled=self.engine.n_filled,
            n_unfilled=self.engine.n_unfilled,
            pair=self.engine.pid,
            account=self.account,
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
        self.account = account or PaperAccount(
            initial_capital=10_000.0, taker_fee=settings.taker_fee, maker_fee=settings.maker_fee
        )
        self.store = store
        self.client = client  # лише для звірки у paper (no-op); live заборонено вище
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
                )
            )

    def step(self, now: pd.Timestamp | None = None) -> str:
        reconcile_exchange_state(self.account, self.client, dry_run=self._dry_run)
        return " || ".join(r.step(now=now, reconcile=False) for r in self.runners)

    def run(self, iterations: int = 10, sleep_sec: int = 300) -> PairsPaperResult:
        actions: list[str] = []
        pts: list[tuple[pd.Timestamp, float]] = []
        for i in range(iterations):
            try:
                action = self.step()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Портфель крок %d: %s", i, exc)
                action = f"error:{exc}"
            actions.append(action)
            pts.append((pd.Timestamp.utcnow().tz_localize(None), self.account.equity))
            if i < iterations - 1:
                time.sleep(sleep_sec)
        filled = sum(r.engine.n_filled for r in self.runners)
        unfilled = sum(r.engine.n_unfilled for r in self.runners)
        return PairsPaperResult(
            equity=pd.Series({t: v for t, v in pts}).sort_index(),
            actions=actions,
            n_filled=filled,
            n_unfilled=unfilled,
            pair="portfolio",
            account=self.account,
        )
