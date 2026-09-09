"""Live-трейдер: цикл "дані → сигнал → ризик → виконання".

Режими:
    - paper: симуляція через PaperAccount (без API-ключів або testnet);
    - testnet: реальні ордери на Binance testnet (API-ключі testnet);
    - live: реальні ордери (DRY_RUN=false + EXCHANGE=binance) — свідомо!

Ризик-контроль перед кожним ордером (книга, гл. 4):
    - ліміт позиції (POSITION_PCT від капіталу);
    - денний ліміт збитків (DAILY_LOSS_LIMIT);
    - пауза після MAX_CONSECUTIVE_LOSSES збитків.

Order lifecycle (C2): у live з maker-виконанням ордер ставиться як
limit+postOnly і реєструється у `pending_orders`; ЛОКАЛЬНА позиція брониться
лише після підтвердженого філа (`poll_pending_orders` → fetch_order). Не-
заповнені ордери скасовуються за сигналом або таймаутом; часткові філи
бронюються за фактичним об'ємом. Paper/testnet market-шлях — миттєвий філ,
як раніше.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings, require_live_credentials
from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.exit_ladders import OneWayTradingLadder
from scalper_hft.live.pending_orders import PendingOrder, PendingOrderManager
from scalper_hft.live.risk_gate import (
    CooldownState,
    decide_entry,
    margin_proximity_ok,
    per_symbol_notional_ok,
)
from scalper_hft.live.trader_bars import as_naive_utc, closed_klines, interval_seconds
from scalper_hft.live.trader_loop import (
    SilentAttritionKillSwitch,
    TradeDecision,
    execute_signal,
    run_trader_once,
)
from scalper_hft.live.ws_user_stream import BinanceUserDataStream, OrderTradeEvent
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


def _intent_key(
    symbol: str,
    side: str,
    kind: str,
    reduce_only: bool,
    interval: str,
    now: pd.Timestamp | None = None,
) -> str:
    """Ключ наміру для ідемпотентного coid: один намір = одне рішення бару.

    Retry у межах того ж бару → той самий ключ (і той самий coid); наступний
    бар → новий ключ, тож stale coid попереднього рішення не успадковується.
    """
    ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    bar_bucket = int(ts.timestamp() // interval_seconds(interval))
    return f"{symbol}:{side}:{kind}:{int(reduce_only)}:{bar_bucket}"


def _is_duplicate_order_id(exc: Exception) -> bool:
    """Чи є виняток помилкою 'такий clientOrderId вже існує' (ccxt/Binance -4116)."""
    try:
        import ccxt

        if isinstance(exc, ccxt.DuplicateOrderId):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "duplicate" in msg and "order" in msg


class LiveTrader:
    """Керує одним символом: свіжі klines → сигнал → ордер (paper/testnet/live).


    Розширення (Спринт 4):
        vol_sizing  — розмір позиції масштабується волатильністю
                      (менша позиція у високій волі): vol_ref/realized_vol;
        hmm_block   — блокувати НОВІ входи у «неспокійному» HMM-режимі
                      (каузальна модель, без lookahead);
        breakeven   — використовуйте strategy.use_breakeven_gate через
                      get_strategy(..., breakeven_gate=True).
    """

    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        interval: str = "1m",
        account: PaperAccount | None = None,
        client: ExchangeClient | None = None,
        vol_sizing: bool = False,
        vol_ref: float | None = None,
        hmm_block: bool = False,
        hmm_states: int = 3,
        hmm_threshold: float = 0.5,
        control_path: Path | str | None = None,
    ) -> None:
        self.settings = get_settings()
        require_live_credentials(self.settings)
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval or "1m"
        self.account = account or PaperAccount(
            initial_capital=self.settings.position_pct * 100_000,
            taker_fee=self.settings.taker_fee,
            maker_fee=self.settings.maker_fee,
        )
        self.client = client or ExchangeClient(
            self.settings.binance_api_key, self.settings.binance_api_secret, self.settings.exchange, auth=True
        )
        self.cost = CostModel(
            maker_fee=self.settings.maker_fee,
            taker_fee=self.settings.taker_fee,
            slippage_frac=self.settings.slippage_frac,
        )
        self.vol_sizing = vol_sizing
        self.vol_ref = vol_ref
        self.hmm_block = hmm_block
        self.hmm_states = hmm_states
        self.hmm_threshold = hmm_threshold
        # Peak-to-trough DrawdownBreaker (Narang: drawdown control) — як у pairs_engine,
        # але для single-symbol LiveTrader. Просідання від історичного піку
        # equity > max_drawdown_pct → halt + АВТО-flatten позиції.
        from scalper_hft.live.risk_gate import DrawdownBreaker

        self.dd_breaker = DrawdownBreaker(
            max_dd_pct=getattr(self.settings, "max_drawdown_pct", 0.10), high_water=self.account.equity
        )
        self.last_signal: int = 0
        self._last_roll_day: object | None = None
        # Тижневий ліміт збитків (ISO-тиждень), як у pairs runner
        self.week_start_equity = self.account.equity
        self._last_roll_week: tuple[int, int] | None = None
        # Silent Attrition: EWMA PnL на угоду → стоп нових входів (PM Ch. 4, 13)
        self.attrition = SilentAttritionKillSwitch()
        self._attrition_seen_trades = 0
        # Control plane (pause / no_new_entries / flatten), читається у run_trader_once
        self.control_path = control_path
        self.cooldown = CooldownState()
        self._entry_size_mult = 1.0
        self._pending = PendingOrderManager(
            self.symbol,
            self.account,
            self.client,
            dry_run=self.settings.dry_run,
            interval=self.interval,
            maker_fill_wait_bars=int(getattr(self.settings, "maker_fill_wait_bars", 1)),
            partial_fill_policy=str(getattr(self.settings, "partial_fill_policy", "cancel")),
        )
        # Ідемпотентність submit: намір → coid, що пережив таймаут (retry з тим
        # самим id). Персистентно (IntentStore, JSON) — переживає і рестарт
        # процесу, інакше retry після краху розмістив би дубль ордера.
        from scalper_hft.live.intent_store import IntentStore

        self._intent_store = IntentStore()
        # Персистентний намір→coid (IntentStore, JSON): retry з тим самим id
        # переживає і таймаут, і рестарт процесу (інакше — дубль ордера).
        from scalper_hft.live.intent_store import IntentStore

        self._intent_store = IntentStore()
        self._intent_coids: dict[str, str] = {}  # гарячий кеш; істина — store  # гарячий кеш; джерело істини — store
        self._last_market_fill_price: float | None = None
        # M4: live-базис equity з біржі (замість фіктивного депозиту)
        self._live_equity_seeded = False
        self.last_live_equity: float | None = None

        self.use_exit_ladders = getattr(self.settings, "use_exit_ladders", False)
        self.ladder: OneWayTradingLadder | None = None
        self.ws_stream: BinanceUserDataStream | None = None
        # TTL-кеш довідкових даних live-кроку (funding/aggTrades): без нього
        # кожен сигнал гребе REST; TTL=300с достатній для барових кроків.
        self._aux_data_cache: dict[str, tuple[float, pd.DataFrame | None]] = {}
        # HMM-гейт делегує до єдиного RegimeDetector (та сама політика fit, що в
        # backtest/RegimeSupervisor: єдине навчання на перших hmm_fit_bars барах,
        # далі filtered_proba forward-only — без lookahead і без розходження
        # backtest↔live). Раньше live мав власний HMM з refit кожні 250 барів,
        # що давав ту саму модель (фіксований seed + ті самі перші 2000 барів),
        # але дублював логіку і ризикував розходженням при зміні політики.
        self._regime_detector: object | None = None
        self._maker_prob_touch = 0.5

    @property
    def pending_orders(self) -> dict[str, PendingOrder]:
        return self._pending.pending_orders

    @property
    def _pending_lock(self) -> threading.RLock:
        return self._pending.lock

    def _sync_pending_state(self) -> None:
        self._pending.client = self.client
        self._pending.dry_run = bool(self.settings.dry_run)
        self._pending.interval = self.interval
        self._pending.maker_fill_wait_bars = int(getattr(self.settings, "maker_fill_wait_bars", 1))

    def _uses_maker_path(self) -> bool:
        """Maker lifecycle (pending limit) для live і paper (dry_run)."""
        return bool(getattr(self.settings, "maker_execution", False))

    def _paper_maker_touch(self, side: str, limit: float, high: float, low: float, bar_idx: int) -> bool:
        """OHLC-touch як у backtest ``_simulate_maker_fills`` (chase, prob_touch)."""
        rng = np.random.default_rng(42)
        if bar_idx > 0:
            rng.random(bar_idx)
        rand = float(rng.random())
        prob = self._maker_prob_touch
        if side == "buy":
            return (low < limit) or (low == limit and rand < prob)
        return (high > limit) or (high == limit and rand < prob)

    def resolve_pending_orders(
        self,
        ts: pd.Timestamp,
        high: float,
        low: float,
        chase_limit: float,
        bar_idx: int,
    ) -> list[str]:
        """Paper maker: філ resting-ордерів за OHLC-touch (parity з backtest chase).

        ``chase_limit`` = close попереднього бару; перевірка на поточному барі
        (low/high vs limit), як у ``_simulate_maker_fills``.
        """
        if not (self.settings.dry_run and self._uses_maker_path()):
            return []
        self._sync_pending_state()
        events: list[str] = []
        wait_bars = max(int(self._pending.maker_fill_wait_bars), 1)
        with self._pending_lock:
            snapshot = list(self.pending_orders.items())
        for coid, po in snapshot:
            with self._pending_lock:
                live = self.pending_orders.get(coid)
                if live is None:
                    continue
                live.bars_waited += 1
                limit_px = float(live.price)
                if self._paper_maker_touch(live.side, limit_px, high, low, bar_idx):
                    fill_px = limit_px
                    self._pending._book_fill_delta(live, live.size, fill_px, ts)
                    self.pending_orders.pop(coid, None)
                    events.append(f"filled:{coid}")
                    continue
                if live.bars_waited >= wait_bars:
                    self.pending_orders.pop(coid, None)
                    events.append(f"timeout_cancel:{coid}")
        return events

    _AUX_DATA_TTL_SEC = 300.0

    def _cached_aux_data(self, key: str, loader: Callable[[], pd.DataFrame | None]) -> pd.DataFrame | None:
        """TTL-кеш для funding/aggTrades у live-кроці."""
        import time

        now = time.monotonic()
        hit = self._aux_data_cache.get(key)
        if hit is not None and now - hit[0] < self._AUX_DATA_TTL_SEC:
            return hit[1]
        data = loader()
        self._aux_data_cache[key] = (now, data)
        return data

    # ── сигнал ───────────────────────────────────────────────────────────────
    def compute_signal(self, df: pd.DataFrame, now: pd.Timestamp | None = None) -> int:
        """Сигнал стратегії на останньому ЗАКРИТОМУ барі (без lookahead)."""
        closed = closed_klines(df, self.interval, now=now)
        if closed is None or len(closed) < 50:
            return 0
        kwargs: dict = {}
        if self.strategy.needs_funding:
            from scalper_hft.data.downloader import download_funding

            kwargs["funding"] = self._cached_aux_data("funding", lambda: download_funding(self.symbol, days=30))
        if self.strategy.needs_trades:
            from scalper_hft.data.downloader import download_agg_trades

            kwargs["trades"] = self._cached_aux_data("trades", lambda: download_agg_trades(self.symbol, days=1))
        signal = self.strategy.generate_signals(closed, **kwargs)
        return int(signal.iloc[-1]) if len(signal) else 0

    # ── волатильність-залежний розмір (Спринт 4: GARCH/realized-vol sizing) ──
    def _realized_vol(self, df: pd.DataFrame, window: int = 60) -> float:
        """Поточна реалізована волатильність (частка ціни) на останньому барі."""
        ret = df["close"].pct_change().dropna()
        if len(ret) < 10:
            return 0.0
        return float(ret.tail(window).std(ddof=0))

    def vol_scaled_size(self, base_size: float, df: pd.DataFrame, window: int = 60) -> float:
        """Розмір позиції з інверсним vol-масштабуванням.

        scale = vol_ref / realized_vol, обмежений [0.25, 3.0]. Якщо
        vol_sizing вимкнено або vol_ref не задано — base_size без змін.
        """
        if not self.vol_sizing or self.vol_ref is None or self.vol_ref <= 0:
            return base_size
        vol = self._realized_vol(df, window)
        if vol <= 0:
            return base_size
        scale = min(max(self.vol_ref / vol, 0.25), 3.0)
        return base_size * scale

    # ── нелінійний soft-penalty розмір (Narang Ch. 4) ─────────────────────────
    def soft_penalty_size(
        self,
        base_size: float,
        signal_strength: float,
        limit_size: float,
        k: float = 1.0,
    ) -> float:
        """Експоненційна штрафна функція розміру ставки замість hard cut-off.

        size = limit * (1 - exp(-k * |signal| / limit)).
        """
        if limit_size <= 0:
            return 0.0
        sig_abs = abs(signal_strength)
        dampened = limit_size * (1.0 - np.exp(-k * sig_abs / limit_size))
        return float(min(base_size, dampened))

    # ── HMM-режимний блок нових входів (Спринт 4, без lookahead) ────────────

    def hmm_blocked(self, df: pd.DataFrame) -> bool:
        """True, якщо поточний HMM-стан «неспокійний» (висока волатильність).

        Делегує до єдиного `RegimeDetector` (той самий, що в backtest/
        RegimeSupervisor): одне навчання на перших `hmm_fit_bars` барах,
        поточна ймовірність — фільтрована (forward-only) → без lookahead і без
        розходження backtest↔live. Блокує лише НОВІ входи; close ніколи не
        блокується (див. `trader_loop.execute_signal`).
        """
        if not self.hmm_block:
            return False
        try:
            from scalper_hft.features.regime_detector import RegimeDetector

            close = df["close"]
            if len(close) < max(self.hmm_states * 20, 80):
                return False
            det = self._regime_detector
            if det is None:
                det = RegimeDetector(
                    n_hmm_states=self.hmm_states,
                    hmm_fit_bars=2000,
                    hmm_seed=42,
                )
                self._regime_detector = det
            state_df = det.detect(close)
            calm = det.hmm_calm_state
            if calm is None:
                return False
            col = f"hmm_p{calm}"
            if col not in state_df.columns:
                return False
            calm_prob = float(state_df[col].iloc[-1])
            return calm_prob < self.hmm_threshold
        except Exception:
            logger.exception("HMM-гейт: помилка моделі — блокуємо нові входи")
            return True

    def maybe_roll_day(self, now: pd.Timestamp | None = None) -> bool:
        """Скинути денний ліміт збитків при зміні UTC-доби.

        Повертає True, якщо відбувся roll. Перший виклик лише запам'ятовує день.
        """
        from scalper_hft.data.downloader import _utc_now

        ts = as_naive_utc(now if now is not None else _utc_now())
        day = ts.date()
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))
        if self._last_roll_week is None:
            self._last_roll_week = week
        elif week != self._last_roll_week:
            self.week_start_equity = self.account.equity
            self._last_roll_week = week
            logger.info("Новий тиждень %s: week_start_equity=%.2f", week, self.week_start_equity)
        if self._last_roll_day is None:
            self._last_roll_day = day
            return False
        if day != self._last_roll_day:
            self.account.roll_to_new_day(self.account.equity)
            self._last_roll_day = day
            logger.info("Новий день %s: day_start_equity=%.2f", day, self.account.day_start_equity)
            return True
        return False

    def _update_attrition(self) -> None:
        """Скормити SilentAttritionKillSwitch нові закриті угоди з журналу."""
        trades = self.account.trades
        if len(trades) <= self._attrition_seen_trades:
            return
        for tr in trades[self._attrition_seen_trades :]:
            notional = float(tr.get("entry_price", 0.0)) * float(tr.get("size", 0.0))
            pnl_pct = float(tr.get("pnl", 0.0)) / notional if notional > 0 else 0.0
            self.attrition.record_trade(pnl_pct)
        self._attrition_seen_trades = len(trades)
        if self.attrition.tripped:
            logger.warning(
                "SilentAttrition: EWMA PnL/trade %.5f ≤ %.5f після %d угод — нові входи заблоковано",
                self.attrition.ewma_pnl,
                self.attrition.threshold_pnl,
                self.attrition.trade_count,
            )

    # ── ризик-контроль (книга, гл. 4) ────────────────────────────────────────
    def risk_check(
        self,
        decision: TradeDecision,
        mark_price: float | None = None,
        now: pd.Timestamp | None = None,
    ) -> tuple[bool, str]:
        """Перевірка ризик-лімітів. Повертає (дозволено?, причина відмови).

        Правило: закриття позицій НІКОЛИ не блокується (зменшення ризику
        завжди дозволено); блокуються лише відкриття нових позицій.
        Після N збитків — cooldown (size×0.5); після max consecutive — halt.
        """
        if decision.action in ("close", "hold"):
            self._entry_size_mult = 1.0
            return True, "ok"
        if mark_price is not None:
            self.account.mark({self.symbol: mark_price})
        from scalper_hft.data.downloader import _utc_now

        ts = as_naive_utc(now if now is not None else _utc_now())
        gate = decide_entry(
            consecutive_losses=self.account.consecutive_losses,
            now=ts,
            cooldown=self.cooldown,
            cooldown_losses=int(getattr(self.settings, "cooldown_losses", 2)),
            max_consecutive_losses=int(self.settings.max_consecutive_losses),
            cooldown_hours=float(getattr(self.settings, "cooldown_hours", 12.0)),
            cooldown_size_mult=float(getattr(self.settings, "cooldown_size_mult", 0.5)),
            flattening=False,
        )
        self.cooldown = gate.cooldown
        self._entry_size_mult = gate.size_mult
        if gate.status == "reject":
            return False, gate.reason
        if self.account.equity <= self.account.day_start_equity * (1 - self.settings.daily_loss_limit):
            return False, "денний ліміт збитків"
        weekly_limit = float(getattr(self.settings, "weekly_loss_limit", 0.06))
        if self.account.equity <= self.week_start_equity * (1 - weekly_limit):
            return False, "тижневий ліміт збитків"
        # Silent Attrition: EWMA PnL на угоду стійко збитковий → стоп входів
        self._update_attrition()
        if self.attrition.tripped:
            return False, "silent attrition: EWMA PnL нижче порогу"
        if len(self.account.positions) >= self.settings.max_open_positions:
            return False, "максимум відкритих позицій"
        # Per-symbol notional cap + margin/liquidation proximity (1D).
        # Ноціонал нового входу = size × ціна; поточний ноціонал позиції символу.
        add_notional = float(decision.size) * float(mark_price if mark_price is not None else 0.0)
        cur_pos = self.account.positions.get(self.symbol)
        cur_notional = 0.0
        if cur_pos is not None:
            cur_notional = abs(float(cur_pos.size)) * float(mark_price if mark_price is not None else 0.0)
        cap_pct = float(getattr(self.settings, "per_symbol_notional_pct", 0.30))
        if not per_symbol_notional_ok(add_notional, self.account.equity, cap_pct):
            return False, "per-symbol notional cap"
        buf = float(getattr(self.settings, "liquidation_proximity_buffer", 0.10))
        max_lev = float(getattr(self.settings, "max_leverage", 3.0))
        if not margin_proximity_ok(cur_notional, add_notional, self.account.equity, max_lev, buf):
            return False, "margin/liquidation proximity"
        if gate.status == "cooldown":
            return True, "cooldown"
        return True, "ok"

    # ── виконання ────────────────────────────────────────────────────────────
    def execute(self, decision: TradeDecision, price: float, ts: pd.Timestamp) -> str:
        self.maybe_roll_day(ts)
        self.account.mark({self.symbol: price})
        allowed, reason = self.risk_check(decision, mark_price=price, now=ts)
        if not allowed:
            logger.warning("Ризик-блок: %s", reason)
            return f"blocked:{reason}"
        if not self._uses_maker_path():
            # Синхронний шлях (paper taker або live market): миттєвий філ.
            if decision.action == "hold":
                return "hold"
            if decision.action == "close":
                ok = self._close_instant(price, ts, size=decision.size)
                return "closed" if ok else "submit_failed:close"
            side = "long" if decision.action == "open_long" else "short"
            size = decision.size * float(getattr(self, "_entry_size_mult", 1.0))
            ok = self._open_instant(side, size, price, ts)
            if not ok:
                return "submit_failed:open"
            return f"opened {side} {size:.6f} @ {price}"

        # ── Live maker: філ приходить пізніше — локальна позиція брониться
        # лише після підтвердження (poll_pending_orders). Тут керуємо наміром.
        with self._pending_lock:
            return self._execute_live_maker(decision, price, ts)

    def _execute_live_maker(self, decision: TradeDecision, price: float, ts: pd.Timestamp) -> str:
        """Live maker-шлях execute під ``_pending_lock`` (без повторного захоплення)."""
        pend = self._pending.find_for_symbol()

        if decision.action == "hold":
            # want == have. Якщо лишився resting-ордер, що суперечить утриманню —
            # скасовуємо: open-pending при flat (сигнал 0) або close-pending при
            # рішенні зберегти позицію.
            if pend is not None:
                if pend.kind == "open":
                    self._pending.cancel(pend, reason="hold_flat")
                elif self.symbol in self.account.positions:
                    self._pending.cancel(pend, reason="hold_keep_position")
            return "hold"

        if decision.action == "close":
            if pend is not None:
                if pend.kind == "open":
                    # закриваємо НАМІР (відміна resting open-ордера) — позиції ще немає
                    self._pending.cancel(pend, reason="close_before_open")
                    return "canceled_pending_open"
                return "hold:closing_pending"  # close вже в роботі — чекаємо філа
            if self.symbol not in self.account.positions:
                return "closed"  # позиції немає — закривати нічого
            pos = self.account.positions[self.symbol]
            side = "sell" if pos.side == "long" else "buy"

            close_size = decision.size if decision.size > 0 else pos.size
            ok, status = self._submit_order(side, close_size, price, reduce_only=True, kind="close", pos_side=pos.side)
            if not ok:
                return "submit_failed:close"
            if status == "pending":
                # локальна позиція лишається, поки close не заповнився
                return "close_pending"
            self.account.close_position(self.symbol, price, ts, is_maker=False, size=close_size)
            return "closed"

        side = "long" if decision.action == "open_long" else "short"
        size = decision.size * float(getattr(self, "_entry_size_mult", 1.0))
        if self.symbol in self.account.positions:
            # страхівка: відкриття поверх відкритої позиції — спершу закрити
            return "hold:need_close_first"
        if pend is not None:
            if pend.kind == "close":
                return "hold:closing_pending"  # не відкриваємо, поки close не заповнився
            if pend.pos_side == side:
                return "hold:open_pending"  # той самий напрямок вже в роботі
            self._pending.cancel(pend, reason="flip_pending_open")  # протилежний — скасовуємо
        ok, status = self._submit_order(
            "buy" if side == "long" else "sell", size, price, reduce_only=False, kind="open", pos_side=side
        )
        if not ok:
            return "submit_failed:open"
        if status == "pending":
            return f"open_pending:{side}:{size:.6f}@{price}"
        self.account.open_position(self.symbol, side, size, price, ts, is_maker=False)
        return f"opened {side} {size:.6f} @ {price}"

    # ── C2: order lifecycle для live maker ────────────────────────────────────

    def _submit_order(
        self,
        side: str,
        size: float,
        price: float,
        *,
        reduce_only: bool = False,
        kind: str = "open",
        pos_side: str = "",
    ) -> tuple[bool, str]:
        """Надіслати ордер. Повертає (ok, status), status ∈ {"filled", "pending"}.

        Paper (dry_run) — миттєвий no-op філ. Live market — філ у відповіді.
        Live maker (limit post_only) — ордер реєструється у `pending_orders`;
        локальна позиція з'явиться лише після підтвердженого філа.

        M5: перед відправкою live-ордера розмір/ціна округлюються до кроків
        біржі (LOT_SIZE/tick) і валідуються (minQty/maxQty/MIN_NOTIONAL) —
        fail-closed: невалідний ордер не летить на біржу.
        """
        if self.settings.dry_run:
            if self.settings.maker_execution:
                from scalper_hft.live.orders import next_client_order_id

                coid = next_client_order_id("sh")
                self._pending.register(
                    coid,
                    PendingOrder(
                        client_order_id=coid,
                        order_id=coid,
                        symbol=self.symbol,
                        side=side,
                        size=size,
                        price=price,
                        reduce_only=reduce_only,
                        kind=kind,
                        pos_side=pos_side,
                        placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
                    ),
                )
                return True, "pending"
            return True, "filled"
        require_live_credentials(self.settings)
        # M5: нормалізація під фільтри біржі (ExchangeClient має sanitize_order;
        # тестові/мінімальні клієнти без нього — пропускають нормалізацію).
        sanitize = getattr(self.client, "sanitize_order", None)
        if sanitize is not None:
            try:
                qty, px, err = sanitize(self.symbol, side, size, price)
            except Exception as exc:  # noqa: BLE001
                logger.error("sanitize_order %s: %s", self.symbol, exc)
                return False, "failed"
            if err is not None:
                logger.error("Ордер відхилено фільтрами біржі: %s %s %s → %s", side, self.symbol, size, err)
                return False, "failed"
            size, price = qty, px if px is not None else price
        params: dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True
        from scalper_hft.live.orders import next_client_order_id

        # Ідемпотентність retry: намір (symbol/side/kind/reduceOnly/бар) тримає
        # свій coid після таймауту. Повторна відправка В МЕЖАХ ТОГО Ж БАРУ йде
        # з ТИМ САМИМ id — біржа дедуплікує (DuplicateOrderId), а ми відновлюємо
        # існуючий ордер замість розміщення дубля. Баровий префікс часу в ключі
        # не дає НОВІЙ уяві (наступний бар, можливо інший розмір/ціна) успадкувати
        # stale coid попередньої — колізія «один намір = одне рішення бару».
        # Персистентність між рестартами процесу забезпечує reconcile з біржею
        # (KillSwitch при розбіжності).
        intent = _intent_key(self.symbol, side, kind, reduce_only, self.interval)
        coid = self._intent_coids.get(intent) or self._intent_store.get(intent) or next_client_order_id("sh")
        try:
            if self.settings.maker_execution:
                resp = self.client.create_order(
                    self.symbol,
                    "limit",
                    side,
                    size,
                    price=price,
                    params=params,
                    post_only=True,
                    client_order_id=coid,
                )
                self._intent_coids.pop(intent, None)
                self._intent_store.pop(intent)
                oid = str((resp or {}).get("id") or coid)
                self._pending.register(
                    coid,
                    PendingOrder(
                        client_order_id=coid,
                        order_id=oid,
                        symbol=self.symbol,
                        side=side,
                        size=size,
                        price=price,
                        reduce_only=reduce_only,
                        kind=kind,
                        pos_side=pos_side,
                        placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
                    ),
                )
                return True, "pending"
            resp = self.client.create_order(self.symbol, "market", side, size, params=params, client_order_id=coid)
            self._intent_coids.pop(intent, None)
            self._intent_store.pop(intent)
            fill_px = float((resp or {}).get("average") or (resp or {}).get("price") or price)
            self._last_market_fill_price = fill_px
            return True, "filled"
        except Exception as exc:
            if _is_duplicate_order_id(exc):
                status = self._recover_duplicate_intent(
                    intent,
                    coid,
                    side=side,
                    size=size,
                    price=price,
                    reduce_only=reduce_only,
                    kind=kind,
                    pos_side=pos_side,
                )
                if status is not None:
                    return True, status
            self._intent_coids[intent] = coid  # наступний retry — з тим самим id
            self._intent_store.put(intent, coid)  # переживає рестарт процесу
            logger.error(
                "Ордер відхилено (стан рахунку не змінено): %s %s %s reduce_only=%s err=%s",
                side,
                self.symbol,
                size,
                reduce_only,
                exc,
            )
            return False, "failed"

    def _recover_duplicate_intent(
        self,
        intent: str,
        coid: str,
        *,
        side: str,
        size: float,
        price: float,
        reduce_only: bool,
        kind: str,
        pos_side: str,
    ) -> str | None:
        """Ордер з цим coid уже існує на біржі (минулий таймаут після accept).

        Знаходимо його серед відкритих і реєструємо локально замість дубля.
        Якщо серед відкритих немає — уже заповнений/скасований: розбіжність
        підхопить reconcile (fail-closed). None — відновлення не вдалося.
        """
        fetch_open = getattr(self.client, "fetch_open_orders", None)
        if not callable(fetch_open):
            return None
        try:
            open_orders = fetch_open(self.symbol) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("recover %s: fetch_open_orders помилка: %s", coid, exc)
            return None
        for o in open_orders:
            if str((o or {}).get("clientOrderId") or "") != coid:
                continue
            oid = str(o.get("id") or coid)
            self._pending.register(
                coid,
                PendingOrder(
                    client_order_id=coid,
                    order_id=oid,
                    symbol=self.symbol,
                    side=side,
                    size=float(o.get("amount") or size),
                    price=float(o.get("price") or price),
                    reduce_only=reduce_only,
                    kind=kind,
                    pos_side=pos_side,
                    placed_ts=pd.Timestamp.now(tz="UTC").tz_localize(None),
                ),
            )
            self._intent_coids.pop(intent, None)
            self._intent_store.pop(intent)
            logger.warning("Відновлено існуючий ордер %s замість дубля (id=%s)", coid, oid)
            return "pending"
        self._intent_coids.pop(intent, None)
        self._intent_store.pop(intent)
        logger.warning("Дубль %s не серед відкритих (заповнений/скасований) — звірка підхопить", coid)
        return None

    def _cancel_pending(self, po: PendingOrder, reason: str = "") -> None:
        """Скасувати resting-ордер на біржі та прибрати з журналу (best effort)."""
        self._sync_pending_state()
        self._pending.cancel(po, reason=reason)

    def cancel_all_pending(self, reason: str = "shutdown") -> int:
        """Скасувати всі робочі resting-ордери (при shutdown або аварії)."""
        self._sync_pending_state()
        return self._pending.cancel_all(reason=reason)

    def shutdown(self, reason: str = "shutdown") -> None:
        """Безпечне завершення роботи трейдера: скасування всіх pending-ордерів
        та страхувальне скасування ордерів на біржі."""
        logger.info("Ініціалізація shutdown для %s (%s)...", self.symbol, reason)
        self.stop_user_stream()
        canceled = self.cancel_all_pending(reason=reason)
        if not self.settings.dry_run:
            cancel_all = getattr(self.client, "cancel_all_orders", None)
            if cancel_all is not None:
                try:
                    cancel_all(self.symbol)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Помилка cancel_all_orders на біржі для %s: %s", self.symbol, exc)
        logger.info("Shutdown завершено: скасовано %d локальних pending-ордерів", canceled)

    def poll_pending_orders(self, now: pd.Timestamp | None = None) -> list[str]:
        """Live maker: опитати resting-ордери → забронити філи, скасувати таймаути."""
        self._sync_pending_state()
        return self._pending.poll(now=now)

    # ── WebSocket User Data Stream ───────────────────────────────────────────

    def on_ws_order_trade(self, event: OrderTradeEvent) -> None:
        """Обробка події ORDER_TRADE_UPDATE з WebSocket User Data Stream."""
        self._pending.on_ws_order_trade(event)

    def start_user_stream(self) -> BinanceUserDataStream | None:
        """Ініціалізувати та повернути клієнт WebSocket User Data Stream."""
        if self.settings.dry_run or not self.client:
            return None
        if self.ws_stream is not None:
            return self.ws_stream
        create_fn = getattr(self.client, "create_listen_key", None)
        keepalive_fn = getattr(self.client, "keepalive_listen_key", None)
        if not callable(create_fn):
            return None
        try:
            listen_key = create_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не вдалося створити listenKey: %s", exc)
            return None
        if not listen_key:
            return None

        self.ws_stream = BinanceUserDataStream(
            listen_key=listen_key,
            testnet="testnet" in str(getattr(self.settings, "exchange", "")).lower(),
            on_order_update=self.on_ws_order_trade,
            keepalive=keepalive_fn if callable(keepalive_fn) else None,
            refresh_listen_key=create_fn,
        )
        logger.info("User Data Stream налаштовано для %s", self.symbol)
        return self.ws_stream

    def stop_user_stream(self) -> None:
        """Зупинити User Data Stream та закрити listenKey."""
        if self.ws_stream is not None:
            stream = self.ws_stream
            self.ws_stream = None
            stream.stop()
            close_fn = getattr(self.client, "close_listen_key", None)
            if callable(close_fn) and stream.listen_key:
                try:
                    close_fn(stream.listen_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Помилка close_listen_key: %s", exc)

    # ── M4: live equity з біржі ──────────────────────────────────────────────

    def sync_live_equity(self, now: pd.Timestamp | None = None) -> float | None:
        """Live: оновити локальний cash з РЕАЛЬНОГО equity біржі.

        Без цього sizing/ризик-гейти рахуються від фіктивного
        PaperAccount(initial_capital = position_pct × 100_000) і перша ж
        розбіжність з реальним балансом ламає розмір позиції. Тут cash
        ребейзиться так, що equity ≈ реальний баланс біржі (wallet + UPNL),
        а day_start_equity фіксується від реального equity при першому sync.

        Paper (dry_run) або клієнт без fetch_usdt_equity — no-op (None).
        """
        if self.settings.dry_run:
            return None
        fetch = getattr(self.client, "fetch_usdt_equity", None)
        if fetch is None:
            return None
        try:
            equity = float(fetch())
        except Exception as exc:  # noqa: BLE001
            logger.warning("sync_live_equity: не вдалося отримати баланс: %s", exc)
            return None
        if not np.isfinite(equity) or equity <= 0:
            return None
        # локальний кеш: equity = cash + unrealized(локальні позиції)
        # → cash = реальний equity − локальний unrealized
        self.account.cash = float(equity) - self.account.unrealized_pnl()
        if not self._live_equity_seeded:
            self.account.day_start_equity = float(equity)
            self._live_equity_seeded = True
        self.last_live_equity = float(equity)
        logger.info("Live equity синхронізовано: %.2f (day_start %.2f)", equity, self.account.day_start_equity)
        return float(equity)

    def _open_instant(self, side: str, size: float, price: float, ts: pd.Timestamp) -> bool:
        """Paper/market шлях: submit + миттєвий філ (як було до C2)."""
        if self.symbol in self.account.positions:
            if not self._close_instant(price, ts):
                return False
        if not self._submit_order("buy" if side == "long" else "sell", size, price, reduce_only=False)[0]:
            return False
        self.account.open_position(self.symbol, side, size, price, ts, is_maker=self.settings.maker_execution)
        return True

    def _close_instant(self, price: float, ts: pd.Timestamp, size: float | None = None) -> bool:
        """Paper/market шлях: submit + миттєвий філ (як було до C2)."""
        if self.symbol not in self.account.positions:
            return True
        pos = self.account.positions[self.symbol]
        side = "sell" if pos.side == "long" else "buy"
        close_sz = size if size is not None and size > 0 else pos.size
        if not self._submit_order(side, close_sz, price, reduce_only=True)[0]:
            return False
        self.account.close_position(self.symbol, price, ts, is_maker=self.settings.maker_execution, size=close_sz)
        return True


__all__ = [
    "LiveTrader",
    "PendingOrder",
    "SilentAttritionKillSwitch",
    "TradeDecision",
    "closed_klines",
    "execute_signal",
    "run_trader_once",
]
