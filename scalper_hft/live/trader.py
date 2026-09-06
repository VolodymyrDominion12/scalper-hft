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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings, require_live_credentials
from scalper_hft.data.client import ExchangeClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.control import ControlState, load_control
from scalper_hft.live.exit_ladders import OneWayTradingLadder
from scalper_hft.live.risk_gate import CooldownState, decide_entry
from scalper_hft.live.ws_user_stream import BinanceUserDataStream, OrderTradeEvent
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


def _as_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _interval_seconds(interval: str) -> float:
    """Тривалість таймфрейму в секундах ('1m' → 60, '1h' → 3600)."""
    unit = interval[-1]
    num = int(interval[:-1])
    per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return float(num * per_unit)


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


def closed_klines(
    df: pd.DataFrame,
    interval: str | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Відкидає формуючий бар (останній рядок REST), щоб не було lookahead.

    Binance OHLCV включає поточну незакриту свічку як iloc[-1]. Сигнал і філл
    беруться лише з барів, чий close-час уже настав. Історичні ряди (останній
    бар у минулому) не змінюються.
    """
    if df is None or df.empty:
        return df
    from scalper_hft.data.downloader import _interval_ms, _utc_now

    interval = interval or "1m"
    now_ts = _as_naive_utc(now if now is not None else _utc_now())
    last = _as_naive_utc(df.index[-1])
    bar_end = last + pd.Timedelta(milliseconds=_interval_ms(interval))
    if now_ts < bar_end:
        return df.iloc[:-1]
    return df


@dataclass
class TradeDecision:
    action: str  # "open_long" | "open_short" | "close" | "hold"
    symbol: str = ""
    size: float = 0.0
    reason: str = ""


@dataclass
class PendingOrder:
    """Live maker-ордер, що ще не заповнився (resting на біржі).

    Локальна позиція брониться ЛИШЕ після підтвердженого філа (fetch_order):
    раніше позиція бронилась одразу після submit → phantom-позиції при
    unfilled/часткових філах (див. C2 у code review).
    """

    client_order_id: str
    order_id: str  # exchange order id (fallback = client id)
    symbol: str
    side: str  # buy | sell (біржовий бік)
    size: float  # ОРИГІНАЛЬНИЙ розмір ордера (не мутує при часткових філах)
    price: float
    reduce_only: bool
    kind: str  # "open" | "close"
    pos_side: str = ""  # long | short (для kind="open")
    placed_ts: pd.Timestamp | None = None
    booked_qty: float = 0.0  # скільки вже зараховано у локальний рахунок (WS/REST дельта-облік)

    @property
    def remaining_qty(self) -> float:
        """Ще не зарахована частина ордера."""
        return max(0.0, self.size - self.booked_qty)


class SilentAttritionKillSwitch:
    """Детектор 'тихого згасання' альфи (Silent Attrition, PM Ch. 4, 13).

    Відстежує EWMA прибутку на угоду (PnL per trade). Якщо EWMA падає нижче
    критичного порогу z-score збитковості — ініціює безпечну зупинку нових входів.
    """

    def __init__(
        self,
        alpha_decay: float = 0.1,
        min_trades: int = 5,
        threshold_pnl: float = -0.005,  # -0.5% на угоду
    ) -> None:
        self.alpha_decay = alpha_decay
        self.min_trades = min_trades
        self.threshold_pnl = threshold_pnl
        self.ewma_pnl: float = 0.0
        self.trade_count: int = 0
        self.tripped: bool = False

    def record_trade(self, pnl_pct: float) -> bool:
        """Реєструє закриту угоду. Повертає True, якщо kill-switch спрацював."""
        self.trade_count += 1
        if self.trade_count == 1:
            self.ewma_pnl = pnl_pct
        else:
            self.ewma_pnl = (1.0 - self.alpha_decay) * self.ewma_pnl + self.alpha_decay * pnl_pct

        if self.trade_count >= self.min_trades and self.ewma_pnl <= self.threshold_pnl:
            self.tripped = True

        return self.tripped

    def reset(self) -> None:
        """Скидання стану після аудиту/перезапуску."""
        self.ewma_pnl = 0.0
        self.trade_count = 0
        self.tripped = False


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
        # C2: live maker-ордери, що ще не заповнились (clientOrderId → PendingOrder)
        self.pending_orders: dict[str, PendingOrder] = {}
        # Ідемпотентність submit: намір → coid, що пережив таймаут (retry з тим самим id)
        self._intent_coids: dict[str, str] = {}
        self._last_market_fill_price: float | None = None
        # M4: live-базис equity з біржі (замість фіктивного депозиту)
        self._live_equity_seeded = False
        self.last_live_equity: float | None = None

        self.use_exit_ladders = getattr(self.settings, "use_exit_ladders", False)
        self.ladder: OneWayTradingLadder | None = None
        self.ws_stream: BinanceUserDataStream | None = None

    # ── сигнал ───────────────────────────────────────────────────────────────
    def compute_signal(self, df: pd.DataFrame, now: pd.Timestamp | None = None) -> int:
        """Сигнал стратегії на останньому ЗАКРИТОМУ барі (без lookahead)."""
        closed = closed_klines(df, self.interval, now=now)
        if closed is None or len(closed) < 50:
            return 0
        if self.strategy.needs_funding:
            from scalper_hft.data.downloader import download_funding

            funding = download_funding(self.symbol, days=30)
            signal = self.strategy.generate_signals(closed, funding=funding)
        elif self.strategy.needs_trades:
            from scalper_hft.data.downloader import download_agg_trades

            trades = download_agg_trades(self.symbol, days=1)
            signal = self.strategy.generate_signals(closed, trades=trades)
        else:
            signal = self.strategy.generate_signals(closed)
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

        Модель навчається на перших 2000 барах, поточна ймовірність —
        фільтрована (forward-only) → без lookahead.
        """
        if not self.hmm_block:
            return False
        try:
            from scalper_hft.features.hmm_regime import GaussianHMM

            close = df["close"]
            ret = close.pct_change().fillna(0.0)
            vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
            obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]
            if len(obs) < self.hmm_states * 20:
                return False
            model = GaussianHMM(n_states=self.hmm_states, seed=42).fit(obs.iloc[:2000].values)
            if model.covars_ is None:
                return False
            calm = int(np.argmin(model.covars_[:, 2]))  # стан з найменшою vol
            post = model.filtered_proba(obs.values)
            return bool(post[-1, calm] < self.hmm_threshold)
        except Exception:
            logger.exception("HMM-гейт: помилка моделі — блокуємо нові входи")
            return True

    def maybe_roll_day(self, now: pd.Timestamp | None = None) -> bool:
        """Скинути денний ліміт збитків при зміні UTC-доби.

        Повертає True, якщо відбувся roll. Перший виклик лише запам'ятовує день.
        """
        from scalper_hft.data.downloader import _utc_now

        ts = _as_naive_utc(now if now is not None else _utc_now())
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

        ts = _as_naive_utc(now if now is not None else _utc_now())
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
        live_maker = (not self.settings.dry_run) and bool(getattr(self.settings, "maker_execution", False))
        if not live_maker:
            # Синхронний шлях (paper або market): миттєвий філ, як і раніше.
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
        pend = next((po for po in self.pending_orders.values() if po.symbol == self.symbol), None)

        if decision.action == "hold":
            # want == have. Якщо лишився resting-ордер, що суперечить утриманню —
            # скасовуємо: open-pending при flat (сигнал 0) або close-pending при
            # рішенні зберегти позицію.
            if pend is not None:
                if pend.kind == "open":
                    self._cancel_pending(pend, reason="hold_flat")
                elif self.symbol in self.account.positions:
                    self._cancel_pending(pend, reason="hold_keep_position")
            return "hold"

        if decision.action == "close":
            if pend is not None:
                if pend.kind == "open":
                    # закриваємо НАМІР (відміна resting open-ордера) — позиції ще немає
                    self._cancel_pending(pend, reason="close_before_open")
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
            self._cancel_pending(pend, reason="flip_pending_open")  # протилежний — скасовуємо
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
        params: dict = {}
        if reduce_only:
            params["reduceOnly"] = True
        from scalper_hft.live.orders import next_client_order_id

        # Ідемпотентність retry: намір (symbol/side/kind/reduceOnly) тримає свій
        # coid після таймауту. Повторна відправка йде з ТИМ САМИМ id — біржа
        # дедуплікує (DuplicateOrderId), а ми відновлюємо існуючий ордер
        # замість розміщення дубля. Персистентність між рестартами процесу
        # забезпечує reconcile з біржею (KillSwitch при розбіжності).
        intent = f"{self.symbol}:{side}:{kind}:{int(reduce_only)}"
        coid = self._intent_coids.get(intent) or next_client_order_id("sh")
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
                oid = str((resp or {}).get("id") or coid)
                self.pending_orders[coid] = PendingOrder(
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
                )
                return True, "pending"
            resp = self.client.create_order(self.symbol, "market", side, size, params=params, client_order_id=coid)
            self._intent_coids.pop(intent, None)
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
            self.pending_orders[coid] = PendingOrder(
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
            )
            self._intent_coids.pop(intent, None)
            logger.warning("Відновлено існуючий ордер %s замість дубля (id=%s)", coid, oid)
            return "pending"
        self._intent_coids.pop(intent, None)
        logger.warning("Дубль %s не серед відкритих (заповнений/скасований) — звірка підхопить", coid)
        return None

    def _cancel_pending(self, po: PendingOrder, reason: str = "") -> None:
        """Скасувати resting-ордер на біржі та прибрати з журналу (best effort)."""
        cancel = getattr(self.client, "cancel_order", None)
        if cancel is not None:
            try:
                cancel(po.order_id, po.symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cancel %s: %s", po.order_id, exc)
        self.pending_orders.pop(po.client_order_id, None)
        logger.info("Скасовано pending %s (%s)", po.client_order_id, reason or "n/a")

    def cancel_all_pending(self, reason: str = "shutdown") -> int:
        """Скасувати всі робочі resting-ордери (при shutdown або аварії)."""
        count = 0
        for po in list(self.pending_orders.values()):
            self._cancel_pending(po, reason=reason)
            count += 1
        return count

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
        """Live maker: опитати resting-ордери → забронити філи, скасувати таймаути.

        Викликається на початку кожного кроку циклу (run_trader_once), ДО звірки:
        заповнені ордери одразу відображаються в локальному рахунку.
        """
        if self.settings.dry_run or not self.pending_orders:
            return []
        fetch = getattr(self.client, "fetch_order", None)
        if fetch is None:
            return ["no_fetch_support"]  # тестовий/мінімальний клієнт
        cancel = getattr(self.client, "cancel_order", None)
        # Вік ордера рахуємо за СТІНОЧНИМ часом (placed_ts = реальний час),
        # а не за бар-міткою now (у тестах/реплеї вони можуть розходитись).
        ts = _as_naive_utc(pd.Timestamp.now(tz="UTC"))
        wait_bars = int(getattr(self.settings, "maker_fill_wait_bars", 1))
        timeout_s = max(_interval_seconds(self.interval), 1.0) * max(wait_bars, 1)
        events: list[str] = []
        for coid, po in list(self.pending_orders.items()):
            try:
                info = fetch(po.order_id, po.symbol)
            except Exception as exc:  # noqa: BLE001
                logger.error("fetch_order %s: %s", po.order_id, exc)
                continue
            status = str((info or {}).get("status") or "").lower()
            filled = float((info or {}).get("filled") or 0.0)
            avg = float((info or {}).get("average") or (info or {}).get("price") or po.price)
            if status in ("closed", "filled") or filled >= po.size - 1e-9:
                # Дельта-облік: частина могла бути зарахована через WS раніше.
                self._book_fill_delta(po, po.size, avg, ts)
                del self.pending_orders[coid]
                events.append(f"filled:{coid}")
                continue
            if filled > 1e-9:  # частковий філ: бронимо дельту, решту скасовуємо
                # PaperAccount підтримує часткове закриття (close_position(size=...))
                # та доливку (add_to_position) — бронюємо рівно заповнене.
                self._book_fill_delta(po, filled, avg, ts)
                if cancel is not None:
                    try:
                        cancel(po.order_id, po.symbol)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cancel partial %s: %s", po.order_id, exc)
                del self.pending_orders[coid]
                events.append(f"partial:{coid}")
                continue
            if status in ("canceled", "cancelled", "expired", "rejected"):
                del self.pending_orders[coid]
                events.append(f"dead:{status}:{coid}")
                continue
            # ще відкритий: перевіряємо таймаут
            age_s = 0.0
            if po.placed_ts is not None:
                age_s = (ts - _as_naive_utc(po.placed_ts)).total_seconds()
            if age_s > timeout_s:
                if cancel is not None:
                    try:
                        cancel(po.order_id, po.symbol)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cancel timeout %s: %s", po.order_id, exc)
                del self.pending_orders[coid]
                events.append(f"timeout_cancel:{coid}")
        return events

    def _book_pending_fill(self, po: PendingOrder, size: float, price: float, ts: pd.Timestamp) -> None:
        """Забронити підтверджений філ у локальному рахунку (maker-комісії).

        Open при вже відкритій позиції — доливка з середньозваженою ціною
        (часткові філи одного ордера), а не тихий пропуск.
        """
        if po.kind == "open":
            existing = self.account.positions.get(self.symbol)
            if existing is None:
                self.account.open_position(self.symbol, po.pos_side, size, price, ts, is_maker=True)
            elif existing.side == po.pos_side:
                self.account.add_to_position(self.symbol, size, price, ts, is_maker=True)
            else:
                # протилежна сторона — не бронюємо; розбіжність зловить reconcile
                logger.error(
                    "Філ %s (%s) при відкритій протилежній позиції %s — пропущено, звірка обов'язкова",
                    po.client_order_id,
                    po.pos_side,
                    existing.side,
                )
        else:
            if self.symbol in self.account.positions:
                self.account.close_position(self.symbol, price, ts, is_maker=True, size=size)

    def _book_fill_delta(self, po: PendingOrder, cumulative_filled: float, price: float, ts: pd.Timestamp) -> float:
        """Забронювати лише ДЕЛЬТУ від авторитетного cumulative (WS або REST).

        Ідемпотентно: дубльовані WS-події або REST-полл після WS-філа не
        подвоюють позицію. Повертає фактично зараховану кількість.
        """
        delta = min(cumulative_filled, po.size) - po.booked_qty
        if delta <= 1e-9:
            return 0.0
        self._book_pending_fill(po, delta, price, ts)
        po.booked_qty += delta
        return delta

    # ── WebSocket User Data Stream ───────────────────────────────────────────

    def on_ws_order_trade(self, event: OrderTradeEvent) -> None:
        """Обробка події ORDER_TRADE_UPDATE з WebSocket User Data Stream.

        Миттєво бронює філи для pending-ордерів без очікування наступного
        REST-поллінгу або закриття бару, мінімізуючи lag виконання.
        """
        coid = event.client_order_id
        if not coid or coid not in self.pending_orders:
            return
        po = self.pending_orders[coid]
        ts = _as_naive_utc(pd.Timestamp.now(tz="UTC"))
        status = event.status.upper()
        if status in ("FILLED", "CLOSED"):
            fill_px = event.last_filled_price if event.last_filled_price > 0 else po.price
            # FILLED → ордер заповнений повністю; бронюємо лише незарахований залишок
            cumulative = event.cumulative_filled_qty if event.cumulative_filled_qty > 0 else po.size
            booked = self._book_fill_delta(po, max(cumulative, po.size), fill_px, ts)
            self.pending_orders.pop(coid, None)
            logger.info("WS fill booked: %s (delta=%.4f, px=%.4f)", coid, booked, fill_px)
        elif status == "PARTIALLY_FILLED":
            cumulative = event.cumulative_filled_qty
            if cumulative <= 1e-9:  # fallback на per-fill qty, якщо cumulative відсутній
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


def execute_signal(
    trader: LiveTrader,
    signal: int,
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    block_new_entries: bool = False,
) -> str:
    """Рішення за сигналом на останньому закритому барі → виконання.

    Реверс (long→short і навпаки): спочатку close (ніколи не блокується
    ризиком), потім open. Інакше max_open_positions=1 блокує фліп назавжди.
    block_new_entries (control.json no_new_entries): закриття дозволене,
    нові входи — ні.
    """
    closed = closed_klines(df, trader.interval, now=now)
    if closed is None or closed.empty:
        return "hold:no_closed_bar"
    trader.maybe_roll_day(now if now is not None else closed.index[-1])
    close = float(closed["close"].iloc[-1])
    ts = closed.index[-1]
    trader.account.mark({trader.symbol: close})
    pos = trader.account.positions.get(trader.symbol)

    want = 1 if signal > 0 else (-1 if signal < 0 else 0)
    have = 0
    if pos is not None:
        have = 1 if pos.side == "long" else -1

    parts: list[str] = []

    ladder_exit = 0.0
    if have != 0 and trader.use_exit_ladders and trader.ladder is not None:
        ladder_exit = trader.ladder.update_price(close)

    if want == 0:
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "сигнал=0"), close, ts))
            trader.ladder = None
        else:
            parts.append(trader.execute(TradeDecision("hold", trader.symbol, 0.0, ""), close, ts))
    elif want == have:
        if ladder_exit > 0 and pos is not None and trader.ladder is not None:
            parts.append(
                trader.execute(TradeDecision("close", trader.symbol, pos.size * ladder_exit, "ladder_exit"), close, ts)
            )
            if trader.ladder.is_fully_closed:
                trader.ladder = None
        else:
            parts.append(trader.execute(TradeDecision("hold", trader.symbol, 0.0, "вже в позиції"), close, ts))
    else:
        # close-before-flip: закриття ніколи не блокується (зменшення ризику)
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "реверс"), close, ts))
            trader.ladder = None
        if block_new_entries:
            parts.append("blocked:no_new_entries")
        # HMM-режимний блок: нові входи лише у «спокійному» стані
        elif trader.hmm_blocked(closed):
            parts.append("blocked:hmm_regime")
        else:
            base_size = trader.settings.position_pct * trader.account.equity / close
            size = trader.vol_scaled_size(base_size, closed)
            action = "open_long" if want > 0 else "open_short"
            parts.append(trader.execute(TradeDecision(action, trader.symbol, size, f"сигнал={signal}"), close, ts))
            if trader.use_exit_ladders:
                trader.ladder = OneWayTradingLadder(
                    close, want, base_step_pct=0.002, num_levels=4, geometric_factor=1.5
                )

    trader.last_signal = signal
    return " | ".join(parts)


def run_trader_once(
    trader: LiveTrader,
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    control: ControlState | None = None,
) -> str:
    """Один крок циклу: control plane → філи live maker-ордерів → звірка
    (live) → сигнал на закритому барі → виконання за close.

    Control plane (control.json): pause — повний стоп кроку; flatten —
    примусовий сигнал 0 (закриття); no_new_entries — лише закриття.
    Poll ПЕРЕД звіркою: щойно заповнені ордери одразу відображаються в
    локальному рахунку, тож звірка з біржею не дає хибний KillSwitch.
    Live: перед сигналом синхронізуємо РЕАЛЬНИЙ equity біржі (M4), щоб
    sizing і денний ліміт збитків рахувались від реального балансу.
    """
    from scalper_hft.live.reconcile import reconcile_exchange_state

    ctrl = control if control is not None else load_control(trader.control_path)
    if ctrl.pause:
        return "hold:paused"
    trader.poll_pending_orders(now=now)
    reconcile_exchange_state(trader.account, trader.client, dry_run=trader.settings.dry_run)
    if not trader.settings.dry_run:
        trader.sync_live_equity(now=now)
    signal = trader.compute_signal(df, now=now)
    if ctrl.flatten:
        signal = 0
    return execute_signal(trader, signal, df, now=now, block_new_entries=ctrl.no_new_entries)
