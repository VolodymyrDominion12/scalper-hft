"""Live-трейдер: цикл "дані → сигнал → ризик → виконання".

Режими:
    - paper: симуляція через PaperAccount (без API-ключів або testnet);
    - testnet: реальні ордери на Binance testnet (API-ключі testnet);
    - live: реальні ордери (DRY_RUN=false + EXCHANGE=binance) — свідомо!

Ризик-контроль перед кожним ордером (книга, гл. 4):
    - ліміт позиції (POSITION_PCT від капіталу);
    - денний ліміт збитків (DAILY_LOSS_LIMIT);
    - пауза після MAX_CONSECUTIVE_LOSSES збитків.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.binance_client import BinanceClient
from scalper_hft.live.account import PaperAccount
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


def _as_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


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
        client: BinanceClient | None = None,
        vol_sizing: bool = False,
        vol_ref: float | None = None,
        hmm_block: bool = False,
        hmm_states: int = 3,
        hmm_threshold: float = 0.5,
    ) -> None:
        self.settings = get_settings()
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval or "1m"
        self.account = account or PaperAccount(
            initial_capital=self.settings.position_pct * 100_000,
            taker_fee=self.settings.taker_fee,
            maker_fee=self.settings.maker_fee,
        )
        self.client = client or BinanceClient(
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
            calm = int(np.argmin(model.covars_[:, 2]))  # стан з найменшою vol
            post = model.filtered_proba(obs.values)
            return bool(post[-1, calm] < self.hmm_threshold)
        except Exception:  # noqa: BLE001
            return False

    def maybe_roll_day(self, now: pd.Timestamp | None = None) -> bool:
        """Скинути денний ліміт збитків при зміні UTC-доби.

        Повертає True, якщо відбувся roll. Перший виклик лише запам'ятовує день.
        """
        from scalper_hft.data.downloader import _utc_now

        ts = _as_naive_utc(now if now is not None else _utc_now())
        day = ts.date()
        if self._last_roll_day is None:
            self._last_roll_day = day
            return False
        if day != self._last_roll_day:
            self.account.roll_to_new_day(self.account.equity)
            self._last_roll_day = day
            logger.info("Новий день %s: day_start_equity=%.2f", day, self.account.day_start_equity)
            return True
        return False

    # ── ризик-контроль (книга, гл. 4) ────────────────────────────────────────
    def risk_check(self, decision: TradeDecision, mark_price: float | None = None) -> tuple[bool, str]:
        """Перевірка ризик-лімітів. Повертає (дозволено?, причина відмови).

        Правило: закриття позицій НІКОЛИ не блокується (зменшення ризику
        завжди дозволено); блокуються лише відкриття нових позицій.
        """
        if decision.action in ("close", "hold"):
            return True, "ok"
        if mark_price is not None:
            self.account.mark({self.symbol: mark_price})
        if self.account.consecutive_losses >= self.settings.max_consecutive_losses:
            return False, "серія збитків — пауза"
        if self.account.equity <= self.account.day_start_equity * (1 - self.settings.daily_loss_limit):
            return False, "денний ліміт збитків"
        if len(self.account.positions) >= self.settings.max_open_positions:
            return False, "максимум відкритих позицій"
        return True, "ok"

    # ── виконання ────────────────────────────────────────────────────────────
    def execute(self, decision: TradeDecision, price: float, ts: pd.Timestamp) -> str:
        self.maybe_roll_day(ts)
        self.account.mark({self.symbol: price})
        allowed, reason = self.risk_check(decision, mark_price=price)
        if not allowed:
            logger.warning("Ризик-блок: %s", reason)
            return f"blocked:{reason}"
        if decision.action == "hold":
            return "hold"
        if decision.action == "close":
            ok = self._close(price, ts)
            return "closed" if ok else "submit_failed:close"
        side = "long" if decision.action == "open_long" else "short"
        ok = self._open(side, decision.size, price, ts)
        if not ok:
            return "submit_failed:open"
        return f"opened {side} {decision.size:.6f} @ {price}"

    def _submit_order(self, side: str, size: float, price: float, *, reduce_only: bool = False) -> bool:
        """Live-ордер: maker → limit+postOnly; інакше market.

        Paper (dry_run) — успішний no-op. Live: fail-closed — False при відмові
        біржі, локальний рахунок не змінюється.
        """
        if self.settings.dry_run:
            return True
        params: dict = {}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            from scalper_hft.live.orders import next_client_order_id

            coid = next_client_order_id("sh")
            if self.settings.maker_execution:
                self.client.create_order(
                    self.symbol,
                    "limit",
                    side,
                    size,
                    price=price,
                    params=params,
                    post_only=True,
                    client_order_id=coid,
                )
            else:
                self.client.create_order(self.symbol, "market", side, size, params=params, client_order_id=coid)
        except Exception as exc:
            logger.error(
                "Ордер відхилено (стан рахунку не змінено): %s %s %s reduce_only=%s err=%s",
                side,
                self.symbol,
                size,
                reduce_only,
                exc,
            )
            return False
        logger.info("LIVE ордер: %s %s %s reduce_only=%s", side, self.symbol, size, reduce_only)
        return True

    def _open(self, side: str, size: float, price: float, ts: pd.Timestamp) -> bool:
        if self.symbol in self.account.positions:
            if not self._close(price, ts):
                return False
        if not self._submit_order("buy" if side == "long" else "sell", size, price, reduce_only=False):
            return False
        self.account.open_position(self.symbol, side, size, price, ts, is_maker=self.settings.maker_execution)
        return True

    def _close(self, price: float, ts: pd.Timestamp) -> bool:
        if self.symbol not in self.account.positions:
            return True
        pos = self.account.positions[self.symbol]
        side = "sell" if pos.side == "long" else "buy"
        if not self._submit_order(side, pos.size, price, reduce_only=True):
            return False
        self.account.close_position(self.symbol, price, ts, is_maker=self.settings.maker_execution)
        return True


def execute_signal(
    trader: LiveTrader,
    signal: int,
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> str:
    """Рішення за сигналом на останньому закритому барі → виконання.

    Реверс (long→short і навпаки): спочатку close (ніколи не блокується
    ризиком), потім open. Інакше max_open_positions=1 блокує фліп назавжди.
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
    if want == 0:
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "сигнал=0"), close, ts))
        else:
            parts.append(trader.execute(TradeDecision("hold", trader.symbol, 0.0, ""), close, ts))
    elif want == have:
        parts.append(trader.execute(TradeDecision("hold", trader.symbol, 0.0, "вже в позиції"), close, ts))
    else:
        # close-before-flip: закриття ніколи не блокується (зменшення ризику)
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "реверс"), close, ts))
        # HMM-режимний блок: нові входи лише у «спокійному» стані
        if trader.hmm_blocked(closed):
            parts.append("blocked:hmm_regime")
        else:
            base_size = trader.settings.position_pct * trader.account.equity / close
            size = trader.vol_scaled_size(base_size, closed)
            action = "open_long" if want > 0 else "open_short"
            parts.append(trader.execute(TradeDecision(action, trader.symbol, size, f"сигнал={signal}"), close, ts))

    trader.last_signal = signal
    return " | ".join(parts)


def run_trader_once(trader: LiveTrader, df: pd.DataFrame, now: pd.Timestamp | None = None) -> str:
    """Один крок циклу: сигнал на останньому закритому барі → виконання за close."""
    signal = trader.compute_signal(df, now=now)
    return execute_signal(trader, signal, df, now=now)
