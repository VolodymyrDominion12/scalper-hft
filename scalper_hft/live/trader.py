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


class LiveTrader:
    """Керує одним символом: свіжі klines → сигнал → ордер (paper/testnet/live)."""

    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        interval: str = "1m",
        account: PaperAccount | None = None,
        client: BinanceClient | None = None,
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
        self.last_signal: int = 0

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
        self.account.mark({self.symbol: price})
        allowed, reason = self.risk_check(decision, mark_price=price)
        if not allowed:
            logger.warning("Ризик-блок: %s", reason)
            return f"blocked:{reason}"
        if decision.action == "hold":
            return "hold"
        if decision.action == "close":
            self._close(price, ts)
            return "closed"
        side = "long" if decision.action == "open_long" else "short"
        self._open(side, decision.size, price, ts)
        return f"opened {side} {decision.size:.6f} @ {price}"

    def _submit_order(self, side: str, size: float, price: float, *, reduce_only: bool = False) -> None:
        """Live-ордер: maker → limit+postOnly; інакше market. Paper (dry_run) — no-op."""
        if self.settings.dry_run:
            return
        params: dict = {}
        if reduce_only:
            params["reduceOnly"] = True
        if self.settings.maker_execution:
            self.client.create_order(
                self.symbol,
                "limit",
                side,
                size,
                price=price,
                params=params,
                post_only=True,
            )
        else:
            self.client.create_order(self.symbol, "market", side, size, params=params)
        logger.info("LIVE ордер: %s %s %s reduce_only=%s", side, self.symbol, size, reduce_only)

    def _open(self, side: str, size: float, price: float, ts: pd.Timestamp) -> None:
        if self.symbol in self.account.positions:
            self._close(price, ts)
        self._submit_order("buy" if side == "long" else "sell", size, price, reduce_only=False)
        self.account.open_position(
            self.symbol, side, size, price, ts, is_maker=self.settings.maker_execution
        )

    def _close(self, price: float, ts: pd.Timestamp) -> None:
        if self.symbol not in self.account.positions:
            return
        pos = self.account.positions[self.symbol]
        side = "sell" if pos.side == "long" else "buy"
        self._submit_order(side, pos.size, price, reduce_only=True)
        self.account.close_position(self.symbol, price, ts, is_maker=self.settings.maker_execution)


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
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "реверс"), close, ts))
        size = trader.settings.position_pct * trader.account.equity / close
        action = "open_long" if want > 0 else "open_short"
        parts.append(
            trader.execute(TradeDecision(action, trader.symbol, size, f"сигнал={signal}"), close, ts)
        )

    trader.last_signal = signal
    return " | ".join(parts)


def run_trader_once(trader: LiveTrader, df: pd.DataFrame, now: pd.Timestamp | None = None) -> str:
    """Один крок циклу: сигнал на останньому закритому барі → виконання за close."""
    signal = trader.compute_signal(df, now=now)
    return execute_signal(trader, signal, df, now=now)
