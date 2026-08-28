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
from scalper_hft.features.indicators import add_standard_features
from scalper_hft.live.account import PaperAccount
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


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
        self.interval = interval
        self.account = account or PaperAccount(self.settings.position_pct * 100_000)
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
    def compute_signal(self, df: pd.DataFrame) -> int:
        """Сигнал стратегії на останньому барі (без lookahead)."""
        if len(df) < 50:
            return 0
        signal = self.strategy.generate_signals(df)
        return int(signal.iloc[-1]) if len(signal) else 0

    # ── ризик-контроль (книга, гл. 4) ────────────────────────────────────────
    def risk_check(self, decision: TradeDecision) -> tuple[bool, str]:
        """Перевірка ризик-лімітів. Повертає (дозволено?, причина відмови)."""
        if self.account.consecutive_losses >= self.settings.max_consecutive_losses:
            return False, "серія збитків — пауза"
        if self.account.equity <= self.account.day_start_equity * (1 - self.settings.daily_loss_limit):
            return False, "денний ліміт збитків"
        if decision.action != "hold" and len(self.account.positions) >= self.settings.max_open_positions:
            return False, "максимум відкритих позицій"
        return True, "ok"

    # ── виконання ────────────────────────────────────────────────────────────
    def execute(self, decision: TradeDecision, price: float, ts: pd.Timestamp) -> str:
        allowed, reason = self.risk_check(decision)
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

    def _open(self, side: str, size: float, price: float, ts: pd.Timestamp) -> None:
        if not self.settings.dry_run:
            self.client.create_order(self.symbol, "market", "buy" if side == "long" else "sell", size)
            logger.info("LIVE ордер: %s %s %s", side, self.symbol, size)
        self.account.open_position(self.symbol, side, size, price, ts)

    def _close(self, price: float, ts: pd.Timestamp) -> None:
        if self.symbol not in self.account.positions:
            return
        if not self.settings.dry_run:
            pos = self.account.positions[self.symbol]
            side = "sell" if pos.side == "long" else "buy"
            self.client.create_order(self.symbol, "market", side, pos.size)
            logger.info("LIVE закриття: %s %s", self.symbol, pos.size)
        self.account.close_position(self.symbol, price, ts)


def run_trader_once(trader: LiveTrader, df: pd.DataFrame) -> str:
    """Один крок циклу: сигнал на останньому барі → виконання за close.

    Повертає рядок результату дії.
    """
    signal = trader.compute_signal(df)
    close = float(df["close"].iloc[-1])
    ts = df.index[-1]
    pos = trader.account.positions.get(trader.symbol)

    if signal == 0:
        if pos is not None:
            decision = TradeDecision("close", trader.symbol, 0.0, "сигнал=0")
        else:
            decision = TradeDecision("hold", trader.symbol, 0.0, "")
    elif signal > 0:
        if pos is not None and pos.side == "long":
            decision = TradeDecision("hold", trader.symbol, 0.0, "вже в лонгу")
        else:
            size = trader.settings.position_pct * trader.account.equity / close
            decision = TradeDecision("open_long", trader.symbol, size, f"сигнал={signal}")
    else:
        if pos is not None and pos.side == "short":
            decision = TradeDecision("hold", trader.symbol, 0.0, "вже в шорті")
        else:
            size = trader.settings.position_pct * trader.account.equity / close
            decision = TradeDecision("open_short", trader.symbol, size, f"сигнал={signal}")

    result = trader.execute(decision, close, ts)
    trader.last_signal = signal
    return result
