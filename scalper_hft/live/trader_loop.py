"""Live trader cycle: signal → risk → execution step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from scalper_hft.live.control import ControlState, load_control
from scalper_hft.live.exit_ladders import OneWayTradingLadder
from scalper_hft.live.trader_bars import closed_klines

if TYPE_CHECKING:
    from scalper_hft.live.trader import LiveTrader


@dataclass
class TradeDecision:
    action: str  # "open_long" | "open_short" | "close" | "hold"
    symbol: str = ""
    size: float = 0.0
    reason: str = ""


class SilentAttritionKillSwitch:
    """Детектор 'тихого згасання' альфи (Silent Attrition, PM Ch. 4, 13)."""

    def __init__(
        self,
        alpha_decay: float = 0.1,
        min_trades: int = 5,
        threshold_pnl: float = -0.005,
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


def execute_signal(
    trader: LiveTrader,
    signal: int,
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    block_new_entries: bool = False,
) -> str:
    """Рішення за сигналом на останньому закритому барі → виконання."""
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

    # Peak-to-trough DrawdownBreaker (Narang) — halt + АВТО-flatten, як у pairs_engine.
    # Просідання від історичного піку equity > max_drawdown_pct → блок нових входів
    # і закриття відкритої позиції. Вихід (close) завжди дозволений: відкрита позиція
    # при triggered авто-flatten'иться; при відсутності позиції — halt (без нових).
    if trader.dd_breaker.check(trader.account.equity):
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "dd_breaker:flatten"), close, ts))
            trader.ladder = None
            parts.append("dd_breaker:flatten")
        else:
            parts.append("dd_breaker:halt")
        trader.last_signal = signal
        return " | ".join(parts)

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
        if have != 0:
            parts.append(trader.execute(TradeDecision("close", trader.symbol, 0.0, "реверс"), close, ts))
            trader.ladder = None
        if block_new_entries:
            parts.append("blocked:no_new_entries")
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
    """Один крок циклу live/paper трейдера."""
    from scalper_hft.live.reconcile import reconcile_exchange_state

    ctrl = control if control is not None else load_control(trader.control_path)
    if ctrl.pause:
        return "hold:paused"
    trader.poll_pending_orders(now=now)
    reconcile_exchange_state(
        trader.account,
        trader.client,
        dry_run=trader.settings.dry_run,
        scope={trader.symbol},
    )
    if not trader.settings.dry_run:
        trader.sync_live_equity(now=now)
    signal = trader.compute_signal(df, now=now)
    if ctrl.flatten:
        signal = 0
    return execute_signal(trader, signal, df, now=now, block_new_entries=ctrl.no_new_entries)
