"""Paper-акаунт: симуляція виконання без реальних грошей.

Використовується в режимі DRY_RUN=true (testnet) або коли API-ключів немає.
Облік: капітал, позиції, комісії, історія угод.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    size: float  # у базовій валюті (e.g. BTC)
    entry_price: float
    entry_ts: pd.Timestamp


@dataclass
class PaperAccount:
    initial_capital: float = 10_000.0
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    cash: float = field(default=10_000.0, init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    trades: list[dict] = field(default_factory=list, init=False)
    realized_pnl: float = 0.0
    consecutive_losses: int = 0
    day_start_equity: float = field(default=10_000.0, init=False)

    @property
    def equity(self) -> float:
        return self.cash + self.realized_pnl

    def open_position(
        self, symbol: str, side: str, size: float, price: float, ts: pd.Timestamp, is_maker: bool = False
    ) -> None:
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * size * fee_rate
        if side == "long":
            self.cash -= price * size + fee
        else:
            self.cash += price * size - fee
        self.positions[symbol] = Position(symbol, side, size, price, ts)

    def close_position(
        self, symbol: str, price: float, ts: pd.Timestamp, is_maker: bool = False
    ) -> dict:
        pos = self.positions.pop(symbol)
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * pos.size * fee_rate
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.size - fee
            self.cash += price * pos.size - fee
        else:
            pnl = (pos.entry_price - price) * pos.size - fee
            self.cash -= price * pos.size + fee
        self.realized_pnl += pnl
        trade = {
            "type": "trade",
            "symbol": symbol,
            "side": pos.side,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "entry_ts": pos.entry_ts,
            "exit_ts": ts,
            "pnl": pnl,
        }
        self.trades.append(trade)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        return trade

    @property
    def is_flat(self) -> bool:
        return len(self.positions) == 0

    def apply_funding(self, symbol: str, rate: float, ts: pd.Timestamp) -> float:
        """Funding-платіж: лонг платить позитивний фандінг, шорт отримує.

        funding_pnl = -side × rate × notional, де side: +1 лонг, -1 шорт.
        Повертає суму платежу (для логування/звіту).
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        notional = pos.entry_price * pos.size
        side = 1.0 if pos.side == "long" else -1.0
        pnl = -side * rate * notional
        self.realized_pnl += pnl
        self.trades.append(
            {
                "type": "funding",
                "symbol": symbol,
                "ts": ts,
                "rate": rate,
                "pnl": pnl,
            }
        )
        return pnl

    def roll_to_new_day(self, equity: float) -> None:
        """Початок нового дня: фіксуємо стартовий капітал для денного ліміту
        збитків і скидаємо лічильник серії збитків (пауза діє лише до кінця дня)."""
        self.day_start_equity = equity
        self.consecutive_losses = 0
