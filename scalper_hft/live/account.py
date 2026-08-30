"""Paper-акаунт: симуляція виконання без реальних грошей.

Облік як USDT-M ф'ючерс (не спот):
    - cash — гаманець (маржа + реалізований PnL); при відкритті списується
      лише комісія, не повний ноціонал;
    - equity = cash + unrealized PnL (mark-to-market);
    - realized_pnl — журнал для звітів, не додається до cash вдруге.
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
    entry_fee: float = 0.0


@dataclass
class PaperAccount:
    initial_capital: float = 10_000.0
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    trades: list[dict] = field(default_factory=list, init=False)
    realized_pnl: float = 0.0
    consecutive_losses: int = 0
    day_start_equity: float = field(init=False)
    _marks: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_capital)
        self.day_start_equity = float(self.initial_capital)

    def mark(self, prices: dict[str, float]) -> None:
        """Оновити mark-ціни для UPNL (ключ — символ)."""
        self._marks.update(prices)

    def unrealized_pnl(self, prices: dict[str, float] | None = None) -> float:
        marks = {**self._marks, **(prices or {})}
        total = 0.0
        for sym, pos in self.positions.items():
            px = marks.get(sym, pos.entry_price)
            if pos.side == "long":
                total += (px - pos.entry_price) * pos.size
            else:
                total += (pos.entry_price - px) * pos.size
        return total

    def equity_at(self, prices: dict[str, float] | None = None) -> float:
        """Капітал з mark-to-market. Без prices — останні mark або ціна входу."""
        return self.cash + self.unrealized_pnl(prices)

    def gross_notional(self, prices: dict[str, float] | None = None) -> float:
        """Сумарний |ціна × розмір| відкритих ніг."""
        marks = {**self._marks, **(prices or {})}
        total = 0.0
        for sym, pos in self.positions.items():
            px = marks.get(sym, pos.entry_price)
            total += abs(px * pos.size)
        return total

    @property
    def equity(self) -> float:
        return self.equity_at()

    def open_position(
        self, symbol: str, side: str, size: float, price: float, ts: pd.Timestamp, is_maker: bool = False
    ) -> None:
        if symbol in self.positions:
            raise ValueError(f"позиція {symbol} вже відкрита — спочатку close")
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * size * fee_rate
        self.cash -= fee
        self.positions[symbol] = Position(symbol, side, size, price, ts, entry_fee=fee)
        self._marks[symbol] = price

    def close_position(self, symbol: str, price: float, ts: pd.Timestamp, is_maker: bool = False) -> dict:
        pos = self.positions.pop(symbol)
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * pos.size * fee_rate
        if pos.side == "long":
            price_pnl = (price - pos.entry_price) * pos.size
        else:
            price_pnl = (pos.entry_price - price) * pos.size
        pnl = price_pnl - fee - pos.entry_fee
        self.cash += price_pnl - fee
        self.realized_pnl += pnl
        self._marks.pop(symbol, None)
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
        self.cash += pnl
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
