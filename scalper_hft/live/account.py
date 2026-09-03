"""Paper-акаунт: симуляція виконання без реальних грошей.

Облік як USDT-M ф'ючерс (не спот):
    - cash — гаманець (маржа + реалізований PnL); при відкритті списується
      лише комісія, не повний ноціонал;
    - equity = cash + unrealized PnL (mark-to-market);
    - realized_pnl — журнал для звітів, не додається до cash вдруге.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    size: float  # у базовій валюті (e.g. BTC)
    entry_price: float
    entry_ts: pd.Timestamp
    entry_fee: float = 0.0

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": float(self.size),
            "entry_price": float(self.entry_price),
            "entry_ts": str(self.entry_ts),
            "entry_fee": float(self.entry_fee),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> Position:
        return cls(
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            size=float(data["size"]),
            entry_price=float(data["entry_price"]),
            entry_ts=pd.Timestamp(data["entry_ts"]),
            entry_fee=float(data.get("entry_fee") or 0.0),
        )


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

    def close_position(
        self, symbol: str, price: float, ts: pd.Timestamp, is_maker: bool = False, size: float | None = None
    ) -> dict:
        if symbol not in self.positions:
            raise ValueError(f"позиція {symbol} не знайдена")
        pos = self.positions[symbol]
        close_sz = size if size is not None else pos.size
        if close_sz <= 0 or close_sz > pos.size + 1e-9:
            raise ValueError(f"некоректний розмір для закриття: {close_sz} (доступно {pos.size})")

        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * close_sz * fee_rate

        if pos.side == "long":
            price_pnl = (price - pos.entry_price) * close_sz
        else:
            price_pnl = (pos.entry_price - price) * close_sz

        # Proportion of entry fee to realize
        frac = close_sz / pos.size
        realized_entry_fee = pos.entry_fee * frac

        pnl = price_pnl - fee - realized_entry_fee
        self.cash += price_pnl - fee
        self.realized_pnl += pnl

        trade = {
            "type": "trade",
            "symbol": symbol,
            "side": pos.side,
            "size": close_sz,
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

        if close_sz >= pos.size - 1e-9:
            self.positions.pop(symbol)
            self._marks.pop(symbol, None)
        else:
            pos.size -= close_sz
            pos.entry_fee -= realized_entry_fee

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

    def to_snapshot(self) -> dict[str, Any]:
        """Стан рахунку для SQLite. Журнал trades не входить — він уже в store."""
        return {
            "initial_capital": float(self.initial_capital),
            "taker_fee": float(self.taker_fee),
            "maker_fee": float(self.maker_fee),
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "consecutive_losses": int(self.consecutive_losses),
            "day_start_equity": float(self.day_start_equity),
            "positions": [p.to_snapshot() for p in self.positions.values()],
            "marks": {k: float(v) for k, v in self._marks.items()},
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> PaperAccount:
        acc = cls(
            initial_capital=float(data.get("initial_capital") or 10_000.0),
            taker_fee=float(data.get("taker_fee") or 0.0005),
            maker_fee=float(data.get("maker_fee") or 0.0002),
        )
        acc.cash = float(data["cash"])
        acc.realized_pnl = float(data.get("realized_pnl") or 0.0)
        acc.consecutive_losses = int(data.get("consecutive_losses") or 0)
        acc.day_start_equity = float(data.get("day_start_equity") or acc.cash)
        acc.positions = {}
        for row in data.get("positions") or []:
            pos = Position.from_snapshot(row)
            acc.positions[pos.symbol] = pos
        acc._marks = {str(k): float(v) for k, v in (data.get("marks") or {}).items()}
        return acc
