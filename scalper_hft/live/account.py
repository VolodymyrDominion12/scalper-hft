"""Paper-акаунт: симуляція виконання без реальних грошей.

Облік як USDT-M ф'ючерс (не спот):
    - cash — гаманець (маржа + реалізований PnL); при відкритті списується
      лише комісія, не повний ноціонал;
    - equity = cash + unrealized PnL (mark-to-market);
    - realized_pnl — журнал для звітів, не додається до cash вдруге.

Грошові поля зберігаються як Decimal (scalper_hft.live.money); API властивостей
повертає float для сумісності з існуючими тестами та REST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from scalper_hft.live.money import as_float, money, quantize, to_decimal


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
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    trades: list[dict[str, Any]] = field(default_factory=list, init=False)
    consecutive_losses: int = 0
    _cash: Decimal = field(init=False, repr=False)
    _realized_pnl: Decimal = field(init=False, repr=False)
    _day_start_equity: Decimal = field(init=False, repr=False)
    _marks: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        cap = money(self.initial_capital)
        self._cash = cap
        self._realized_pnl = money(0)
        self._day_start_equity = cap

    @property
    def cash(self) -> float:
        return as_float(self._cash)

    @cash.setter
    def cash(self, value: float) -> None:
        self._cash = money(value)

    @property
    def realized_pnl(self) -> float:
        return as_float(self._realized_pnl)

    @realized_pnl.setter
    def realized_pnl(self, value: float) -> None:
        self._realized_pnl = money(value)

    @property
    def day_start_equity(self) -> float:
        return as_float(self._day_start_equity)

    @day_start_equity.setter
    def day_start_equity(self, value: float) -> None:
        self._day_start_equity = money(value)

    def mark(self, prices: dict[str, float]) -> None:
        """Оновити mark-ціни для UPNL (ключ — символ)."""
        self._marks.update(prices)

    def unrealized_pnl(self, prices: dict[str, float] | None = None) -> float:
        marks = {**self._marks, **(prices or {})}
        total = money(0)
        for sym, pos in self.positions.items():
            px = money(marks.get(sym, pos.entry_price))
            sz = money(pos.size)
            ep = money(pos.entry_price)
            if pos.side == "long":
                total += quantize((px - ep) * sz)
            else:
                total += quantize((ep - px) * sz)
        return as_float(total)

    def equity_at(self, prices: dict[str, float] | None = None) -> float:
        """Капітал з mark-to-market. Без prices — останні mark або ціна входу."""
        return as_float(self._cash + money(self.unrealized_pnl(prices)))

    def gross_notional(self, prices: dict[str, float] | None = None) -> float:
        """Сумарний |ціна × розмір| відкритих ніг."""
        marks = {**self._marks, **(prices or {})}
        total = money(0)
        for sym, pos in self.positions.items():
            px = money(marks.get(sym, pos.entry_price))
            total += quantize(abs(px * money(pos.size)))
        return as_float(total)

    @property
    def equity(self) -> float:
        return self.equity_at()

    def _fee_rate(self, is_maker: bool) -> Decimal:
        return to_decimal(self.maker_fee if is_maker else self.taker_fee)

    def open_position(
        self, symbol: str, side: str, size: float, price: float, ts: pd.Timestamp, is_maker: bool = False
    ) -> None:
        if symbol in self.positions:
            raise ValueError(f"позиція {symbol} вже відкрита — спочатку close")
        px = money(price)
        sz = money(size)
        fee = quantize(px * sz * self._fee_rate(is_maker))
        self._cash -= fee
        self.positions[symbol] = Position(symbol, side, as_float(sz), as_float(px), ts, entry_fee=as_float(fee))
        self._marks[symbol] = as_float(px)

    def add_to_position(self, symbol: str, size: float, price: float, ts: pd.Timestamp, is_maker: bool = False) -> None:
        """Доливка до існуючої позиції (часткові філи одного ордера).

        Середньозважена ціна входу; комісія додається до entry_fee.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            raise ValueError(f"позиція {symbol} не знайдена — спочатку open")
        add_sz = money(size)
        if add_sz <= 0:
            raise ValueError(f"некоректний розмір доливки: {size}")
        px = money(price)
        fee = quantize(px * add_sz * self._fee_rate(is_maker))
        self._cash -= fee
        old_sz = money(pos.size)
        old_ep = money(pos.entry_price)
        total = old_sz + add_sz
        pos.entry_price = as_float(quantize((old_ep * old_sz + px * add_sz) / total))
        pos.size = as_float(total)
        pos.entry_fee = as_float(money(pos.entry_fee) + fee)
        pos.entry_ts = ts if ts < pos.entry_ts else pos.entry_ts
        self._marks[symbol] = as_float(px)

    def close_position(
        self, symbol: str, price: float, ts: pd.Timestamp, is_maker: bool = False, size: float | None = None
    ) -> dict[str, Any]:
        if symbol not in self.positions:
            raise ValueError(f"позиція {symbol} не знайдена")
        pos = self.positions[symbol]
        pos_sz = money(pos.size)
        close_sz = money(size if size is not None else pos.size)
        if close_sz <= 0 or close_sz > pos_sz + money("0.00000001"):
            raise ValueError(f"некоректний розмір для закриття: {close_sz} (доступно {pos_sz})")

        px = money(price)
        fee = quantize(px * close_sz * self._fee_rate(is_maker))
        ep = money(pos.entry_price)

        if pos.side == "long":
            price_pnl = quantize((px - ep) * close_sz)
        else:
            price_pnl = quantize((ep - px) * close_sz)

        frac = close_sz / pos_sz
        realized_entry_fee = quantize(money(pos.entry_fee) * frac)

        pnl = quantize(price_pnl - fee - realized_entry_fee)
        self._cash += price_pnl - fee
        self._realized_pnl += pnl

        trade = {
            "type": "trade",
            "symbol": symbol,
            "side": pos.side,
            "size": as_float(close_sz),
            "entry_price": pos.entry_price,
            "exit_price": as_float(px),
            "entry_ts": pos.entry_ts,
            "exit_ts": ts,
            "pnl": as_float(pnl),
        }
        self.trades.append(trade)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if close_sz >= pos_sz - money("0.00000001"):
            self.positions.pop(symbol)
            self._marks.pop(symbol, None)
        else:
            pos.size = as_float(pos_sz - close_sz)
            pos.entry_fee = as_float(money(pos.entry_fee) - realized_entry_fee)

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
        notional = quantize(money(pos.entry_price) * money(pos.size))
        side = to_decimal(1) if pos.side == "long" else to_decimal(-1)
        pnl = quantize(-side * to_decimal(rate) * notional)
        self._cash += pnl
        self._realized_pnl += pnl
        self.trades.append(
            {
                "type": "funding",
                "symbol": symbol,
                "ts": ts,
                "rate": rate,
                "pnl": as_float(pnl),
            }
        )
        return as_float(pnl)

    def roll_to_new_day(self, equity: float) -> None:
        """Початок нового дня: фіксуємо стартовий капітал для денного ліміту
        збитків і скидаємо лічильник серії збитків (пауза діє лише до кінця дня)."""
        self._day_start_equity = money(equity)
        self.consecutive_losses = 0

    def to_snapshot(self) -> dict[str, Any]:
        """Стан рахунку для SQLite. Журнал trades не входить — він уже в store."""
        return {
            "initial_capital": float(self.initial_capital),
            "taker_fee": float(self.taker_fee),
            "maker_fee": float(self.maker_fee),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "consecutive_losses": int(self.consecutive_losses),
            "day_start_equity": self.day_start_equity,
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
