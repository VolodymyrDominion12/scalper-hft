"""Податковий аудит-лог для України (Законопроєкт 10225-д, дослідження §8).

Дослідження §8 наголошує: для фінмоніторингу потрібен повний аудит API-логів
угод з бірж для документального підтвердження «витрат на придбання».

Ключові норми Закону 10225-д (станом на 2026):
    1. Податок лише на ПРИБУТОК (різниця доходу від продажу і витрат на
       придбання), не на оборот.
    2. Крипто-обмін (одна крипта → інша крипта) НЕ оподатковується. Подія
       оподаткування виникає лише при «виході у фіат» (USDT→UAH/USD/EUR) або
       оплаті товарів/послуг криптою.
    3. Ставка: 18% ПДФО + 5% ВЗ = 23% ефективної ставки на чистий прибуток.

Цей модуль — **аудит-дані для бухгалтера**, не податкова порада. Він:
    - веде cost-basis за FIFO per symbol;
    - розрізняє «обмін» (crypto→crypto, не оподатковується) від «вихід у фіат»
      (опудатковується);
    - експортує CSV/JSON для податкового звіту.

⚠ Не дає податкових порад — лише структурує дані угод. Для прийняття рішень
звертатись до кваліфікованого бухгалтера/юриста.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# «Фіатні» валюти для цілей оподаткування (вихід у ці валюти = taxable event).
# USDT/USDC зазвичай трактуються як crypto (не фіат) у Україні — але це
# залежить від тлумачення; залишаємо як параметр.
FIAT_CURRENCIES = frozenset({"UAH", "USD", "EUR", "GBP", "PLN", "CZK"})

# Ефективна податкова ставка (ПДФО 18% + ВЗ 5%).
DEFAULT_TAX_RATE = 0.23


@dataclass
class Fill:
    """Один філ (виконаний ордер) для податкового аудиту.

    Attributes:
        timestamp: ISO-рядок або datetime.
        symbol: Базовий актив (напр. BTC, ETH, LINK).
        side: "buy" | "sell".
        qty: Кількість базового активу.
        price: Ціна в одиницях quote.
        quote: Валюта котирування (USDT, USD, UAH, BTC тощо).
        pair: Торговельна пара (напр. BTCUSDT) — для довідки.
    """

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    quote: str
    pair: str = ""


@dataclass
class TaxEvent:
    """Податкова подія: реалізований прибуток при виході у фіат.

    Для crypto→crypto обмінів taxable=False (не податкова подія, лише cost-basis
    перенос). Для crypto→fiat taxable=True (опудатковується за 23%).
    """

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    quote: str
    proceeds: float  # виручка (qty × price)
    cost_basis: float  # FIFO витрати на придбання
    realized_pnl: float  # proceeds − cost_basis
    taxable: bool  # True якщо вихід у фіат
    tax_estimate: float  # 0 якщо не taxable
    pair: str = ""


@dataclass
class TaxReport:
    """Зведений податковий звіт за період."""

    events: list[TaxEvent] = field(default_factory=list)
    total_proceeds: float = 0.0
    total_cost_basis: float = 0.0
    total_realized_pnl: float = 0.0
    total_tax_estimate: float = 0.0
    taxable_events_count: int = 0
    exchange_events_count: int = 0  # crypto→crypto (не оподатковується)

    def to_dict(self) -> dict:
        return {
            "events": [asdict(e) for e in self.events],
            "summary": {
                "total_proceeds": round(self.total_proceeds, 2),
                "total_cost_basis": round(self.total_cost_basis, 2),
                "total_realized_pnl": round(self.total_realized_pnl, 2),
                "total_tax_estimate": round(self.total_tax_estimate, 2),
                "taxable_events_count": self.taxable_events_count,
                "exchange_events_count": self.exchange_events_count,
            },
        }


class FifoCostBasis:
    """FIFO облік собівартості per symbol.

    Купівлі додають лот у чергу (qty, price). Продажі знімають з початку
    черги (FIFO) і рахують realized_pnl = (sell_price − lot_price) × qty_used.
    """

    def __init__(self) -> None:
        self._lots: dict[str, list[tuple[float, float]]] = {}
        # lots[symbol] = [(qty, price), ...] — черга FIFO

    def buy(self, symbol: str, qty: float, price: float) -> None:
        if qty <= 0:
            return
        self._lots.setdefault(symbol, []).append((qty, price))

    def sell(self, symbol: str, qty: float, price: float) -> tuple[float, float, float]:
        """Продаж qty символу за price. Повертає (cost_basis, proceeds, pnl).

        Знімає лоти FIFO з початку черги. Якщо лотів не вистачає — cost_basis=0
        (нема придбання → весь прибуток оподатковується).
        """
        if qty <= 0:
            return 0.0, 0.0, 0.0
        lots = self._lots.get(symbol, [])
        remaining = qty
        cost = 0.0
        while remaining > 0 and lots:
            lot_qty, lot_price = lots[0]
            used = min(remaining, lot_qty)
            cost += used * lot_price
            remaining -= used
            if used >= lot_qty:
                lots.pop(0)
            else:
                lots[0] = (lot_qty - used, lot_price)
        proceeds = qty * price
        pnl = proceeds - cost
        return cost, proceeds, pnl

    def remaining_basis(self, symbol: str) -> float:
        """Залишкова собівартість для символу (сума qty×price по всіх лотах)."""
        return sum(q * p for q, p in self._lots.get(symbol, []))


def build_tax_report(
    fills: Iterable[Fill],
    *,
    fiat_currencies: frozenset[str] = FIAT_CURRENCIES,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> TaxReport:
    """Побудувати податковий звіт зі списку філів.

    Логіка (Закон 10225-д, дослідження §8):
        - buy → додаємо в FIFO cost-basis.
        - sell з quote ∈ fiat_currencies → taxable event (вихід у фіат):
          realized_pnl = proceeds − cost_basis (FIFO); tax = pnl × rate.
        - sell з quote ∉ fiat_currencies (crypto→crypto) → не taxable:
          перенос cost-basis у нову крипту (за ціною продажу як новою cost-basis).

    Returns:
        TaxReport з подіями та зведенням.
    """
    basis = FifoCostBasis()
    report = TaxReport()

    for f in fills:
        if f.side == "buy":
            basis.buy(f.symbol, f.qty, f.price)
            # купівля — не податкова подія, лише cost-basis
            continue
        # sell
        is_fiat_exit = f.quote.upper() in fiat_currencies
        cost, proceeds, pnl = basis.sell(f.symbol, f.qty, f.price)
        taxable = is_fiat_exit
        tax = pnl * tax_rate if taxable and pnl > 0 else 0.0
        event = TaxEvent(
            timestamp=f.timestamp,
            symbol=f.symbol,
            side=f.side,
            qty=f.qty,
            price=f.price,
            quote=f.quote,
            proceeds=proceeds,
            cost_basis=cost,
            realized_pnl=pnl,
            taxable=taxable,
            tax_estimate=tax,
            pair=f.pair,
        )
        report.events.append(event)
        report.total_proceeds += proceeds
        report.total_cost_basis += cost
        report.total_realized_pnl += pnl
        report.total_tax_estimate += tax
        if taxable:
            report.taxable_events_count += 1
        else:
            report.exchange_events_count += 1
        # crypto→crypto: переносимо cost-basis у нову крипту (якщо quote не фіат).
        if not taxable and f.quote.upper() not in fiat_currencies:
            basis.buy(f.quote, proceeds, 1.0)  # отримали `quote` крипту вартістю proceeds

    return report


def export_csv(report: TaxReport, path: str | Path) -> Path:
    """Експорт подій у CSV для бухгалтера."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "side",
                "qty",
                "price",
                "quote",
                "proceeds",
                "cost_basis",
                "realized_pnl",
                "taxable",
                "tax_estimate",
                "pair",
            ]
        )
        for e in report.events:
            writer.writerow(
                [
                    e.timestamp,
                    e.symbol,
                    e.side,
                    f"{e.qty:.8f}",
                    f"{e.price:.8f}",
                    e.quote,
                    f"{e.proceeds:.2f}",
                    f"{e.cost_basis:.2f}",
                    f"{e.realized_pnl:.2f}",
                    str(e.taxable),
                    f"{e.tax_estimate:.2f}",
                    e.pair,
                ]
            )
    logger.info("Tax CSV exported: %s (%d events)", path, len(report.events))
    return path


def export_json(report: TaxReport, path: str | Path) -> Path:
    """Експорт звіту у JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    logger.info("Tax JSON exported: %s", path)
    return path


__all__ = [
    "FIAT_CURRENCIES",
    "DEFAULT_TAX_RATE",
    "Fill",
    "TaxEvent",
    "TaxReport",
    "FifoCostBasis",
    "build_tax_report",
    "export_csv",
    "export_json",
]
