"""Подієвий рушій бектесту для мікроструктурних стратегій.

Використовується для:
    - market_maker (пасивні лімітні ордери, інвентар, adverse selection);
    - ob_imbalance на снапшотах стакана (колонка 'imbalance' у df).

Модель market maker (спрощена, але чесна щодо обмежень):
    - на кожному барі котируємо bid/ask навколо mid (offset = спред × mult);
    - філл відбувається, якщо бар своїм high/low перетинає рівень котирування;
    - ціна філа = рівень нашого ліміту (buy — bid, sell — ask): resting limit
      виконується за своєю ціною навіть при гепі крізь рівень; ціни «краще за
      ліміт» не буває (раніше haircut давав фантомний прибуток всередину спреду);
    - adverse selection моделюється «толом» за кожен філ
      (adverse_sel_haircut × спред) + mark-to-market за поточний close бару
      (геп, що заповнив наш bid, одразу видно в equity);
    - інвентар обмежений inventory_cap; при досягненні капу котируємо лише
      у протилежний бік (книга, гл. 15 — NCMM inventory management);
    - облік: середня ціна входу, реалізований PnL при частковому закритті
      (знак шорт-покриття коректний: (entry − exit)/entry), mark-to-market
      відкритої частини; комісії maker за кожен філл.

⚠ Недоліки моделі: немає queue position, спред оцінюється з барів,
глибина стакана не моделюється. Для реальної оцінки MM —
nautilus_trader + L2 дані (Tardis.dev).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.strategies.base import Strategy


@dataclass
class EventBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return self.metrics.summary()


def run_event_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    cost: CostModel | None = None,
    quote_size_pct: float = 0.02,
    inventory_cap: float = 1.0,
    spread_offset_mult: float = 0.5,
    adverse_sel_haircut: float = 0.5,
    spread_frac: float | None = None,
) -> EventBacktestResult:
    """Подієвий бектест пасивного market maker на свічкових даних.

    df: open/high/low/close/volume (краще 1s/5s — реалістичніші перетини).
    spread_frac: РЕАЛЬНИЙ відносний спред (ask−bid)/mid. За замовчуванням
        0.0002 (2 bps — типово для BTCUSDT перп). НЕ використовувати
        діапазон бару як проксі спреду — це завищує прибуток.
    """
    cost = cost or CostModel()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    avg_spread_frac = spread_frac if spread_frac is not None else 0.0002
    if not np.isfinite(avg_spread_frac) or avg_spread_frac <= 0:
        avg_spread_frac = 0.0002

    half = close * avg_spread_frac * spread_offset_mult / 2.0
    bid_q = close - half
    ask_q = close + half

    # Одиниці: inventory у "філл-одиницях"; ноціонал однієї одиниці фіксований
    notional_per_unit = quote_size_pct * initial_capital

    inventory = 0.0  # у філл-одиницях (signed)
    avg_cost = 0.0  # середня ціна входу ($)
    realized_pnl = 0.0
    fees_paid = 0.0
    trades: list[dict] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []
    n_fills_total = 0

    def _pnl(entry_px: float, exit_px: float, units: float) -> float:
        return (exit_px - entry_px) / entry_px * notional_per_unit * units if entry_px else 0.0

    def _short_pnl(entry_px: float, exit_px: float, units: float) -> float:
        """PnL закриття шорта: (entry − exit)/entry (виграш, коли викупили дешевше)."""
        return (entry_px - exit_px) / entry_px * notional_per_unit * units if entry_px else 0.0

    def _open_trade(side: int, ts: pd.Timestamp, price: float) -> None:
        trades.append({"entry_ts": ts, "exit_ts": ts, "side": side, "ret": 0.0, "entry_price": price})

    def _close_trade(ts: pd.Timestamp, price: float, pnl: float, units: float) -> None:
        if trades:
            t = trades[-1]
            t["exit_ts"] = ts
            t["exit_price"] = price
            t["ret"] = pnl / (notional_per_unit * units) if units else 0.0

    for i in range(1, len(df)):
        ts = df.index[i]
        prev_bid, prev_ask = bid_q.iloc[i - 1], ask_q.iloc[i - 1]

        fills: list[float] = []
        # наш bid заповнений: бар торкнувся/перетнув наш bid. Філ — РІВНО за наш
        # ліміт (prev_bid): resting limit виконується за своєю ціною навіть при
        # гепі крізь рівень. Ціни «краще за ліміт» (всередину спреду) не буває —
        # раніше haircut давав фантомний прибуток (buy нижче bid, sell вище ask).
        if inventory < inventory_cap and low.iloc[i] <= prev_bid:
            fills.append(prev_bid)  # купуємо
        # наш ask заповнений: бар торкнувся/перетнув наш ask
        if inventory > -inventory_cap and high.iloc[i] >= prev_ask:
            fills.append(-prev_ask)  # продаємо

        for f in fills:
            n_fills_total += 1
            size = 1.0  # одна філл-одиниця
            fee = cost.maker_cost_per_side() * notional_per_unit
            fees_paid += fee
            # adverse selection: після філа ринок рухається проти нас — сплачуємо
            # "тол" за кожен філ (частка спреду), окремо від mark-to-close.
            adverse_toll = adverse_sel_haircut * avg_spread_frac * notional_per_unit * size
            fees_paid += adverse_toll
            if f > 0:  # buy
                if inventory < 0:  # закриваємо шорт
                    closed = min(size, -inventory)
                    pnl = _short_pnl(avg_cost, f, closed)
                    realized_pnl += pnl
                    _close_trade(ts, f, pnl, closed)
                    inventory += closed
                    size -= closed
                if size > 0:
                    if inventory == 0:
                        _open_trade(1, ts, f)
                    total_notional = abs(inventory) * avg_cost + size * f
                    avg_cost = total_notional / (abs(inventory) + size)
                    inventory += size
            else:  # sell
                px = -f
                if inventory > 0:  # закриваємо лонг
                    closed = min(size, inventory)
                    pnl = _pnl(avg_cost, px, closed)
                    realized_pnl += pnl
                    _close_trade(ts, px, pnl, closed)
                    inventory -= closed
                    size -= closed
                if size > 0:
                    if inventory == 0:
                        _open_trade(-1, ts, px)
                    total_notional = abs(inventory) * avg_cost + size * px
                    avg_cost = total_notional / (abs(inventory) + size)
                    inventory -= size

        # mark-to-market за ПОТОЧНИЙ close бару i: геп, що заповнив наш bid,
        # одразу відображається в equity (раніше — prev_mid, тобто марка
        # запізнювалась на бар і геп-прибуток був фантомним).
        unrealized = _pnl(avg_cost, close.iloc[i], inventory) if inventory != 0 else 0.0
        equity = initial_capital + realized_pnl - fees_paid + unrealized
        equity_points.append((ts, equity))

    # Залишковий інвентар на кінець серії: синтетичне закриття за останній close,
    # щоб n_trades/статистика угод не занижували активність MM. Equity не змінюється:
    # остання точка вже промаркована за останній close (unrealized → realized).
    if inventory != 0 and len(df) > 1:
        last_close = float(close.iloc[-1])
        pnl = _pnl(avg_cost, last_close, inventory) if inventory > 0 else _short_pnl(avg_cost, last_close, -inventory)
        realized_pnl += pnl
        if trades:
            trades[-1]["exit_ts"] = df.index[-1]
            trades[-1]["exit_price"] = last_close
            trades[-1]["ret"] = pnl / (notional_per_unit * abs(inventory)) if inventory else 0.0
        inventory = 0.0

    equity_series = pd.Series(dict(equity_points)).sort_index()
    if equity_series.empty or len(equity_series) < 2:
        raise ValueError("Подієвий бектест не дав жодної точки equity")

    trades_df = (
        pd.DataFrame(trades, columns=["entry_ts", "exit_ts", "side", "ret", "entry_price", "exit_price"])
        if trades
        else pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "ret", "entry_price", "exit_price"])
    )
    metrics = compute_metrics(equity_series, trades=trades_df, exposure=0.0, turnover=0.0)
    return EventBacktestResult(
        equity=equity_series,
        trades=trades_df,
        metrics=metrics,
        params={
            "strategy": strategy.name,
            "quote_size_pct": quote_size_pct,
            "inventory_cap": inventory_cap,
            "spread_offset_mult": spread_offset_mult,
            "adverse_sel_haircut": adverse_sel_haircut,
            "n_fills": n_fills_total,
        },
    )
