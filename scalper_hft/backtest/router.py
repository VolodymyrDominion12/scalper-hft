"""Роутер рушія: векторний vs подієвий (market_maker).

Подієвий рушій (`event_engine`) — це симулятор ПАСИВНОГО market maker
(котирування обох сторін, інвентар, adverse selection): він доречний лише
для `market_maker`, сигнали якого (`generate_signals` → 0) не керують
напрямком — його облік живе в самому рушії, а параметри стратегії
(спред, інвентар-кап, haircut, розмір котирування) тепер реально
передаються в рушій.

`ob_imbalance` — НАПРЯМКОВА стратегія (власний `generate_signals` на
колонці imbalance/buy_ratio): для неї коректний векторний рушій
(`run_backtest`, сигнал на close t → виконання t+1 з maker/taker
комісіями). Раніше вона потрапляла в event_engine, який ігнорує сигнали —
результат був константою, що не залежить від її параметрів.
"""

from __future__ import annotations

from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.event_engine import EventBacktestResult, run_event_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy

# Стратегії, чиє виконання моделює ЛИШЕ подієвий рушій (пасивний MM).
EVENT_STRATEGIES = frozenset({"market_maker"})


def run_strategy_backtest(
    df,
    strategy: Strategy,
    *,
    cost: CostModel | None = None,
    trades=None,
    funding=None,
    position_pct: float = 0.01,
    is_maker: bool = False,
    initial_capital: float = 10_000.0,
    trace: bool = False,
) -> BacktestResult | EventBacktestResult:
    name = getattr(strategy, "name", "")
    if name in EVENT_STRATEGIES:
        # Параметри стратегії → параметри рушія (раніше лишались дефолти,
        # тож sweep/Optuna по market_maker повертали константу).
        return run_event_backtest(
            df,
            strategy,
            initial_capital=initial_capital,
            cost=cost,
            quote_size_pct=float(strategy.get("quote_size_pct", position_pct)),
            inventory_cap=float(strategy.get("inventory_cap", 1.0)),
            spread_offset_mult=float(strategy.get("spread_offset_mult", 0.5)),
            adverse_sel_haircut=float(strategy.get("adverse_sel_haircut", 0.5)),
        )
    return run_backtest(
        df,
        strategy,
        initial_capital=initial_capital,
        cost=cost,
        position_pct=position_pct,
        trades=trades,
        funding=funding,
        is_maker=is_maker,
        trace=trace,
    )
