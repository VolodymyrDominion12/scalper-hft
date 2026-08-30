"""Роутер рушія: векторний vs подієвий (ob_imbalance / market_maker)."""

from __future__ import annotations

from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.event_engine import EventBacktestResult, run_event_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy

EVENT_STRATEGIES = frozenset({"ob_imbalance", "market_maker"})


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
) -> BacktestResult | EventBacktestResult:
    name = getattr(strategy, "name", "")
    if name in EVENT_STRATEGIES:
        return run_event_backtest(
            df,
            strategy,
            initial_capital=initial_capital,
            cost=cost,
            quote_size_pct=position_pct,
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
    )
