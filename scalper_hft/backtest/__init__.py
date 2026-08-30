"""Бектест: виконання, метрики, рушії (книга, гл. 5 — Transaction Cost Models; гл. 9 — Testing)."""

from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.event_engine import EventBacktestResult, run_event_backtest
from scalper_hft.backtest.execution import CostModel, ImplementationShortfallTracker, apply_breakeven_gate
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.backtest.micro_price import QueuePositionModel, calculate_micro_price, estimate_order_book_imbalance

__all__ = [
    "BacktestResult",
    "run_backtest",
    "EventBacktestResult",
    "run_event_backtest",
    "CostModel",
    "ImplementationShortfallTracker",
    "apply_breakeven_gate",
    "BacktestMetrics",
    "compute_metrics",
    "calculate_micro_price",
    "estimate_order_book_imbalance",
    "QueuePositionModel",
]

