"""Бектест: виконання, метрики, рушії (книга, гл. 5 — Transaction Cost Models; гл. 9 — Testing)."""

from scalper_hft.backtest.engine import BacktestResult, run_backtest
from scalper_hft.backtest.event_engine import EventBacktestResult, run_event_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics

__all__ = [
    "BacktestResult",
    "run_backtest",
    "EventBacktestResult",
    "run_event_backtest",
    "CostModel",
    "BacktestMetrics",
    "compute_metrics",
]
