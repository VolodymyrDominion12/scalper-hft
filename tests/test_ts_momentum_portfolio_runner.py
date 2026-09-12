"""Тести paper-портфеля ts_momentum."""

from __future__ import annotations

import dataclasses

from scalper_hft.config import get_settings, set_settings
from scalper_hft.live.ts_momentum_portfolio_runner import (
    TS_MOMENTUM_PAPER_PARAMS,
    TS_MOMENTUM_PAPER_SYMBOLS,
    TsMomentumPortfolioPaperRunner,
)


def test_portfolio_runner_init_splits_capital() -> None:
    orig = get_settings()
    set_settings(dataclasses.replace(orig, dry_run=True))
    try:
        runner = TsMomentumPortfolioPaperRunner(
            symbols=("BTCUSDT", "ETHUSDT"),
            initial_capital=10_000.0,
            store=None,
        )
        assert len(runner.runners) == 2
        assert runner.runners[0].account.initial_capital == 5_000.0
        assert runner.strategy_params == TS_MOMENTUM_PAPER_PARAMS
        assert runner.runners[0].strategy.get("allow_short") is False
    finally:
        set_settings(orig)


def test_default_universe_has_15_symbols() -> None:
    assert len(TS_MOMENTUM_PAPER_SYMBOLS) == 15
