"""Тести benchmark та audit_extensions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.validation.benchmark import buy_and_hold_sharpe, strategy_beats_benchmark
from scalper_hft.validation.cell_audit import cell_verdict


def test_buy_and_hold_sharpe_positive_trend() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame({"close": close, "open": close, "high": close, "low": close, "volume": 1.0})
    sr = buy_and_hold_sharpe(df)
    assert sr > 0


def test_cell_verdict_benchmark_fail() -> None:
    from scalper_hft.validation.cell_audit import CellAudit

    audit = CellAudit(
        symbol="BTCUSDT",
        interval="1h",
        strategy="x",
        status="ok",
        avg_oos_sharpe=0.5,
        oos_pos_frac=0.6,
        dsr=0.97,
        smoothness=0.4,
        n_trades_oos=40,
        bt_sharpe=0.1,
        benchmark_sharpe=0.5,
        quintile_pass=True,
        time_decay_pass=True,
        stress_pass=True,
    )
    label, reasons = cell_verdict(audit, mode="exploratory")
    assert label == "FAIL"
    assert "benchmark" in reasons


def test_strategy_beats_benchmark() -> None:
    assert strategy_beats_benchmark(1.0, 0.5)
    assert not strategy_beats_benchmark(0.5, 1.0)
