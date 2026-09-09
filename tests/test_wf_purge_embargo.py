"""Тести WF purge/embargo (1C) — прогін між train/test та ембарго між OOS-вікнами.

Перевіряє:
- purge_bars>0 додає прогін між кінцем train і початком test (test_start зміщений);
- embargo_bars>0 додає зазор між послідовними OOS-вікнами;
- дефолт (0/0) = ідентична попередній поведінці (non-overlapping contiguous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy
from scalper_hft.validation.walk_forward import run_walk_forward


def _df(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx
    )


class _Flat(Strategy):
    name = "flat"
    param_space: dict = {}
    needs_trades = False
    needs_funding = False

    def generate_signals(self, df, trades=None, funding=None):
        return pd.Series(0, index=df.index)


def test_wf_default_no_purge_contiguous() -> None:
    df = _df(600)
    res = run_walk_forward(df, _Flat(), train_bars=200, test_bars=100, cost=CostModel())
    # test-вікна contiguous non-overlapping
    for i in range(1, len(res.windows)):
        assert res.windows[i].test_start == res.windows[i - 1].test_end
    # train_end == test_start (без прогіну)
    assert all(w.test_start == w.train_end for w in res.windows)


def test_wf_purge_shifts_test_start() -> None:
    df = _df(700)
    purge = 20
    res = run_walk_forward(df, _Flat(), train_bars=200, test_bars=100, cost=CostModel(), purge_bars=purge)
    for w in res.windows:
        assert w.test_start == w.train_end + purge  # прогін між train і test
        assert w.test_end == w.test_start + 100


def test_wf_embargo_creates_gap_between_test_windows() -> None:
    df = _df(900)
    embargo = 30
    res = run_walk_forward(df, _Flat(), train_bars=200, test_bars=100, cost=CostModel(), embargo_bars=embargo)
    for i in range(1, len(res.windows)):
        gap = res.windows[i].test_start - res.windows[i - 1].test_end
        assert gap == embargo  # зазор між OOS-вікнами


def test_wf_purge_reduces_window_count() -> None:
    df = _df(600)
    n0 = len(run_walk_forward(df, _Flat(), train_bars=200, test_bars=100, cost=CostModel()).windows)
    n1 = len(
        run_walk_forward(
            df, _Flat(), train_bars=200, test_bars=100, cost=CostModel(), purge_bars=50, embargo_bars=50
        ).windows
    )
    assert n1 < n0  # прогін+ембарго зменшують кількість вікон
