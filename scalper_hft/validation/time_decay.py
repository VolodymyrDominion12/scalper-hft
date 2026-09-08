"""Time-decay тест: прибуток при затримці входу на 0..N барів (Narang гл. 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy


class LagMetricList(list):
    """Список метрик за лагами, що підтримує як list-інтерфейс, так і dict (.items(), .keys(), .values())."""

    def items(self) -> list[tuple[int, float]]:
        return list(enumerate(self))

    def keys(self) -> list[int]:
        return list(range(len(self)))

    def values(self) -> list[float]:
        return list(self)


@dataclass
class TimeDecayResult:
    lags: list[int]
    sharpes: list[float]
    returns: list[float]

    def __post_init__(self) -> None:
        if not isinstance(self.sharpes, LagMetricList):
            self.sharpes = LagMetricList(self.sharpes)
        if not isinstance(self.returns, LagMetricList):
            self.returns = LagMetricList(self.returns)

    def summary(self) -> str:
        rows = " | ".join(f"lag{lag} SR={s:+.3f}" for lag, s in zip(self.lags, self.sharpes))
        return f"Time-decay: {rows}"


class _Lagged(Strategy):
    name = "lagged"
    param_space: dict[str, tuple[float, float, float]] = {}
    needs_trades: bool = False
    needs_funding: bool = False

    def __init__(self, base: Strategy, lag: int) -> None:
        super().__init__()
        self.base = base
        self.lag = lag
        self.name = f"{base.name}_lag{lag}"
        self.needs_trades = getattr(base, "needs_trades", False)
        self.needs_funding = getattr(base, "needs_funding", False)

    def generate_signals(self, df, trades=None, funding=None):
        if self.needs_funding:
            sig = self.base.generate_signals(df, funding=funding)
        elif self.needs_trades:
            sig = self.base.generate_signals(df, trades=trades)
        else:
            sig = self.base.generate_signals(df)
        if self.lag <= 0:
            return sig
        return sig.shift(self.lag).fillna(0)

    def exit_levels(self, df):
        """Делегування рівнів SL/TP базовій стратегії (якщо вони є)."""
        fn = getattr(self.base, "exit_levels", None)
        return fn(df) if fn is not None else None


def time_decay_test(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    max_lag: int = 3,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
) -> TimeDecayResult:
    lags = list(range(max_lag + 1))
    sharpes: list[float] = []
    rets: list[float] = []
    for lag in lags:
        wrapped = _Lagged(strategy, lag)
        res = run_backtest(
            df, wrapped, cost=cost or CostModel(), trades=trades, funding=funding, position_pct=position_pct
        )
        sharpes.append(float(res.metrics.sharpe_hourly))
        rets.append(float(res.metrics.total_return))
    return TimeDecayResult(lags=lags, sharpes=sharpes, returns=rets)


def pairs_time_decay(
    signals: pd.Series,
    spread: pd.Series,
    max_lag: int = 3,
) -> TimeDecayResult:
    """Те саме для pairs: pos = signal.shift(1+lag), PnL = −pos Δspread."""
    d = spread.diff().fillna(0.0)
    lags, sharpes, rets = [], [], []
    for lag in range(max_lag + 1):
        pos = signals.astype(float).shift(1 + lag).fillna(0.0)
        pnl = -pos * d
        sr = float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(len(pnl))) if pnl.std(ddof=0) else 0.0
        lags.append(lag)
        sharpes.append(sr)
        rets.append(float(np.prod(1.0 + pnl.to_numpy(dtype=float))) - 1.0)
    return TimeDecayResult(lags=lags, sharpes=sharpes, returns=rets)
