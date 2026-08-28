"""Чутливість до параметрів: плато vs ізольований пік (ознака перенавчання).

Здорова стратегія має "плато" хороших параметрів: сусідні значення дають
схожі результати. Ізольований вузький пік — класична ознака підгонки під
шум (книга, гл. 11 — data mining; гл. 12 — evaluating the edge).

Повертає матрицю метрик по сітці параметрів + коефіцієнти гладкості.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy


@dataclass
class SensitivityResult:
    grid: pd.DataFrame  # колонки: параметри + metric
    best_params: dict
    smoothness: float  # 1.0 = ідеальне плато; ~0 = вузький пік
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Sensitivity: {len(self.grid)} комбінацій; smoothness={self.smoothness:.3f}\n"
            f"best: {self.best_params}\n"
            "smoothness < 0.3 → ймовірне перенавчання (ізольований пік)\n"
            f"{self.grid.sort_values('metric', ascending=False).head(8).to_string(index=False)}"
        )


def parameter_sensitivity(
    df: pd.DataFrame,
    strategy: Strategy,
    param_name: str,
    values: list[float],
    secondary_param: str | None = None,
    secondary_values: list[float] | None = None,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> SensitivityResult:
    """Прогін бектесту по сітці одного (або двох) параметрів.

    smoothness: середня |Δ metric| між сусідніми значеннями, нормована.
    """
    rows: list[dict] = []
    values = sorted(values)
    sec_values = secondary_values or [None]

    for v in values:
        for s in sec_values:
            p = dict(strategy.params)
            p[param_name] = v
            if secondary_param and s is not None:
                p[secondary_param] = s
            strat = type(strategy)(**p)
            try:
                res = run_backtest(df, strat, cost=cost, trades=trades, funding=funding)
                row: dict = {param_name: v}
                if secondary_param and s is not None:
                    row[secondary_param] = s
                row["metric"] = res.metrics.sharpe
                rows.append(row)
            except Exception:  # noqa: BLE001
                continue

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise ValueError("Жодної комбінації не відпрацювало")

    # гладкість уздовж головного параметра
    smooth = 1.0
    if secondary_param is None:
        g = grid.sort_values(param_name)
        metric = g["metric"].values
        if len(metric) > 2 and metric.std() > 0:
            diffs = np.abs(np.diff(metric))
            smooth = 1.0 - min(1.0, float(diffs.mean() / max(metric.std(), 1e-9)))

    best = grid.loc[grid["metric"].idxmax()]
    best_params = {param_name: best[param_name]}
    if secondary_param and secondary_param in grid.columns:
        best_params[secondary_param] = best[secondary_param]

    return SensitivityResult(grid=grid, best_params=best_params, smoothness=smooth)
