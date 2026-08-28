"""Портфельний бектест парних стратегій (XRP/BTC + BTC/ETH + ...).

Кожна пара запускається незалежно (run_pairs_backtest), прибутковості
комбінуються з вагами алокації:
    portfolio_ret = Σ w_i × pair_ret_i

Це показує диверсифікацію режимів: пари мають різну концентрацію прибутку
(XRP/BTC — рівномірна, BTC/ETH — Q4), тож портфель має бути гладкішим.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import compute_metrics
from scalper_hft.backtest.pairs import run_pairs_backtest
from scalper_hft.strategies.base import Strategy


@dataclass
class PairsPortfolioResult:
    equity: pd.Series
    pair_equities: dict[str, pd.Series]
    metrics: object
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Портфель пар: ret={m.total_return:+.3%} | SRh={m.sharpe_hourly:+.3f} | maxDD={m.max_drawdown:.2%}",
            "  Пари (вага → річний внесок):",
        ]
        for name, eq in self.pair_equities.items():
            r = eq.iloc[-1] / eq.iloc[0] - 1
            lines.append(f"    {name}: {r:+.3%}")
        lines.append(m.summary())
        return "\n".join(lines)


def run_pairs_portfolio(
    data: dict[str, pd.DataFrame],
    pair_configs: list[dict],
    weights: list[float] | None = None,
    position_pct: float = 0.3,
    cost: CostModel | None = None,
    initial_capital: float = 10_000.0,
    maker_execution: bool = True,
) -> PairsPortfolioResult:
    """Портфель пар на спільному часовому індексі.

    data: {symbol: klines_df}.
    pair_configs: [{leg1, leg2, strategy, funding1?, funding2?}, ...].
    weights: ваги алокації (за замовч. рівні).
    """
    cost = cost or CostModel()
    if weights is None:
        weights = [1.0 / len(pair_configs)] * len(pair_configs)
    if len(weights) != len(pair_configs):
        raise ValueError("weights і pair_configs мають бути однієї довжини")

    pair_equities: dict[str, pd.Series] = {}
    returns: list[pd.Series] = []

    for cfg, w in zip(pair_configs, weights):
        leg1, leg2 = cfg["leg1"], cfg["leg2"]
        strategy: Strategy = cfg["strategy"]
        f1 = cfg.get("funding1")
        f2 = cfg.get("funding2")
        res = run_pairs_backtest(
            data[leg1], data[leg2], strategy, f1, f2,
            position_pct=position_pct, cost=cost, maker_execution=maker_execution,
        )
        name = f"{leg1}/{leg2}"
        pair_equities[name] = res.equity
        returns.append((res.equity.pct_change().fillna(0.0)) * w)

    common_idx = returns[0].index
    for r in returns[1:]:
        common_idx = common_idx.intersection(r.index)
    port_ret = pd.Series(0.0, index=common_idx)
    for r in returns:
        port_ret = port_ret + r.reindex(common_idx).fillna(0.0)

    equity = (1.0 + port_ret).cumprod() * initial_capital
    metrics = compute_metrics(equity, exposure=0.0, turnover=0.0)
    return PairsPortfolioResult(
        equity=equity,
        pair_equities=pair_equities,
        metrics=metrics,
        details={"weights": weights, "position_pct": position_pct},
    )
