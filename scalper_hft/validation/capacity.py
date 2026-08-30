"""Capacity-тест стратегії: як Sharpe деградує з ростом капіталу.

Джерело: Predictive Marketing (Artun & Levin), Ch.4 — «share of wallet»:
аналогія — скільки капіталу сигнал може поглинути, поки edge не зруйнується
(масштабування позицій збільшує market impact → витрати).

Метод: бектест стратегії з масштабом позицій ×1, ×2, … ×K і побудова
кривої Sharpe(scale). Точка насичення — де Sharpe падає нижче порогу
(наприклад, нижче 50% від пікового або нижче 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel, estimate_impact_k_from_bars


def capacity_curve(
    df: pd.DataFrame,
    strategy,
    scales: list[float] | None = None,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.01,
    is_maker: bool = False,
) -> pd.DataFrame:
    """Крива Sharpe(scale) для стратегії.

    На кожному масштабі s позиція множиться на s, а market impact на бари
    додається за Square-Root Law (I = k·σ̂·√(Q/ADV)), калібруючи k з даних.

    Returns:
        DataFrame: scale, total_return, sharpe, max_drawdown, impact_bps.
    """
    from scalper_hft.backtest.engine import run_backtest

    cost = cost or CostModel()
    scales = scales or [1.0, 2.0, 5.0, 10.0, 20.0]
    k = estimate_impact_k_from_bars(df)
    sigma = float(df["close"].pct_change().rolling(288).std().median()) or 0.01

    rows: dict[str, list] = {"scale": [], "total_return": [], "sharpe": [], "max_drawdown": [], "impact_bps": []}
    for s in scales:
        cost_s = CostModel(
            maker_fee=cost.maker_fee,
            taker_fee=cost.taker_fee,
            slippage_frac=cost.slippage_frac,
            impact_k=cost.impact_k,
            vol_ref=cost.vol_ref,
            vol_exp=cost.vol_exp,
        )
        # impact зростає з розміром: I = k·σ·√(s·Q/ADV), Q/ADV нормуємо на scale 1
        share = min(s * position_pct, 1.0)
        impact = k * sigma * np.sqrt(share)
        # impact сплачується на turnover — наближено: на кожну зміну позиції
        cost_s = CostModel(
            maker_fee=cost.maker_fee,
            taker_fee=cost.taker_fee,
            slippage_frac=cost.slippage_frac,
            impact_frac=float(impact) if s > 1.0 else 0.0,
            impact_k=cost.impact_k,
            vol_ref=cost.vol_ref,
            vol_exp=cost.vol_exp,
        )
        res = run_backtest(
            df,
            strategy,
            cost=cost_s,
            trades=trades,
            funding=funding,
            position_pct=position_pct * s,
            is_maker=is_maker,
        )
        rows["scale"].append(s)
        rows["total_return"].append(res.metrics.total_return)
        rows["sharpe"].append(res.metrics.sharpe)
        rows["max_drawdown"].append(res.metrics.max_drawdown)
        rows["impact_bps"].append(impact * 1e4)
    return pd.DataFrame(rows)


def saturation_scale(
    curve: pd.DataFrame,
    metric: str = "sharpe",
    drop_threshold: float = 0.5,
) -> float:
    """Найбільший масштаб, де метрика ≥ (1−drop_threshold)·пік.

    Returns:
        scale насичення (перший масштаб, де метрика падає нижче порогу;
        якщо не падає — останній масштаб у кривій).
    """
    if curve is None or curve.empty or metric not in curve.columns:
        return 1.0
    vals = curve[metric].astype(float)
    peak = float(vals.max())
    if peak <= 0:
        return 1.0
    threshold = peak * (1.0 - drop_threshold)
    below = curve.index[vals < threshold]
    if len(below) == 0:
        return float(curve["scale"].iloc[-1])
    return float(curve["scale"].iloc[below[0] - 1]) if below[0] > 0 else float(curve["scale"].iloc[0])


def capacity_report(df, strategy, **kwargs) -> str:
    """Людиночитабельний звіт capacity-тесту."""
    curve = capacity_curve(df, strategy, **kwargs)
    sat = saturation_scale(curve)
    lines = [
        "Capacity-тест (Sharpe при масштабуванні позицій):",
        "",
        curve.round(4).to_string(index=False),
        "",
        f"Точка насичення: ×{sat:g}",
        f"→ edge зберігається до ×{sat:.0f} капіталу",
    ]
    return "\n".join(lines)


__all__ = ["capacity_curve", "saturation_scale", "capacity_report"]
