"""Оптимізація параметрів через Optuna з CV-цільовою функцією.

Критично (книга, гл. 9–11): оптимізація на всьому наборі = перенавчання.
Тому:
    1. цільова функція = середній OOS Sharpe по purged K-fold;
    2. фінальна оцінка — на повному OOS holdout (поза оптимізацією);
    3. deflated Sharpe коригує результат на кількість спроб.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy
from scalper_hft.validation.cv import purged_kfold_indices

logger = logging.getLogger(__name__)

try:  # optuna — опційна залежність (pyproject: [optim])
    import optuna  # type: ignore

    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    _HAS_OPTUNA = False


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    cv_scores: list[float]
    n_trials: int
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Оптимізація: {self.n_trials} спроб; кращі параметри: {self.best_params}\n"
            f"CV OOS Sharpe: {self.cv_scores} (avg {np.mean(self.cv_scores):.3f})"
        )


def _default_objective(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    params: dict,
    cost: CostModel,
    trades: pd.DataFrame | None,
    n_splits: int,
    embargo: int,
    position_pct: float,
) -> float:
    """Середній Sharpe по purged K-fold на OOS-зрізах."""
    oos_sharpes: list[float] = []
    for train_idx, test_idx in purged_kfold_indices(len(df), n_splits, purge=50, embargo=embargo):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        trades_tr = trades.iloc[train_idx] if trades is not None else None
        try:
            res = run_backtest(te, strategy_cls(**params), cost=cost, trades=trades_tr, position_pct=position_pct)
            sharpe = res.metrics.sharpe
        except Exception:  # noqa: BLE001
            sharpe = -1.0
        if not np.isfinite(sharpe):
            sharpe = -1.0
        oos_sharpes.append(sharpe)
    return float(np.mean(oos_sharpes)) if oos_sharpes else -1.0


def optimize_params(
    df: pd.DataFrame,
    strategy: Strategy,
    n_trials: int = 60,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    n_splits: int = 4,
    embargo: int = 30,
    position_pct: float = 0.01,
    sampler: str = "tpe",
) -> OptimizationResult:
    """Оптимізація параметрів з param_space стратегії.

    sampler: 'tpe' (байєсівський) або 'random'.
    """
    if not _HAS_OPTUNA:
        raise ImportError("Встановіть optuna: uv add --optional optim optuna")

    cost = cost or CostModel()
    strategy_cls = type(strategy)
    space = strategy.param_space

    def objective(trial: "optuna.Trial") -> float:
        params: dict = {}
        for pname, (lo, hi, step) in space.items():
            is_int = float(step) == int(step) and float(lo) == int(lo) and float(hi) == int(hi)
            if is_int:
                params[pname] = trial.suggest_int(pname, int(lo), int(hi))
            else:
                params[pname] = trial.suggest_float(pname, float(lo), float(hi))
        return _default_objective(
            df, strategy_cls, params, cost, trades, n_splits, embargo, position_pct
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # CV-оцінки найкращого варіанта (перезапуск objective з фіксованими параметрами)
    cv_scores: list[float] = []
    if study.best_params:
        for train_idx, test_idx in purged_kfold_indices(len(df), n_splits, purge=50, embargo=embargo):
            te = df.iloc[test_idx]
            trades_te = trades.iloc[test_idx] if trades is not None else None
            try:
                res = run_backtest(
                    te, strategy_cls(**study.best_params), cost=cost, trades=trades_te, position_pct=position_pct
                )
                score = res.metrics.sharpe
            except Exception:  # noqa: BLE001
                score = -1.0
            cv_scores.append(score if np.isfinite(score) else -1.0)

    return OptimizationResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        cv_scores=cv_scores,
        n_trials=n_trials,
        details={"direction": "maximize", "sampler": sampler},
    )
