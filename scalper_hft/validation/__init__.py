"""Валідація стратегій: захист від перенавчання (книга, гл. 9 — Testing; гл. 11 — Data Mining).

Набір інструментів:
    - walk_forward   — ковзні IS/OOS вікна (реальна перевірка генералізації);
    - cv             — time-series K-fold з purging/embargo (López de Prado);
    - deflated_sharpe— Deflated Sharpe Ratio (Bailey & López de Prado) та PBO;
    - sensitivity    — чутливість до параметрів (плато vs ізольований пік);
    - optimize       — Optuna-пошук параметрів з CV-цільовою функцією.
"""

from scalper_hft.validation.cv import purged_kfold_indices, time_series_split
from scalper_hft.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    estimate_n_trials,
    probability_of_backtest_overfitting,
)
from scalper_hft.validation.optimize import optimize_params
from scalper_hft.validation.sensitivity import parameter_sensitivity
from scalper_hft.validation.walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "purged_kfold_indices",
    "time_series_split",
    "deflated_sharpe_ratio",
    "estimate_n_trials",
    "probability_of_backtest_overfitting",
    "optimize_params",
    "parameter_sensitivity",
    "WalkForwardResult",
    "run_walk_forward",
]
