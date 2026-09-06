"""Валідація стратегій: захист від перенавчання (книга, гл. 9 — Testing; гл. 11 — Data Mining).

Набір інструментів:
    - walk_forward   — ковзні IS/OOS вікна (реальна перевірка генералізації);
    - cv             — time-series K-fold з purging/embargo (López de Prado);
    - deflated_sharpe— Deflated Sharpe Ratio (Bailey & López de Prado) та PBO;
    - sensitivity    — чутливість до параметрів (плато vs ізольований пік);
    - optimize       — Optuna-пошук параметрів з CV-цільовою функцією.
"""

from scalper_hft.validation.capacity import capacity_curve, capacity_report, saturation_scale
from scalper_hft.validation.cell_audit import (
    CellAudit,
    audit_cell,
    cell_verdict,
    default_train_test,
    min_trades_for,
    resolve_wf_windows,
)
from scalper_hft.validation.cohort import cohort_decay, cohort_metrics, cohort_report
from scalper_hft.validation.cscv import CscvResult, combinatorial_splits, pbo_cscv, variant_returns
from scalper_hft.validation.cv import PurgedKFold, purged_kfold_indices, time_series_split
from scalper_hft.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    estimate_n_trials,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from scalper_hft.validation.experiments import (
    Verdict,
    compare_to_baseline,
    qualifies_for_oos,
)
from scalper_hft.validation.forensics import analyze_trades, compute_mfe_mae
from scalper_hft.validation.lift import decile_lift, feature_lift_report, lift_summary
from scalper_hft.validation.optimize import optimize_params
from scalper_hft.validation.sensitivity import parameter_sensitivity
from scalper_hft.validation.stress import SCENARIOS, apply_stress, stress_report, stress_test
from scalper_hft.validation.survival import (
    kaplan_meier,
    median_survival_time,
    survival_by_feature,
    trade_durations,
)
from scalper_hft.validation.walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "CellAudit",
    "audit_cell",
    "cell_verdict",
    "default_train_test",
    "min_trades_for",
    "resolve_wf_windows",
    "CscvResult",
    "combinatorial_splits",
    "pbo_cscv",
    "variant_returns",
    "purged_kfold_indices",
    "time_series_split",
    "PurgedKFold",
    "cohort_metrics",
    "cohort_decay",
    "cohort_report",
    "deflated_sharpe_ratio",
    "estimate_n_trials",
    "probability_of_backtest_overfitting",
    "probabilistic_sharpe_ratio",
    "Verdict",
    "compare_to_baseline",
    "qualifies_for_oos",
    "analyze_trades",
    "compute_mfe_mae",
    "decile_lift",
    "feature_lift_report",
    "lift_summary",
    "optimize_params",
    "parameter_sensitivity",
    "SCENARIOS",
    "apply_stress",
    "stress_test",
    "stress_report",
    "capacity_curve",
    "capacity_report",
    "saturation_scale",
    "kaplan_meier",
    "median_survival_time",
    "survival_by_feature",
    "trade_durations",
    "WalkForwardResult",
    "run_walk_forward",
]
