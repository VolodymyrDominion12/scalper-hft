"""ML-шар: класифікатори напрямку з walk-forward тренуванням.

Ідея (FreqAI-підхід; дослідження: LightGBM у трейдингу):
    - фічі: стандартні індикатори + мікроструктурні (CVD, buy_ratio);
    - таргет: напрямок наступного бару (клас +1/-1) з порогом t;
    - навчання: ковзне вікно (train) → прогноз на OOS без рефіту;
    - оцінка: accuracy, logloss, OOS Sharpe симуляції на прогнозах.

AFML-розширення (Marcos López de Prado):
    - labeling: Triple-Barrier Labeling (Ch.3)
    - sample_weights: Uniqueness + Time-Decay + Sequential Bootstrap (Ch.4)
    - frac_diff: Fractional Differentiation FFD (Ch.5)
"""

from scalper_hft.ml.bet_sizing import (
    calibrate_omega,
    discretize,
    limit_price,
    meta_size,
    prob_to_size,
    sigmoid_size,
    target_size,
)
from scalper_hft.ml.clustered_importance import (
    CfiResult,
    cluster_features,
    clustered_mda,
)
from scalper_hft.ml.feature_importance import (
    feature_importance_report,
    mda,
    mdi,
    pca_importance_corr,
    sfi,
)
from scalper_hft.ml.features import build_labeled_dataset
from scalper_hft.ml.frac_diff import add_frac_diff, find_min_d, frac_diff_features, frac_diff_ffd
from scalper_hft.ml.labeling import get_events, get_labels, label_from_ohlcv
from scalper_hft.ml.ood import apply_ood_veto, dissimilarity_index, fit_ood_stats, ood_mask
from scalper_hft.ml.sample_weights import compute_sample_weights, get_ind_matrix, seq_bootstrap
from scalper_hft.ml.trainer import (
    MlResult,
    cpcv_validate_returns,
    predict,
    train_from_ohlcv,
    train_walk_forward,
    train_walk_forward_meta,
)

__all__ = [
    "build_labeled_dataset",
    "train_walk_forward",
    "train_walk_forward_meta",
    "train_from_ohlcv",
    "predict",
    "MlResult",
    "cpcv_validate_returns",
    "clustered_mda",
    "cluster_features",
    "CfiResult",
    # AFML
    "label_from_ohlcv",
    "get_events",
    "get_labels",
    "compute_sample_weights",
    "seq_bootstrap",
    "get_ind_matrix",
    "frac_diff_ffd",
    "find_min_d",
    "add_frac_diff",
    "frac_diff_features",
    "fit_ood_stats",
    "dissimilarity_index",
    "ood_mask",
    "apply_ood_veto",
    # Bet sizing (AFML Ch.10.3, 10.6)
    "prob_to_size",
    "discretize",
    "meta_size",
    "sigmoid_size",
    "calibrate_omega",
    "target_size",
    "limit_price",
    # Feature importance (AFML Ch.8)
    "mdi",
    "mda",
    "sfi",
    "pca_importance_corr",
    "feature_importance_report",
]
