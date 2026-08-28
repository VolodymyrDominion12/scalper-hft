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

from scalper_hft.ml.features import build_labeled_dataset
from scalper_hft.ml.trainer import train_walk_forward, predict
from scalper_hft.ml.labeling import label_from_ohlcv, get_events, get_labels
from scalper_hft.ml.sample_weights import compute_sample_weights, seq_bootstrap, get_ind_matrix
from scalper_hft.ml.frac_diff import frac_diff_ffd, find_min_d, add_frac_diff, frac_diff_features

__all__ = [
    "build_labeled_dataset",
    "train_walk_forward",
    "predict",
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
]
