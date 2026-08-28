"""ML-шар: класифікатори напрямку з walk-forward тренуванням.

Ідея (FreqAI-підхід; дослідження: LightGBM у трейдингу):
    - фічі: стандартні індикатори + мікроструктурні (CVD, buy_ratio);
    - таргет: напрямок наступного бару (клас +1/-1) з порогом t;
    - навчання: ковзне вікно (train) → прогноз на OOS без рефіту;
    - оцінка: accuracy, logloss, OOS Sharpe симуляції на прогнозах.
"""

from scalper_hft.ml.features import build_labeled_dataset
from scalper_hft.ml.trainer import train_walk_forward, predict

__all__ = ["build_labeled_dataset", "train_walk_forward", "predict"]
