"""Абстракція ML-бэкенду: LightGBM / XGBoost (дослідження §2.1).

Дослідження §2.1: XGBoost і LightGBM — обидва еталони градієнтного бустингу.
Bake-off — не «заміна», а перевірка чи XGBoost дає ΔSharpe > 0 на тій самій
OOS-вибірці (Narang гл.9 value-added).

Цей модуль надає єдиний інтерфейс `ModelBackend.fit/predict_proba` поверх
`lightgbm.LGBMClassifier` та `xgboost.XGBClassifier`. Ваги зразків (AFML
uniqueness+decay) передаються обом бекендам.

XGBoost — опційна залежність (`[ml]` extras). Якщо не встановлено —
`make_backend("xgboost")` піднімає `ImportError` (викликач ловить і fallback).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False


class ModelBackend(Protocol):
    """Єдиний інтерфейс для LightGBM/XGBoost класифікаторів."""

    classes_: np.ndarray

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> None: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...
    def feature_importances_(self) -> np.ndarray: ...


def make_backend(
    name: str = "lightgbm",
    params: dict | None = None,
) -> ModelBackend:
    """Створити ML-бекенд за ім'ям.

    Args:
        name: "lightgbm" (дефолт) або "xgboost".
        params: гіперпараметри (передаються конструктору).

    Returns:
        ModelBackend з єдиним інтерфейсом fit/predict_proba.

    Raises:
        ImportError: якщо бекенд не встановлено.
        ValueError: якщо ім'я невідоме.
    """
    backend = str(name).lower().strip()
    params = dict(params or {})
    if backend == "lightgbm":
        if not _HAS_LGBM:
            raise ImportError("lightgbm не встановлено (uv pip install -e '.[ml]')")
        return LGBMClassifier(**params)
    if backend == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost не встановлено (uv pip install -e '.[ml]')")
        # XGBoost вимагає лейбли 0/1 (не -1/+1); конвертацію робить trainer.
        if "verbosity" in params and params["verbosity"] < 0:
            params["verbosity"] = 0
        return XGBClassifier(**params)
    raise ValueError(f"Невідомий ML-бэкенд: '{name}'. Доступні: lightgbm, xgboost")


def is_available(name: str) -> bool:
    """Чи доступний бекенд (встановлено)."""
    backend = str(name).lower().strip()
    if backend == "lightgbm":
        return _HAS_LGBM
    if backend == "xgboost":
        return _HAS_XGB
    return False


__all__ = ["ModelBackend", "make_backend", "is_available"]
