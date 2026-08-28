"""Тренування ML-класифікатора з walk-forward (без lookahead і без leakage).

Схема:
    для кожного ковзного вікна:
        model.fit(X_train, y_train)          # лише минуле
        preds = model.predict_proba(X_test)  # OOS прогноз
    → конкатенація OOS-прогнозів = чесна оцінка генералізації.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
    _HAS_LGBM = False


@dataclass
class MlResult:
    oos_accuracy: float
    oos_logloss: float
    oos_sharpe: float  # Sharpe симуляції: позиція = знак прогнозу
    predictions: pd.Series
    n_train: int
    n_test: int
    n_windows: int
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ML walk-forward: {self.n_windows} вікон; train {self.n_train}, test {self.n_test}\n"
            f"OOS accuracy: {self.oos_accuracy:.3f} | logloss: {self.oos_logloss:.4f} | OOS Sharpe: {self.oos_sharpe:.3f}"
        )


def _simulate_sharpe(preds: pd.Series, close: pd.Series) -> float:
    """Sharpe стратегії 'позиція = знак прогнозу' на OOS-даних."""
    pos = np.sign(preds).astype(float)
    ret = close.pct_change().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret
    if strat.std() == 0 or len(strat) < 2:
        return 0.0
    return float(strat.mean() / strat.std(ddof=0) * np.sqrt(len(strat)))


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int = 2000,
    test_size: int = 500,
    params: dict | None = None,
    close: pd.Series | None = None,
) -> MlResult:
    """Walk-forward навчання LightGBM.

    X/y: повернуті build_labeled_dataset (вирівняні індекси).
    close: ряд цін для симуляції Sharpe (індекс як у X).
    """
    if not _HAS_LGBM:
        raise ImportError("Встановіть lightgbm: uv add --optional ml lightgbm scikit-learn")

    params = params or {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
    }
    oos_preds: list[pd.Series] = []
    oos_proba: list[np.ndarray] = []
    windows = 0
    start = 0
    while start + train_size + test_size <= len(X):
        X_tr, y_tr = X.iloc[start : start + train_size], y.iloc[start : start + train_size]
        X_te = X.iloc[start + train_size : start + train_size + test_size]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        classes = list(model.classes_)
        # ймовірність класу +1
        if 1 in classes:
            p_pos = proba[:, classes.index(1)]
        else:
            p_pos = 0.5 - np.zeros(len(X_te))
        pred = pd.Series(np.where(p_pos >= 0.5, 1, -1), index=X_te.index)
        oos_preds.append(pred)
        oos_proba.append(np.asarray(p_pos, dtype=float))
        windows += 1
        start += test_size

    if not oos_preds:
        raise ValueError("Не вийшло жодного вікна — збільшіть train_size/test_size")

    preds = pd.concat(oos_preds).sort_index()
    y_oos = y.reindex(preds.index)
    acc = float((preds == y_oos).mean())
    # logloss: середнє -log(p(істинний клас))
    eps = 1e-9
    p_pos_all = np.concatenate(oos_proba) if oos_proba else np.zeros(len(preds))
    p_true = np.where(y_oos.values == 1, p_pos_all, 1 - p_pos_all)
    logloss = float(-np.mean(np.log(np.clip(p_true, eps, 1 - eps))))

    close_aligned = close.reindex(preds.index) if close is not None else None
    sharpe = _simulate_sharpe(preds, close_aligned) if close_aligned is not None else 0.0

    return MlResult(
        oos_accuracy=acc,
        oos_logloss=logloss,
        oos_sharpe=sharpe,
        predictions=preds,
        n_train=train_size,
        n_test=len(preds),
        n_windows=windows,
    )


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    """Прогноз напрямку (+1/-1) для нових даних."""
    proba = model.predict_proba(X)  # type: ignore[attr-defined]
    classes = list(model.classes_)  # type: ignore[attr-defined]
    p_pos = proba[:, classes.index(1)] if 1 in classes else 0.5 - np.zeros(len(X))
    return np.where(p_pos >= 0.5, 1, -1)
