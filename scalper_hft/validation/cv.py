"""Time-series CV з purging та embargo (López de Prado, "Advances in Financial ML" Ch.7).

Проблема звичайного K-fold для часових рядів: сусідні спостереження
автокорельовані → витік інформації між train/test. Рішення:
    - purge: викинути з train спостереження, що перекриваються з test;
    - embargo: викинути додатковий буфер після test (для ковзних фіч).

Ключове розширення (AFML):
    PurgedKFold враховує реальні spans лейблів [t0, t1] щоб визначити
    які train-зразки містять lookahead відносно test-барів.
    Це критично для triple-barrier labels де t1 > t0.

Також надаємо простий expanding-window split для швидких перевірок.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd


def time_series_split(
    n: int, n_splits: int = 5, embargo: int = 0, gap: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window train/test split (sklearn-подібний, з gap та embargo)."""
    indices = np.arange(n)
    test_size = n // (n_splits + 1)
    for i in range(n_splits):
        test_start = (i + 1) * test_size
        train_end = test_start - gap - embargo
        if train_end <= 0:
            continue
        yield indices[:train_end], indices[test_start : test_start + test_size]


def purged_kfold_indices(
    n: int,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold з purging (видалення перекриття train/test) та embargo.

    purge: скільки рядків на межі train/test викинути з train.
    embargo: додатковий буфер після test (для ковзних індикаторів).
    """
    indices = np.arange(n)
    fold_size = n // n_splits
    for k in range(n_splits):
        test_start = k * fold_size
        test_end = min(test_start + fold_size, n)
        test = indices[test_start:test_end]
        train = np.concatenate([indices[: max(0, test_start - purge - embargo)], indices[test_end + embargo :]])
        if len(train) == 0:
            continue
        yield train, test


def split_by_index(
    df: pd.DataFrame, n_splits: int = 5, purge: int = 0, embargo: int = 0
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Те саме, але повертає DataFrame-зрізи."""
    for train_idx, test_idx in purged_kfold_indices(len(df), n_splits, purge, embargo):
        yield df.iloc[train_idx], df.iloc[test_idx]


# ── PurgedKFold (AFML Ch.7) ───────────────────────────────────────────────────


class PurgedKFold:
    """Purged K-Fold Cross-Validator з урахуванням spanів triple-barrier лейблів.

    На відміну від purged_kfold_indices(), ця реалізація знає про
    реальні проміжки [t0, t1] подій і видаляє з train усі зразки,
    чиї t1 потрапляють в test-період (а не лише t0, як у звичайному purge).

    Args:
        n_splits: кількість фолдів.
        embargo_pct: частка даних (0..1) додатково видалена після test.

    Usage:
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in pkf.split(X, t1=events['t1']):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            ...
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        t1: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Генерує (train_indices, test_indices) з purging по t1.

        Args:
            X: матриця фіч (DatetimeIndex).
            t1: Series t0 → t1 (з events[\'t1\']).
            sample_weight: ігнорується (для sklearn-сумісності).
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X повинен мати DatetimeIndex")

        indices = np.arange(len(X))
        embargo = int(len(X) * self.embargo_pct)
        fold_size = len(X) // self.n_splits

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, len(X))
            test_idx = indices[test_start:test_end]

            # Часові межі test-фолду
            t_test_start = X.index[test_start]

            # Purging: видаляємо train-зразки чиї t1 потрапляє в test-період
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[test_start:test_end] = False  # сам test

            # embargo після test
            embargo_end = min(test_end + embargo, len(X))
            train_mask[test_end:embargo_end] = False

            # purge: для кожного train-зразка перевірити чи t1 у test-регіоні
            # fillna: якщо t1 відсутня для зразку → вважаємо t1 = t0 (лейбл миттєвий)
            t1_aligned = t1.reindex(X.index)
            # заповнюємо NaN власним індексом (t1 = t0 означає миттєвий лейбл)
            idx_series = pd.Series(X.index, index=X.index)
            t1_aligned = t1_aligned.fillna(idx_series)
            for i in indices[:test_start]:  # ліва частина train
                if train_mask[i] and t1_aligned.iloc[i] > t_test_start:
                    train_mask[i] = False

            train_idx = indices[train_mask]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def cross_val_score(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series,
        sample_weight: pd.Series | None = None,
        scoring: str = "accuracy",
    ) -> np.ndarray:
        """Cross-validation score з purging.

        Args:
            estimator: sklearn-сумісний класифікатор (fit/predict_proba).
            X: фічі.
            y: лейбли.
            t1: Series t0 → t1.
            sample_weight: ваги (передаються в fit).
            scoring: 'accuracy' | 'neg_log_loss'.

        Returns:
            np.ndarray з score по кожному фолду.
        """
        scores = []
        for train_idx, test_idx in self.split(X, t1=t1, sample_weight=sample_weight):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]

            fit_kwargs: dict = {}
            if sample_weight is not None:
                w_tr = sample_weight.iloc[train_idx].values
                s = w_tr.sum()
                fit_kwargs["sample_weight"] = w_tr / s if s > 0 else w_tr

            estimator.fit(X_tr, y_tr, **fit_kwargs)  # type: ignore[union-attr]

            if scoring == "accuracy":
                preds = estimator.predict(X_te)  # type: ignore[union-attr]
                scores.append(float((preds == y_te.values).mean()))
            elif scoring == "neg_log_loss":
                proba = estimator.predict_proba(X_te)  # type: ignore[union-attr]
                classes = list(estimator.classes_)  # type: ignore[union-attr]
                p_pos = proba[:, classes.index(1)] if 1 in classes else np.full(len(X_te), 0.5)
                eps = 1e-9
                p_true = np.where(y_te.values == 1, p_pos, 1 - p_pos)
                ll = -np.mean(np.log(np.clip(p_true, eps, 1 - eps)))
                scores.append(-ll)
            else:
                raise ValueError(f"scoring={scoring!r} не підтримується")

        return np.array(scores)
