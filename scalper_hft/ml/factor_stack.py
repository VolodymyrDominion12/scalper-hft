"""Factor stacking з purged CV (Phase 2D).

Мета-модель над (strategy signals + regime features): замість простого
голосування/середнього — навчена модель, що знає, якому сигналу коли вірити.
Навчання з purged K-fold CV (AFML Ch.7): між train/test блоками витаваються
`purge_bars` + `embargo_bars`, щоб лейбли/позиції не перетікали між блоками.

Без lookahead: модель на барі t навчається лише на блоках, що закінчуються
строго до t (з purge/embargo). OOS-прогноз на тестовому блоці.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FactorStackResult:
    """Результат factor stacking: OOS-прогнози + ваги фіч."""

    oos_signal: pd.Series  # OOS-стекed сигнал у [-1, 1] на кожному барі
    feature_importance: pd.Series  # |coef|/importance per feature
    n_folds: int


def _purged_folds(n: int, k: int, purge_bars: int, embargo_bars: int) -> list[tuple[int, int, int, int]]:
    """K-fold split з purge+embargo: (train_start, train_end, test_start, test_end).

    Тестові блоки — послідовні неперетинні шматки. Train = усе до test_start
    мінус embargo; після test_end + purge йде наступний train.
    """
    fold_size = n // k
    folds: list[tuple[int, int, int, int]] = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        train_end = max(test_start - embargo_bars, 0)
        train_start = 0
        folds.append((train_start, train_end, test_start, test_end))
    return folds


def factor_stack(
    feature_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    *,
    n_folds: int = 5,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    clip: float = 1.0,
) -> FactorStackResult:
    """Навчити лінійну мета-модель над фічами з purged K-fold CV (2D).

    Args:
        feature_matrix: (T × F) фічі (strategy signals + regime features),
            індексовані часом.
        forward_returns: (T,) дохідність наступного бару (label).
        n_folds: число K-fold блоків.
        purge_bars/embargo_bars: прогін/ембарго між train/test (AFML Ch.7).
        clip: кліп OOS-сигналу у [-clip, clip].

    Returns:
        FactorStackResult: OOS-стекed сигнал на кожному барі (out-of-fold
        прогнози) + importance фіч (|стандартизований coef|).

    Без lookahead: прогноз на тестовому блоці робиться моделлю, навченою на
    train-блоці, що закінчується строго до test (з embargo).
    """
    X = feature_matrix.copy()
    y = forward_returns.reindex(X.index).fillna(0.0)
    X = X.fillna(0.0)
    n = len(X)
    if n < n_folds * 2:
        # замало даних → рівний OOS-сигнал 0
        return FactorStackResult(
            oos_signal=pd.Series(0.0, index=X.index),
            feature_importance=pd.Series(0.0, index=X.columns),
            n_folds=0,
        )

    folds = _purged_folds(n, n_folds, purge_bars, embargo_bars)
    oos = np.zeros(n, dtype=float)
    coef_acc = np.zeros(X.shape[1], dtype=float)
    n_used = 0

    for train_start, train_end, test_start, test_end in folds:
        if train_end <= train_start or test_end <= test_start:
            continue
        Xtr = X.iloc[train_start:train_end].values
        ytr = y.iloc[train_start:train_end].values
        Xte = X.iloc[test_start:test_end].values
        if len(Xtr) < 2 or len(Xte) == 0:
            continue
        # Ridge-регресія (замкнене рішення): β = (XᵀX + λI)⁻¹Xᵀy
        lam = 1.0
        XtX = Xtr.T @ Xtr + lam * np.eye(X.shape[1])
        try:
            beta = np.linalg.solve(XtX, Xtr.T @ ytr)
        except np.linalg.LinAlgError:
            continue
        pred = Xte @ beta
        # сигнал = sign(pred) × min(|pred|/scale, 1); scale = std(train target)
        scale = float(np.std(ytr)) or 1.0
        sig = np.sign(pred) * np.minimum(np.abs(pred) / scale, 1.0)
        oos[test_start:test_end] = sig
        coef_acc += np.abs(beta)
        n_used += 1

    importance = pd.Series(coef_acc / max(n_used, 1), index=X.columns)
    oos_series = pd.Series(oos, index=X.index).clip(-clip, clip).fillna(0.0)
    return FactorStackResult(
        oos_signal=oos_series,
        feature_importance=importance.sort_values(ascending=False),
        n_folds=n_used,
    )


__all__ = ["FactorStackResult", "factor_stack"]
