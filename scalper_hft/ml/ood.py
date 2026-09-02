"""OOD / Dissimilarity Index veto для ML-сигналів (ідея FreqAI DI_threshold).

Блокер, не альфа: якщо рядок фіч далеко від train-розподілу — сигнал = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ood_stats(x_train: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Середнє і std по колонках train (std кліпається знизу, щоб не ділити на 0)."""
    arr = np.asarray(x_train, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def dissimilarity_index(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """RMS стандартизованої відстані (DI). inf, якщо розмірності не збігаються."""
    row = np.asarray(x, dtype=float).reshape(-1)
    if row.size == 0 or mean.size == 0 or row.size != mean.size:
        return float("inf")
    z = (row - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return float(np.sqrt(np.mean(z * z)))


def ood_mask(
    x: pd.DataFrame | np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """True = in-distribution (можна торгувати). threshold <= 0 вимикає фільтр."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    if threshold <= 0 or mean.size == 0:
        return np.ones(n, dtype=bool)
    di = np.array([dissimilarity_index(arr[i], mean, std) for i in range(n)], dtype=float)
    return di <= float(threshold)


def apply_ood_veto(signals: pd.Series, in_distribution: pd.Series | np.ndarray) -> pd.Series:
    """Зануляє сигнали на OOD-барах. Close/flat (0) не змінюється додатково."""
    mask = pd.Series(in_distribution, index=signals.index)
    return signals.where(mask.reindex(signals.index, fill_value=True).astype(bool), other=0.0)
