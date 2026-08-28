"""Time-series CV з purging та embargo (López de Prado, "Advances in Financial ML").

Проблема звичайного K-fold для часових рядів: сусідні спостереження
автокорельовані → витік інформації між train/test. Рішення:
    - purge: викинути з train спостереження, що перекриваються з test;
    - embargo: викинути додатковий буфер після test (для ковзних фіч).

Також надаємо простий expanding-window split для швидких перевірок.
"""

from __future__ import annotations

from typing import Iterator

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
