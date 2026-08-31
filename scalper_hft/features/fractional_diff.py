"""Fractional Differentiation (AFML Chapter 5).

Реалізація фракційного диференціювання для стаціонаризації фіч
з максимальним збереженням пам'яті часового ряду.
Делегує до розширеного модуля scalper_hft.ml.frac_diff.
"""

from __future__ import annotations

import numpy as np

from scalper_hft.ml.frac_diff import (
    _get_weights_ffd,
    add_frac_diff,
    find_min_d,
    frac_diff_expanding,
    frac_diff_ffd,
)


def get_weights(d: float, size: int) -> np.ndarray:
    """Отримує ваги для фракційного диференціювання."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w)[::-1].reshape(-1, 1)


def get_weights_ffd(d: float, thres: float = 1e-4) -> np.ndarray:
    """Отримує ваги для Fixed-Width Window Fractional Differentiation (FFD)."""
    return _get_weights_ffd(d, threshold=thres).reshape(-1, 1)


__all__ = [
    "add_frac_diff",
    "find_min_d",
    "frac_diff_expanding",
    "frac_diff_ffd",
    "get_weights",
    "get_weights_ffd",
]

