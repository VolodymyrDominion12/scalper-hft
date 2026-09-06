"""Квінтильний / monotonicity тест сили спреду (Narang гл. 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QuintileResult:
    means: pd.Series
    spearman: float
    monotonic: bool

    def summary(self) -> str:
        flag = "монотонно" if self.monotonic else "не монотонно"
        return f"Quintile z: Spearman ρ={self.spearman:+.3f} ({flag})\n{self.means.to_string()}"


def quintile_spread_study(
    z: pd.Series,
    forward: pd.Series,
    n_bins: int = 5,
) -> QuintileResult:
    """Середній forward-return (напр. −Δspread) по квінтилях |z| або z."""
    aligned = pd.concat({"z": z, "fwd": forward}, axis=1).dropna()
    if len(aligned) < n_bins * 5:
        raise ValueError("Замало точок для квінтилів")
    try:
        bins = pd.qcut(aligned["z"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(aligned["z"], n_bins, labels=False)
    means = aligned["fwd"].groupby(bins).mean()
    means.index = [f"Q{int(i) + 1}" for i in means.index]
    ranks = np.arange(len(means))
    means_arr = means.to_numpy(dtype=float)
    if len(means) >= 2:
        spearman = float(pd.Series(ranks).corr(pd.Series(means_arr), method="spearman"))
    else:
        spearman = 0.0
    diffs = np.diff(means_arr)
    monotonic = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
    return QuintileResult(means=means, spearman=spearman, monotonic=monotonic)
