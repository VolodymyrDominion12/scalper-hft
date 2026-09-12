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
    monotonic = bool(np.all(diffs >= -1e-12) or np.all(diffs <= -1e-12))
    return QuintileResult(means=means, spearman=spearman, monotonic=monotonic)


@dataclass
class DiscreteSignalResult:
    """Інформаційний тест для дискретного сигналу {-1, 0, +1} (не пари).

    Квінтильний тест на такому сигналі вироджується: `qcut` по 3 значеннях дає
    2–3 біни, а Spearman двох-трьох точок — це ±1 незалежно від того, чи сигнал
    взагалі щось передбачає (перевірено: випадковий forward-return з дискретним
    z дає ρ=+1.0, monotonic=True). Тому для directional-стратегій рахуємо те, що
    справді має сенс: чи середній forward-return монотонний за знаком сигналу і
    чи спред крайніх кошиків статистично відрізняється від нуля (Welch t).
    """

    buckets: pd.Series
    counts: pd.Series
    spearman: float
    monotonic: bool
    spread: float
    t_stat: float

    @property
    def pass_(self) -> bool:
        """Монотонність + значущий спред крайніх кошиків (|t| > 2)."""
        return bool(self.monotonic and abs(self.t_stat) > 2.0)

    def summary(self) -> str:
        return (
            f"Дискретний сигнал: спред країв={self.spread:+.5f}, t={self.t_stat:+.2f}, "
            f"монотонно={self.monotonic}, ρ={self.spearman:+.3f}\n"
            f"{self.buckets.to_string()}\n{self.counts.to_string()}"
        )


def discrete_signal_study(sig: pd.Series, forward: pd.Series) -> DiscreteSignalResult:
    """Середній forward-return по кошиках знака сигналу + Welch t для країв.

    Порожні/одноточкові кошики відкидаються. Кидає ValueError, якщо лишається
    менше двох кошиків (тест незастосовний — це не FAIL, а N/A).
    """
    aligned = pd.concat({"s": sig, "f": forward}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("немає вирівняних точок сигнал/forward")
    aligned["s"] = aligned["s"].round(6)
    grouped = aligned.groupby("s")["f"]
    means = grouped.mean()
    counts = grouped.size()
    keep = counts[counts >= 20].index
    means, counts = means.loc[keep], counts.loc[keep]
    if len(means) < 2:
        raise ValueError("менше двох непорожніх кошиків сигналу")
    means = means.sort_index()
    ranks = np.arange(len(means))
    means_arr = means.to_numpy(dtype=float)
    spearman = float(pd.Series(ranks).corr(pd.Series(means_arr), method="spearman"))
    diffs = np.diff(means_arr)
    monotonic = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))

    lo_val, hi_val = means.index[0], means.index[-1]
    lo = aligned.loc[aligned["s"] == lo_val, "f"].to_numpy(dtype=float)
    hi = aligned.loc[aligned["s"] == hi_val, "f"].to_numpy(dtype=float)
    spread = float(hi.mean() - lo.mean())
    denom = float(np.sqrt(hi.var(ddof=1) / max(len(hi), 1) + lo.var(ddof=1) / max(len(lo), 1)))
    t_stat = float(spread / denom) if denom > 0 else 0.0
    return DiscreteSignalResult(
        buckets=means,
        counts=counts,
        spearman=spearman,
        monotonic=monotonic,
        spread=spread,
        t_stat=t_stat,
    )
