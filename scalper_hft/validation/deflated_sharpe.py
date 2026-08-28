"""Deflated Sharpe Ratio та PBO (Bailey & López de Prado, "The Deflated Sharpe Ratio").

Проблема: якщо перебрати N варіантів стратегії, найкращий Sharpe буде
завищений випадково. Deflated Sharpe коригує спостережуваний Sharpe на
кількість спроб (trials) і дає статистично обґрунтований поріг.

Формули (Bailey & López de Prado 2014):
    SR_0 = sqrt(V[SR_n]) × ((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e)))
    де V[SR_n] — дисперсія оцінки Sharpe, γ — коеф. Ейлера-Маскероні.
    DSR = Φ((SR − SR_0) × sqrt(n−1) / sqrt(1 − skew×SR + (kurt−1)/4×SR²))
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Обернена функція нормального розподілу (алгоритм Acklam)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416, 1.0)
    plow, phigh = 0.02425, 1 - 0.02425
    q = 0.0
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + d[4]
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + d[4]
    )


def estimate_n_trials(param_combinations: int, backtests_per_combo: int = 1, multiplier: float = 1.0) -> int:
    """Оцінка ефективної кількості незалежних спроб (trials).

    Кожна комбінація параметрів × число бектестів × коефіцієнт (за замовч.
    1.0; можна збільшити, якщо дослідник перебирав варіанти ітеративно).
    """
    return max(1, int(param_combinations * backtests_per_combo * multiplier))


def deflated_sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    n_trials: int,
    skew: float | None = None,
    kurtosis: float | None = None,
    sr_benchmark: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio.

    returns: послідовність прибутковостей стратегії (на період).
    n_trials: кількість спроб (див. estimate_n_trials).
    sr_benchmark: Sharpe еталону (за замовч. 0 — безризикова).
    Повертає DSR ∈ [0,1] — імовірність, що справжній Sharpe > 0 після
    коригування на множинне тестування.
    """
    ret = np.asarray(returns, dtype=float)
    n = len(ret)
    if n < 2:
        return 0.0
    sr = ret.mean() / ret.std(ddof=1) if ret.std(ddof=1) > 0 else 0.0
    if skew is None:
        skew = float(pd.Series(ret).skew())
    if kurtosis is None:
        kurtosis = float(pd.Series(ret).kurt())  # надлишковий ексцес

    # дисперсія оцінки Sharpe (Lo 2002 / Bailey-LdP):
    # V[SR] = (1 − skew×SR + (kurt−1)/4×SR²) / (n−1)
    var_sr = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n - 1)
    var_sr = max(var_sr, 1e-12)

    if n_trials <= 1:
        sr0 = sr_benchmark
    else:
        # очікуваний максимум з N спроб (Bailey & López de Prado 2014, eq. 7)
        z = (1.0 - _EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n_trials) + _EULER_GAMMA * _norm_ppf(
            1.0 - 1.0 / (n_trials * math.e)
        )
        sr0 = sr_benchmark + math.sqrt(var_sr) * z

    if sr <= sr0:
        return 0.0
    denominator = math.sqrt(var_sr)
    z_stat = (sr - sr0) / denominator if denominator > 0 else 0.0
    return _norm_cdf(z_stat)


def probability_of_backtest_overfitting(
    oos_sharpes: Sequence[float],
    n_trials: int | None = None,
    threshold: float = 0.0,
) -> float:
    """Оцінка PBO за розподілом OOS Sharpe.

    Проста евристика: частка OOS вікон, де Sharpe нижчий за поріг (0 за
    замовч.), скоригована на кількість спроб. Повна CSCV-оцінка (López de
    Prado) потребує комбінаторних розбиттів — реалізована спрощено через
    бутстреп OOS Sharpe.

    Повертає PBO ∈ [0,1] — імовірність, що "краща" стратегія на IS
    виявиться гіршою за медіану на OOS.
    """
    arr = np.asarray(oos_sharpes, dtype=float)
    if len(arr) == 0:
        return 1.0
    if n_trials is None:
        n_trials = max(1, len(arr))
    # бутстреп-оцінка ймовірності того, що випадково обраний IS-кращий
    # варіант має OOS Sharpe нижчий за медіану
    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(2000, len(arr)), replace=True)
    best_idx = samples.argmax(axis=1)  # індекс "найкращого" у кожному бутстрепі
    best_oos = samples[np.arange(2000), best_idx]
    pbo = float((best_oos < threshold).mean())
    # коригування на trials: з більшою кількістю спроб PBO зростає
    return min(1.0, pbo * math.log2(max(n_trials, 2)) / math.log2(32))
