"""Fractional Differentiation (AFML Chapter 5).

Fractional differentiation дозволяє зберегти "пам'ять" часового ряду
(стаціонарний, але не позбавлений довгострокових трендів) на відміну від
цілочисельного диференціювання (I(1) → I(0) = повна втрата пам'яті).

Алгоритм (FFD — Fixed-Width Window Fractional Differencing):
  y_t = Σ_{k=0}^{∞} w_k * x_{t-k}
  w_0 = 1, w_k = w_{k-1} * (d - k + 1) / k

Практично: обрізаємо w при |w_k| < threshold.

Функції:
  frac_diff_ffd      — основна реалізація FFD
  find_min_d         — бінарний пошук мінімального d при якому ряд стаціонарний
  add_frac_diff      — додає frac_diff фічу до DataFrame (для pipeline)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ── Ваги ─────────────────────────────────────────────────────────────────────

def _get_weights_ffd(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Ваги для FFD (Fixed-Width Window).

    w_0 = 1
    w_k = w_{k-1} * (d - k + 1) / k
    Обрізаємо, коли |w_k| < threshold (відносно w_0=1).
    threshold=1e-4 дає ~20-30 ваг для d=0.4, що практично для 300-bar серій.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # реверс: w[-1] * x_t + w[-2] * x_{t-1} + ...


# ── FFD ───────────────────────────────────────────────────────────────────────

def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Fixed-Width Window Fractional Differencing.

    Args:
        series: вхідний часовий ряд (ціни або лог-ціни).
        d: ступінь диференціювання ∈ (0, 1].
        threshold: відсікання малих ваг.

    Returns:
        Series тієї ж довжини зі збереженим індексом.
        Перші len(w)−1 значень = NaN (необхідний контекст).
    """
    w = _get_weights_ffd(d, threshold)
    width = len(w) - 1
    output = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.values.astype(float)

    for i in range(width, len(values)):
        window = values[i - width : i + 1]
        if np.isnan(window).any():
            continue
        output.iloc[i] = np.dot(w, window)

    return output


def frac_diff_expanding(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Expanding-window fractional differencing (повна версія).

    Повільніше, але без NaN на початку (окрім першого рядка).
    Корисна для коротких рядів.
    """
    values = series.values.astype(float)
    n = len(values)
    output = np.full(n, np.nan)

    for t in range(1, n):
        w_full = _get_weights_ffd(d, threshold=0.0)
        # обрізаємо до доступних спостережень
        k = min(len(w_full), t + 1)
        w_slice = w_full[-k:]
        # нормуємо на суму ваг
        w_slice = w_slice / w_slice.sum() if w_slice.sum() != 0 else w_slice
        window = values[max(0, t - k + 1) : t + 1]
        if len(window) == len(w_slice):
            output[t] = np.dot(w_slice, window)

    return pd.Series(output, index=series.index)


# ── Пошук мінімального d ─────────────────────────────────────────────────────

def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    adf_threshold: float = 0.05,
    threshold: float = 1e-4,
) -> float:
    """Бінарний пошук мінімального d, при якому ADF відкидає нульову гіпотезу.

    Ціль: знайти найменший d, що дає стаціонарний ряд.
    Менший d → більше збереженої пам'яті.

    Args:
        series: вхідний ряд (log-ціни).
        d_range: діапазон пошуку (lo, hi).
        step: крок пошуку.
        adf_threshold: p-value порог (0.05).
        threshold: обрізання ваг FFD.

    Returns:
        Мінімальний d або -1.0 якщо жодне d не дає стаціонарності.
    """
    lo, hi = d_range
    results = {}

    for d in np.arange(lo, hi + step, step):
        diff = frac_diff_ffd(series, d=round(d, 4), threshold=threshold).dropna()
        if len(diff) < 20:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(diff, maxlag=1, regression="c", autolag=None)
        results[round(d, 4)] = adf[1]  # p-value

    for d_val in sorted(results):
        if results[d_val] < adf_threshold:
            return d_val

    return -1.0


# ── Pipeline helper ───────────────────────────────────────────────────────────

def add_frac_diff(
    df: pd.DataFrame,
    col: str = "close",
    d: float | None = None,
    use_log: bool = True,
    threshold: float = 1e-4,
    adf_threshold: float = 0.05,
) -> pd.DataFrame:
    """Додає frac_diff фічу до DataFrame.

    Args:
        df: OHLCV DataFrame.
        col: колонка для диференціювання (default: 'close').
        d: ступінь диференціювання; якщо None — автоматичний пошук.
        use_log: якщо True — диференціює log(price).
        threshold: обрізання ваг FFD.
        adf_threshold: для авто-пошуку d.

    Returns:
        DataFrame з новою колонкою 'fd_{col}' та 'd_used' attr.
    """
    out = df.copy()
    src = np.log(df[col]) if use_log else df[col]

    if d is None:
        d = find_min_d(src, threshold=threshold, adf_threshold=adf_threshold)
        if d < 0:
            # fallback: d=0.4 (правило великого пальця)
            d = 0.4

    fd = frac_diff_ffd(src, d=d, threshold=threshold)
    out[f"fd_{col}"] = fd
    out.attrs["d_used"] = d
    return out


def frac_diff_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    d: float = 0.4,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Застосовує FFD до кількох колонок одночасно.

    Args:
        df: OHLCV DataFrame.
        cols: колонки для обробки (default: ['close', 'volume']).
        d: фіксований ступінь диференціювання.
        threshold: обрізання ваг.

    Returns:
        DataFrame з доданими fd_* колонками.
    """
    if cols is None:
        cols = ["close", "volume"]

    out = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        src = np.log(df[col].replace(0, np.nan)).dropna() if col in ("close", "volume") else df[col]
        fd = frac_diff_ffd(src, d=d, threshold=threshold)
        out[f"fd_{col}"] = fd.reindex(df.index)
    return out
