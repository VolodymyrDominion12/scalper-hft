"""Sample Weights (AFML Chapter 4).

Два типи ваг:
  1. Uniqueness-ваги (concurrency): w_i ∝ 1/c̄_i де c̄_i — середня одночасність
     лейблу i. Зменшує вплив барів що перекриваються з іншими.
  2. Time-decay ваги: w_i помножується на монотонно зростаючий decay
     (старіші спостереження важать менше).

Sequential Bootstrap (AFML §4.5):
  Замість i.i.d. bootstrap — семплюємо так, щоб мінімізувати одночасність
  між вибраними зразками (наближення до IID).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Матриця індикаторів ───────────────────────────────────────────────────────

def get_ind_matrix(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.DataFrame:
    """Матриця φ[i,j] = 1, якщо бар i міститься в лейблі j.

    Args:
        bar_index: DatetimeIndex всіх барів (рядки матриці).
        t1: Series t0→t1 (повернення з get_events або label_from_ohlcv).

    Returns:
        DataFrame (bars × labels) з булевими значеннями.
    """
    ind_m = pd.DataFrame(0, index=bar_index, columns=range(len(t1)))
    for j, (t0, t_end) in enumerate(t1.items()):
        if pd.isna(t_end):
            continue
        ind_m.loc[t0:t_end, j] = 1.0
    return ind_m


def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    """Середня унікальність кожного лейблу.

    c_t = кількість одночасних лейблів у барі t.
    u_t,j = φ[t,j] / c_t  — частка унікальності лейблу j у барі t.
    ū_j = середнє u_t,j по всіх барах де φ[t,j] = 1.
    """
    c = ind_m.sum(axis=1)  # одночасність у кожному барі
    u = ind_m.div(c.replace(0, np.nan), axis=0)  # u[t,j]
    avg_u = u[ind_m > 0].mean()
    return avg_u


# ── Sequential Bootstrap ──────────────────────────────────────────────────────

def seq_bootstrap(
    ind_m: pd.DataFrame,
    s_length: int | None = None,
) -> list[int]:
    """Sequential Bootstrap: обирає індекси зразків з мінімальною одночасністю.

    Args:
        ind_m: матриця індикаторів (bars × labels).
        s_length: розмір вибірки (default = кількість лейблів).

    Returns:
        Список індексів колонок ind_m (тобто індексів лейблів).
    """
    if s_length is None:
        s_length = ind_m.shape[1]

    phi = []
    for _ in range(s_length):
        avg_u = pd.Series(dtype=float)
        for i in ind_m.columns:
            phi_ = phi + [i]
            avg_u[i] = get_avg_uniqueness(ind_m[phi_]).iloc[-1]
        # вибираємо лейбл з найвищою унікальністю (менш корельований з вже вибраними)
        prob = avg_u / avg_u.sum()
        phi.append(np.random.choice(prob.index, p=prob.values))
    return phi


def get_sample_tw(
    t1: pd.Series,
    num_co_events: pd.Series,
) -> pd.Series:
    """Ваги на основі одночасності подій.

    w_i = 1 / ū_i де ū_i = середня кількість одночасних лейблів.

    Args:
        t1: Series t0→t1.
        num_co_events: кількість одночасних подій (з get_num_co_events).

    Returns:
        Series ваг (нормовано до суми = кількість зразків).
    """
    w = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            w[t_in] = 0.0
            continue
        window = num_co_events[t_in:t_out]
        w[t_in] = (1.0 / window.replace(0, np.nan)).mean()
    w = w.fillna(0.0)
    # нормуємо
    total = w.sum()
    if total > 0:
        w = w * (len(w) / total)
    return w


def get_num_co_events(
    close_index: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Кількість одночасних лейблів для кожного бару.

    Returns:
        Series: bar_timestamp → count of concurrent labels.
    """
    t1 = t1.fillna(close_index[-1])
    if molecule is None:
        molecule = t1.index
    # всі бари між першою подією і останньою
    iloc = close_index.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_index[iloc[0] : iloc[1] + 1])
    for t_in, t_out in t1.loc[molecule].items():
        count.loc[t_in:t_out] += 1
    return count.loc[t1.index[0] : t1.max()]


# ── Time-Decay ваги ───────────────────────────────────────────────────────────

def get_time_decay_weights(
    t1: pd.Series,
    num_co_events: pd.Series,
    decay: float = 1.0,
) -> pd.Series:
    """Поєднує uniqueness-ваги з time-decay.

    decay ∈ (0, 1]: 1.0 = без decay; 0.5 = найдавніший зразок важить вдвічі менше.
    """
    w = get_sample_tw(t1, num_co_events)
    if decay >= 1.0:
        return w / w.sum()

    # лінійний time-decay
    cw = w.sort_index().cumsum()
    if decay >= 0.0:
        slope = (1.0 - decay) / cw.iloc[-1]
        const = 1.0 - slope * cw.iloc[-1]
    else:
        slope = 1.0 / ((decay + 1) * cw.iloc[-1])
        const = 1.0 / (decay + 1)

    w_decay = const + slope * cw
    w_decay[w_decay < 0.0] = 0.0
    w_decay = w_decay / w_decay.sum()
    return w_decay.reindex(w.index).fillna(0.0)


# ── Комбінований helper ───────────────────────────────────────────────────────

def compute_sample_weights(
    events: pd.DataFrame,
    close: pd.Series,
    decay: float = 0.9,
) -> pd.Series:
    """Головна функція: від events → ваги для кожного зразку.

    Args:
        events: DataFrame з колонками [t1, target, side, label] (з labeling.py).
        close: серія цін закриття.
        decay: time-decay коефіцієнт (1.0 = без decay).

    Returns:
        Series ваг, вирівняна за events.index.
    """
    t1 = events["t1"].dropna()
    if t1.empty:
        return pd.Series(1.0, index=events.index)

    num_co = get_num_co_events(close.index, t1)
    w = get_time_decay_weights(t1, num_co, decay=decay)
    return w.reindex(events.index).fillna(0.0)
