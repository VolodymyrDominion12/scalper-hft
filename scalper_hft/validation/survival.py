"""Survival analysis угод: Kaplan–Meier (Predictive Marketing Ch.7/12/13).

Аналогія: «churn за віком відносин» = вихід з позиції залежно від віку
трейду. Замість фіксованого вертикального бар'єра (holding_bars у
triple-barrier) — емпірична крива «дожиття» позиції та медіанний час
утримання залежно від умов входу.

⚠ Чесно: вихід з позиції — це РІШЕННЯ стратегії, а не екзогенна подія,
тож цензурування «брудне». Використовувати як описову аналітику для
калібрування holding_bars, не як істину про оптимальний вихід.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_durations(
    trades: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """Тривалості угод у барах + цензурування.

    trades: DataFrame з колонками entry_ts, exit_ts (формат _extract_trades).
    event = 1 для закритих угод; 0 — якщо угода «дожила» до кінця вибірки
    (цензурована — остання незакрита позиція).

    Returns:
        DataFrame з колонками duration (бари), event (0/1).
    """
    if trades is None or trades.empty or "entry_ts" not in trades.columns:
        return pd.DataFrame(columns=["duration", "event"])
    df = trades.copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df["duration"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / _freq_seconds(freq)
    df["duration"] = df["duration"].clip(lower=1.0)
    # цензурування: угода, що досі «відкрита» на останньому барі — event=0
    df["event"] = 1
    last = df["exit_ts"].max()
    df.loc[df["exit_ts"] >= last, "event"] = 0
    return df[["duration", "event"]]


def _freq_seconds(freq: str) -> float:
    import re

    m = re.match(r"(\d+)([smhd])", freq.strip().lower())
    if not m:
        return 60.0
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "s":
        return float(n)
    if unit == "m":
        return float(n * 60)
    if unit == "h":
        return float(n * 3600)
    return float(n * 86400)


def kaplan_meier(
    durations: np.ndarray | pd.Series, events: np.ndarray | pd.Series, max_time: int | None = None
) -> pd.DataFrame:
    """Крива виживання Kaplan–Meier (без lifelines, на numpy).

    S(t) = Π_{t_i ≤ t} (1 − d_i/n_i), де d_i — події в момент t_i, n_i —
    кількість «під ризиком».

    Returns:
        DataFrame: time, n_risk, n_events, survival (S(t)), cum_events.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    if len(d) == 0:
        return pd.DataFrame(columns=["time", "n_risk", "n_events", "survival", "cum_events"])
    times = np.sort(np.unique(d))
    if max_time is not None:
        times = times[times <= max_time]
    survival = 1.0
    rows: dict[str, list] = {"time": [], "n_risk": [], "n_events": [], "survival": [], "cum_events": []}
    cum = 0
    for t in times:
        at_risk = int(np.sum(d >= t))
        n_events = int(np.sum((d == t) & (e == 1)))
        if at_risk > 0:
            survival *= 1.0 - n_events / at_risk
        cum += n_events
        rows["time"].append(float(t))
        rows["n_risk"].append(at_risk)
        rows["n_events"].append(n_events)
        rows["survival"].append(survival)
        rows["cum_events"].append(cum)
    return pd.DataFrame(rows)


def median_survival_time(km: pd.DataFrame) -> float:
    """Медіанний час «життя» позиції: найменший t, де S(t) ≤ 0.5."""
    if km is None or km.empty or "survival" not in km.columns:
        return 0.0
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float(km["time"].iloc[-1])


def survival_by_feature(
    trades: pd.DataFrame,
    feature: pd.Series,
    n_bins: int = 3,
    freq: str = "1min",
) -> pd.DataFrame:
    """Медіанний час утримання за бінами фічі входу (описово).

    trades: з entry_ts/exit_ts; feature: Series значень фічі, індексована
    за entry_ts (значення на вході). Returns DataFrame: bin, n, median_hold,
    survival@25% (частка позицій, що «живуть» довше 25% медіани... ні —
    S при медіанному часі всіх угод).
    """
    dur = trade_durations(trades, freq=freq)
    if dur.empty:
        return pd.DataFrame(columns=["bin", "n", "median_hold", "survival_10"])
    feats = feature.reindex(pd.to_datetime(trades["entry_ts"]))
    df = dur.copy()
    df["feat"] = feats.values
    df = df.dropna(subset=["feat"])
    if df.empty or df["feat"].nunique() < 2:
        return pd.DataFrame(columns=["bin", "n", "median_hold", "survival_10"])
    try:
        df["bin"] = pd.qcut(df["feat"], q=min(n_bins, df["feat"].nunique()), duplicates="drop")
    except ValueError:
        df["bin"] = pd.qcut(df["feat"].rank(method="first"), q=min(n_bins, df["feat"].nunique()), duplicates="drop")
    rows: dict[str, list] = {"bin": [], "n": [], "median_hold": [], "survival_10": []}
    overall_median = float(df["duration"].median())
    for b, g in df.groupby("bin", observed=True):
        km = kaplan_meier(g["duration"].values, g["event"].values)
        surv_at = 0.0
        if not km.empty and overall_median > 0:
            interp = km[km["time"] <= overall_median]
            surv_at = float(interp["survival"].iloc[-1]) if len(interp) else 1.0
        rows["bin"].append(str(b))
        rows["n"].append(len(g))
        rows["median_hold"].append(float(g["duration"].median()))
        rows["survival_10"].append(surv_at)
    return pd.DataFrame(rows)


__all__ = ["trade_durations", "kaplan_meier", "median_survival_time", "survival_by_feature"]
