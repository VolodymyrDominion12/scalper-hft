"""Triple-Barrier Labeling (AFML Chapter 3).

Реалізація мета-лейблінгу за Marcos López de Prado:
  - Вертикальний бар'єр: t₀ + max_holding
  - Горизонтальний: ±pt/sl у відносних одиницях ATR або ціни

Без lookahead: таргет обчислюється з t₁ (наступний бар) до t₁+max_holding.
Повертає:
  events   — DataFrame з t0, t1, side, target
  labels   — Series {−1, 0, +1} (0 = вертикальний бар'єр / timeout)
  returns  — Series фактичної прибутковості [t0→t1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Допоміжні ────────────────────────────────────────────────────────────────


def _daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Денна (per-bar) волатильність для встановлення бар'єрів."""
    ret = np.log(close / close.shift(1))
    return ret.ewm(span=span, min_periods=span // 2).std()


# ── Бар'єри ───────────────────────────────────────────────────────────────────


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Генерує DataFrame подій (t0 → t1) з triple-barrier labeling.

    Args:
        close: серія цін закриття.
        t_events: точки входу (індекс close).
        pt_sl: (profit-take множник, stop-loss множник) відносно target.
                Якщо 0 — бар'єр відключено.
        target: волатильність (або ATR) у тих же одиницях, що й close.
        min_ret: мінімальна абс. прибутковість для включення події.
        vertical_barrier_times: опціонально — крайній час для кожної t0.
        side: опціонально — напрямок (+1 лонг / −1 шорт); якщо None — обидва.

    Returns:
        DataFrame з колонками [t1, target, side].
    """
    # фільтрувати target мінімальною прибутковістю
    target = target.reindex(t_events)
    if min_ret > 0.0:
        target = target[target >= min_ret]
    if target.empty:
        return pd.DataFrame(columns=["t1", "target", "side"])

    # вертикальний бар'єр
    if vertical_barrier_times is None:
        t1 = pd.Series(pd.NaT, index=target.index)
    else:
        t1 = vertical_barrier_times.reindex(target.index)

    if side is None:
        side_ = pd.Series(1.0, index=target.index)  # обидва напрямки
    else:
        side_ = side.reindex(target.index)

    events = pd.concat({"t1": t1, "target": target, "side": side_}, axis=1).dropna(subset=["target"])
    return _apply_pt_sl(close=close, events=events, pt_sl=pt_sl)


def _apply_pt_sl(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """Знаходить перший час, коли ціна торкнулася будь-якого бар'єра."""
    out = events[["t1"]].copy()
    pt, sl = pt_sl

    for loc, t0 in events.index.to_series().items():
        df0 = close[loc:]
        # обмежуємо вертикальним бар'єром
        if pd.notna(events.at[loc, "t1"]):
            df0 = df0[: events.at[loc, "t1"]]
        if df0.empty:
            out.at[loc, "t1"] = events.at[loc, "t1"]
            continue

        df0 = (df0 / close[loc] - 1) * events.at[loc, "side"]  # ret * side
        t_pt = df0[df0 >= pt * events.at[loc, "target"]].index.min() if pt > 0 else pd.NaT
        t_sl = df0[df0 <= -sl * events.at[loc, "target"]].index.min() if sl > 0 else pd.NaT

        out.at[loc, "t1"] = (
            min(t for t in [t_pt, t_sl, events.at[loc, "t1"]] if pd.notna(t))
            if any(pd.notna(t) for t in [t_pt, t_sl, events.at[loc, "t1"]])
            else pd.NaT
        )

    return out.join(events[["target", "side"]])


def get_labels(
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.Series:
    """Лейбли: +1 (PT hit), −1 (SL hit), 0 (вертикальний бар'єр / timeout).

    Якщо side ≠ None (мета-лейблінг), returns 0/1 замість −1/0/+1.
    """
    events = events.dropna(subset=["t1"])
    px = events.index.union(events["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")

    out = pd.Series(index=events.index, dtype=int)
    for loc, t1 in events["t1"].items():
        if pd.isna(t1) or loc not in px.index or t1 not in px.index:
            out.at[loc] = 0
            continue
        ret = (px[t1] - px[loc]) / px[loc]
        out.at[loc] = int(np.sign(ret * events.at[loc, "side"]))
    return out


def get_t_events(close: pd.Series, threshold: float = 0.005) -> pd.DatetimeIndex:
    """CUSUM filter (AFML Ch.2.5.2): події лише на інформативних зсувах ціни."""
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    events: list[pd.Timestamp] = []
    s_pos = 0.0
    s_neg = 0.0
    for t, r in log_ret.items():
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos > threshold or s_neg < -threshold:
            events.append(pd.Timestamp(t))
            s_pos = 0.0
            s_neg = 0.0
    return pd.DatetimeIndex(events)


# ── Вертикальний бар'єр (простий helper) ─────────────────────────────────────


def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_days: int = 1,
) -> pd.Series:
    """Вертикальний бар'єр через `num_days` барів (не календарних днів).

    Ім'я аргумента історичне (AFML: num_days); для 1m/5m `holding_bars=10`
    означає 10 свічок, а не 10 діб.
    """
    n_bars = int(num_days)
    if n_bars < 1:
        raise ValueError("num_days (бари горизонту) має бути >= 1")
    locs = close.index.searchsorted(t_events)
    t1_locs = locs + n_bars
    valid = (locs >= 0) & (t1_locs < len(close.index))
    if not np.any(valid):
        return pd.Series(dtype="datetime64[ns]")
    events = pd.DatetimeIndex(t_events)[valid]
    return pd.Series(close.index[t1_locs[valid]], index=events)


# ── Швидкий helper: від OHLCV до labeled dataset ──────────────────────────────


def label_from_ohlcv(
    df: pd.DataFrame,
    pt: float = 1.0,
    sl: float = 1.0,
    holding_bars: int = 10,
    vol_span: int = 100,
    min_ret: float = 0.0,
    event_filter: str = "all",
    cusum_threshold: float = 0.005,
) -> pd.DataFrame:
    """Одна функція: OHLCV → events + labels.

    event_filter: 'all' — кожен бар; 'cusum' — лише CUSUM-події (AFML 2.5.2).
    """
    close = df["close"]
    if event_filter == "cusum":
        t_events = get_t_events(close, threshold=cusum_threshold)
        if len(t_events) == 0:
            t_events = close.index
    else:
        t_events = close.index
    vol = _daily_vol(close, span=vol_span)
    t1 = add_vertical_barrier(t_events, close, num_days=holding_bars)

    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=(pt, sl),
        target=vol,
        min_ret=min_ret,
        vertical_barrier_times=t1,
    )
    labels = get_labels(events, close)
    events["label"] = labels

    # фактичний ret
    def _ret(row):
        if pd.isna(row["t1"]) or row.name not in close.index or row["t1"] not in close.index:
            return np.nan
        return (close[row["t1"]] - close[row.name]) / close[row.name]

    events["ret"] = events.apply(_ret, axis=1)
    return events
