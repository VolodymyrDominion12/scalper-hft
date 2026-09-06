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
    high: pd.Series | None = None,
    low: pd.Series | None = None,
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
        high/low: якщо задані — шлях бар'єрів по внутрішньобарових екстремумах
                (точніше для скальпінгу); інакше — close-to-close.

    Returns:
        DataFrame з колонками [t1, target, side, barrier],
        barrier ∈ {"pt", "sl", "vb"} — який бар'єр спрацював (side-adjusted).
    """
    # фільтрувати target мінімальною прибутковістю
    target = target.reindex(t_events)
    if min_ret > 0.0:
        target = target[target >= min_ret]
    if target.empty:
        return pd.DataFrame(columns=["t1", "target", "side", "barrier"])

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
    return _apply_pt_sl(close=close, events=events, pt_sl=pt_sl, high=high, low=low)


def _apply_pt_sl(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> pd.DataFrame:
    """Знаходить перший час, коли ціна торкнулася будь-якого бар'єра.

    З high/low — внутрішньобаровий шлях: для лонга PT торкається high,
    SL — low (для шорта дзеркально через side-adjusted дохідності).
    Одночасне торкання обох бар'єрів одним баром → песимістично SL.
    """
    out = events[["t1"]].copy()
    barrier = pd.Series(index=events.index, dtype=object)
    pt, sl = pt_sl
    use_wicks = high is not None and low is not None

    for loc, t0 in events.index.to_series().items():
        vb = events.at[loc, "t1"]
        side = events.at[loc, "side"]
        p0 = close[loc]
        # Шлях починається з НАСТУПНОГО бару: high/low бару входу частково
        # передують входу на close — їх використання було б lookahead.
        pos0 = close.index.get_loc(loc)
        start = int(pos0) + 1 if not isinstance(pos0, slice) else 1

        if use_wicks:
            # сприятливий екстремум: для лонга — high, для шорта — low
            fav = high if side > 0 else low
            adv = low if side > 0 else high
            up_path = (fav.iloc[start:] / p0 - 1) * side
            dn_path = (adv.iloc[start:] / p0 - 1) * side
        else:
            path = (close.iloc[start:] / p0 - 1) * side
            up_path = path
            dn_path = path
        # обмежуємо вертикальним бар'єром
        if pd.notna(vb):
            up_path = up_path[:vb]
            dn_path = dn_path[:vb]
        if up_path.empty:
            out.at[loc, "t1"] = vb
            barrier.at[loc] = "vb" if pd.notna(vb) else None
            continue

        target = events.at[loc, "target"]
        t_pt = up_path[up_path >= pt * target].index.min() if pt > 0 else pd.NaT
        t_sl = dn_path[dn_path <= -sl * target].index.min() if sl > 0 else pd.NaT

        # одночасне торкання обох одним баром → песимістично SL
        if pd.notna(t_pt) and pd.notna(t_sl) and t_pt == t_sl:
            t_pt = pd.NaT

        candidates = [(t, b) for t, b in [(t_pt, "pt"), (t_sl, "sl"), (vb, "vb")] if pd.notna(t)]
        if candidates:
            t_best, b_best = min(candidates, key=lambda x: x[0])
            out.at[loc, "t1"] = t_best
            barrier.at[loc] = b_best
        else:
            out.at[loc, "t1"] = pd.NaT

    out["barrier"] = barrier
    return out.join(events[["target", "side"]])


def get_labels(
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.Series:
    """Лейбли: +1 (PT hit), −1 (SL hit), 0 (вертикальний бар'єр / timeout).

    Якщо events мають колонку "barrier" (з _apply_pt_sl) — лейбл з неї
    (side-adjusted: PT → +1, SL → −1, VB → 0). Інакше — fallback на знак
    дохідності close-to-close (legacy-поведінка для зовнішніх events).
    """
    events = events.dropna(subset=["t1"])

    if "barrier" in events.columns:
        mapping = {"pt": 1, "sl": -1, "vb": 0}
        labels = events["barrier"].map(mapping)
        return labels.fillna(0).astype(int)

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

    # Шлях бар'єрів по high/low, якщо колонки є (точніше для скальпінгу)
    has_wicks = {"high", "low"}.issubset(df.columns)
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=(pt, sl),
        target=vol,
        min_ret=min_ret,
        vertical_barrier_times=t1,
        high=df["high"] if has_wicks else None,
        low=df["low"] if has_wicks else None,
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
