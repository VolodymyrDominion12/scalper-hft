"""Фічі та фільтри режиму ринку (книга, гл. 4 — Risk Models; гл. 10 — Regime Risk).

Скальпінг-стратегії чутливі до режиму: у тренді працює моментум, у флеті —
mean-reversion. Фільтри режиму відсікають несприятливі стани і тим самим
зменшують кількість збиткових трейдів (покращують win rate та PF).

Іменований стан ринку — два каузальні виміри (лише дані ≤ t):
    структура: range | trend_up | trend_down
    волатильність: low | normal | high
Це не «бичий/ведмежий ринок» на денному ТФ, а локальна структура для скальпінгу.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np
import pandas as pd

MarketStructure = Literal["range", "trend_up", "trend_down"]
VolRegime = Literal["low", "normal", "high"]

STRUCTURE_LABELS: Final[frozenset[str]] = frozenset(get_args(MarketStructure))
VOL_LABELS: Final[frozenset[str]] = frozenset(get_args(VolRegime))
STRUCTURE_ORDER: Final[tuple[str, ...]] = ("range", "trend_up", "trend_down")
VOL_ORDER: Final[tuple[str, ...]] = ("low", "normal", "high")
COMPOSITE_ORDER: Final[tuple[str, ...]] = tuple(f"{s}|{v}" for s in STRUCTURE_ORDER for v in VOL_ORDER)
DEFAULT_TREND_THRESHOLD: Final[float] = 0.35


def volatility_regime(close: pd.Series, lookback: int = 60, percentile_window: int = 500) -> pd.Series:
    """Режим волатильності: low / normal / high за процентилем останньої реалізованої волатильності.

    rolling-ранг рахується на numpy-масивах (raw=True): той самий результат,
    що й pandas-варіант, але без конструювання Series на кожне вікно
    (на 1m×90д порядку ~0.3s замість ~17s).
    """
    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
    rv = log_ret.rolling(lookback, min_periods=lookback // 2).std()
    pct = rv.rolling(percentile_window, min_periods=percentile_window // 2).apply(
        lambda x: (x[-1] >= x).mean(), raw=True
    )
    regime = pd.Series("normal", index=close.index, dtype=object)
    regime[pct > 0.8] = "high"
    regime[pct < 0.2] = "low"
    return regime


def trend_strength(close: pd.Series, ema_fast: int = 9, ema_slow: int = 50) -> pd.Series:
    """Міра сили тренду в [0, 1]: нормалізована відстань між EMA."""
    f = close.ewm(span=ema_fast, adjust=False).mean()
    s = close.ewm(span=ema_slow, adjust=False).mean()
    dist = (f - s).abs() / close.rolling(ema_slow, min_periods=ema_slow).std().replace(0, np.nan)
    return dist.clip(upper=1.0).fillna(0.0)


def funding_filter(funding: pd.DataFrame, max_abs_rate: float = 0.001) -> pd.Series:
    """Маска, коли фандінг у межах норми (не перегрітий ринок перекосів)."""
    if funding is None or funding.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=bool)
    return funding["fundingRate"].abs() < max_abs_rate


def session_filter(index: pd.DatetimeIndex, start_hour: int = 0, end_hour: int = 24) -> pd.Series:
    """Маска активної сесії (години UTC). Скальпінг — лише в ліквідні години."""
    hours = index.hour
    if start_hour <= end_hour:
        mask = (hours >= start_hour) & (hours < end_hour)
    else:
        mask = (hours >= start_hour) | (hours < end_hour)
    return pd.Series(mask, index=index, dtype=bool)


def market_structure(
    close: pd.Series,
    ema_fast: int = 9,
    ema_slow: int = 50,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> pd.Series:
    """Локальна структура ринку: range / trend_up / trend_down.

    Каузально: EMA та сила тренду на закритті бару t. Якщо сила < порога —
    діапазон; інакше знак (EMA_fast − EMA_slow) задає напрям.
    """
    if trend_threshold < 0.0:
        raise ValueError(f"trend_threshold must be >= 0, got {trend_threshold}")
    strength = trend_strength(close, ema_fast=ema_fast, ema_slow=ema_slow)
    fast = close.ewm(span=ema_fast, adjust=False).mean()
    slow = close.ewm(span=ema_slow, adjust=False).mean()
    trending = strength >= trend_threshold
    up = fast > slow
    out = pd.Series("range", index=close.index, dtype=object)
    out.loc[trending & up] = "trend_up"
    out.loc[trending & ~up] = "trend_down"
    return out


def htf_market_structure(
    close: pd.Series,
    htf: str = "1d",
    ema_fast: int = 3,
    ema_slow: int = 20,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> pd.Series:
    """Напрямок ринку на СТАРШОМУ таймфреймі (causal, без lookahead).

    Проблема (iter2-діагностика): структура на робочому ТФ (1h EMA 9/50)
    запізнюється відносно тренду старшого ТФ → напрямкові гейти на ній ріжуть
    правильні лонги. `htf_market_structure` рахує EMA-структуру на зресемпленому
    старшому ТФ (напр. 4h бари → htf='1d') і мапить на кожен бар робочого ТФ
    через ОСТАННІЙ ЗАКРИТИЙ htf-бар ≤ t (searchsorted по часах закриття вікон).

    Returns: Series('range'|'trend_up'|'trend_down'), індексована як close.
    """
    if trend_threshold < 0.0:
        raise ValueError(f"trend_threshold must be >= 0, got {trend_threshold}")
    off = pd.tseries.frequencies.to_offset(htf)
    htf_close = close.resample(off).last().dropna()
    if len(htf_close) < ema_slow + 5:
        return pd.Series("range", index=close.index, dtype=object)

    fast = htf_close.ewm(span=ema_fast, adjust=False).mean()
    slow = htf_close.ewm(span=ema_slow, adjust=False).mean()
    strength = (fast - slow).abs() / htf_close.rolling(ema_slow, min_periods=ema_slow).std().replace(0, np.nan)
    up = (fast > slow) & (strength >= trend_threshold)
    down = (fast < slow) & (strength >= trend_threshold)
    htf_labels = pd.Series("range", index=htf_close.index, dtype=object)
    htf_labels.loc[up] = "trend_up"
    htf_labels.loc[down] = "trend_down"

    # Часи ЗАКРИТТЯ htf-вікон; бар t бачить лише вікна з end <= t.
    ends = htf_close.index + off
    ts = close.index
    pos = ends.searchsorted(ts.to_numpy(), side="right") - 1
    out = pd.Series("range", index=close.index, dtype=object)
    valid = pos >= 0
    out.loc[valid] = htf_labels.iloc[pos[valid]].to_numpy()
    return out


def composite_regime_label(structure: pd.Series, vol: pd.Series) -> pd.Series:
    """Складена мітка `structure|vol`, наприклад `range|low`."""
    return structure.astype(str) + "|" + vol.astype(str)


def apply_min_dwell(series: pd.Series, min_dwell: int) -> pd.Series:
    """Гістерезис категоріального ряду: зміна значення приймається лише після
    `min_dwell` послідовних барів нового значення.

    Мета (дослідження: Krishhiv HMM_TR_Alg; Kiploks guide) — режимний churn:
    без гістерезису швидкі фліпи range↔trend змушують мета-стратегію постійно
    міняти ваги суб-стратегій (зайві комісії, втрата трендового edge).
    Приклад: min_dwell=2 для 1h барів = перемикання щонайменше через 2 год
    стабільного нового режиму. min_dwell=0 → оригінальний ряд без змін.
    Каузальний: рішення на барі t використовує лише значення ≤ t.
    """
    if min_dwell <= 0 or len(series) < 2:
        return series.copy()
    vals = series.to_numpy()
    out = np.empty(len(vals), dtype=object)
    out[0] = vals[0]
    cur = vals[0]
    cand: object | None = None
    run = 0
    for i in range(1, len(vals)):
        v = vals[i]
        if v == cur:
            cand = None
            run = 0
            out[i] = cur
            continue
        if cand is None or v != cand:
            cand = v
            run = 1
        else:
            run += 1
        if run >= min_dwell:
            cur = cand
            cand = None
            run = 0
        out[i] = cur
    return pd.Series(out, index=series.index, dtype=object)


def apply_regime_gates(
    signal: pd.Series,
    regime_df: pd.DataFrame,
    *,
    vol_high_veto: bool = False,
    trend_direction_gate: bool = False,
) -> pd.Series:
    """Режимні гейти на готовий сигнал мета-стратегії (каузальні, без lookahead).

    - vol_high_veto: у режимі high-vol позиція = 0 (дослідження: у вибуховій
      волатильності короткий 1h-горизонт домінує реверсія, а моментум ловить
      «momentum crash»; плюс комісійний бюджет на угоду);
    - trend_direction_gate: у trend_up заборонені шорти, у trend_down — лонги
      (не торгуємо проти явного тренду; range — без обмежень).
    Regime-мітки (structure/vol) беруться з regime_df, обчисленого на закритих
    барах; рушій зсуває сигнал на 1 — виконання з t+1.
    """
    if not (vol_high_veto or trend_direction_gate) or regime_df is None or regime_df.empty:
        return signal.copy()
    out = signal.copy()
    vol = regime_df.reindex(signal.index)["vol"].fillna("normal")
    structure = regime_df.reindex(signal.index)["structure"].fillna("range")
    if vol_high_veto:
        out = out.mask(vol == "high", 0.0)
    if trend_direction_gate:
        out = out.mask((structure == "trend_up") & (out < 0), 0.0)
        out = out.mask((structure == "trend_down") & (out > 0), 0.0)
    return out


def named_market_state(
    close: pd.Series,
    *,
    ema_fast: int = 9,
    ema_slow: int = 50,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    vol_lookback: int = 60,
    vol_percentile_window: int = 500,
) -> pd.DataFrame:
    """Каузальний іменований стан: колонки structure, vol, label.

    label складається як structure|vol. Не HMM і не бик/ведмідь на старшому ТФ.
    """
    structure = market_structure(
        close,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        trend_threshold=trend_threshold,
    )
    vol = volatility_regime(close, lookback=vol_lookback, percentile_window=vol_percentile_window)
    return pd.DataFrame(
        {
            "structure": structure,
            "vol": vol,
            "label": composite_regime_label(structure, vol),
        },
        index=close.index,
    )
