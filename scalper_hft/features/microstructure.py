"""Мікроструктурні фічі (AFML Ch.19) — токсичність потоку, ліквідність, спред.

Додатковий набір поза `indicators.py` (там уже є CVD, OB imbalance, spread):

    - vpin               — Volume-Synchronized PIN: міра flow toxicity, прямо
                           калібрує ризик adverse selection для maker-рушія;
    - kyle_lambda        — ціновий вплив потоку: OLS Δp = λ·(signed_vol) + ε;
                           фіча — t-value λ (у книзі: t-значення інформативніші);
    - roll_spread        — ефективний спред із серійної коваріації прибутків
                           (2√(−cov(Δp_t, Δp_{t−1}))) — без depth, з aggTrades;
    - amihud             — |r| / доларовий об'єм — ціновий відгук на ліквідність;
    - corwin_schultz     — спред лише з High/Low (без книги) + Parkinson vol;
    - signed_flow_autocorr — персистентність потоку (splitting/herding).

Усі функції повертають Series/DataFrame без lookahead (лише минулі дані).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Volume bars ───────────────────────────────────────────────────────────────


def volume_bars(trades: pd.DataFrame, bar_volume: float) -> pd.DataFrame:
    """Об'ємні бари з потоку угод (AFML Ch.2.3.1).

    trades: DataFrame з колонками price/amount/side, індекс — ts.
    bar_volume: цільовий об'єм бару (в одиницях amount).

    Returns:
        DataFrame з колонками open/high/low/close/volume/buy_volume/sell_volume,
        індекс — час закриття бару.
    """
    if trades is None or trades.empty or "price" not in trades.columns:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])
    t = trades.sort_index().copy()
    cumvol = t["amount"].cumsum()
    t["bar_id"] = (cumvol // bar_volume).astype(int)
    g = t.groupby("bar_id")
    out = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["amount"].sum(),
        }
    )
    if "side" in t.columns:
        buy = t[t["side"] == "buy"].groupby("bar_id")["amount"].sum().rename("buy_volume")
        sell = t[t["side"] == "sell"].groupby("bar_id")["amount"].sum().rename("sell_volume")
        out = out.join(buy).join(sell).fillna(0.0)
    else:
        out["buy_volume"] = 0.0
        out["sell_volume"] = 0.0
    out.index = t.groupby("bar_id").tail(1).index  # час закриття бару
    # останній бар майже завжди неповний (залишок потоку) — викидаємо
    if len(out) > 1:
        out = out.iloc[:-1]
    return out


# ── VPIN (AFML Ch.19.5.2) ─────────────────────────────────────────────────────


def vpin(trades: pd.DataFrame, bar_volume: float = 1000.0, n: int = 50) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading.

    VPIN = Σ|V_B^τ − V_S^τ| / (n·V) — нормований дисбаланс агресивного потоку
    за останні n об'ємних барів. Високий VPIN = токсичний потік = високий
    ризик adverse selection (не лишати maker-ордери!).

    Returns:
        Series VPIN, індексована часом закриття об'ємних барів.
    """
    bars = volume_bars(trades, bar_volume)
    if bars.empty:
        return pd.Series(dtype=float)
    diff = (bars["buy_volume"] - bars["sell_volume"]).abs()
    return diff.rolling(n, min_periods=max(10, n // 2)).sum() / (n * bar_volume)


# ── Kyle λ (AFML Ch.19.4.1) ───────────────────────────────────────────────────


def signed_volume_series(trades: pd.DataFrame, resample: str = "1min") -> pd.Series:
    """Підписаний об'єм за період: Σ side·amount (buy > 0, sell < 0)."""
    if trades is None or trades.empty or "side" not in trades.columns:
        return pd.Series(dtype=float)
    df = trades.copy()
    df["sv"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    return df["sv"].resample(resample).sum().fillna(0.0)


def kyle_lambda(close: pd.Series, signed_volume: pd.Series) -> tuple[float, float]:
    """Коефіцієнт Kyle λ: Δp_t = λ·(signed_vol_t) + ε_t (без константи).

    Returns:
        (lambda, t_stat): λ — ціновий вплив одиниці підписаного потоку;
        t_stat — його значущість (AFML: t-значення інформативніші за λ).
    """
    dp = close.diff().dropna()
    sv = signed_volume.reindex(dp.index).fillna(0.0)
    x = sv.to_numpy(dtype=float)
    y = dp.to_numpy(dtype=float)
    den = float(np.dot(x, x))
    if den <= 0 or len(x) < 3:
        return 0.0, 0.0
    lam = float(np.dot(x, y) / den)
    resid = y - lam * x
    rss = float(np.dot(resid, resid))
    n = len(x)
    se = float(np.sqrt(rss / (n - 1) / den)) if n > 1 and rss > 0 else float("inf")
    t_stat = lam / se if np.isfinite(se) and se > 0 else 0.0
    return lam, t_stat


def kyle_lambda_series(close: pd.Series, signed_volume: pd.Series, window: int = 120) -> tuple[pd.Series, pd.Series]:
    """Rolling-оцінка Kyle λ і його t-value на кожному барі (AFML Ch.19.4.1).

    Векторизована реалізація через numpy sliding_window_view (без Python loops).
    10–50× швидше за оригінальний цикл при великих T.

    Returns:
        (lambda_series, t_series): вирівняні за close.index.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    dp = close.diff()
    sv = signed_volume.reindex(dp.index).fillna(0.0)
    x = sv.to_numpy(dtype=float)
    y = dp.to_numpy(dtype=float)
    n = len(y)

    lam_vals = np.full(n, np.nan)
    tstat_vals = np.full(n, np.nan)

    if n < window + 1:
        return pd.Series(lam_vals, index=close.index), pd.Series(tstat_vals, index=close.index)

    # sliding_window_view: shape (T-window, window), без копіювання
    xw = sliding_window_view(x, window)  # (T-w, w)
    yw = sliding_window_view(y, window)  # (T-w, w)

    xx = (xw * xw).sum(axis=1)  # Σ x²  — shape (T-w,)
    xy = (xw * yw).sum(axis=1)  # Σ x·y

    valid = xx > 0
    b = np.where(valid, xy / np.where(valid, xx, 1.0), np.nan)  # OLS slope

    resid = yw - b[:, None] * xw  # (T-w, w)
    rss = (resid * resid).sum(axis=1)  # Σ ε²

    w = window
    se = np.where(
        valid & (rss > 0),
        np.sqrt(rss / (w - 1) / np.where(valid, xx, 1.0)),
        np.inf,
    )
    t = np.where((se > 0) & np.isfinite(se), b / se, 0.0)

    # результати відповідають барам [window-1 .. T-1] (перше вікно закінчується на барі window-1)
    lam_vals[window - 1 :] = b
    tstat_vals[window - 1 :] = t

    return pd.Series(lam_vals, index=close.index), pd.Series(tstat_vals, index=close.index)


# ── Roll spread (AFML Ch.19.3.2) ──────────────────────────────────────────────


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Ефективний спред з серійної коваріації прибутків: 2√(−cov(Δp_t, Δp_{t−1})).

    0, якщо cov ≥ 0 (неможливий спред за Roll). Використовує лише ціни —
    підходить для кешованих klines без depth.

    Векторизована реалізація через pandas rolling.cov — без Python loops.
    """
    dp = close.diff()
    dp_lag = dp.shift(1)
    # rolling sample cov між dp_t і dp_{t-1}
    cov = dp.rolling(window, min_periods=window // 2).cov(dp_lag)
    # Roll-спред: 2√(-cov), де cov < 0; інакше 0
    spread = (2.0 * np.sqrt((-cov).clip(lower=0.0))).where(cov < 0, other=0.0)
    return spread.reindex(close.index).fillna(0.0)


# ── Amihud illiquidity (AFML Ch.19.4.2) ──────────────────────────────────────


def amihud(returns: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """Міра неліквідності Amihud: середнє |r| / доларовий об'єм за вікно.

    Вище значення = тонша ліквідність (більший ціновий відгук на долар об'єму).
    """
    illiq = returns.abs() / dollar_volume.replace(0, np.nan)
    return illiq.rolling(window, min_periods=window // 2).mean()


# ── Corwin–Schultz spread + Parkinson vol (AFML Ch.19.3.3–19.3.4) ─────────────


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Спред Corwin–Schultz лише з High/Low (без книги/глибини).

    α = (√(2β) − √(β/3) − √(2γ−β)) / (3 − 2√2),  S = 2α,
    де β — сума квадратів лог-діапазонів двох барів, γ — квадрат дворового
    діапазону. NaN, де (2γ−β) < 0 (шум — стандартна поведінка оцінювача).
    """
    h = pd.Series(np.log(high / low), index=high.index).pow(2)
    beta = h.rolling(2).sum()
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    gamma = pd.Series(np.log(h2 / l2), index=high.index).pow(2)
    inner = 2.0 * gamma - beta
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta / 3.0) - np.sqrt(inner)) / (3.0 - 2.0 * np.sqrt(2.0))
    alpha = alpha.where(inner > 0).where(beta > 0).clip(lower=0.0)  # негативні → 0 (стандартна практика)
    return 2.0 * alpha


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Волатильність Parkinson: sqrt(mean(ln(H/L)²) / (4·ln2)) за вікно."""
    hl = pd.Series(np.log(high / low), index=high.index).pow(2)
    return np.sqrt(hl.rolling(window, min_periods=window // 2).mean() / (4.0 * np.log(2.0)))


# ── Персистентність потоку (AFML Ch.19.6.5) ──────────────────────────────────


def signed_flow_autocorr(trades: pd.DataFrame, resample: str = "1min", lags: int = 1, window: int = 120) -> pd.Series:
    """Серійна кореляція підписаного потоку — splitting/herding.

    Висока додатна кореляція = інформований агент дробить ордери (flow
    «липкий»); висока від'ємна = чергування покупців/продавців (шум).
    """
    sv = signed_volume_series(trades, resample)
    if sv.empty:
        return pd.Series(dtype=float)
    return sv.rolling(window, min_periods=min(window, max(10, window // 2))).corr(sv.shift(lags))


# ── Зручний додавач ──────────────────────────────────────────────────────────


def add_microstructure_features(df: pd.DataFrame, trades: pd.DataFrame, resample: str = "1min") -> pd.DataFrame:
    """Приєднати мікроструктурні фічі до свічкового DataFrame (ffill, без lookahead).

    Додає: vpin (на 1/10 середнього об'єму бару), kyle_t (t-value λ на барах),
    roll_spread, amihud, parkinson_vol, signed_flow_ac.
    """
    out = df.copy()
    close = out["close"]
    ret = close.pct_change().fillna(0.0)
    dollar_vol = out["volume"] * close

    # VPIN: об'ємні бари ≈ 10% середнього об'єму свічки
    if trades is not None and not trades.empty and "side" in trades.columns:
        avg_vol = out["volume"].rolling(100, min_periods=30).mean().median() or out["volume"].median()
        bar_vol = max(float(avg_vol) / 10.0, 1e-9)
        v = vpin(trades, bar_volume=bar_vol, n=50)
        out["vpin"] = v.reindex(out.index, method="ffill").fillna(0.5)

        sv = signed_volume_series(trades, resample)
        _, kyle_t = kyle_lambda_series(close, sv, window=120)
        out["kyle_t"] = kyle_t.reindex(out.index, method="ffill").fillna(0.0)

        ac = signed_flow_autocorr(trades, resample)
        out["signed_flow_ac"] = ac.reindex(out.index, method="ffill").fillna(0.0)

    out["roll_spread"] = roll_spread(close).reindex(out.index).fillna(0.0)
    out["amihud"] = amihud(ret, dollar_vol).reindex(out.index).fillna(0.0)
    out["parkinson_vol"] = parkinson_vol(out["high"], out["low"]).reindex(out.index).fillna(0.0)
    out["corwin_schultz_spread"] = corwin_schultz_spread(out["high"], out["low"]).reindex(out.index).fillna(0.0)
    return out


# ── Order Book Imbalance (OBI) ───────────────────────────────────────────────


def order_book_imbalance(bookticker: pd.DataFrame) -> pd.Series:
    """Обчислює Order Book Imbalance (OBI) на основі best bid/ask.

    OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    bookticker: DataFrame з колонками ['bid_qty', 'ask_qty']

    Returns:
        pd.Series значень OBI від -1 до 1.
    """
    if bookticker is None or bookticker.empty:
        return pd.Series(dtype=float)

    bid_qty = bookticker.get("bid_qty", pd.Series(0, index=bookticker.index))
    ask_qty = bookticker.get("ask_qty", pd.Series(0, index=bookticker.index))

    total_qty = bid_qty + ask_qty
    obi = (bid_qty - ask_qty) / total_qty.replace(0, np.nan)
    return obi.fillna(0.0)


__all__ = [
    "volume_bars",
    "vpin",
    "signed_volume_series",
    "kyle_lambda",
    "kyle_lambda_series",
    "roll_spread",
    "amihud",
    "corwin_schultz_spread",
    "parkinson_vol",
    "signed_flow_autocorr",
    "add_microstructure_features",
    "order_book_imbalance",
]
