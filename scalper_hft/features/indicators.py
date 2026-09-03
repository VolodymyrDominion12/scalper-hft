"""Фічі стратегій: OHLCV-індикатори + мікроструктурні фічі.

Мікроструктурні фічі (книга, гл. 14–15; дослідження 2023–25):
    - CVD (cumulative volume delta) — агресія покупців/продавців з aggTrades;
    - Order Book Imbalance — дисбаланс глибини стакана (з bookTicker/снапшотів);
    - spread — маржа між кращим bid/ask;
    - rolled волатильність, ATR — фільтр режиму ринку.

Усі функції повертають Series/DataFrame, вирівняні за індексом вхідних даних,
без lookahead (використовують лише минуле та поточний бар).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── OHLCV-індикатори ─────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + num_std * std, "bb_low": mid - num_std * std})


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Зважена за об'ємом середня ціна у ковзному вікні (без lookahead)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.rolling(window, min_periods=window).sum() / df["volume"].rolling(window, min_periods=window).sum()


def realized_vol(close: pd.Series, window: int = 30, bars_per_year: float | None = None) -> pd.Series:
    """Річна волатильність лог-прибутковостей у вікні.

    bars_per_year: кількість барів у році для ануалізації. Якщо None,
    визначається автоматично з частоти DatetimeIndex, або fallback на 525 600 (1m).
    """
    log_ret = np.log(close / close.shift(1))
    if bars_per_year is None:
        if isinstance(close.index, pd.DatetimeIndex) and len(close) > 1:
            diffs = close.index.to_series().diff().dropna()
            if len(diffs) and diffs.gt(pd.Timedelta(0)).all():
                delta_s = diffs.median().total_seconds()
                bars_per_year = (365.0 * 24.0 * 3600.0) / max(delta_s, 1.0)
            else:
                bars_per_year = 365.0 * 24.0 * 60.0
        else:
            bars_per_year = 365.0 * 24.0 * 60.0
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(bars_per_year)


# ── Мікроструктурні фічі ─────────────────────────────────────────────────────
def cvd_from_trades(trades: pd.DataFrame, resample: str = "1min") -> pd.DataFrame:
    """Cumulative Volume Delta з aggTrades.

    delta = об'єм покупок − об'єм продажів за період; cvd = кумулятивна сума.
    Вхід: DataFrame з колонками price/amount/side (side ∈ {buy, sell}).
    """
    if trades.empty or "side" not in trades.columns:
        raise ValueError("trades має містити колонку 'side'")
    df = trades.copy()
    df["delta"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    g = df["delta"].resample(resample).sum().fillna(0.0)
    vol = df["amount"].resample(resample).sum().fillna(0.0)
    out = pd.DataFrame({"delta": g, "volume": vol})
    out["cvd"] = out["delta"].cumsum()
    # ковзні нормовані агрегації
    out["cvd_mom"] = out["cvd"].diff(12)
    out["buy_ratio"] = (out["delta"] + out["volume"]) / (2 * out["volume"].replace(0, np.nan))
    return out


def resample_trades_to_klines(trades: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    """Приєднати per-bar CVD-фічі до свічкового DataFrame."""
    cvd = cvd_from_trades(trades, resample=_infer_resample(klines))
    merged = klines.join(cvd[["cvd", "cvd_mom", "buy_ratio"]], how="left")
    merged["cvd"] = merged["cvd"].ffill()
    merged["cvd_mom"] = merged["cvd_mom"].ffill()
    merged["buy_ratio"] = merged["buy_ratio"].ffill()
    return merged


def _infer_resample(klines: pd.DataFrame) -> str:
    if len(klines) < 2:
        return "1min"
    delta = klines.index[1] - klines.index[0]
    minutes = delta.total_seconds() / 60.0
    if minutes < 1:
        return f"{int(max(1, minutes * 60))}s"
    return f"{int(minutes)}min"


def order_book_imbalance(bid_depth: pd.Series, ask_depth: pd.Series) -> pd.Series:
    """Дисбаланс стакана: (bid − ask) / (bid + ask) ∈ [−1, 1].

    > 0 — тиск покупців; < 0 — тиск продавців. Застосовується до снапшотів
    bookTicker (best bid/ask depth) або глибших рівнів стакана.
    """
    denom = (bid_depth + ask_depth).replace(0.0, np.nan)
    return ((bid_depth - ask_depth) / denom).fillna(0.0)


def relative_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Відносний спред: (ask − bid) / mid."""
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid.replace(0.0, np.nan)).fillna(0.0)


def zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """z-оцінка у ковзному вікні."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).fillna(0.0)


# ── Depth-фічі (з depth5/bookTicker записів) ────────────────────────────────
def weighted_depth_imbalance(
    depth: pd.DataFrame,
    levels: int = 5,
    weight_scheme: str = "harmonic",
) -> pd.Series:
    """Зважений дисбаланс глибини стакана.

    depth: DataFrame з колонками bid{i}, bid{i}_qty, ask{i}, ask{i}_qty (i=1..levels)
           або bookTicker-формат (bid/bid_qty/ask/ask_qty → 1 рівень).
    weight_scheme:
        'harmonic' — ваги 1/i (дальші рівні менш значущі);
        'equal'    — всі рівні рівні;
        'linear'   — ваги (levels-i+1).
    Стабільніший за top-of-book imbalance: усереднює стіни по рівнях.
    """
    # bookTicker-формат → один рівень
    if "bid_qty" in depth.columns and "bid1_qty" not in depth.columns:
        return order_book_imbalance(depth["bid_qty"], depth["ask_qty"])

    bids: list[pd.Series] = []
    asks: list[pd.Series] = []
    for i in range(1, levels + 1):
        bcol, acol = f"bid{i}_qty", f"ask{i}_qty"
        if bcol not in depth.columns or acol not in depth.columns:
            break
        bids.append(depth[bcol].fillna(0.0))
        asks.append(depth[acol].fillna(0.0))
    if not bids:
        raise ValueError(f"Немає depth-колонок bid{{1..{levels}}}_qty у даних")

    n = len(bids)
    if weight_scheme == "harmonic":
        w = np.array([1.0 / (i + 1) for i in range(n)])
    elif weight_scheme == "linear":
        w = np.array([float(n - i) for i in range(n)])
    else:  # equal
        w = np.ones(n)
    w = w / w.sum()

    wbid = sum(b * wi for b, wi in zip(bids, w))
    wask = sum(a * wi for a, wi in zip(asks, w))
    return order_book_imbalance(wbid, wask)


def ema_smooth(series: pd.Series, span: int = 30) -> pd.Series:
    """EMA-згладжування (для шумних мікроструктурних фіч)."""
    return series.ewm(span=span, adjust=False).mean()


def add_depth_to_klines(
    df: pd.DataFrame,
    depth: pd.DataFrame,
    levels: int = 5,
    weight_scheme: str = "harmonic",
    smooth_span: int = 30,
    resample: str = "1min",
) -> pd.DataFrame:
    """Приєднати зважений depth-imbalance до свічкового DataFrame.

    depth-снапшоти усереднюються за період бару (resample), imbalance
    згладжується EMA і вирівнюється на klines через ffill (без lookahead:
    використовуються лише снапшоти до кінця бару).
    """
    if isinstance(depth.index, pd.RangeIndex) or not isinstance(depth.index, pd.DatetimeIndex):
        depth = depth.copy()
        depth.index = pd.to_datetime(depth.index)
    imb_raw = weighted_depth_imbalance(depth, levels=levels, weight_scheme=weight_scheme)
    imb_bar = imb_raw.resample(resample).mean().dropna()
    imb_smooth = ema_smooth(imb_bar, smooth_span)
    out = df.copy()
    out["imbalance"] = imb_smooth.reindex(out.index, method="ffill").fillna(0.0)
    out["imbalance_raw"] = imb_bar.reindex(out.index, method="ffill").fillna(0.0)
    return out


def add_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Канонічний набір фіч для свічкових стратегій (без lookahead)."""
    out = df.copy()
    close = out["close"]
    out["rsi_14"] = rsi(close, 14)
    out["ema_9"] = ema(close, 9)
    out["ema_21"] = ema(close, 21)
    out["ema_50"] = ema(close, 50)
    out["atr_14"] = atr(out, 14)
    bb = bollinger(close, 20, 2.0)
    out = out.join(bb)
    out["bb_width"] = (out["bb_up"] - out["bb_low"]) / out["bb_mid"].replace(0, np.nan)
    out["vwap_20"] = rolling_vwap(out, 20)
    out["realized_vol_30"] = realized_vol(close, 30)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(20, min_periods=20).mean().replace(0, np.nan)
    return out
