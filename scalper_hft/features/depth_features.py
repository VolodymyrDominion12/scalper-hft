"""Фічі з глибини стакана (depth5 snapshots, рекордер bookticker_recorder).

Джерело даних: `data/{SYMBOL}_depth5.parquet` — снапшоти top-5 bid/ask
(колонки bid1..bid5, ask1..ask5 + _qty, індекс ts, UTC, ~100 мс),
накопичуються на VPS (systemd scalper-record@*.service).

Функції:
    depth_imbalance        — depth-weighted order book imbalance ∈ [-1, 1];
    top_of_book_imbalance  — OBI лише за найкращою парою (для порівняння);
    depth_spread_bps       — (ask1 − bid1)/mid у базисних пунктах;
    snapshot_quality       — швидкий звіт якості снапшотів (rate/gaps).

Без lookahead: усі фічі рахуються за рядком снапшота t (лише дані ≤ t).
Рівні, яких бракує (qty = NaN), трактуються як 0 обсягу.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BID_QTY = [f"bid{i}_qty" for i in range(1, 6)]
_ASK_QTY = [f"ask{i}_qty" for i in range(1, 6)]
_BID_PX = [f"bid{i}" for i in range(1, 6)]
_ASK_PX = [f"ask{i}" for i in range(1, 6)]


def _level_weights(levels: int, scheme: str = "linear") -> np.ndarray:
    """Ваги рівнів: uniform | linear | sqrt (глибші рівні = менша вага)."""
    idx = np.arange(1, levels + 1, dtype=float)
    if scheme == "uniform":
        w = np.ones(levels)
    elif scheme == "linear":
        w = levels - idx + 1.0
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(idx)
    else:
        raise ValueError(f"невідома схема ваг: {scheme}")
    return w / w.sum()


def depth_imbalance(depth: pd.DataFrame, levels: int = 5, weights: str = "linear") -> pd.Series:
    """Depth-weighted order book imbalance ∈ [-1, 1].

    imb = (Σ w_i·bid_qty_i − Σ w_i·ask_qty_i) / (Σ w_i·(bid_qty_i + ask_qty_i))
    Додатне = тиск покупців. levels ≤ 5 (дані depth5). Каузально, рядково.
    """
    if depth is None or depth.empty:
        return pd.Series(dtype=float)
    levels = max(1, min(int(levels), 5))
    bq = depth.reindex(columns=[c for c in _BID_QTY[:levels] if c in depth.columns]).fillna(0.0)
    aq = depth.reindex(columns=[c for c in _ASK_QTY[:levels] if c in depth.columns]).fillna(0.0)
    w = _level_weights(levels, weights)
    wb = (bq.values * w[: bq.shape[1]]).sum(axis=1)
    wa = (aq.values * w[: aq.shape[1]]).sum(axis=1)
    denom = wb + wa
    out = np.divide(wb - wa, denom, out=np.zeros_like(denom), where=denom > 0)
    return pd.Series(out, index=depth.index)


def top_of_book_imbalance(depth: pd.DataFrame) -> pd.Series:
    """OBI лише за level-1 (bid1_qty vs ask1_qty) — baseline для порівняння."""
    return depth_imbalance(depth, levels=1, weights="uniform")


def depth_spread_bps(depth: pd.DataFrame) -> pd.Series:
    """(ask1 − bid1)/mid у б.п. (1e4). NaN, якщо немає level-1."""
    if depth is None or depth.empty:
        return pd.Series(dtype=float)
    bid1 = depth.get("bid1", pd.Series(np.nan, index=depth.index)).astype(float)
    ask1 = depth.get("ask1", pd.Series(np.nan, index=depth.index)).astype(float)
    mid = (bid1 + ask1) / 2.0
    spread = (ask1 - bid1) / mid * 1e4
    return spread.where(mid > 0)


def snapshot_quality(depth: pd.DataFrame) -> dict:
    """Швидкий звіт якості снапшотів: кількість, span, rate, найбільший геп."""
    if depth is None or depth.empty:
        return {"rows": 0, "span_sec": 0.0, "rows_per_sec": 0.0, "max_gap_sec": 0.0, "duplicates": 0}
    idx = depth.index
    ts = pd.Series(np.asarray(idx, dtype="datetime64[ns]"), index=idx)
    diff = ts.diff().dropna().dt.total_seconds()
    span = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())
    return {
        "rows": int(len(depth)),
        "span_sec": round(span, 1),
        "rows_per_sec": round(len(depth) / span, 2) if span > 0 else 0.0,
        "max_gap_sec": round(float(diff.max()), 2) if len(diff) else 0.0,
        "duplicates": int(idx.duplicated().sum()),
    }


__all__ = ["depth_imbalance", "top_of_book_imbalance", "depth_spread_bps", "snapshot_quality"]
