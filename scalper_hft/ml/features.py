"""Побудова labeled-датасету для ML.

Без lookahead: фічі на барі t (закриття t) → таргет напрямку від t до t+h.
Таргет зважений на волатильність: якщо рух менший за noise_threshold —
приклад відкидається (важко прогнозувати шум).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.features.indicators import add_standard_features, cvd_from_trades

_FEATURE_COLS = [
    "rsi_14",
    "ema_9",
    "ema_21",
    "ema_50",
    "atr_14",
    "bb_width",
    "vwap_20",
    "realized_vol_30",
    "ret_1",
    "ret_5",
    "vol_ratio",
    "cvd_mom",
    "buy_ratio",
]


def build_labeled_dataset(
    df: pd.DataFrame,
    horizon: int = 3,
    noise_threshold: float = 0.0005,
    trades: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Повертає (X, y): фічі та бінарний таргет напрямку через `horizon` барів.

    y = +1, якщо close[t+h] / close[t] − 1 > noise_threshold
    y = −1, якщо close[t+h] / close[t] − 1 < −noise_threshold
    y = 0  (відкидається), якщо рух у межах шуму.
    """
    f = add_standard_features(df)
    if trades is not None and not trades.empty:
        cvd = cvd_from_trades(trades, resample=_infer(df))
        f = f.join(cvd[["cvd_mom", "buy_ratio"]], how="left")
        f["cvd_mom"] = f["cvd_mom"].ffill().fillna(0.0)
        f["buy_ratio"] = f["buy_ratio"].ffill().fillna(0.5)
    else:
        f["cvd_mom"] = 0.0
        f["buy_ratio"] = 0.5

    close = df["close"]
    fwd = close.shift(-horizon) / close - 1.0
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd > noise_threshold] = 1
    y[fwd < -noise_threshold] = -1

    X = f[_FEATURE_COLS].iloc[:-horizon]
    y = y.iloc[:-horizon]
    mask = y != 0
    return X[mask], y[mask]


def _infer(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "1min"
    delta = df.index[1] - df.index[0]
    minutes = delta.total_seconds() / 60.0
    return f"{int(minutes)}min" if minutes >= 1 else f"{int(max(1, minutes * 60))}s"
