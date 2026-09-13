"""Hawkes Processes для оцінки мікроструктурних властивостей та токсичності потоку.

Використовує само-збуджувальні точкові процеси для моделювання інтенсивності 
потоку ордерів. Корисне для визначення кластеризації угод та дисбалансу агресорів.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hawkes_intensity(
    timestamps: pd.Series,
    alpha: float = 0.1,
    beta: float = 0.5,
) -> pd.Series:
    """Обчислює інтенсивність Hawkes процесу для серії подій (наприклад, угод).
    
    Модель: λ(t) = μ + h(t)
    де h(t_i) = h(t_{i-1}) * exp(-β * (t_i - t_{i-1})) + α
    
    Args:
        timestamps: Series з мітками часу подій (в секундах або частках секунд).
                    Важливо щоб beta відповідав шкалі часу.
        alpha: стрибок інтенсивності при кожній події (jump size).
        beta: швидкість затухання (decay rate).
        
    Returns:
        pd.Series значень інтенсивності (само-збуджувальної компоненти h(t)) 
        на момент кожної події (після її врахування).
    """
    ts = timestamps.astype(float).values
    if len(ts) == 0:
        return pd.Series(dtype=float, index=timestamps.index)
        
    h = np.zeros(len(ts))
    h[0] = alpha
    
    for i in range(1, len(ts)):
        dt = max(ts[i] - ts[i - 1], 0.0)
        h[i] = h[i - 1] * np.exp(-beta * dt) + alpha
        
    return pd.Series(h, index=timestamps.index)


def order_flow_toxicity_hawkes(
    trades: pd.DataFrame,
    alpha: float = 0.1,
    beta: float = 0.5,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Обчислює Hawkes інтенсивності для покупок і продажів для оцінки токсичності.
    
    Toxic flow виникає, коли інтенсивність покупок значно перевищує інтенсивність продажів
    або навпаки, вказуючи на спрямований агресивний потік (наприклад, інформовані трейдери).
    
    Args:
        trades: DataFrame угод, повинен мати колонки 'side' та (опціонально) 'timestamp'/'time'.
                Якщо time_col не вказано, використовується index (має бути datetime/numeric).
        alpha: стрибок інтенсивності.
        beta: швидкість затухання.
    
    Returns:
        DataFrame з колонками 'hawkes_buy', 'hawkes_sell', 'hawkes_imbalance'.
    """
    if trades is None or trades.empty or "side" not in trades.columns:
        return pd.DataFrame()
        
    df = trades.copy()
    if time_col and time_col in df.columns:
        ts = df[time_col].astype(float)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index.astype(int) / 10**9, index=df.index)  # в секунди
    else:
        ts = pd.Series(df.index.astype(float), index=df.index)
        
    buy_mask = df["side"] == "buy"
    sell_mask = df["side"] == "sell"
    
    h_buy = hawkes_intensity(ts[buy_mask], alpha, beta)
    h_sell = hawkes_intensity(ts[sell_mask], alpha, beta)
    
    out = pd.DataFrame(index=df.index)
    out.loc[buy_mask, "hawkes_buy"] = h_buy.values
    out.loc[sell_mask, "hawkes_sell"] = h_sell.values
    
    # ffill наближення: для потоку з високою частотою це прийнятно.
    out["hawkes_buy"] = out["hawkes_buy"].ffill().fillna(0.0)
    out["hawkes_sell"] = out["hawkes_sell"].ffill().fillna(0.0)
    
    tot = out["hawkes_buy"] + out["hawkes_sell"]
    out["hawkes_imbalance"] = (out["hawkes_buy"] - out["hawkes_sell"]) / tot.replace(0.0, np.nan)
    out["hawkes_imbalance"] = out["hawkes_imbalance"].fillna(0.0)
    
    return out


__all__ = ["hawkes_intensity", "order_flow_toxicity_hawkes"]
