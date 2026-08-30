"""AFML Bar Generation: Volume, Dollar, and Tick Imbalance Bars.
Цей модуль реалізує створення альтернативних барів (замість часових)
на основі даних aggTrades (price, amount, side) згідно з
Advances in Financial Machine Learning (López de Prado).
"""

import numpy as np
import pandas as pd


def _compute_bars(trades: pd.DataFrame, threshold_col: str, threshold: float) -> pd.DataFrame:
    """Загальна функція для формування барів по досягненню порогу."""
    if trades.empty:
        return pd.DataFrame()

    # Створюємо копію та обчислюємо кумулятивні значення
    df = trades.copy()

    # Визначаємо індекси розривів (де кумулятивна сума перетинає поріг)
    df["cum_val"] = df[threshold_col].cumsum()
    df["group"] = (df["cum_val"] // threshold).astype(int)

    # Обчислюємо доларовий об'єм перед групуванням
    df["dollar_volume"] = df["price"] * df["amount"]

    # Агрегуємо дані по групах
    grouped = df.groupby("group")

    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["amount"].sum(),
            "dollar_volume": grouped["dollar_volume"].sum(),
        }
    )

    # Використовуємо timestamp останнього трейду в барі як індекс
    bars.index = grouped.apply(lambda x: x.index[-1])

    return bars


def create_volume_bars(trades: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    """Формує бари на основі фіксованого об'єму торгів (Volume Bars)."""
    return _compute_bars(trades, threshold_col="amount", threshold=volume_threshold)


def create_dollar_bars(trades: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """Формує бари на основі фіксованого доларового об'єму (Dollar Bars)."""
    trades_copy = trades.copy()
    trades_copy["dollar_val"] = trades_copy["price"] * trades_copy["amount"]
    return _compute_bars(trades_copy, threshold_col="dollar_val", threshold=dollar_threshold)


def create_tick_imbalance_bars(
    trades: pd.DataFrame, expected_imbalance_window: int = 1000, ewma_window: int = 100
) -> pd.DataFrame:
    """Формує Tick Imbalance Bars (TIB).

    TIB формуються коли кумулятивний імбаланс тіків (різниця між buy і sell)
    перевищує динамічний поріг, що залежить від очікуваного розміру бару
    та очікуваної ймовірності напрямку.
    """
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()

    # 1. Розрахунок Tick Rule (напрямок угоди: 1 якщо ціна зросла, -1 якщо впала, інакше попередній)
    df["price_diff"] = df["price"].diff()
    df["tick_rule"] = np.where(df["price_diff"] > 0, 1, np.where(df["price_diff"] < 0, -1, 0))
    # Заповнюємо нулі попереднім значенням (forward fill)
    df["tick_rule"] = df["tick_rule"].replace(0, np.nan).ffill().fillna(1)

    # Якщо є стовпець side (buy/sell), можна використовувати його замість tick_rule
    # Зазвичай у Binance: side == 'buy' -> 1, side == 'sell' -> -1.
    # Припускаємо, що tick_rule надійніше, якщо side не стандартизовано.

    bar_indices = []

    # Початкові оцінки
    expected_T = expected_imbalance_window  # очікувана кількість тіків у барі
    expected_imbalance = df["tick_rule"].ewm(span=ewma_window).mean()

    theta_t = 0.0
    last_bar_idx = 0

    for i in range(1, len(df)):
        theta_t += df["tick_rule"].iloc[i]

        # Динамічний поріг: E_0[T] * |2*P[b_t=1] - 1|
        # що еквівалентно E_0[T] * |E_0[b_t]|
        threshold = expected_T * abs(expected_imbalance.iloc[i])

        if abs(theta_t) >= threshold:
            bar_indices.append(i)
            # Оновлюємо очікувану довжину бару
            expected_T = (expected_T * ewma_window + (i - last_bar_idx)) / (ewma_window + 1)
            last_bar_idx = i
            theta_t = 0.0

    # Формуємо бари
    if not bar_indices:
        return pd.DataFrame()

    bars = []
    prev_idx = 0
    for idx in bar_indices:
        subset = df.iloc[prev_idx : idx + 1]
        bars.append(
            {
                "timestamp": subset.index[-1],
                "open": subset["price"].iloc[0],
                "high": subset["price"].max(),
                "low": subset["price"].min(),
                "close": subset["price"].iloc[-1],
                "volume": subset["amount"].sum(),
                "dollar_volume": (subset["price"] * subset["amount"]).sum(),
            }
        )
        prev_idx = idx + 1

    result = pd.DataFrame(bars)
    result.set_index("timestamp", inplace=True)
    return result
