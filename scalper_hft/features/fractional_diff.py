"""Fractional Differentiation (AFML Chapter 5)
Реалізація фракційного диференціювання для стаціонаризації фіч 
з максимальним збереженням пам'яті часового ряду.
"""

import numpy as np
import pandas as pd


def get_weights(d: float, size: int) -> np.ndarray:
    """Отримує ваги для фракційного диференціювання.
    
    Args:
        d: Порядок диференціювання (0 < d < 1)
        size: Кількість ваг
    Returns:
        Масив ваг [w_0, w_1, ..., w_{size-1}]
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    
    # Реверс масиву для конволюції
    w = np.array(w)[::-1].reshape(-1, 1)
    return w


def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    """Отримує ваги для Fixed-Width Window Fractional Differentiation (FFD).
    Обчислює ваги, поки вони не стануть меншими за заданий поріг.
    
    Args:
        d: Порядок диференціювання (0 < d < 1)
        thres: Поріг відсікання ваг (напр. 1e-5)
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w)[::-1].reshape(-1, 1)
    return w


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Застосовує Fixed-Width Window Fractional Differentiation (FFD) до часового ряду.
    
    Args:
        series: Вхідний часовий ряд (ціни)
        d: Порядок диференціювання
        thres: Поріг для відсікання ваг
    Returns:
        pd.Series: Стаціонарний ряд із збереженням пам'яті
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    # Вирівнюємо ряд
    df = series.to_frame()
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    res = pd.Series(index=df.index, dtype=float)
    
    # Обчислюємо згортку (конволюцію)
    for i in range(width, len(df)):
        # Беремо вікно
        window = df.iloc[i - width:i + 1]
        # Скалярний добуток ваг і значень вікна
        val = np.dot(w.T, window)[0, 0]
        res.iloc[i] = val
        
    return res


def find_min_d(series: pd.Series, min_d: float = 0.0, max_d: float = 1.0, 
               step: float = 0.1, thres: float = 1e-5, p_val_thres: float = 0.05) -> float:
    """Знаходить мінімальне d, при якому ряд стає стаціонарним за тестом ADF.
    (Потребує statsmodels)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("Потрібно встановити statsmodels для використання find_min_d")
        
    for d in np.arange(min_d, max_d + step, step):
        df_d = frac_diff_ffd(series, d, thres)
        df_d = df_d.dropna()
        if len(df_d) < 10:
            continue
        
        res = adfuller(df_d.values, maxlag=1, regression='c', autolag=None)
        p_val = res[1]
        
        if p_val < p_val_thres:
            return d
            
    return max_d
