import pytest
import numpy as np
import pandas as pd

from scalper_hft.features.fractional_diff import frac_diff_ffd, get_weights_ffd
from scalper_hft.features.signal_processing import apply_kalman_filter

def test_fractional_diff():
    # Створюємо простий тренд
    np.random.seed(42)
    x = np.cumsum(np.random.randn(100) + 0.1)
    series = pd.Series(x)
    
    # Застосовуємо frac diff
    diff_series = frac_diff_ffd(series, d=0.5, thres=1e-3)
    
    # Початок буде NaN, але далі мають бути значення
    assert diff_series.isna().sum() > 0
    assert not diff_series.dropna().empty
    
def test_kalman_filter():
    x = np.linspace(0, 10, 100) + np.random.randn(100) * 0.5
    series = pd.Series(x)
    
    filtered = apply_kalman_filter(series, process_variance=1e-2, measurement_variance=1.0)
    
    assert len(filtered) == len(series)
    # Фільтрований ряд має мати меншу дисперсію різниць (бути більш плавним)
    assert filtered.diff().var() < series.diff().var()
