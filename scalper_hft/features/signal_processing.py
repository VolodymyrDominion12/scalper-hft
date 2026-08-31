"""Signal Processing: Kalman Filter & Wavelet Denoising.
Використовується для фільтрації мікроструктурного шуму
та оцінки справедливої ціни без внесення значної затримки.
"""

import numpy as np
import pandas as pd


class SimpleKalmanFilter:
    """Простий 1D Фільтр Калмана для трекінгу ціни (mid-price estimation)."""

    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-4):
        self.q = process_variance  # Q: Дисперсія шуму процесу
        self.r = measurement_variance  # R: Дисперсія шуму вимірювань
        self.p = 1.0  # Початкова дисперсія помилки оцінки
        self.x: float | None = None  # Поточна оцінка стану (ціни)

    def update(self, measurement: float) -> float:
        """Оновлює стан фільтра на основі нового вимірювання (ціни)."""
        if self.x is None:
            self.x = measurement
            return self.x

        # Prediction update
        # State prediction: x = x (ми припускаємо модель random walk для ціни)
        p_pred = self.p + self.q

        # Measurement update
        # Kalman Gain
        k = p_pred / (p_pred + self.r)

        # State update
        self.x = self.x + k * (measurement - self.x)

        # Error covariance update
        self.p = (1 - k) * p_pred

        return self.x


def apply_kalman_filter(
    series: pd.Series, process_variance: float = 1e-5, measurement_variance: float = 1e-4
) -> pd.Series:
    """Застосовує фільтр Калмана до всього часового ряду."""
    kf = SimpleKalmanFilter(process_variance, measurement_variance)
    filtered = series.apply(kf.update)
    return pd.Series(filtered, index=series.index, name="kalman_filtered")


def apply_wavelet_denoising(series: pd.Series, wavelet: str = "db4", level: int = 1) -> pd.Series:
    """Зменшує шум часового ряду за допомогою дискретного вейвлет-перетворення (DWT).
    Використовує PyWavelets (pywt).
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("Потрібно встановити PyWavelets (pywt) для wavelet_denoising")

    # Розклад на коефіцієнти
    coeffs = pywt.wavedec(series.values, wavelet, mode="per", level=level)

    # Видалення високочастотних деталей (останній рівень коефіцієнтів = 0)
    # Зберігаємо апроксимацію і відкидаємо деталі
    coeffs[1:] = [np.zeros_like(v) for v in coeffs[1:]]

    # Відновлення сигналу
    reconstructed = pywt.waverec(coeffs, wavelet, mode="per")

    # Якщо довжина не співпадає через padding
    if len(reconstructed) > len(series):
        reconstructed = reconstructed[: len(series)]

    return pd.Series(reconstructed, index=series.index, name=f"wavelet_{wavelet}")
