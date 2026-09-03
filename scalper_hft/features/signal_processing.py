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


# ── 2D Kalman Filter для динамічного hedge ratio пар ─────────────────────────


class KalmanHedgeRatio:
    """2D State-Space Фільтр Калмана для динамічного трекінгу коефіцієнта хеджування.

    Модель спостереження:
        y_t = beta_t * x_t + alpha_t + e_t
        де y_t = ln(P1_t), x_t = ln(P2_t), e_t ~ N(0, R)
    Модель переходу стану:
        theta_t = [beta_t, alpha_t]^T = theta_{t-1} + w_t, w_t ~ N(0, Q)
    """

    def __init__(
        self,
        q_beta: float = 1e-5,
        q_alpha: float = 1e-5,
        r: float = 1e-3,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
    ) -> None:
        self.q = np.array([[q_beta, 0.0], [0.0, q_alpha]], dtype=float)
        self.r = float(r)
        self.theta = np.array([initial_beta, initial_alpha], dtype=float)  # [beta, alpha]
        self.p = np.eye(2, dtype=float) * 1.0

    def update(self, y: float, x: float) -> tuple[float, float, float]:
        """Оновлює стан для спостереження (y, x).

        Повертає (beta, alpha, spread), де spread = y - (beta * x + alpha).
        """
        # Time update (Prediction)
        theta_pred = self.theta
        p_pred = self.p + self.q

        # Measurement matrix H = [x, 1.0]
        h = np.array([x, 1.0], dtype=float)

        # Innovation / residual
        y_hat = float(np.dot(h, theta_pred))
        error = float(y - y_hat)

        # Innovation covariance
        s = float(np.dot(h, np.dot(p_pred, h)) + self.r)

        # Kalman gain
        k = np.dot(p_pred, h) / s

        # Measurement update (Correction)
        self.theta = theta_pred + k * error
        self.p = (np.eye(2, dtype=float) - np.outer(k, h)) @ p_pred

        beta, alpha = float(self.theta[0]), float(self.theta[1])
        spread = float(y - (beta * x + alpha))
        return beta, alpha, spread


def dynamic_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    q_beta: float = 1e-5,
    q_alpha: float = 1e-5,
    r: float = 1e-3,
) -> pd.DataFrame:
    """Обчислює динамічні бета, альфа та спред за 2D фільтром Калмана для двох рядів.

    Повертає DataFrame з колонками: beta, alpha, spread.
    Сигнали не мають lookahead (кожна точка використовує лише дані до t).
    """
    common = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if common.empty:
        return pd.DataFrame(columns=["beta", "alpha", "spread"], index=y.index)

    kf = KalmanHedgeRatio(q_beta=q_beta, q_alpha=q_alpha, r=r)
    n = len(common)
    betas = np.empty(n, dtype=float)
    alphas = np.empty(n, dtype=float)
    spreads = np.empty(n, dtype=float)

    y_vals = common["y"].values
    x_vals = common["x"].values

    for i in range(n):
        b, a, s = kf.update(float(y_vals[i]), float(x_vals[i]))
        betas[i] = b
        alphas[i] = a
        spreads[i] = s

    res = pd.DataFrame({"beta": betas, "alpha": alphas, "spread": spreads}, index=common.index)
    return res.reindex(y.index).ffill()


def estimate_half_life(spread: pd.Series, min_obs: int = 20) -> float:
    """Оцінює Ornstein-Uhlenbeck Half-Life: tau = -ln(2) / theta.

    theta отримується з OLS: Delta(spread_t) = theta * spread_{t-1} + const.
    Повертає inf, якщо процес не демонструє повернення до середнього (theta >= 0).
    """
    s = spread.dropna()
    if len(s) < min_obs:
        return float("inf")
    x = s.shift(1).iloc[1:].values
    y = s.diff().iloc[1:].values
    if np.std(x) < 1e-12:
        return float("inf")
    b = float(np.polyfit(x, y, 1)[0])
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)

