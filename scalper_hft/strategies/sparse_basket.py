"""Sparse Mean-Reverting Basket Arbitrage (FSPML Ch. 3 — Akansu et al.).

Побудова розрідженого кошика активів (Sparse Mean-Reverting Basket):
    S_t = Σ w_i * Y_{i,t}

Максимізація швидкості повернення до середнього (Ornstein–Uhlenbeck θ)
шляхом знаходження мінімального власного вектора коваріації відносно дисперсії приростів
(Box–Tiao decomposition) з l1-регуляризацією для отримання розріджених ваг w.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy


def estimate_ou_parameters(spread: pd.Series) -> dict[str, float]:
    """Оцінка параметрів процесу Орнштейна–Уленбека: dS_t = θ(μ - S_t)dt + σ dW_t.

    Через регресію AR(1): S_t = a * S_{t-1} + b + ε_t
        θ = -ln(a) / Δt, μ = b / (1 - a), σ = std(ε) * sqrt(-2 ln(a) / (1 - a^2))
    """
    s = spread.dropna().values
    if len(s) < 30:
        return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "half_life": float("inf")}

    x = s[:-1]
    y = s[1:]
    a, b = np.polyfit(x, y, 1)

    if a >= 1.0 or a <= 0.0:
        return {"theta": 0.0, "mu": float(np.mean(s)), "sigma": float(np.std(s)), "half_life": float("inf")}

    theta = -float(np.log(a))
    mu = float(b / (1.0 - a))
    residuals = y - (a * x + b)
    sigma = float(np.std(residuals) * np.sqrt(-2.0 * np.log(a) / max(1.0 - a**2, 1e-6)))
    half_life = float(np.log(2.0) / max(theta, 1e-6))

    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": half_life}


def compute_sparse_basket_weights(
    prices: pd.DataFrame,
    sparsity_k: int = 3,
    l1_penalty: float = 0.05,
) -> pd.Series:
    """Знаходження розріджених ваг кошика з максимальною швидкістю mean-reversion.

    Використовує Box–Tiao декомпозицію: співвідношення коваріації рівнів до коваріації приростів
    з soft-thresholding для занулення шумових компонентів.
    """
    df = prices.dropna()
    if df.shape[1] < 2 or len(df) < 30:
        return pd.Series(1.0 / max(df.shape[1], 1), index=prices.columns)

    # Логарифмічні ціни
    log_p = np.log(df.values)
    log_p_centered = log_p - log_p.mean(axis=0)

    # Коваріація рівнів Gamma0 і приростів Delta
    gamma0 = np.cov(log_p_centered, rowvar=False) + np.eye(df.shape[1]) * 1e-6
    diffs = np.diff(log_p, axis=0)
    gamma_diff = np.cov(diffs, rowvar=False) + np.eye(df.shape[1]) * 1e-6

    # Власні вектори M = Gamma0^(-1) * Gamma_diff (найменше власне значення -> найсильніший mean-reversion)
    try:
        from scipy.linalg import eigh

        eigvals, eigvecs = eigh(gamma0, gamma_diff)
        w_raw = eigvecs[:, 0]  # перший компонент з найменшим eigval
    except Exception:
        w_raw = np.ones(df.shape[1])
        w_raw[1:] = -1.0 / (df.shape[1] - 1)

    # Soft thresholding (L1 sparsity)
    w_thresh = np.sign(w_raw) * np.maximum(np.abs(w_raw) - l1_penalty, 0.0)

    # Залишаємо top-k за абсолютною величиною
    if np.count_nonzero(w_thresh) > sparsity_k:
        top_k_indices = np.argsort(np.abs(w_thresh))[-sparsity_k:]
        mask = np.zeros_like(w_thresh, dtype=bool)
        mask[top_k_indices] = True
        w_thresh[~mask] = 0.0

    # Нормалізація суми абсолютних ваг
    norm = np.sum(np.abs(w_thresh))
    if norm > 1e-12:
        w_final = w_thresh / norm
    else:
        w_final = np.zeros(df.shape[1])
        w_final[0] = 1.0
        w_final[1] = -1.0
        w_final /= 2.0

    return pd.Series(w_final, index=prices.columns)


class SparseBasketArb(Strategy):
    """Стратегія арбітражу розрідженого кошика (Sparse Mean-Reverting Basket)."""

    name = "sparse_basket"

    param_space = {
        "lookback": (30, 200, 10),
        "entry_z": (1.0, 3.0, 0.2),
        "exit_z": (0.0, 0.8, 0.1),
        "sparsity_k": (2, 5, 1),
    }

    def __init__(
        self,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.2,
        sparsity_k: int = 3,
        l1_penalty: float = 0.05,
    ) -> None:
        super().__init__(
            lookback=int(lookback),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            sparsity_k=int(sparsity_k),
            l1_penalty=float(l1_penalty),
        )
        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.sparsity_k = int(sparsity_k)
        self.l1_penalty = float(l1_penalty)


    def generate_signals(
        self,
        df: pd.DataFrame,
        basket_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Генерує торгові сигнали по розрідженому кошику.

        Якщо передано basket_df (DataFrame цін кількох активів), розраховує
        динамічний спред кошика та z-score.
        """
        if basket_df is None or basket_df.empty:
            # Фолбек на одиночний ряд (mean-reversion на close)
            close = df["close"]
            ma = close.rolling(self.lookback, min_periods=20).mean()
            std = close.rolling(self.lookback, min_periods=20).std().replace(0, np.nan)
            z = (close - ma) / std
        else:
            weights = compute_sparse_basket_weights(
                basket_df.tail(self.lookback * 2),
                sparsity_k=self.sparsity_k,
                l1_penalty=self.l1_penalty,
            )
            # Спред кошика (rolling)
            log_prices = np.log(basket_df)
            spread = (log_prices * weights).sum(axis=1)
            ma = spread.rolling(self.lookback, min_periods=20).mean()
            std = spread.rolling(self.lookback, min_periods=20).std().replace(0, np.nan)
            z = (spread - ma) / std

        z = z.fillna(0.0)
        signal = pd.Series(0, index=df.index, dtype=int)

        # Лонг спреду коли z < -entry_z, Шорт спреду коли z > entry_z
        pos = 0
        signals_list = []
        for val in z.values:
            if pos == 0:
                if val <= -self.entry_z:
                    pos = 1
                elif val >= self.entry_z:
                    pos = -1
            elif pos == 1 and val >= -self.exit_z:
                pos = 0
            elif pos == -1 and val <= self.exit_z:
                pos = 0
            signals_list.append(pos)

        signal.iloc[:] = signals_list
        return signal
