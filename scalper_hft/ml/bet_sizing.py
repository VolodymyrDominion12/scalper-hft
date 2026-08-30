"""Bet sizing з передбачених імовірностей (AFML Chapter 10.3).

Замість фіксованого розміру позиції (±1) розмір ставки виводиться з
впевненості моделі. Формули (López de Prado, Ch.10.3):

    z = ln(p̂ / (1 − p̂))            # логіт передбаченої ймовірності
    m = 2·Φ(z) − 1 ∈ (−1, 1)        # Φ — CDF стандартного нормального

Властивості:
    p̂ = 0.5  → m = 0   (не торгуємо)
    p̂ → 1    → m → +1  (повний розмір)
    p̂ → 0    → m → −1  (повний протилежний розмір)

Додатково: дискретизація (Ch.10.5) — прибирає jitter-переторговку.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as _norm

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _norm = None
    _HAS_SCIPY = False


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Φ(z) без scipy: 0.5·(1 + erf(z/√2))."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def prob_to_size(p: np.ndarray | pd.Series, method: str = "logit") -> np.ndarray:
    """Мапа імовірності p̂ ∈ [0, 1] → розмір ставки m ∈ (−1, 1).

    Args:
        p: ймовірність «позитивного» результату (клас +1 / мета-мітка 1).
        method: 'logit' — z = ln(p/(1−p)); 'probit' — z = Φ⁻¹(p).
            За замовчуванням 'logit' (AFML Ch.10.3).

    Returns:
        m: розмір ставки у (−1, 1); знак = напрямок (для primary),
           для мета-лейблінгу береться abs(m).
    """
    arr = np.asarray(p, dtype=float)
    eps = 1e-6
    arr = np.clip(arr, eps, 1.0 - eps)
    if method == "probit":
        if not _HAS_SCIPY:
            raise ImportError("probit потребує scipy: uv add scipy")
        z = _norm.ppf(arr)
    else:  # logit
        z = np.log(arr / (1.0 - arr))
    if not _HAS_SCIPY:
        m = _norm_cdf(z) * 2.0 - 1.0
    else:
        m = 2.0 * _norm.cdf(z) - 1.0
    return m


def discretize(m: np.ndarray | pd.Series, step: float = 0.2) -> np.ndarray:
    """Дискретизація розміру ставки (AFML Ch.10.5): m_d = round(m/d)·d.

    Прибирає дрібний jitter у сигналі (зменшує turnover → комісії).
    """
    arr = np.asarray(m, dtype=float)
    if step <= 0:
        return arr
    return np.round(arr / step) * step


def meta_size(p_meta: np.ndarray | pd.Series, method: str = "logit") -> np.ndarray:
    """Розмір позиції з мета-моделі P(meta=1) ∈ [0, 1].

    Клас 0 у мета-лейблінгу = пропуск угоди (AFML Ch.3.7), тому:
        p_meta < 0.5 → m = 0 (не торгуємо);
        p_meta ≥ 0.5 → m = prob_to_size(p_meta) ∈ [0, 1).

    Returns:
        m ∈ [0, 1) — множник розміру; сторона задається primary-моделлю.
    """
    m = prob_to_size(np.asarray(p_meta, dtype=float), method=method)
    m = np.clip(m, 0.0, 1.0)
    m = np.where(np.asarray(p_meta, dtype=float) < 0.5, 0.0, m)
    return m


# ── Динамічний розмір і лімітна ціна (AFML Ch.10.6) ──────────────────────────


def sigmoid_size(x: np.ndarray | float, omega: float) -> np.ndarray:
    """Сигмоїдний розмір: m[ω,x] = 2/(1 + e^{−ωx}) − 1 ∈ (−1, 1).

    x = f_i − p_t — розбіжність прогнозу (fair value) і поточної ціни.
    x > 0 → лонг (ціна нижча за прогноз), x < 0 → шорт.
    """
    return 2.0 / (1.0 + np.exp(-omega * np.asarray(x, dtype=float))) - 1.0


def calibrate_omega(x_star: float, m_star: float) -> float:
    """Калібровка ω: при розбіжності x_star бажаємо розмір m_star.

    ω = ln((1+m*)/(1−m*)) / x* (AFML Ch.10.6): напр., m*=0.95 при x*=10.
    """
    if x_star == 0 or not (-1.0 < m_star < 1.0):
        raise ValueError("x_star ≠ 0 і −1 < m_star < 1")
    return float(np.log((1.0 + m_star) / (1.0 - m_star)) / x_star)


def target_size(Q: float, omega: float, fair_value: np.ndarray | float, price: np.ndarray | float) -> np.ndarray:
    """Цільовий розмір позиції: q* = Int[Q·m[ω, f − p]].

    Q — максимальний розмір (лотів/ноціоналу); m — сигмоїдний множник.
    """
    m = sigmoid_size(np.asarray(fair_value, dtype=float) - np.asarray(price, dtype=float), omega)
    return np.trunc(Q * m).astype(int)


def limit_price(omega: float, m: np.ndarray | float, fair_value: np.ndarray | float) -> np.ndarray:
    """Лімітна ціна для maker-ордера: L = f − (1/ω)·ln((1+m)/(1−m)).

    Це breakeven-рівень між ринком і прогнозом: коли ціна наближається до
    fair value, позиція закривається з прибутком без додаткових сигналів.
    """
    m = np.clip(np.asarray(m, dtype=float), -1.0 + 1e-9, 1.0 - 1e-9)
    return np.asarray(fair_value, dtype=float) - np.log((1.0 + m) / (1.0 - m)) / omega
